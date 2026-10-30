"""Evidence-filtered, dual-gated representation revision.

The v0.1 controller only checked whether the *new* task improved. That is not
enough: an update can improve the new task while destroying earlier competence.
This module adds three mechanisms on top of the same fit/rollback primitive:

* :class:`EvidenceValidator` rejects records that are inconsistent with the
  learner's current predictions using a robust (median/MAD) residual scale.
* :class:`RetentionRevisionController` gates an update on **both** new-task
  improvement and a predeclared old-task retention budget.
* Every attempt is recorded as a structured, JSON-serialisable revision event.

The legacy :class:`~genesis.learning.revision.RevisionController` is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from genesis.experience.observation import select_labeled
from genesis.experience.schema import Observation
from genesis.learning.learner import Learner

_MAD_TO_STD = 1.4826


class EvidenceConfig(BaseModel):
    """Thresholds for the evidence filter, fixed before a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_residual_z: float = Field(default=6.0, gt=0)
    min_batch_size: int = Field(default=3, ge=1)
    min_accepted_fraction: float = Field(default=0.5, ge=0, le=1)
    scale_floor: float = Field(default=1e-6, gt=0)

    @field_validator("min_accepted_fraction")
    @classmethod
    def _fraction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("min_accepted_fraction must be within [0, 1]")
        return value


class RetentionPolicy(BaseModel):
    """Predeclared acceptance policy for one revision attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=50, ge=1)
    min_new_task_improvement: float = Field(default=1e-6, ge=0)
    old_task_max_ratio: float = Field(default=4.0, ge=1.0)
    old_task_mse_floor_fraction: float = Field(default=0.20, ge=0.0)
    min_accepted_fraction: float = Field(default=0.5, ge=0, le=1)


@dataclass
class EvidenceReport:
    """Outcome of filtering an update batch against current predictions."""

    accepted: list[Observation] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    n_total: int = 0
    residual_center: float = 0.0
    residual_scale: float = 0.0
    sufficient: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    @property
    def accepted_fraction(self) -> float:
        return self.n_accepted / self.n_total if self.n_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_accepted": self.n_accepted,
            "n_rejected": len(self.rejected),
            "accepted_fraction": self.accepted_fraction,
            "residual_center": self.residual_center,
            "residual_scale": self.residual_scale,
            "sufficient": self.sufficient,
            "reasons": list(self.reasons),
            "rejected": list(self.rejected),
        }


class EvidenceValidator:
    """Robust outlier filter for proposed update records.

    Residuals are computed with the *current* learner. The scale is the median
    absolute deviation, so a small number of corrupted labels cannot inflate
    the threshold that is supposed to catch them. Batches smaller than
    ``min_batch_size`` are treated as insufficient evidence and rejected.
    """

    def __init__(self, config: EvidenceConfig | None = None) -> None:
        self.config = config or EvidenceConfig()

    def validate(
        self, learner: Learner, records: list[Observation] | tuple[Observation, ...]
    ) -> EvidenceReport:
        labeled = select_labeled(records)
        report = EvidenceReport(n_total=len(labeled))
        if not labeled:
            report.reasons.append("no labeled records")
            return report

        features = learner.encoder.transform(labeled)
        predictions = learner.predict(features)
        outcomes = np.asarray([record.outcome for record in labeled], dtype=float)
        residuals = outcomes - predictions

        if not np.all(np.isfinite(residuals)):
            report.reasons.append("non-finite residuals")
            return report

        center = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - center)))
        scale = max(_MAD_TO_STD * mad, self.config.scale_floor)
        report.residual_center = center
        report.residual_scale = scale

        if len(labeled) < self.config.min_batch_size:
            report.reasons.append(
                f"batch smaller than min_batch_size={self.config.min_batch_size}"
            )
            return report

        z_scores = np.abs(residuals - center) / scale
        for record, residual, z_score in zip(labeled, residuals, z_scores, strict=True):
            if z_score <= self.config.max_residual_z:
                report.accepted.append(record)
            else:
                report.rejected.append(
                    {
                        "id": record.id,
                        "residual": float(residual),
                        "z_score": float(z_score),
                    }
                )

        report.sufficient = report.accepted_fraction >= self.config.min_accepted_fraction
        if not report.sufficient:
            report.reasons.append(
                "accepted fraction below configured minimum "
                f"({report.accepted_fraction:.3f} < {self.config.min_accepted_fraction:.3f})"
            )
        return report


@dataclass
class RetentionRevisionResult:
    """Auditable result of one dual-gated revision attempt."""

    accepted: bool
    reason: str
    before_new: float
    after_new: float
    before_old: float
    after_old: float
    retention_budget: float
    old_outcome_variance: float
    evidence: EvidenceReport
    policy: RetentionPolicy
    n_update_samples: int
    n_replay_samples: int
    state_restored: bool
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def new_task_relative_improvement(self) -> float:
        if self.before_new <= 0:
            return 0.0 if self.after_new <= self.before_new else -np.inf
        return (self.before_new - self.after_new) / self.before_new

    @property
    def old_task_relative_change(self) -> float:
        if self.before_old <= 0:
            return 0.0 if self.after_old <= self.before_old else np.inf
        return (self.after_old - self.before_old) / self.before_old

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "before_new": self.before_new,
            "after_new": self.after_new,
            "before_old": self.before_old,
            "after_old": self.after_old,
            "new_task_relative_improvement": self.new_task_relative_improvement,
            "old_task_relative_change": self.old_task_relative_change,
            "retention_budget": self.retention_budget,
            "old_outcome_variance": self.old_outcome_variance,
            "n_update_samples": self.n_update_samples,
            "n_replay_samples": self.n_replay_samples,
            "state_restored": self.state_restored,
            "evidence": self.evidence.to_dict(),
            "policy": self.policy.model_dump(mode="json"),
            "details": dict(self.details),
        }


class RetentionRevisionController:
    """Dual-gated updates with evidence filtering, replay and rollback."""

    def __init__(
        self,
        policy: RetentionPolicy | None = None,
        evidence: EvidenceConfig | None = None,
    ) -> None:
        self.policy = policy or RetentionPolicy()
        self.validator = EvidenceValidator(evidence)
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _mse(learner: Learner, records: list[Observation]) -> float:
        features = learner.encoder.transform(records)
        predictions = learner.predict(features)
        outcomes = np.asarray([record.outcome for record in records], dtype=float)
        return float(np.mean((predictions - outcomes) ** 2))

    def controlled_update(
        self,
        learner: Learner,
        update_records: list[Observation] | tuple[Observation, ...],
        new_validation_records: list[Observation] | tuple[Observation, ...],
        old_validation_records: list[Observation] | tuple[Observation, ...],
        *,
        replay_records: list[Observation] | tuple[Observation, ...] | None = None,
    ) -> RetentionRevisionResult:
        if not learner.fitted:
            raise ValueError("learner must be fitted before controlled revision")

        new_validation = select_labeled(new_validation_records)
        old_validation = select_labeled(old_validation_records)
        if not new_validation or not old_validation:
            raise ValueError("both new-task and old-task validation records are required")

        evidence = self.validator.validate(learner, update_records)
        replay = select_labeled(replay_records or [])
        snapshot = learner.snapshot()

        before_new = self._mse(learner, new_validation)
        before_old = self._mse(learner, old_validation)
        old_outcomes = np.asarray(
            [record.outcome for record in old_validation], dtype=float
        )
        old_variance = float(np.var(old_outcomes))
        retention_budget = max(
            before_old * self.policy.old_task_max_ratio,
            self.policy.old_task_mse_floor_fraction * old_variance,
        )

        if not evidence.accepted:
            return self._finish(
                accepted=False,
                reason="update rejected before training: insufficient evidence",
                before_new=before_new,
                after_new=before_new,
                before_old=before_old,
                after_old=before_old,
                retention_budget=retention_budget,
                old_variance=old_variance,
                evidence=evidence,
                n_update=len(select_labeled(update_records)),
                n_replay=len(replay),
                restored=False,
                details={"gate": "evidence"},
            )

        training_records = [*evidence.accepted, *replay]
        learner.fit(training_records, warm_start=True, epochs=self.policy.epochs)
        after_new = self._mse(learner, new_validation)
        after_old = self._mse(learner, old_validation)

        new_improved = after_new < before_new - self.policy.min_new_task_improvement
        retention_ok = after_old <= retention_budget
        evidence_ok = evidence.accepted_fraction >= self.policy.min_accepted_fraction
        accepted = bool(new_improved and retention_ok and evidence_ok)

        if accepted:
            reasons = ["new task improved and retention budget respected"]
        else:
            reasons = []
            if not new_improved:
                reasons.append("new-task validation did not improve")
            if not retention_ok:
                reasons.append("old-task retention budget exceeded")
            if not evidence_ok:
                reasons.append("accepted evidence fraction below minimum")

        restored = False
        if not accepted:
            learner.restore(snapshot)
            restored = True

        return self._finish(
            accepted=accepted,
            reason="; ".join(reasons),
            before_new=before_new,
            after_new=after_new,
            before_old=before_old,
            after_old=after_old,
            retention_budget=retention_budget,
            old_variance=old_variance,
            evidence=evidence,
            n_update=len(evidence.accepted),
            n_replay=len(replay),
            restored=restored,
            details={
                "gate": "dual",
                "new_task_improved": bool(new_improved),
                "retention_ok": bool(retention_ok),
                "evidence_ok": bool(evidence_ok),
            },
        )

    def _finish(
        self,
        *,
        accepted: bool,
        reason: str,
        before_new: float,
        after_new: float,
        before_old: float,
        after_old: float,
        retention_budget: float,
        old_variance: float,
        evidence: EvidenceReport,
        n_update: int,
        n_replay: int,
        restored: bool,
        details: dict[str, Any],
    ) -> RetentionRevisionResult:
        result = RetentionRevisionResult(
            accepted=accepted,
            reason=reason,
            before_new=before_new,
            after_new=after_new,
            before_old=before_old,
            after_old=after_old,
            retention_budget=retention_budget,
            old_outcome_variance=old_variance,
            evidence=evidence,
            policy=self.policy,
            n_update_samples=n_update,
            n_replay_samples=n_replay,
            state_restored=restored,
            details=details,
        )
        self.history.append(result.to_dict())
        return result
