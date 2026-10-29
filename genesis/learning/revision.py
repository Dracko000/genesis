"""Controlled representation revision.

A prediction error alone is not evidence that the representation is wrong. The
revision controller therefore:

1. snapshots the current parameters,
2. measures validation performance before the update,
3. applies a candidate update,
4. measures validation performance after the update,
5. keeps the update only if it improves validation error by a margin,
   otherwise rolls back.

This makes isolated anomalous observations difficult to absorb silently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from genesis.experience.observation import select_labeled
from genesis.experience.schema import Observation
from genesis.learning.learner import Learner


@dataclass
class RevisionResult:
    """Outcome of one controlled revision attempt."""

    accepted: bool
    before: float
    after: float
    reason: str
    n_update_samples: int
    n_validation_samples: int
    details: dict[str, Any] = field(default_factory=dict)


class RevisionController:
    """Validation-gated parameter updates with rollback."""

    def __init__(self, min_improvement: float = 1e-6, epochs: int = 50) -> None:
        if min_improvement < 0:
            raise ValueError("min_improvement must be non-negative")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self.min_improvement = min_improvement
        self.epochs = epochs

    @staticmethod
    def _mse(learner: Learner, records: list[Observation]) -> float:
        features = learner.encoder.transform(records)
        predictions = learner.predict(features)
        outcomes = np.asarray([record.outcome for record in records], dtype=float)
        return float(np.mean((predictions - outcomes) ** 2))

    def controlled_update(
        self,
        learner: Learner,
        update_records: Iterable[Observation],
        validation_records: Iterable[Observation],
        *,
        replay_records: Iterable[Observation] | None = None,
    ) -> RevisionResult:
        """Try to improve a fitted learner on new experience.

        ``replay_records`` mixes earlier labeled experience into the candidate
        update. They are training data only; validation remains untouched.
        """

        if not learner.fitted:
            raise ValueError("learner must be fitted before controlled revision")

        update = select_labeled(update_records)
        validation = select_labeled(validation_records)
        if not update:
            return RevisionResult(False, 0.0, 0.0, "no labeled update records", 0, len(validation))
        if not validation:
            raise ValueError("validation records are required for controlled revision")

        replay = select_labeled(replay_records or [])
        snapshot = learner.snapshot()
        before = self._mse(learner, validation)

        training_records = [*update, *replay]
        learner.fit(training_records, warm_start=True, epochs=self.epochs)
        after = self._mse(learner, validation)

        improved = after < before - self.min_improvement
        if not improved:
            learner.restore(snapshot)
            return RevisionResult(
                accepted=False,
                before=before,
                after=after,
                reason="validation error did not improve; rolled back",
                n_update_samples=len(update),
                n_validation_samples=len(validation),
                details={
                    "restored_validation_mse": self._mse(learner, validation),
                    "replay_samples": len(replay),
                },
            )

        return RevisionResult(
            accepted=True,
            before=before,
            after=after,
            reason="validation error improved",
            n_update_samples=len(update),
            n_validation_samples=len(validation),
            details={"replay_samples": len(replay)},
        )
