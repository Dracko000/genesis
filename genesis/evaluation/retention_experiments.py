"""E003-v2: dual-gated revision with explicit retention reporting.

This runner is deliberately separate from :func:`genesis.evaluation.experiments.run_e003`.
The legacy runner and its committed result stay untouched so the pre-upgrade
negative remains auditable. Here, every condition shares the same data, model,
seed and adaptation budget; only the update gate differs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.evaluation.experiments import (
    ExperimentConfig,
    _clone_learner,
    _neural_metrics,
    _observations,
)
from genesis.evaluation.metrics import summarize
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig
from genesis.learning.retention import (
    EvidenceConfig,
    RetentionPolicy,
    RetentionRevisionController,
)
from genesis.learning.revision import RevisionController


class E003RetentionConfig(ExperimentConfig):
    """Configuration fixed before the comparison run."""

    stable_rule: str = "interaction"
    changed_rule: str = "changed"
    n_train: int = 400
    n_new: int = 120
    n_new_val: int = 100
    n_test: int = 300
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    adaptation_epochs: list[int] = Field(default_factory=lambda: [5, 10, 25, 50])
    anomaly_magnitude: float = 25.0
    neural: NeuralConfig = Field(default_factory=lambda: NeuralConfig(epochs=300, seed=0))

    # Predeclared gate thresholds. Do not tune after inspecting the run.
    min_new_task_improvement: float = 1e-6
    old_task_max_ratio: float = 4.0
    old_task_mse_floor_fraction: float = 0.20
    min_accepted_fraction: float = 0.5
    max_residual_z: float = 6.0
    min_batch_size: int = 3

    # Predeclared success criterion for the dual-gated replay condition.
    success_new_task_improvement: float = 0.10


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _rate(values: list[bool]) -> float:
    return float(np.mean([1.0 if value else 0.0 for value in values])) if values else 0.0


def _policy(config: E003RetentionConfig, epochs: int) -> RetentionPolicy:
    return RetentionPolicy(
        epochs=epochs,
        min_new_task_improvement=config.min_new_task_improvement,
        old_task_max_ratio=config.old_task_max_ratio,
        old_task_mse_floor_fraction=config.old_task_mse_floor_fraction,
        min_accepted_fraction=config.min_accepted_fraction,
    )


def _evidence(config: E003RetentionConfig) -> EvidenceConfig:
    return EvidenceConfig(
        max_residual_z=config.max_residual_z,
        min_batch_size=config.min_batch_size,
        min_accepted_fraction=config.min_accepted_fraction,
    )


def run_e003_retention(config: E003RetentionConfig | None = None) -> dict[str, Any]:
    """Compare legacy gating, dual gating, and dual gating with replay."""

    config = config or E003RetentionConfig()
    per_seed: list[dict[str, Any]] = []

    for seed in config.seeds:
        stable_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.stable_rule, seed=seed)
        )
        changed_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.changed_rule, seed=seed + 300)
        )
        train, old_gate, test = stable_env.generate_split(
            config.n_train, max(50, config.n_train // 5), config.n_test, seed=seed
        )
        new_train, new_gate, new_test = changed_env.generate_split(
            config.n_new, config.n_new_val, config.n_test, seed=seed + 300
        )

        neural_config = config.neural.model_copy(update={"seed": seed})
        base = Learner(
            LearnerConfig(
                feature_names=list(stable_env.feature_names), neural=neural_config
            )
        )
        base.fit(
            _observations(stable_env, train, seed),
            validation_records=_observations(stable_env, old_gate, seed + 1),
        )

        old_gate_records = _observations(stable_env, old_gate, seed + 1)
        old_test_records = _observations(stable_env, test, seed + 2)
        new_train_records = _observations(changed_env, new_train, seed + 300)
        new_gate_records = _observations(changed_env, new_gate, seed + 302)
        new_test_records = _observations(changed_env, new_test, seed + 301)

        old_before = _neural_metrics(base, test, old_test_records)
        new_before = _neural_metrics(base, new_test, new_test_records)

        curve: list[dict[str, Any]] = []
        for epochs in config.adaptation_epochs:
            conditions = {
                "legacy_gate": (RevisionController(epochs=epochs), None),
                "dual_gate": (
                    RetentionRevisionController(
                        _policy(config, epochs), _evidence(config)
                    ),
                    None,
                ),
                "dual_gate_replay": (
                    RetentionRevisionController(
                        _policy(config, epochs), _evidence(config)
                    ),
                    _observations(stable_env, train, seed),
                ),
            }
            for name, (controller, replay) in conditions.items():
                candidate = _clone_learner(base)
                if name == "legacy_gate":
                    legacy_result = controller.controlled_update(  # type: ignore[union-attr]
                        candidate, new_train_records, new_gate_records
                    )
                    accepted = legacy_result.accepted
                    evidence = None
                    retention_budget = None
                    reason = legacy_result.reason
                    candidate_new = legacy_result.after
                    candidate_old = None
                    candidate_new_improvement = None
                    candidate_old_change = None
                else:
                    result = controller.controlled_update(  # type: ignore[union-attr]
                        candidate,
                        new_train_records,
                        new_gate_records,
                        old_gate_records,
                        replay_records=replay,
                    )
                    accepted = result.accepted
                    evidence = result.evidence.to_dict()
                    retention_budget = result.retention_budget
                    reason = result.reason
                    candidate_new = result.after_new
                    candidate_old = result.after_old
                    candidate_new_improvement = result.new_task_relative_improvement
                    candidate_old_change = result.old_task_relative_change

                after_new = _neural_metrics(candidate, new_test, new_test_records)
                after_old = _neural_metrics(candidate, test, old_test_records)
                curve.append(
                    {
                        "epochs": epochs,
                        "condition": name,
                        "accepted": bool(accepted),
                        "reason": reason,
                        "new_task_mse": after_new["mse"],
                        "old_task_mse": after_old["mse"],
                        "candidate_new_validation_mse": candidate_new,
                        "candidate_old_validation_mse": candidate_old,
                        "candidate_new_task_improvement": candidate_new_improvement,
                        "candidate_old_task_change": candidate_old_change,
                        "evidence": evidence,
                        "retention_budget": retention_budget,
                    }
                )

        # Robustness probes run only for the dual-gated conditions.
        anomaly = new_train_records[0].model_copy(
            update={
                "outcome": float(
                    np.mean([record.outcome for record in new_train_records])
                    + config.anomaly_magnitude
                ),
                "id": f"anomaly-{seed}",
            }
        )
        max_epochs = max(config.adaptation_epochs)

        isolated = _clone_learner(base)
        isolated_controller = RetentionRevisionController(
            _policy(config, max_epochs), _evidence(config)
        )
        isolated_result = isolated_controller.controlled_update(
            isolated, [anomaly], new_gate_records, old_gate_records
        )

        contaminated = _clone_learner(base)
        contaminated_controller = RetentionRevisionController(
            _policy(config, max_epochs), _evidence(config)
        )
        contaminated_result = contaminated_controller.controlled_update(
            contaminated,
            [*new_train_records, anomaly],
            new_gate_records,
            old_gate_records,
            replay_records=_observations(stable_env, train, seed),
        )
        rejected_ids = {
            entry["id"] for entry in contaminated_result.evidence.to_dict()["rejected"]
        }

        per_seed.append(
            {
                "seed": seed,
                "old_task_before": old_before,
                "new_task_before": new_before,
                "curve": curve,
                "isolated_anomaly": {
                    "accepted": isolated_result.accepted,
                    "reason": isolated_result.reason,
                    "evidence": isolated_result.evidence.to_dict(),
                },
                "contaminated_batch": {
                    "accepted": contaminated_result.accepted,
                    "outlier_rejected": anomaly.id in rejected_ids,
                    "n_accepted": contaminated_result.evidence.n_accepted,
                    "n_total": contaminated_result.evidence.n_total,
                    "new_task_mse": _neural_metrics(
                        contaminated, new_test, new_test_records
                    )["mse"],
                    "old_task_mse": _neural_metrics(
                        contaminated, test, old_test_records
                    )["mse"],
                },
                "history": contaminated_controller.history,
            }
        )

    conditions = ["legacy_gate", "dual_gate", "dual_gate_replay"]
    summary: dict[str, Any] = {}
    for condition in conditions:
        summary[condition] = {}
        for epochs in config.adaptation_epochs:
            entries = [
                next(
                    item
                    for item in seed_entry["curve"]
                    if item["condition"] == condition and item["epochs"] == epochs
                )
                for seed_entry in per_seed
            ]
            candidate_new = [
                item["candidate_new_validation_mse"]
                for item in entries
                if item["candidate_new_validation_mse"] is not None
            ]
            candidate_old = [
                item["candidate_old_validation_mse"]
                for item in entries
                if item["candidate_old_validation_mse"] is not None
            ]
            summary[condition][str(epochs)] = {
                "new_task_mse": summarize(item["new_task_mse"] for item in entries),
                "old_task_mse": summarize(item["old_task_mse"] for item in entries),
                "acceptance_rate": _rate([item["accepted"] for item in entries]),
                "candidate_new_validation_mse": summarize(candidate_new) if candidate_new else None,
                "candidate_old_validation_mse": summarize(candidate_old) if candidate_old else None,
            }

    max_key = str(max(config.adaptation_epochs))
    replay_final = summary["dual_gate_replay"][max_key]
    new_before_mean = _mean(
        [entry["new_task_before"]["mse"] for entry in per_seed]
    )
    new_after_mean = replay_final["new_task_mse"]["mean"]
    relative_improvement = (
        (new_before_mean - new_after_mean) / new_before_mean
        if new_before_mean > 0
        else 0.0
    )
    retention_ok = all(
        next(
            item
            for item in entry["curve"]
            if item["condition"] == "dual_gate_replay"
            and item["epochs"] == max(config.adaptation_epochs)
        )["old_task_mse"]
        <= (
            next(
                item
                for item in entry["curve"]
                if item["condition"] == "dual_gate_replay"
                and item["epochs"] == max(config.adaptation_epochs)
            )["retention_budget"]
            or 0.0
        )
        for entry in per_seed
    )
    isolated_rate = _rate(
        [entry["isolated_anomaly"]["accepted"] for entry in per_seed]
    )
    contaminated_reject_rate = _rate(
        [entry["contaminated_batch"]["outlier_rejected"] for entry in per_seed]
    )
    criterion = {
        "statement": (
            "At the largest adaptation budget, dual-gated replay must improve "
            "new-task MSE by at least 10%, keep mean old-task MSE within the "
            "predeclared retention budget, reject every isolated anomaly, and "
            "filter the injected outlier from every contaminated batch."
        ),
        "new_task_relative_improvement": relative_improvement,
        "retention_within_budget": bool(retention_ok),
        "isolated_anomaly_acceptance_rate": isolated_rate,
        "contaminated_outlier_rejection_rate": contaminated_reject_rate,
        "met": bool(
            relative_improvement >= config.success_new_task_improvement
            and retention_ok
            and isolated_rate == 0.0
            and contaminated_reject_rate == 1.0
        ),
    }

    return {
        "experiment": "E003-v2",
        "title": "Dual-gated representation revision with retention",
        "research_question": (
            "Can the learner adapt to a rule change while keeping earlier "
            "competence within a predeclared budget?"
        ),
        "hypothesis": (
            "Evidence filtering plus a dual new/old gate rejects isolated and "
            "embedded outliers, and replay keeps old-task error inside the "
            "retention budget at a modest cost to new-task adaptation."
        ),
        "config": config.model_dump(mode="json"),
        "summary": summary,
        "criterion": criterion,
        "per_seed": per_seed,
        "limitations": [
            (
                "The retention budget is a design choice, not a universal law; it "
                "is declared before the run and reported with every condition."
            ),
            "Old-task validation and test come from the same synthetic rule.",
            "The synthetic rule-change pair is low-dimensional.",
            (
                "Replay uses the full earlier training set here; bounded buffers "
                "are tested separately by the ReplayBuffer unit tests."
            ),
        ],
    }
