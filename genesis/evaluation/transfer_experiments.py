"""E002-v2: fair transfer diagnostics across related and unrelated rule pairs.

The legacy E002 used a single source/target pair (``linear`` -> ``interaction``)
which share little structure, and it did not report a related-pair control. This
runner keeps the target-task budget identical for warm-start and fresh models,
reuses the exact same architecture and seed, and sweeps a small matrix of
related and unrelated pairs. Pretraining compute is reported separately rather
than being folded into the target budget.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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


class TransferPair(BaseModel):
    """One source -> target transfer condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    source_rule: str
    target_rule: str
    related: bool


def _default_pairs() -> list[TransferPair]:
    return [
        TransferPair(
            label="additive_related",
            source_rule="linear",
            target_rule="linear_alt",
            related=True,
        ),
        TransferPair(
            label="interaction_related",
            source_rule="interaction",
            target_rule="interaction_alt",
            related=True,
        ),
        TransferPair(
            label="legacy_linear_to_interaction",
            source_rule="linear",
            target_rule="interaction",
            related=False,
        ),
        TransferPair(
            label="unrelated_linear_to_quadratic",
            source_rule="linear",
            target_rule="quadratic",
            related=False,
        ),
    ]


class E002DiagnosticsConfig(ExperimentConfig):
    """Predeclared configuration for the transfer matrix."""

    pairs: list[TransferPair] = Field(default_factory=_default_pairs)
    unrelated_rule: str = "quadratic"
    n_source: int = 500
    n_target: int = 120
    n_target_test: int = 300
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    source_epochs: int = 250
    target_epochs: int = 50
    full_target_epochs: int = 250
    neural: NeuralConfig = Field(default_factory=lambda: NeuralConfig(seed=0))
    success_relative_improvement: float = 0.10

    def pair_labels(self) -> list[str]:
        return [pair.label for pair in self.pairs]


def _mse(learner: Learner, env: SyntheticWorld, dataset: Any, seed: int) -> dict[str, float]:
    return _neural_metrics(learner, dataset, _observations(env, dataset, seed))


def _relative_improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference if reference > 0 else 0.0


def run_e002_diagnostics(
    config: E002DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Run the matched-budget transfer matrix."""

    config = config or E002DiagnosticsConfig()
    per_seed: list[dict[str, Any]] = []

    for seed in config.seeds:
        source_cache: dict[str, Learner] = {}
        pair_entries: list[dict[str, Any]] = []

        for pair in config.pairs:
            source_key = pair.source_rule
            if source_key not in source_cache:
                source_env = SyntheticWorld(
                    SyntheticWorldConfig(rule=pair.source_rule, seed=seed)
                )
                source_train, source_val, source_test = source_env.generate_split(
                    config.n_source,
                    max(50, config.n_source // 5),
                    config.n_target_test,
                    seed=seed,
                )
                source_learner = Learner(
                    LearnerConfig(
                        feature_names=list(source_env.feature_names),
                        neural=config.neural.model_copy(update={"seed": seed}),
                    )
                )
                source_learner.fit(
                    _observations(source_env, source_train, seed),
                    validation_records=_observations(source_env, source_val, seed + 1),
                    epochs=config.source_epochs,
                )
                source_cache[source_key] = source_learner
            source_learner = source_cache[source_key]

            source_env = SyntheticWorld(
                SyntheticWorldConfig(rule=pair.source_rule, seed=seed)
            )
            target_env = SyntheticWorld(
                SyntheticWorldConfig(rule=pair.target_rule, seed=seed + 100)
            )
            target_train, _target_val, target_test = target_env.generate_split(
                config.n_target,
                max(30, config.n_target // 4),
                config.n_target_test,
                seed=seed + 100,
            )
            source_test = source_env.generate_split(
                config.n_source,
                max(50, config.n_source // 5),
                config.n_target_test,
                seed=seed,
            )[2]

            source_before = _mse(source_learner, source_env, source_test, seed + 2)
            target_before = _mse(source_learner, target_env, target_test, seed + 101)

            # Warm-start transfer: identical target data, epochs and architecture.
            transfer = _clone_learner(source_learner)
            transfer.fit(
                _observations(target_env, target_train, seed + 100),
                warm_start=True,
                epochs=config.target_epochs,
            )
            transfer_target = _mse(transfer, target_env, target_test, seed + 101)
            transfer_source = _mse(transfer, source_env, source_test, seed + 2)

            # Fresh equal-budget control: same architecture seed, same epochs.
            fresh = Learner(
                LearnerConfig(
                    feature_names=list(target_env.feature_names),
                    neural=config.neural.model_copy(update={"seed": seed}),
                )
            )
            fresh.fit(
                _observations(target_env, target_train, seed + 100),
                epochs=config.target_epochs,
            )
            fresh_target = _mse(fresh, target_env, target_test, seed + 101)

            # Full-budget reference on the target task.
            full = Learner(
                LearnerConfig(
                    feature_names=list(target_env.feature_names),
                    neural=config.neural.model_copy(update={"seed": seed}),
                )
            )
            full.fit(
                _observations(target_env, target_train, seed + 100),
                epochs=config.full_target_epochs,
            )
            full_target = _mse(full, target_env, target_test, seed + 101)

            # Mismatched prior control: unrelated source, same target budget.
            unrelated_env = SyntheticWorld(
                SyntheticWorldConfig(rule=config.unrelated_rule, seed=seed + 200)
            )
            unrelated_train = unrelated_env.generate_split(
                config.n_target, 30, 30, seed=seed + 200
            )[0]
            unrelated = Learner(
                LearnerConfig(
                    feature_names=list(unrelated_env.feature_names),
                    neural=config.neural.model_copy(update={"seed": seed}),
                )
            )
            unrelated.fit(
                _observations(unrelated_env, unrelated_train, seed + 200),
                epochs=config.full_target_epochs,
            )
            unrelated.fit(
                _observations(target_env, target_train, seed + 100),
                warm_start=True,
                epochs=config.target_epochs,
            )
            unrelated_target = _mse(unrelated, target_env, target_test, seed + 101)

            pair_entries.append(
                {
                    "label": pair.label,
                    "source_rule": pair.source_rule,
                    "target_rule": pair.target_rule,
                    "related": pair.related,
                    "condition_metrics": {
                        "source_only": source_before,
                        "target_before_transfer": target_before,
                        "transfer_small_budget": transfer_target,
                        "fresh_small_budget": fresh_target,
                        "fresh_full_budget": full_target,
                        "unrelated_prior_small_budget": unrelated_target,
                        "transfer_source_retention": transfer_source,
                    },
                    "relative_improvement_vs_fresh": _relative_improvement(
                        fresh_target["mse"], transfer_target["mse"]
                    ),
                    "relative_improvement_vs_unrelated": _relative_improvement(
                        unrelated_target["mse"], transfer_target["mse"]
                    ),
                    "source_retention_mse_change": (
                        transfer_source["mse"] - source_before["mse"]
                    ),
                }
            )

        per_seed.append({"seed": seed, "pairs": pair_entries})

    summary: dict[str, Any] = {}
    for pair in config.pairs:
        entries = [
            next(
                item
                for item in seed_entry["pairs"]
                if item["label"] == pair.label
            )
            for seed_entry in per_seed
        ]
        summary[pair.label] = {
            "related": pair.related,
            "source_rule": pair.source_rule,
            "target_rule": pair.target_rule,
            "metrics": {
                condition: {
                    metric: summarize(
                        entry["condition_metrics"][condition][metric] for entry in entries
                    )
                    for metric in ("mse", "mae", "r2")
                }
                for condition in entries[0]["condition_metrics"]
            },
            "relative_improvement_vs_fresh": summarize(
                entry["relative_improvement_vs_fresh"] for entry in entries
            ),
            "relative_improvement_vs_unrelated": summarize(
                entry["relative_improvement_vs_unrelated"] for entry in entries
            ),
            "source_retention_mse_change": summarize(
                entry["source_retention_mse_change"] for entry in entries
            ),
        }

    related_labels = [pair.label for pair in config.pairs if pair.related]
    unrelated_labels = [pair.label for pair in config.pairs if not pair.related]
    related_improvements = [
        entry["relative_improvement_vs_fresh"]
        for seed_entry in per_seed
        for entry in seed_entry["pairs"]
        if entry["related"]
    ]
    unrelated_improvements = [
        entry["relative_improvement_vs_fresh"]
        for seed_entry in per_seed
        for entry in seed_entry["pairs"]
        if not entry["related"]
    ]
    related_mean = float(np.mean(related_improvements)) if related_improvements else 0.0
    unrelated_mean = (
        float(np.mean(unrelated_improvements)) if unrelated_improvements else 0.0
    )

    criterion = {
        "statement": (
            "Averaged over related source/target pairs and seeds, warm-start "
            "transfer must reduce target MSE by at least 10% relative to an "
            "equal-budget fresh model with the same architecture seed. "
            "Unrelated pairs are a negative control and are reported separately."
        ),
        "related_pairs": related_labels,
        "unrelated_pairs": unrelated_labels,
        "related_relative_improvement": related_mean,
        "unrelated_relative_improvement": unrelated_mean,
        "related_pair_improvements": {
            label: summary[label]["relative_improvement_vs_fresh"]["mean"]
            for label in related_labels
        },
        "secondary_each_related_pair_met": {
            label: bool(
                summary[label]["relative_improvement_vs_fresh"]["mean"]
                >= config.success_relative_improvement
            )
            for label in related_labels
        },
        "secondary_all_related_pairs_met": bool(
            all(
                summary[label]["relative_improvement_vs_fresh"]["mean"]
                >= config.success_relative_improvement
                for label in related_labels
            )
        ),
        "met": bool(related_mean >= config.success_relative_improvement),
    }

    return {
        "experiment": "E002-v2",
        "title": "Fair transfer diagnostics across rule pairs",
        "research_question": (
            "Does prior experience reduce target-task learning cost when the "
            "source and target share structure, after matching architecture, "
            "seed, data and target epochs?"
        ),
        "hypothesis": (
            "Warm-start transfer helps on related rule pairs and does not show "
            "the same advantage on unrelated pairs."
        ),
        "config": config.model_dump(mode="json"),
        "summary": summary,
        "criterion": criterion,
        "per_seed": per_seed,
        "fairness_controls": {
            "same_architecture_seed": True,
            "same_target_train_records": True,
            "same_target_epochs": True,
            "same_target_test_records": True,
            "source_pretraining_epochs": config.source_epochs,
            "source_pretraining_samples": config.n_source,
            "target_epochs": config.target_epochs,
            "target_samples": config.n_target,
            "budget_note": (
                "Target-task epochs are matched; source pretraining compute is "
                "reported separately and is not disguised as target budget."
            ),
        },
        "limitations": [
            (
                "Relatedness is defined by the synthetic rule family, not by a "
                "domain-independent metric."
            ),
            (
                "MSE on a low-dimensional synthetic task is not evidence of "
                "human-like structure reuse."
            ),
            (
                "The unrelated-prior control shares the same feature surface, so "
                "it isolates rule mismatch rather than input-distribution shift."
            ),
        ],
    }
