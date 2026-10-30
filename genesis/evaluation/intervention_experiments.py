"""E005: representation interventions after transfer.

Question: are the hidden units that are most active on the source task causally
important for it, and does the same intervention matter as much on the target
task? A matched random ablation is the control. Linear CKA reports how similar
the source and target representations are, layer by layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.evaluation.experiments import (
    ExperimentConfig,
    _clone_learner,
    _observations,
)
from genesis.evaluation.metrics import regression_report, summarize
from genesis.learning.interventions import (
    ablation_effect,
    layer_cka,
    matched_random_units,
    mean_absolute_activation,
    top_units,
)
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig


class InterventionConfig(ExperimentConfig):
    """Predeclared configuration for the intervention probe."""

    source_rule: str = "linear"
    target_rule: str = "linear_alt"
    n_source: int = 400
    n_target: int = 120
    n_target_test: int = 200
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    source_epochs: int = 200
    target_epochs: int = 50
    ablation_fraction: float = Field(default=0.2, gt=0, le=1)
    neural: NeuralConfig = Field(default_factory=lambda: NeuralConfig(seed=0))
    success_selectivity_ratio: float = 1.5


def _array_metrics(learner: Learner, features: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    return regression_report(outcomes, learner.predict(features))


def run_intervention_probe(
    config: InterventionConfig | None = None,
) -> dict[str, Any]:
    config = config or InterventionConfig()
    per_seed: list[dict[str, Any]] = []

    for seed in config.seeds:
        source_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.source_rule, seed=seed)
        )
        target_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.target_rule, seed=seed + 100)
        )
        source_train, source_val, source_test = source_env.generate_split(
            config.n_source, max(50, config.n_source // 5), config.n_target_test, seed=seed
        )
        target_train, _, target_test = target_env.generate_split(
            config.n_target,
            max(30, config.n_target // 4),
            config.n_target_test,
            seed=seed + 100,
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
        transfer = _clone_learner(source_learner)
        transfer.fit(
            _observations(target_env, target_train, seed + 100),
            warm_start=True,
            epochs=config.target_epochs,
        )

        source_records = _observations(source_env, source_test, seed + 2)
        target_records = _observations(target_env, target_test, seed + 101)
        source_features = transfer.encoder.transform(source_records)
        target_features = transfer.encoder.transform(target_records)
        source_outcomes = np.asarray([record.outcome for record in source_records])
        target_outcomes = np.asarray([record.outcome for record in target_records])

        cka = layer_cka(transfer, source_features, target_features)
        source_profile = mean_absolute_activation(transfer, source_features)
        layer_entries: list[dict[str, Any]] = []
        for layer, profile in enumerate(source_profile):
            width = int(profile.shape[0])
            k = max(1, round(config.ablation_fraction * width))
            selected = top_units(profile, k)
            random_units = matched_random_units(width, k, seed + 10_000 + layer)

            source_specialized_source = ablation_effect(
                transfer, source_features, source_outcomes, layer, selected
            )
            source_specialized_target = ablation_effect(
                transfer, target_features, target_outcomes, layer, selected
            )
            random_source = ablation_effect(
                transfer, source_features, source_outcomes, layer, random_units
            )
            random_target = ablation_effect(
                transfer, target_features, target_outcomes, layer, random_units
            )

            layer_entries.append(
                {
                    "layer": layer,
                    "width": width,
                    "k": k,
                    "cka_source_target": cka[layer],
                    "selected_units": selected,
                    "random_units": random_units,
                    "source_specialized": {
                        "source": source_specialized_source,
                        "target": source_specialized_target,
                    },
                    "matched_random": {
                        "source": random_source,
                        "target": random_target,
                    },
                }
            )

        per_seed.append(
            {
                "seed": seed,
                "source_only": _array_metrics(
                    source_learner, source_features, source_outcomes
                ),
                "transfer_source": _array_metrics(
                    transfer, source_features, source_outcomes
                ),
                "transfer_target": _array_metrics(
                    transfer, target_features, target_outcomes
                ),
                "layers": layer_entries,
            }
        )

    layer_count = len(per_seed[0]["layers"])
    summary: dict[str, Any] = {}
    for layer in range(layer_count):
        entries = [seed_entry["layers"][layer] for seed_entry in per_seed]
        special_source = [
            entry["source_specialized"]["source"]["mse_delta"] for entry in entries
        ]
        random_source = [entry["matched_random"]["source"]["mse_delta"] for entry in entries]
        special_target = [
            entry["source_specialized"]["target"]["mse_delta"] for entry in entries
        ]
        random_target = [entry["matched_random"]["target"]["mse_delta"] for entry in entries]
        mean_abs_special_source = float(
            np.mean([abs(value) for value in special_source])
        )
        mean_abs_random_source = float(np.mean([abs(value) for value in random_source]))
        source_ratio = (
            mean_abs_special_source / mean_abs_random_source
            if mean_abs_random_source > 1e-12
            else float("inf")
        )
        mean_abs_special_target = float(
            np.mean([abs(value) for value in special_target])
        )
        mean_abs_random_target = float(np.mean([abs(value) for value in random_target]))
        target_ratio = (
            mean_abs_special_target / mean_abs_random_target
            if mean_abs_random_target > 1e-12
            else float("inf")
        )
        summary[str(layer)] = {
            "cka_source_target": summarize(
                entry["cka_source_target"] for entry in entries
            ),
            "source_specialized_source_mse_delta": summarize(special_source),
            "matched_random_source_mse_delta": summarize(random_source),
            "source_specialized_target_mse_delta": summarize(special_target),
            "matched_random_target_mse_delta": summarize(random_target),
            "source_specialized_source_abs_delta_mean": mean_abs_special_source,
            "matched_random_source_abs_delta_mean": mean_abs_random_source,
            "source_specialized_target_abs_delta_mean": mean_abs_special_target,
            "matched_random_target_abs_delta_mean": mean_abs_random_target,
            "selectivity_ratio_source": source_ratio,
            "selectivity_ratio_target": target_ratio,
        }

    best_layer = max(
        summary, key=lambda key: summary[key]["selectivity_ratio_source"]
    )
    best_ratio = summary[best_layer]["selectivity_ratio_source"]
    criterion = {
        "statement": (
            "Zeroing the source-most-active hidden units must change source MSE "
            "by at least 1.5x the magnitude of a matched random ablation, "
            "averaged over seeds, for at least one layer. The sign of the change "
            "is reported rather than assumed, and target effects and CKA have no "
            "success threshold."
        ),
        "best_layer": int(best_layer),
        "selectivity_ratio_source": best_ratio,
        "met": bool(best_ratio >= config.success_selectivity_ratio),
    }

    return {
        "experiment": "E005",
        "title": "Representation interventions after transfer",
        "research_question": (
            "Are source-specialised hidden units causally important for the "
            "source task, and how does that importance transfer to a related "
            "target task?"
        ),
        "hypothesis": (
            "Ablating the most active source units has a larger causal effect "
            "than a matched random ablation, while CKA and target effects "
            "quantify how much of that structure is reused."
        ),
        "config": config.model_dump(mode="json"),
        "summary": summary,
        "criterion": criterion,
        "per_seed": per_seed,
        "limitations": [
            (
                "Unit ablation is one causal probe; redundancy across units can "
                "understate importance."
            ),
            "CKA measures linear similarity, not semantic equivalence.",
            (
                "The probe compares one related rule pair on a low-dimensional "
                "synthetic task."
            ),
        ],
    }
