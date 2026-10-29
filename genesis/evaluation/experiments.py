"""Reproducible experiment runners for E001, E002 and E003.

Each runner is a pure function of its configuration: the same config and the
same software produce the same result dictionary. Success criteria are declared
in the result payload *before* observations are inspected, and the runner
reports whether the criterion was met rather than asserting success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from genesis.environments.base import Dataset
from genesis.environments.synthetic_world import RULES, SyntheticWorld, SyntheticWorldConfig
from genesis.evaluation.benchmarks import (
    LinearRidgeBaseline,
    MeanBaseline,
    NearestNeighbourBaseline,
    OracleBaseline,
)
from genesis.evaluation.datasets import split_indices
from genesis.evaluation.metrics import regression_report, summarize
from genesis.experience.schema import Observation
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig
from genesis.learning.revision import RevisionController
from genesis.learning.symbolic import SymbolicConfig, SymbolicRuleLearner


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class E001Config(ExperimentConfig):
    """Emergent structure: compositional generalisation against baselines."""

    rule: str = "interaction"
    n_train: int = 400
    n_test: int = 200
    held_out_combinations: list[tuple[int, int]] = Field(default_factory=lambda: [(1, 1)])
    noise_std: float = 0.0
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    neural: NeuralConfig = Field(
        default_factory=lambda: NeuralConfig(epochs=300, hidden_sizes=[32, 32], seed=0)
    )
    sample_sizes: list[int] = Field(default_factory=lambda: [50, 100, 200])
    noise_levels: list[float] = Field(default_factory=lambda: [0.0, 0.1, 0.3])
    run_noise_study: bool = True


class E002Config(ExperimentConfig):
    """Structure reuse: warm-start transfer against equal-budget controls."""

    source_rule: str = "linear"
    target_rule: str = "interaction"
    unrelated_rule: str = "quadratic"
    n_source: int = 500
    n_target: int = 120
    n_target_test: int = 300
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    warm_epochs: int = 25
    full_epochs: int = 250
    neural: NeuralConfig = Field(default_factory=lambda: NeuralConfig(seed=0))


class HybridAblationConfig(ExperimentConfig):
    """Neural-only vs symbolic-basis-only vs validation-selected hybrid."""

    rule: str = "interaction"
    n_train: int = 400
    n_val: int = 200
    n_test: int = 400
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    neural: NeuralConfig = Field(
        default_factory=lambda: NeuralConfig(epochs=300, hidden_sizes=[32, 32], seed=0)
    )
    symbolic: SymbolicConfig = Field(default_factory=SymbolicConfig)


class E003Config(ExperimentConfig):
    """Representation revision after a hidden rule change."""

    stable_rule: str = "interaction"
    changed_rule: str = "changed"
    n_train: int = 400
    n_new: int = 120
    n_new_val: int = 100
    n_test: int = 300
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    adaptation_epochs: list[int] = Field(default_factory=lambda: [5, 10, 25, 50])
    anomaly_magnitude: float = 25.0
    neural: NeuralConfig = Field(
        default_factory=lambda: NeuralConfig(epochs=300, seed=0)
    )


def _observations(env: SyntheticWorld, dataset: Dataset, seed: int) -> list[Observation]:
    return env.to_observations(dataset, seed=seed)


def _neural_metrics(learner: Learner, dataset: Dataset, records: list[Observation]) -> dict[str, float]:
    predictions = learner.predict(learner.encoder.transform(records))
    return regression_report(dataset.outcomes, predictions)


def _baseline_metrics(
    baseline: Any, train: Dataset, test: Dataset
) -> dict[str, float]:
    baseline.fit(train.features, train.outcomes)
    return regression_report(test.outcomes, baseline.predict(test.features))


def _fit_neural(
    env: SyntheticWorld,
    train: Dataset,
    test: Dataset,
    neural: NeuralConfig,
    *,
    seed: int,
    epochs: int | None = None,
    warm_from: Learner | None = None,
) -> tuple[Learner, dict[str, float]]:
    config = LearnerConfig(
        feature_names=list(env.feature_names),
        neural=neural.model_copy(update={"seed": seed}),
    )
    if warm_from is None:
        learner = Learner(config)
        learner.fit(
            _observations(env, train, seed), epochs=epochs
        )
    else:
        learner = _clone_learner(warm_from)
        learner.fit(
            _observations(env, train, seed), warm_start=True, epochs=epochs
        )
    test_records = _observations(env, test, seed + 5000)
    return learner, _neural_metrics(learner, test, test_records)


def _clone_learner(source: Learner) -> Learner:
    clone = Learner(source.config)
    clone.restore(source.snapshot())
    clone.checkpoint_id = source.checkpoint_id
    return clone


def _aggregate(per_seed: list[dict[str, Any]], method: str, metric: str) -> dict[str, float]:
    return summarize(entry["metrics"][method][metric] for entry in per_seed)


def _method_names(per_seed: list[dict[str, Any]]) -> list[str]:
    if not per_seed:
        return []
    return list(per_seed[0]["metrics"].keys())


def run_e001(config: E001Config | None = None) -> dict[str, Any]:
    """Run E001 and return a complete, JSON-serialisable result."""

    config = config or E001Config()
    per_seed: list[dict[str, Any]] = []
    efficiency: dict[int, dict[str, list[float]]] = {
        size: {} for size in config.sample_sizes
    }
    noise_study: dict[float, dict[str, list[float]]] = {
        level: {} for level in config.noise_levels
    }

    for seed in config.seeds:
        env = SyntheticWorld(
            SyntheticWorldConfig(
                rule=config.rule, noise_std=config.noise_std, seed=seed
            )
        )
        train, test = env.generate_compositional_split(
            config.n_train,
            config.n_test,
            seed=seed,
            held_out_combinations=config.held_out_combinations,
        )
        train_records = _observations(env, train, seed)

        neural_config = config.neural.model_copy(update={"seed": seed})
        learner = Learner(
            LearnerConfig(feature_names=list(env.feature_names), neural=neural_config)
        )
        learner.fit(train_records)

        metrics: dict[str, dict[str, float]] = {
            "neural": _neural_metrics(learner, test, _observations(env, test, seed + 1)),
            "nearest_neighbour": _baseline_metrics(
                NearestNeighbourBaseline(), train, test
            ),
            "linear_ridge": _baseline_metrics(LinearRidgeBaseline(), train, test),
            "mean": _baseline_metrics(MeanBaseline(), train, test),
            "oracle": _baseline_metrics(OracleBaseline(config.rule), train, test),
        }

        permutation = list(range(len(env.feature_names)))[::-1]
        permuted = env.permute_features(test.features, permutation)
        metrics["neural_permuted_features"] = regression_report(
            test.outcomes, learner.predict(learner.encoder.transform_matrix(permuted))
        )

        # Sample efficiency: same held-out test, increasing training subsets.
        for size in config.sample_sizes:
            if size > len(train_records):
                continue
            subset = Dataset(train.features[:size], train.outcomes[:size], train.feature_names)
            _, subset_metrics = _fit_neural(
                env,
                subset,
                test,
                config.neural,
                seed=seed,
                epochs=max(50, config.neural.epochs // 2),
            )
            efficiency[size].setdefault("neural", []).append(subset_metrics["mse"])
            nn_metrics = _baseline_metrics(NearestNeighbourBaseline(), subset, test)
            efficiency[size].setdefault("nearest_neighbour", []).append(nn_metrics["mse"])
            ridge_metrics = _baseline_metrics(LinearRidgeBaseline(), subset, test)
            efficiency[size].setdefault("linear_ridge", []).append(ridge_metrics["mse"])
            oracle_metrics = _baseline_metrics(OracleBaseline(config.rule), subset, test)
            efficiency[size].setdefault("oracle", []).append(oracle_metrics["mse"])

        per_seed.append({"seed": seed, "metrics": metrics})

    if config.run_noise_study:
        for level in config.noise_levels:
            for seed in config.seeds:
                env = SyntheticWorld(
                    SyntheticWorldConfig(rule=config.rule, noise_std=level, seed=seed)
                )
                train, test = env.generate_compositional_split(
                    config.n_train,
                    config.n_test,
                    seed=seed,
                    held_out_combinations=config.held_out_combinations,
                )
                _, neural_metrics = _fit_neural(
                    env, train, test, config.neural, seed=seed
                )
                noise_study[level].setdefault("neural", []).append(neural_metrics["mse"])
                nn_metrics = _baseline_metrics(NearestNeighbourBaseline(), train, test)
                noise_study[level].setdefault("nearest_neighbour", []).append(nn_metrics["mse"])
                oracle_metrics = _baseline_metrics(OracleBaseline(config.rule), train, test)
                noise_study[level].setdefault("oracle", []).append(oracle_metrics["mse"])

    methods = _method_names(per_seed)
    summary = {
        method: {
            metric: _aggregate(per_seed, method, metric)
            for metric in ("mse", "mae", "r2")
        }
        for method in methods
    }
    efficiency_summary = {
        size: {method: summarize(values) for method, values in by_method.items() if values}
        for size, by_method in efficiency.items()
    }
    noise_summary = {
        level: {method: summarize(values) for method, values in by_method.items() if values}
        for level, by_method in noise_study.items()
    }

    neural_mse = summary["neural"]["mse"]["mean"]
    nn_mse = summary["nearest_neighbour"]["mse"]["mean"]
    neural_r2 = summary["neural"]["r2"]["mean"]
    improvement = (nn_mse - neural_mse) / nn_mse if nn_mse > 0 else 0.0
    criterion = {
        "statement": (
            "Neural held-out MSE is at least 10% lower than the nearest-neighbour "
            "baseline AND neural mean R^2 >= 0.5."
        ),
        "neural_vs_nearest_relative_improvement": improvement,
        "met": bool(improvement >= 0.10 and neural_r2 >= 0.5),
    }

    return {
        "experiment": "E001",
        "title": "Emergent structure: compositional generalisation",
        "research_question": (
            "Can a learner form representations that improve prediction on unseen "
            "feature combinations without being given high-level concepts?"
        ),
        "hypothesis": (
            "A neural learner with raw features predicts held-out sign-combinations "
            "better than memorisation, approaching a linear model only when the rule "
            "is additive."
        ),
        "config": config.model_dump(mode="json"),
        "summary": summary,
        "sample_efficiency": efficiency_summary,
        "noise_sensitivity": noise_summary,
        "criterion": criterion,
        "per_seed": per_seed,
        "limitations": [
            "Held-out sign combinations are only one operationalisation of compositional generalisation.",
            "Improved prediction is not evidence of human-like concept formation.",
            "The oracle uses the true rule and is an upper reference only.",
        ],
    }


def run_e002(config: E002Config | None = None) -> dict[str, Any]:
    """Run E002: does prior experience reduce target-task learning cost?"""

    config = config or E002Config()
    per_seed: list[dict[str, Any]] = []

    for seed in config.seeds:
        source_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.source_rule, seed=seed)
        )
        target_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.target_rule, seed=seed + 100)
        )
        unrelated_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.unrelated_rule, seed=seed + 200)
        )

        source_train, source_val, source_test = source_env.generate_split(
            config.n_source, max(50, config.n_source // 5), config.n_target_test, seed=seed
        )
        target_train, target_val, target_test = target_env.generate_split(
            config.n_target, max(30, config.n_target // 4), config.n_target_test, seed=seed + 100
        )
        unrelated_train, _, _ = unrelated_env.generate_split(
            config.n_target, 30, 30, seed=seed + 200
        )

        neural_config = config.neural.model_copy(update={"seed": seed})
        base_learner = Learner(
            LearnerConfig(feature_names=list(source_env.feature_names), neural=neural_config)
        )
        base_learner.fit(
            _observations(source_env, source_train, seed),
            validation_records=_observations(source_env, source_val, seed + 1),
        )
        source_test_metrics = _neural_metrics(
            base_learner, source_test, _observations(source_env, source_test, seed + 2)
        )

        # Transfer: warm-start from source, small target budget.
        transfer = _clone_learner(base_learner)
        transfer.fit(
            _observations(target_env, target_train, seed + 100),
            warm_start=True,
            epochs=config.warm_epochs,
        )
        transfer_metrics = _neural_metrics(
            transfer, target_test, _observations(target_env, target_test, seed + 101)
        )
        transfer_retention = _neural_metrics(
            transfer, source_test, _observations(source_env, source_test, seed + 2)
        )

        # Equal small budget from scratch.
        fresh = Learner(
            LearnerConfig(feature_names=list(target_env.feature_names), neural=neural_config)
        )
        fresh.fit(
            _observations(target_env, target_train, seed + 100),
            epochs=config.warm_epochs,
        )
        fresh_metrics = _neural_metrics(
            fresh, target_test, _observations(target_env, target_test, seed + 101)
        )

        # Full target budget reference.
        fresh_full = Learner(
            LearnerConfig(feature_names=list(target_env.feature_names), neural=neural_config)
        )
        fresh_full.fit(
            _observations(target_env, target_train, seed + 100),
            epochs=config.full_epochs,
        )
        fresh_full_metrics = _neural_metrics(
            fresh_full, target_test, _observations(target_env, target_test, seed + 101)
        )

        # Unrelated prior with the same small target budget.
        unrelated = Learner(
            LearnerConfig(feature_names=list(unrelated_env.feature_names), neural=neural_config)
        )
        unrelated.fit(
            _observations(unrelated_env, unrelated_train, seed + 200),
            epochs=config.full_epochs,
        )
        unrelated.fit(
            _observations(target_env, target_train, seed + 100),
            warm_start=True,
            epochs=config.warm_epochs,
        )
        unrelated_metrics = _neural_metrics(
            unrelated, target_test, _observations(target_env, target_test, seed + 101)
        )

        per_seed.append(
            {
                "seed": seed,
                "metrics": {
                    "transfer_small_budget": transfer_metrics,
                    "fresh_small_budget": fresh_metrics,
                    "fresh_full_budget": fresh_full_metrics,
                    "unrelated_prior_small_budget": unrelated_metrics,
                    "source_only": source_test_metrics,
                    "transfer_source_retention": transfer_retention,
                },
            }
        )

    methods = _method_names(per_seed)
    summary = {
        method: {
            metric: _aggregate(per_seed, method, metric)
            for metric in ("mse", "mae", "r2")
        }
        for method in methods
    }
    transfer_mse = summary["transfer_small_budget"]["mse"]["mean"]
    fresh_mse = summary["fresh_small_budget"]["mse"]["mean"]
    improvement = (fresh_mse - transfer_mse) / fresh_mse if fresh_mse > 0 else 0.0
    retention_before = summary["source_only"]["mse"]["mean"]
    retention_after = summary["transfer_source_retention"]["mse"]["mean"]

    return {
        "experiment": "E002",
        "title": "Structure reuse: transfer to a related task",
        "research_question": (
            "Does experience from one task reduce the amount of learning needed for "
            "a related task?"
        ),
        "hypothesis": (
            "Warm-start transfer from a related rule lowers target-task error at equal "
            "target-task budget compared with fresh initialisation."
        ),
        "config": config.model_dump(mode="json"),
        "summary": summary,
        "criterion": {
            "statement": (
                "Transfer target MSE is at least 10% lower than the equal-budget fresh "
                "initialisation."
            ),
            "relative_improvement_vs_fresh": improvement,
            "source_retention_mse_deterioration": retention_after - retention_before,
            "met": bool(improvement >= 0.10),
        },
        "per_seed": per_seed,
        "limitations": [
            "Budget parity is measured in target epochs; pretraining compute is not included.",
            "Renamed-surface robustness is probed in E001 via feature permutation; a dedicated protocol is pending.",
            "A single source/target rule pair does not establish general transferability.",
        ],
    }


def run_e003(config: E003Config | None = None) -> dict[str, Any]:
    """Run E003: adapt to a changed rule while monitoring retention."""

    config = config or E003Config()
    per_seed: list[dict[str, Any]] = []

    for seed in config.seeds:
        stable_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.stable_rule, seed=seed)
        )
        changed_env = SyntheticWorld(
            SyntheticWorldConfig(rule=config.changed_rule, seed=seed + 300)
        )
        train, val, test = stable_env.generate_split(
            config.n_train, max(50, config.n_train // 5), config.n_test, seed=seed
        )
        new_train, new_val, new_test = changed_env.generate_split(
            config.n_new, config.n_new_val, config.n_test, seed=seed + 300
        )

        neural_config = config.neural.model_copy(update={"seed": seed})
        old_train_records = _observations(stable_env, train, seed)
        learner = Learner(
            LearnerConfig(feature_names=list(stable_env.feature_names), neural=neural_config)
        )
        learner.fit(
            old_train_records,
            validation_records=_observations(stable_env, val, seed + 1),
        )
        old_before = _neural_metrics(
            learner, test, _observations(stable_env, test, seed + 2)
        )
        new_before = _neural_metrics(
            learner, new_test, _observations(changed_env, new_test, seed + 301)
        )

        new_train_records = _observations(changed_env, new_train, seed + 300)
        new_val_records = _observations(changed_env, new_val, seed + 302)
        old_test_records = _observations(stable_env, test, seed + 2)
        new_test_records = _observations(changed_env, new_test, seed + 301)

        # Adaptation-speed curve: identical update data, increasing budgets.
        adaptation_curve: list[dict[str, Any]] = []
        for epochs in config.adaptation_epochs:
            candidate = _clone_learner(learner)
            controller = RevisionController(epochs=epochs)
            result = controller.controlled_update(
                candidate, new_train_records, new_val_records
            )
            after = _neural_metrics(candidate, new_test, new_test_records)
            retention = _neural_metrics(candidate, test, old_test_records)
            adaptation_curve.append(
                {
                    "epochs": epochs,
                    "accepted": result.accepted,
                    "validation_before": result.before,
                    "validation_after": result.after,
                    "new_task_mse": after["mse"],
                    "old_task_mse": retention["mse"],
                }
            )

        # Main controlled update at the largest declared budget.
        main = _clone_learner(learner)
        controller = RevisionController(epochs=max(config.adaptation_epochs))
        main_result = controller.controlled_update(
            main, new_train_records, new_val_records
        )
        old_after = _neural_metrics(main, test, old_test_records)
        new_after = _neural_metrics(main, new_test, new_test_records)

        # Exploratory consolidation variant: replay earlier experience while
        # adapting to the changed rule.
        replay_candidate = _clone_learner(learner)
        replay_controller = RevisionController(epochs=max(config.adaptation_epochs))
        replay_result = replay_controller.controlled_update(
            replay_candidate,
            new_train_records,
            new_val_records,
            replay_records=old_train_records,
        )
        replay_new = _neural_metrics(replay_candidate, new_test, new_test_records)
        replay_old = _neural_metrics(replay_candidate, test, old_test_records)

        # Isolated anomalous observation: should not be accepted on its own.
        anomaly = new_train_records[0].model_copy(
            update={
                "outcome": float(
                    np.mean([record.outcome for record in new_train_records])
                    + config.anomaly_magnitude
                ),
                "id": f"anomaly-{seed}",
            }
        )
        anomaly_candidate = _clone_learner(learner)
        anomaly_controller = RevisionController(epochs=max(config.adaptation_epochs))
        anomaly_result = anomaly_controller.controlled_update(
            anomaly_candidate, [anomaly], new_val_records
        )
        anomaly_metrics = _neural_metrics(
            anomaly_candidate, new_test, new_test_records
        )

        # Contaminated batch: measure the effect of one outlier inside real data.
        contaminated = _clone_learner(learner)
        contaminated_controller = RevisionController(epochs=max(config.adaptation_epochs))
        contaminated_result = contaminated_controller.controlled_update(
            contaminated, [*new_train_records, anomaly], new_val_records
        )
        contaminated_metrics = _neural_metrics(
            contaminated, new_test, new_test_records
        )

        per_seed.append(
            {
                "seed": seed,
                "old_task_before": old_before,
                "new_task_before": new_before,
                "old_task_after": old_after,
                "new_task_after": new_after,
                "update_accepted": main_result.accepted,
                "adaptation_curve": adaptation_curve,
                "single_anomaly": {
                    "accepted": anomaly_result.accepted,
                    "new_task_mse": anomaly_metrics["mse"],
                },
                "contaminated_batch": {
                    "accepted": contaminated_result.accepted,
                    "new_task_mse": contaminated_metrics["mse"],
                },
                "replay_consolidation": {
                    "accepted": replay_result.accepted,
                    "new_task_mse": replay_new["mse"],
                    "old_task_mse": replay_old["mse"],
                },
            }
        )

    new_before = summarize(entry["new_task_before"]["mse"] for entry in per_seed)
    new_after = summarize(entry["new_task_after"]["mse"] for entry in per_seed)
    old_before = summarize(entry["old_task_before"]["mse"] for entry in per_seed)
    old_after = summarize(entry["old_task_after"]["mse"] for entry in per_seed)
    acceptance_rate = float(
        np.mean([1.0 if entry["update_accepted"] else 0.0 for entry in per_seed])
    )
    anomaly_rate = float(
        np.mean([1.0 if entry["single_anomaly"]["accepted"] else 0.0 for entry in per_seed])
    )
    adaptation_improvement = (new_before["mean"] - new_after["mean"]) / new_before["mean"]

    return {
        "experiment": "E003",
        "title": "Representation revision under a changed rule",
        "research_question": (
            "Can the learner adapt when the underlying rule changes while preserving "
            "performance on the earlier task?"
        ),
        "hypothesis": (
            "Validation-gated updates reduce new-task error after a rule change and "
            "reject an isolated anomalous observation, with bounded old-task deterioration."
        ),
        "config": config.model_dump(mode="json"),
        "summary": {
            "new_task_before": new_before,
            "new_task_after": new_after,
            "old_task_before": old_before,
            "old_task_after": old_after,
            "old_task_deterioration": old_after["mean"] - old_before["mean"],
            "accepted_update_rate": acceptance_rate,
            "single_anomaly_acceptance_rate": anomaly_rate,
            "replay_consolidation_new_task_mse": summarize(
                entry["replay_consolidation"]["new_task_mse"] for entry in per_seed
            ),
            "replay_consolidation_old_task_mse": summarize(
                entry["replay_consolidation"]["old_task_mse"] for entry in per_seed
            ),
        },
        "criterion": {
            "statement": (
                "New-task MSE improves by at least 10% after the controlled update "
                "and an isolated anomalous observation is not accepted."
            ),
            "relative_new_task_improvement": adaptation_improvement,
            "met": bool(adaptation_improvement >= 0.10 and anomaly_rate == 0.0),
            "retention_reported_not_gated": True,
        },
        "per_seed": per_seed,
        "limitations": [
            "Retention is reported without a success threshold; the trade-off is exploratory.",
            "Rule-change pairs are synthetic and low-dimensional.",
            "A single anomalous record is a narrow robustness probe.",
            "Replay consolidation is an exploratory variant, not part of the gated criterion.",
        ],
    }


def save_result(result: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return target


def run_hybrid_ablation(config: HybridAblationConfig | None = None) -> dict[str, Any]:
    """Compare neural-only, symbolic-basis-only and a validation-selected hybrid."""

    config = config or HybridAblationConfig()
    per_seed: list[dict[str, Any]] = []

    for seed in config.seeds:
        env = SyntheticWorld(SyntheticWorldConfig(rule=config.rule, seed=seed))
        train, val, test = env.generate_split(
            config.n_train, config.n_val, config.n_test, seed=seed
        )
        train_records = _observations(env, train, seed)
        val_records = _observations(env, val, seed + 1)
        test_records = _observations(env, test, seed + 2)

        neural_config = config.neural.model_copy(update={"seed": seed})
        learner = Learner(
            LearnerConfig(feature_names=list(env.feature_names), neural=neural_config)
        )
        learner.fit(train_records, validation_records=val_records)

        symbolic = SymbolicRuleLearner(config.symbolic).fit(train.features, train.outcomes)

        neural_val = learner.predict(learner.encoder.transform(val_records))
        symbolic_val = symbolic.predict(val.features)
        neural_val_mse = float(np.mean((neural_val - val.outcomes) ** 2))
        symbolic_val_mse = float(np.mean((symbolic_val - val.outcomes) ** 2))
        selected = "symbolic" if symbolic_val_mse < neural_val_mse else "neural"

        neural_test = learner.predict(learner.encoder.transform(test_records))
        symbolic_test = symbolic.predict(test.features)
        hybrid_test = symbolic_test if selected == "symbolic" else neural_test

        metrics = {
            "neural_only": regression_report(test.outcomes, neural_test),
            "symbolic_only": regression_report(test.outcomes, symbolic_test),
            "hybrid_selected": regression_report(test.outcomes, hybrid_test),
            "oracle": regression_report(test.outcomes, RULES[config.rule].fn(test.features)),
        }
        per_seed.append(
            {
                "seed": seed,
                "metrics": metrics,
                "validation_mse": {
                    "neural": neural_val_mse,
                    "symbolic": symbolic_val_mse,
                },
                "selected": selected,
                "symbolic_expression": symbolic.expression(),
            }
        )

    methods = _method_names(per_seed)
    summary = {
        method: {
            metric: _aggregate(per_seed, method, metric)
            for metric in ("mse", "mae", "r2")
        }
        for method in methods
    }
    best_single = min(
        summary["neural_only"]["mse"]["mean"], summary["symbolic_only"]["mse"]["mean"]
    )
    hybrid_mse = summary["hybrid_selected"]["mse"]["mean"]
    ratio = hybrid_mse / best_single if best_single > 0 else float("inf")
    selections = [entry["selected"] for entry in per_seed]

    return {
        "experiment": "HYBRID",
        "title": "Neural vs symbolic-basis vs validation-selected hybrid",
        "research_question": (
            "Does a hybrid of the neural baseline and a low-level symbolic basis "
            "improve held-out prediction relative to either alone?"
        ),
        "hypothesis": (
            "Validation-selected hybridisation is no worse than the better single "
            "method on held-out data."
        ),
        "config": config.model_dump(mode="json"),
        "summary": summary,
        "criterion": {
            "statement": (
                "Hybrid held-out MSE is at most 5% worse than the better single method."
            ),
            "hybrid_to_best_single_ratio": ratio,
            "selected_counts": {
                "neural": selections.count("neural"),
                "symbolic": selections.count("symbolic"),
            },
            "met": bool(ratio <= 1.05),
        },
        "per_seed": per_seed,
        "limitations": [
            "The symbolic basis is a fixed prior over low-level operators, not emergent symbols.",
            "Budget parity is approximate: the symbolic fit is closed-form, the neural fit is iterative.",
            "Selection uses a single validation split and can overfit that split.",
        ],
    }
