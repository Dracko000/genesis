"""Background experiment execution for the API.

Long-running research runs are executed on daemon worker threads so that normal
request handling is not blocked. Status transitions are persisted immediately:
``running`` -> ``completed`` or ``failed`` with the error text preserved.
"""

from __future__ import annotations

import threading
from typing import Any

from genesis.evaluation.experiments import (
    E001Config,
    E002Config,
    E003Config,
    run_e001,
    run_e002,
    run_e003,
)
from genesis.evaluation.intervention_experiments import (
    InterventionConfig,
    run_intervention_probe,
)
from genesis.evaluation.retention_experiments import (
    E003RetentionConfig,
    run_e003_retention,
)
from genesis.evaluation.transfer_experiments import (
    E002DiagnosticsConfig,
    run_e002_diagnostics,
)
from genesis.storage.repository import ExperimentRecord, Repository
from services.api.settings import Settings

RUNNERS = {
    "E001": (E001Config, run_e001),
    "E002": (E002Config, run_e002),
    "E002-DIAG": (E002DiagnosticsConfig, run_e002_diagnostics),
    "E003": (E003Config, run_e003),
    "E003-RETENTION": (E003RetentionConfig, run_e003_retention),
    "E005": (InterventionConfig, run_intervention_probe),
}


def build_config(kind: str, overrides: dict[str, Any]) -> Any:
    if kind not in RUNNERS:
        raise ValueError(f"unknown experiment {kind!r}")
    config_cls, _ = RUNNERS[kind]
    return config_cls.model_validate(overrides)


def _flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        if "mean" in value and isinstance(value["mean"], (int, float)):
            flattened[prefix or "value"] = float(value["mean"])
            return flattened
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_metrics(child, name))
    elif isinstance(value, (int, float)):
        flattened[prefix or "value"] = float(value)
    return flattened


def _execute(
    repository: Repository,
    settings: Settings,
    experiment_id: str,
    kind: str,
    overrides: dict[str, Any],
) -> None:
    try:
        config = build_config(kind, overrides)
        _, runner = RUNNERS[kind]
        result = runner(config)
        for name, value in _flatten_metrics(result.get("summary", {})).items():
            repository.add_metric(
                experiment_id=experiment_id,
                name=f"{kind}.{name}",
                value=value,
                payload={"experiment": kind},
            )
        if kind == "E003":
            for entry in result.get("per_seed", []):
                repository.add_revision(
                    experiment_id=experiment_id,
                    payload={
                        "experiment": "E003",
                        "seed": entry.get("seed"),
                        "update_accepted": entry.get("update_accepted"),
                        "new_task_before_mse": entry.get("new_task_before", {}).get("mse"),
                        "new_task_after_mse": entry.get("new_task_after", {}).get("mse"),
                        "old_task_before_mse": entry.get("old_task_before", {}).get("mse"),
                        "old_task_after_mse": entry.get("old_task_after", {}).get("mse"),
                        "single_anomaly": entry.get("single_anomaly"),
                        "adaptation_curve": entry.get("adaptation_curve"),
                        "replay_consolidation": entry.get("replay_consolidation"),
                    },
                )
        if kind == "E003-RETENTION":
            for entry in result.get("per_seed", []):
                repository.add_revision(
                    experiment_id=experiment_id,
                    payload={
                        "experiment": "E003-v2",
                        "seed": entry.get("seed"),
                        "curve": entry.get("curve"),
                        "isolated_anomaly": entry.get("isolated_anomaly"),
                        "contaminated_batch": entry.get("contaminated_batch"),
                        "criterion": result.get("criterion"),
                    },
                )
        repository.add_log(
            level="INFO",
            message=f"{kind} experiment completed",
            payload={"experiment_id": experiment_id},
        )
        # Publish completion last so a client that sees ``completed`` can read
        # every metric and revision row already persisted for the run.
        repository.update_experiment(
            experiment_id, status="completed", result=result
        )
    except Exception as exc:  # noqa: BLE001 - status must record every failure
        repository.update_experiment(
            experiment_id, status="failed", error=f"{type(exc).__name__}: {exc}"
        )
        repository.add_log(
            level="ERROR",
            message=f"{kind} experiment failed",
            payload={"experiment_id": experiment_id, "error": str(exc)},
        )


def launch_experiment(
    repository: Repository,
    settings: Settings,
    record: ExperimentRecord,
    kind: str,
    overrides: dict[str, Any],
) -> threading.Thread:
    thread = threading.Thread(
        target=_execute,
        args=(repository, settings, record.id, kind, overrides),
        name=f"genesis-{kind}-{record.id}",
        daemon=True,
    )
    thread.start()
    return thread
