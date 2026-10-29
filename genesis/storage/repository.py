"""Storage-agnostic repository contract.

The interface deliberately exposes only operations the research workflow
needs. Experiments and checkpoints are append-only from the caller's point of
view: there is no delete operation, and checkpoint creation refuses to replace
an existing id. This makes an accidental request unable to erase history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from genesis.experience.schema import Observation
from genesis.representation.checkpoint import CheckpointMetadata


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ExperimentRecord:
    id: str
    name: str
    status: str
    config: dict[str, Any]
    seed: int | None = None
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class MetricRecord:
    id: int
    experiment_id: str
    name: str
    value: float
    step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    path: str
    metadata: dict[str, Any]
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class LogRecord:
    id: int
    level: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)


class Repository(ABC):
    """Persistence contract; PostgreSQL can implement this later unchanged."""

    @abstractmethod
    def add_experience(self, observation: Observation) -> Observation: ...

    @abstractmethod
    def get_experience(self, observation_id: str) -> Observation | None: ...

    @abstractmethod
    def list_experiences(self, *, limit: int = 100, offset: int = 0) -> list[Observation]: ...

    @abstractmethod
    def create_experiment(
        self,
        *,
        experiment_id: str,
        name: str,
        status: str,
        config: dict[str, Any],
        seed: int | None = None,
    ) -> ExperimentRecord: ...

    @abstractmethod
    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> ExperimentRecord: ...

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None: ...

    @abstractmethod
    def list_experiments(self, *, limit: int = 100) -> list[ExperimentRecord]: ...

    @abstractmethod
    def add_metric(
        self,
        *,
        experiment_id: str,
        name: str,
        value: float,
        step: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> MetricRecord: ...

    @abstractmethod
    def list_metrics(self, experiment_id: str) -> list[MetricRecord]: ...

    @abstractmethod
    def add_checkpoint(self, *, path: str, metadata: CheckpointMetadata) -> CheckpointRecord: ...

    @abstractmethod
    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None: ...

    @abstractmethod
    def list_checkpoints(self, *, limit: int = 100) -> list[CheckpointRecord]: ...

    @abstractmethod
    def add_revision(self, *, experiment_id: str | None, payload: dict[str, Any]) -> int: ...

    @abstractmethod
    def list_revisions(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def add_log(
        self,
        *,
        level: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> LogRecord: ...

    @abstractmethod
    def list_logs(self, *, limit: int = 200) -> list[LogRecord]: ...
