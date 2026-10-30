"""Request/response schemas for the Genesis API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from genesis.experience.schema import Observation
from genesis.learning.neural import NeuralConfig


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    database: str


class VersionResponse(BaseModel):
    name: str = "genesis"
    version: str
    api_schema: int = 1


class LearningStepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(default="manual_learning_step", max_length=120)
    feature_names: list[str] | None = None
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    validation_fraction: float = Field(default=0.2, ge=0.0, lt=0.9)
    save_checkpoint: bool = True
    seed: int | None = None


class LearningStepResponse(BaseModel):
    experiment_id: str
    status: str
    metrics: dict[str, float]
    checkpoint_id: str | None
    n_train: int
    n_validation: int
    learner: dict[str, Any]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float]
    action: float | None = None
    actual: float | None = None
    checkpoint_id: str | None = None


class PredictResponse(BaseModel):
    prediction: float
    actual: float | None = None
    error: float | None = None
    checkpoint_id: str | None = None
    feature_names: list[str]


class ExperimentStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: Literal[
        "E001", "E002", "E002-DIAG", "E003", "E003-RETENTION", "E005"
    ]
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentView(BaseModel):
    id: str
    name: str
    status: str
    config: dict[str, Any]
    seed: int | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentView]


class MetricView(BaseModel):
    name: str
    value: float
    step: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RevisionView(BaseModel):
    id: int
    experiment_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RevisionListResponse(BaseModel):
    revisions: list[RevisionView]


class CheckpointView(BaseModel):
    checkpoint_id: str
    path: str
    metadata: dict[str, Any]
    created_at: datetime


class CheckpointListResponse(BaseModel):
    checkpoints: list[CheckpointView]


class RestoreResponse(BaseModel):
    checkpoint_id: str
    restored: bool
    learner: dict[str, Any]
    metadata: dict[str, Any]


class ExperienceListResponse(BaseModel):
    experiences: list[Observation]


class LogView(BaseModel):
    id: int
    level: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ResearchLogResponse(BaseModel):
    markdown: str
    logs: list[LogView]
