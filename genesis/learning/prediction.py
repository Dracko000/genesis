"""Typed prediction records.

A prediction is stored *before* the outcome is observed whenever the pipeline
supports it. The error is therefore an auditable observation-level fact rather
than a reconstructed value.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionRecord(BaseModel):
    """One prediction, its target, and the checkpoint that produced it."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str | None = None
    prediction: float
    actual: float | None = None
    error: float | None = None
    checkpoint_id: str | None = None
    model_version: str = "0.1.0"
    created_at: datetime = Field(default_factory=_utc_now)
    features: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


def make_prediction_record(
    prediction: float,
    *,
    actual: float | None = None,
    observation_id: str | None = None,
    checkpoint_id: str | None = None,
    model_version: str = "0.1.0",
    features: dict[str, float] | None = None,
    metadata: dict[str, object] | None = None,
) -> PredictionRecord:
    error = None if actual is None else float(prediction) - float(actual)
    return PredictionRecord(
        observation_id=observation_id,
        prediction=float(prediction),
        actual=None if actual is None else float(actual),
        error=error,
        checkpoint_id=checkpoint_id,
        model_version=model_version,
        features=features or {},
        metadata=metadata or {},
    )
