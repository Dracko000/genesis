"""Experience schemas.

An :class:`Observation` is a single structured experience: input features,
an optional action, an optional observed outcome, optional context and
provenance. Outcomes are optional because unlabeled experience is still
useful, but the learner will only train on records that carry a target.

The schema enforces the distinction between observations and inferred
explanations. A record whose provenance says ``inferred=True`` must have
``kind=inference`` and vice versa. This makes it impossible to silently store
a hypothesis as a verified observation.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from genesis.experience.provenance import Provenance


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExperienceKind(str, Enum):
    """Whether a record is observed or inferred."""

    OBSERVATION = "observation"
    INFERENCE = "inference"


class Observation(BaseModel):
    """A single structured experience record."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=80)
    episode_id: str | None = Field(default=None, max_length=80)
    sequence: int | None = Field(default=None, ge=0)
    timestamp: datetime = Field(default_factory=_utc_now)

    features: dict[str, float] = Field(
        default_factory=dict,
        description="Named numeric input features. Target values must never appear here.",
    )
    text: str | None = Field(default=None, max_length=100_000)
    action: float | None = None
    action_label: str | None = Field(default=None, max_length=200)
    context: dict[str, Any] = Field(default_factory=dict)

    outcome: float | None = Field(
        default=None,
        description="Observed scalar target. None means the record is unlabeled.",
    )
    feedback: float | None = Field(
        default=None,
        description="Optional scalar human/teacher feedback, kept separate from outcome.",
    )

    provenance: Provenance = Field(default_factory=Provenance)
    kind: ExperienceKind = ExperienceKind.OBSERVATION

    @field_validator("features")
    @classmethod
    def _validate_features(cls, values: dict[str, float]) -> dict[str, float]:
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("feature names must be non-empty strings")
            if not math.isfinite(value):
                raise ValueError(f"feature {name!r} must be finite, got {value!r}")
        return values

    @field_validator("outcome", "action", "feedback")
    @classmethod
    def _validate_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError(f"value must be finite, got {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_kind_and_provenance(self) -> "Observation":
        if self.kind is ExperienceKind.INFERENCE and not self.provenance.inferred:
            raise ValueError(
                "inference records require provenance.inferred=True; "
                "never store a hypothesis as a verified observation"
            )
        if self.kind is ExperienceKind.OBSERVATION and self.provenance.inferred:
            raise ValueError(
                "provenance.inferred=True is incompatible with kind='observation'"
            )
        return self

    @property
    def has_target(self) -> bool:
        """True when this record can be used as supervised training signal."""

        return self.kind is ExperienceKind.OBSERVATION and self.outcome is not None

    def feature_vector(self, feature_names: list[str]) -> list[float]:
        """Return features in a fixed order, raising if a feature is missing."""

        missing = [name for name in feature_names if name not in self.features]
        if missing:
            raise KeyError(f"observation {self.id!r} is missing features: {missing}")
        return [float(self.features[name]) for name in feature_names]
