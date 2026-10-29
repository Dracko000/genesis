"""Input-encoding interface and the initial inspectable numeric encoder.

The encoder is deliberately narrow: it maps only the named features (and an
optional action) to a vector. Targets are never read. Normalisation statistics
are fitted on training data only and then frozen, which prevents evaluation
leakage.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from genesis.experience.schema import Observation


@dataclass(frozen=True)
class FeatureSpec:
    """The ordered feature contract of a representation."""

    feature_names: tuple[str, ...]
    include_action: bool = False

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")

    @property
    def input_dim(self) -> int:
        return len(self.feature_names) + (1 if self.include_action else 0)


@dataclass
class NumericFeatureEncoder:
    """Standardise named numeric features using frozen training statistics."""

    spec: FeatureSpec
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))
    _fitted: bool = False

    def __post_init__(self) -> None:
        if not self._fitted:
            self.mean = np.zeros(self.spec.input_dim)
            self.std = np.ones(self.spec.input_dim)

    @staticmethod
    def _vector(record: Observation, spec: FeatureSpec) -> np.ndarray:
        missing = [name for name in spec.feature_names if name not in record.features]
        if missing:
            raise KeyError(f"observation {record.id!r} is missing features: {missing}")
        values = [float(record.features[name]) for name in spec.feature_names]
        if spec.include_action:
            values.append(0.0 if record.action is None else float(record.action))
        vector = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"observation {record.id!r} contains non-finite inputs")
        return vector

    def fit(self, records: Iterable[Observation]) -> "NumericFeatureEncoder":
        """Fit normalisation on training records only."""

        matrix = self._matrix(records)
        if matrix.shape[0] == 0:
            raise ValueError("cannot fit encoder on zero records")
        self.mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std < 1e-8] = 1.0
        self.std = std
        self._fitted = True
        return self

    def _matrix(self, records: Iterable[Observation]) -> np.ndarray:
        rows = [self._vector(record, self.spec) for record in records]
        if not rows:
            return np.zeros((0, self.spec.input_dim), dtype=float)
        return np.vstack(rows)

    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.spec.input_dim:
            raise ValueError(
                f"expected a matrix with {self.spec.input_dim} columns, got {matrix.shape}"
            )
        return (matrix - self.mean) / self.std

    def fit_transform(self, records: Iterable[Observation]) -> np.ndarray:
        self.fit(records)
        return self.transform(records)

    def transform(self, records: Iterable[Observation]) -> np.ndarray:
        return self.transform_matrix(self._matrix(records))

    def state(self) -> dict[str, object]:
        return {
            "feature_names": list(self.spec.feature_names),
            "include_action": self.spec.include_action,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fitted": self._fitted,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "NumericFeatureEncoder":
        names = state.get("feature_names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("invalid encoder state: feature_names")
        spec = FeatureSpec(tuple(names), bool(state.get("include_action", False)))
        encoder = cls(spec)
        encoder.mean = np.asarray(state.get("mean", np.zeros(spec.input_dim)), dtype=float)
        encoder.std = np.asarray(state.get("std", np.ones(spec.input_dim)), dtype=float)
        if encoder.mean.shape != (spec.input_dim,) or encoder.std.shape != (spec.input_dim,):
            raise ValueError("invalid encoder state: normalisation shape mismatch")
        if np.any(encoder.std <= 0):
            raise ValueError("invalid encoder state: std must be positive")
        encoder._fitted = bool(state.get("fitted", True))
        return encoder

    def to_spec(self) -> dict[str, Sequence[float] | str | bool | list[str]]:
        return self.state()
