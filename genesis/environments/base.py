"""Environment interface shared by synthetic worlds.

Environments expose inputs and outcomes as arrays. Ground-truth rules live in
the environment/evaluation layer and are not part of a learner's input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """A feature matrix, target vector and neutral feature names."""

    features: np.ndarray
    outcomes: np.ndarray
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=float)
        outcomes = np.asarray(self.outcomes, dtype=float).reshape(-1)
        if features.ndim != 2:
            raise ValueError("features must be 2-D")
        if features.shape[0] != outcomes.shape[0]:
            raise ValueError("features and outcomes must have the same length")
        if self.feature_names and len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names length must match feature width")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "outcomes", outcomes)

    def __len__(self) -> int:
        return self.features.shape[0]


class Environment(ABC):
    """Minimal environment contract."""

    feature_names: tuple[str, ...]

    @abstractmethod
    def generate(self, n: int, seed: int) -> Dataset:
        """Generate ``n`` samples deterministically from ``seed``."""

    @abstractmethod
    def generate_split(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        seed: int,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Return disjoint training, validation and test datasets."""
