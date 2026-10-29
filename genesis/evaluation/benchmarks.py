"""Baselines used by the experiments.

Baselines share a tiny protocol: ``fit(X, y)`` and ``predict(X)``. They receive
the same feature matrix and targets as the neural learner. The oracle baseline
is for upper-reference reporting only and is never given to the learner.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from genesis.environments.synthetic_world import RULES


class Baseline(Protocol):
    name: str

    def fit(self, features: np.ndarray, outcomes: np.ndarray) -> "Baseline": ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


class MeanBaseline:
    """Predict the training-set mean (a sanity floor)."""

    name = "mean"

    def __init__(self) -> None:
        self._mean = 0.0
        self._fitted = False

    def fit(self, features: np.ndarray, outcomes: np.ndarray) -> "MeanBaseline":
        outcomes = np.asarray(outcomes, dtype=float).reshape(-1)
        if outcomes.size == 0:
            raise ValueError("cannot fit on zero samples")
        self._mean = float(outcomes.mean())
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("baseline must be fitted before prediction")
        features = np.asarray(features, dtype=float)
        return np.full(features.shape[0], self._mean)


class NearestNeighbourBaseline:
    """1-nearest-neighbour regression over standardised features.

    This is the memorisation / nearest-example baseline: it can only reuse
    training outcomes and cannot extrapolate to unseen regions by construction.
    """

    name = "nearest_neighbour"

    def __init__(self) -> None:
        self._features: np.ndarray | None = None
        self._outcomes: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, features: np.ndarray, outcomes: np.ndarray) -> "NearestNeighbourBaseline":
        features = np.asarray(features, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float).reshape(-1)
        if features.ndim != 2 or features.shape[0] != outcomes.shape[0]:
            raise ValueError("invalid training arrays")
        if features.shape[0] == 0:
            raise ValueError("cannot fit on zero samples")
        self._mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-8] = 1.0
        self._std = std
        self._features = (features - self._mean) / self._std
        self._outcomes = outcomes
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._features is None or self._outcomes is None:
            raise RuntimeError("baseline must be fitted before prediction")
        features = np.asarray(features, dtype=float)
        scaled = (features - self._mean) / self._std
        distances = np.sum((scaled[:, None, :] - self._features[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1)
        return self._outcomes[nearest]


class LinearRidgeBaseline:
    """Closed-form ridge regression (a simple parametric control)."""

    name = "linear_ridge"

    def __init__(self, l2: float = 1e-3) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.l2 = l2
        self._weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, outcomes: np.ndarray) -> "LinearRidgeBaseline":
        features = np.asarray(features, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float).reshape(-1)
        if features.ndim != 2 or features.shape[0] != outcomes.shape[0]:
            raise ValueError("invalid training arrays")
        design = np.hstack([np.ones((features.shape[0], 1)), features])
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        self._weights = np.linalg.solve(design.T @ design + penalty, design.T @ outcomes)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("baseline must be fitted before prediction")
        features = np.asarray(features, dtype=float)
        design = np.hstack([np.ones((features.shape[0], 1)), features])
        return design @ self._weights


class OracleBaseline:
    """Upper reference: predict with the noiseless ground-truth rule.

    Only evaluation code may construct this baseline. With label noise the
    oracle's MSE equals the irreducible noise variance, not zero.
    """

    name = "oracle"

    def __init__(self, rule: str) -> None:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}")
        self.rule = rule
        self._fn = RULES[rule].fn

    def fit(self, features: np.ndarray, outcomes: np.ndarray) -> "OracleBaseline":
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        return self._fn(features)
