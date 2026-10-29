"""A simple symbolic-regression baseline and hybrid selector.

This is not a concept-discovery engine. It fits a sparse linear model over a
small, explicitly declared basis of low-level operators (linear terms,
pairwise products, squares and trigonometric transforms). The basis is a prior
that must be reported honestly: the symbolic baseline only "proposes rules"
inside that basis. It exists to make the neural-vs-symbolic-vs-hybrid ablation
measurable, not to claim emergent symbol formation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from genesis.learning.neural import NeuralPredictor


class SymbolicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    l2: float = Field(default=1e-6, ge=0)
    coefficient_threshold: float = Field(default=1e-4, ge=0)
    include_products: bool = True
    include_squares: bool = True
    include_trigonometric: bool = False


@dataclass
class SymbolicRuleLearner:
    """Ridge regression with explicit term names and magnitude pruning."""

    config: SymbolicConfig = field(default_factory=SymbolicConfig)
    name: str = "symbolic_basis"
    _terms: list[str] = field(default_factory=list)
    _weights: np.ndarray | None = None
    _fitted: bool = False

    def _design(self, features: np.ndarray) -> tuple[np.ndarray, list[str]]:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError("features must be 2-D")
        n, d = features.shape
        columns = [np.ones(n)]
        names = ["1"]
        for i in range(d):
            columns.append(features[:, i])
            names.append(f"f{i + 1}")
        if self.config.include_squares:
            for i in range(d):
                columns.append(features[:, i] ** 2)
                names.append(f"f{i + 1}^2")
        if self.config.include_products:
            for i in range(d):
                for j in range(i + 1, d):
                    columns.append(features[:, i] * features[:, j])
                    names.append(f"f{i + 1}*f{j + 1}")
        if self.config.include_trigonometric:
            for i in range(d):
                columns.append(np.sin(features[:, i]))
                names.append(f"sin(f{i + 1})")
        return np.column_stack(columns), names

    def fit(self, features: np.ndarray, outcomes: np.ndarray) -> "SymbolicRuleLearner":
        design, names = self._design(features)
        outcomes = np.asarray(outcomes, dtype=float).reshape(-1)
        if design.shape[0] != outcomes.shape[0] or design.shape[0] == 0:
            raise ValueError("invalid training arrays")
        penalty = np.eye(design.shape[1]) * self.config.l2
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ outcomes)
        active = np.abs(weights) >= self.config.coefficient_threshold
        active[0] = True
        weights = np.where(active, weights, 0.0)
        self._terms = names
        self._weights = weights
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted or self._weights is None:
            raise RuntimeError("symbolic learner must be fitted before prediction")
        design, _ = self._design(features)
        return design @ self._weights

    def expression(self) -> list[dict[str, float | str]]:
        if not self._fitted or self._weights is None:
            raise RuntimeError("symbolic learner must be fitted before inspection")
        return [
            {"term": term, "coefficient": float(weight)}
            for term, weight in zip(self._terms, self._weights, strict=True)
            if abs(weight) > 0
        ]


@dataclass
class HybridLearner:
    """Select the model with the lower validation error, then predict with it."""

    symbolic: SymbolicRuleLearner = field(default_factory=SymbolicRuleLearner)
    neural: NeuralPredictor | None = None
    selected: str = "neural"
    _symbolic_mse: float = float("inf")
    _neural_mse: float = float("inf")

    def fit(
        self,
        train_features: np.ndarray,
        train_outcomes: np.ndarray,
        validation_features: np.ndarray,
        validation_outcomes: np.ndarray,
        neural: NeuralPredictor,
    ) -> "HybridLearner":
        self.neural = neural
        self.symbolic.fit(train_features, train_outcomes)
        neural.fit(train_features, train_outcomes)
        self._symbolic_mse = float(
            np.mean((self.symbolic.predict(validation_features) - validation_outcomes) ** 2)
        )
        self._neural_mse = float(
            np.mean((self.neural.predict(validation_features) - validation_outcomes) ** 2)
        )
        self.selected = "symbolic" if self._symbolic_mse < self._neural_mse else "neural"
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.neural is None:
            raise RuntimeError("hybrid learner must be fitted before prediction")
        if self.selected == "symbolic":
            return self.symbolic.predict(features)
        return self.neural.predict(features)

    @property
    def validation_errors(self) -> dict[str, float]:
        return {"symbolic": self._symbolic_mse, "neural": self._neural_mse}
