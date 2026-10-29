"""Container describing a learned representation at a point in time.

A :class:`LatentState` bundles parameter tensors with the exact configuration
and input contract needed to reproduce predictions. It is intentionally not a
database row: persistence is handled by :mod:`genesis.representation.checkpoint`
and :mod:`genesis.storage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from genesis.representation.interface import FeatureSpec


@dataclass
class LatentState:
    """Learned parameters plus their reproducibility contract."""

    spec: FeatureSpec
    state_dict: dict[str, Any]
    model_config: dict[str, Any] = field(default_factory=dict)
    target_mean: float = 0.0
    target_std: float = 1.0
    encoder_state: dict[str, Any] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_std <= 0:
            raise ValueError("target_std must be positive")
        if list(self.spec.feature_names) != list(self.encoder_state.get("feature_names", self.spec.feature_names)):
            raise ValueError("encoder feature names must match the representation spec")

    def summary(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.spec.feature_names),
            "include_action": self.spec.include_action,
            "input_dim": self.spec.input_dim,
            "model_config": dict(self.model_config),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "training_summary": dict(self.training_summary),
        }

    @staticmethod
    def normalize_target(target: np.ndarray, mean: float, std: float) -> np.ndarray:
        return (np.asarray(target, dtype=float) - mean) / std

    @staticmethod
    def denormalize_target(target: np.ndarray, mean: float, std: float) -> np.ndarray:
        return np.asarray(target, dtype=float) * std + mean
