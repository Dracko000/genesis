"""Representation interventions: similarity and causal unit ablation.

These helpers answer a narrow question: are the hidden features learned on one
task the *same* features used on another? Similarity is measured with linear
CKA, and causal relevance is probed by zeroing units during the forward pass.
Nothing here assumes that a representation is symbolic or interpretable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from genesis.learning.learner import Learner


def _center(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True)


def linear_cka(features_x: np.ndarray, features_y: np.ndarray) -> float:
    """Linear centred kernel alignment between two representation matrices.

    Returns a value in ``[0, 1]`` (1 means identical linear structure up to an
    orthogonal transform). Rows must correspond to the same examples.
    """

    x = np.asarray(features_x, dtype=float)
    y = np.asarray(features_y, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("features must be 2-D matrices")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features must have the same number of rows")
    if x.shape[0] < 2:
        raise ValueError("CKA requires at least two examples")

    x_centered = _center(x)
    y_centered = _center(y)
    cross = np.linalg.norm(x_centered.T @ y_centered, ord="fro") ** 2
    norm_x = np.linalg.norm(x_centered.T @ x_centered, ord="fro")
    norm_y = np.linalg.norm(y_centered.T @ y_centered, ord="fro")
    denominator = norm_x * norm_y
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(cross / denominator, 0.0, 1.0))


def layer_cka(
    learner: Learner, features_x: np.ndarray, features_y: np.ndarray
) -> list[float]:
    """Per-hidden-layer CKA between two feature sets."""

    activations_x = learner.predictor.hidden_activations(features_x)
    activations_y = learner.predictor.hidden_activations(features_y)
    return [
        linear_cka(activation_x, activation_y)
        for activation_x, activation_y in zip(
            activations_x, activations_y, strict=True
        )
    ]


def mean_absolute_activation(learner: Learner, features: np.ndarray) -> list[np.ndarray]:
    """Mean absolute activation per hidden unit, one vector per layer."""

    activations = learner.predictor.hidden_activations(features)
    return [np.mean(np.abs(activation), axis=0) for activation in activations]


def top_units(profile: np.ndarray, k: int) -> list[int]:
    if k < 0:
        raise ValueError("k must be non-negative")
    k = min(k, profile.shape[0])
    return [int(index) for index in np.argsort(profile)[::-1][:k]]


def matched_random_units(width: int, k: int, seed: int) -> list[int]:
    if k < 0 or k > width:
        raise ValueError("k must be within [0, width]")
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(width, size=k, replace=False))


def ablation_effect(
    learner: Learner,
    features: np.ndarray,
    outcomes: np.ndarray,
    layer: int,
    units: Sequence[int],
) -> dict[str, Any]:
    """Compare MSE with and without zeroing selected hidden units."""

    outcomes = np.asarray(outcomes, dtype=float).reshape(-1)
    baseline_predictions = learner.predict(features)
    ablated_predictions = learner.predictor.predict_with_ablation(features, layer, units)
    baseline_mse = float(np.mean((baseline_predictions - outcomes) ** 2))
    ablated_mse = float(np.mean((ablated_predictions - outcomes) ** 2))
    return {
        "layer": layer,
        "units": list(units),
        "mse_before": baseline_mse,
        "mse_after": ablated_mse,
        "mse_delta": ablated_mse - baseline_mse,
        "relative_delta": (
            (ablated_mse - baseline_mse) / baseline_mse if baseline_mse > 0 else 0.0
        ),
    }
