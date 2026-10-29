"""Dataset splitting helpers with explicit target/input separation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from genesis.environments.base import Dataset


def split_indices(n: int, fractions: Sequence[float], seed: int = 7) -> list[np.ndarray]:
    """Split ``range(n)`` into disjoint index arrays."""

    if n < 1:
        raise ValueError("n must be >= 1")
    if not fractions or any(f < 0 for f in fractions):
        raise ValueError("fractions must be non-negative")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    permutation = np.random.default_rng(seed).permutation(n)
    sizes = [int(round(f * n)) for f in fractions]
    sizes[-1] = n - sum(sizes[:-1])
    if any(size < 0 for size in sizes):
        raise ValueError("fractions produce a negative split")
    splits: list[np.ndarray] = []
    cursor = 0
    for size in sizes:
        splits.append(permutation[cursor : cursor + size])
        cursor += size
    return splits


def subset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    return Dataset(
        dataset.features[indices], dataset.outcomes[indices], dataset.feature_names
    )


def train_val_test(
    dataset: Dataset,
    *,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 7,
) -> tuple[Dataset, Dataset, Dataset]:
    indices = split_indices(len(dataset), fractions, seed=seed)
    return tuple(subset(dataset, idx) for idx in indices)  # type: ignore[return-value]


def concat(left: Dataset, right: Dataset) -> Dataset:
    if left.feature_names != right.feature_names:
        raise ValueError("cannot concatenate datasets with different feature names")
    return Dataset(
        np.vstack([left.features, right.features]),
        np.concatenate([left.outcomes, right.outcomes]),
        left.feature_names,
    )


def assert_no_target_in_features(features: np.ndarray, outcomes: np.ndarray) -> None:
    """Guard against an accidental exact target copy inside the input matrix.

    This is a cheap structural check, not a semantic proof: it rejects the
    common leakage bug where the target column is appended to the features.
    """

    features = np.asarray(features, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float).reshape(-1)
    for column in range(features.shape[1]):
        if np.array_equal(features[:, column], outcomes):
            raise ValueError(f"feature column {column} is an exact copy of the target")
