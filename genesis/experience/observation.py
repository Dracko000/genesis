"""Helpers for turning arrays and environment samples into observations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from genesis.experience.provenance import Provenance
from genesis.experience.schema import Observation


def observations_from_arrays(
    features: np.ndarray,
    outcomes: np.ndarray | Sequence[float] | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    provenance: Provenance | None = None,
    episode_id: str | None = None,
    start_sequence: int = 0,
) -> list[Observation]:
    """Build validated observations from a 2-D feature matrix.

    The outcome column is passed separately and is never merged into the
    feature dictionary, which keeps the target/target-leakage boundary explicit.
    """

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2-D array")

    names = list(feature_names) if feature_names is not None else [
        f"f{i + 1}" for i in range(features.shape[1])
    ]
    if len(names) != features.shape[1]:
        raise ValueError("feature_names length must match feature matrix width")

    outcome_values: list[float | None]
    if outcomes is None:
        outcome_values = [None] * features.shape[0]
    else:
        outcome_array = np.asarray(outcomes, dtype=float).reshape(-1)
        if outcome_array.shape[0] != features.shape[0]:
            raise ValueError("outcomes length must match feature matrix rows")
        outcome_values = [float(value) for value in outcome_array]

    prov = provenance or Provenance(source="array", created_by="genesis.observation")
    records: list[Observation] = []
    for index in range(features.shape[0]):
        row = {name: float(value) for name, value in zip(names, features[index], strict=True)}
        records.append(
            Observation(
                features=row,
                outcome=outcome_values[index],
                provenance=prov,
                episode_id=episode_id,
                sequence=start_sequence + index,
            )
        )
    return records


def select_labeled(records: Iterable[Observation]) -> list[Observation]:
    """Keep only observed records that carry a target."""

    return [record for record in records if record.has_target]
