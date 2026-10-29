"""Replay-based consolidation.

Continuing to train only on new experience can erase earlier competence. A
simple reservoir replay buffer retains a bounded, deterministic mixture of
earlier observations so that new updates can be trained alongside them.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from genesis.experience.observation import select_labeled
from genesis.experience.schema import Observation
from genesis.learning.learner import Learner


class ReplayBuffer:
    """Deterministic reservoir sampling of labeled observations."""

    def __init__(self, capacity: int, seed: int = 7) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._records: list[Observation] = []
        self._seen = 0

    def __len__(self) -> int:
        return len(self._records)

    def add(self, records: Iterable[Observation]) -> None:
        for record in select_labeled(records):
            self._seen += 1
            if len(self._records) < self.capacity:
                self._records.append(record)
            else:
                index = self._rng.randrange(self._seen)
                if index < self.capacity:
                    self._records[index] = record

    def sample(self, n: int) -> list[Observation]:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n >= len(self._records):
            return list(self._records)
        return self._rng.sample(self._records, n)

    def all(self) -> list[Observation]:
        return list(self._records)


def consolidate(
    learner: Learner,
    new_records: Iterable[Observation],
    buffer: ReplayBuffer,
    *,
    replay_fraction: float = 0.5,
) -> dict[str, object]:
    """Refit on new experience mixed with replay samples.

    ``replay_fraction`` is the share of the combined training set drawn from
    the buffer. The caller decides whether to keep the resulting parameters;
    see :class:`~genesis.learning.revision.RevisionController`.
    """

    if not 0.0 <= replay_fraction <= 1.0:
        raise ValueError("replay_fraction must be within [0, 1]")
    new = select_labeled(new_records)
    if not new:
        raise ValueError("no labeled new records")
    replay_count = int(round(len(new) * replay_fraction / max(1e-9, 1.0 - replay_fraction)))
    replay = buffer.sample(replay_count)
    combined = [*new, *replay]
    return learner.fit(combined)
