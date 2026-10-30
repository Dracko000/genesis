"""A small synthetic world with configurable, hidden ground-truth rules.

Design notes
------------
* Features are sampled uniformly from ``[-1, 1]`` and have neutral names
  (``f1``, ``f2``, ...) to reduce accidental semantic leakage.
* Ground-truth rules are functions defined here and in evaluation only. The
  learner receives feature values and targets, never the rule.
* Every random draw comes from ``numpy.random.default_rng(seed)`` so runs are
  reproducible across platforms for the same NumPy version.
* Three modes are supported: stable-rule, changed-rule (via
  :meth:`SyntheticWorld.with_rule`) and compositional splits. Controlled
  label noise is configurable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from genesis.environments.base import Dataset, Environment
from genesis.experience.observation import observations_from_arrays
from genesis.experience.provenance import Provenance
from genesis.experience.schema import Observation


@dataclass(frozen=True)
class RuleSpec:
    """A documented hidden rule mapping feature rows to outcomes."""

    name: str
    description: str
    fn: Callable[[np.ndarray], np.ndarray]


def _linear(x: np.ndarray) -> np.ndarray:
    return 0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.2 * x[:, 2] + 0.3


def _interaction(x: np.ndarray) -> np.ndarray:
    return 1.2 * x[:, 0] * x[:, 1] + 0.6 * x[:, 2]


def _trigonometric(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * x[:, 0]) + 0.5 * x[:, 1] * x[:, 2] - 0.3


def _quadratic(x: np.ndarray) -> np.ndarray:
    return x[:, 0] ** 2 - x[:, 1] ** 2 + 0.5 * x[:, 2]


def _changed(x: np.ndarray) -> np.ndarray:
    return -0.7 * x[:, 0] + 0.9 * x[:, 1] * x[:, 2] + 0.1


def _linear_alt(x: np.ndarray) -> np.ndarray:
    return -1.1 * x[:, 0] + 0.7 * x[:, 1] - 0.4 * x[:, 2] + 0.2


def _interaction_alt(x: np.ndarray) -> np.ndarray:
    return -1.1 * x[:, 0] * x[:, 1] + 0.4 * x[:, 2] - 0.2


def _partial_interaction(x: np.ndarray) -> np.ndarray:
    return 1.2 * x[:, 0] * x[:, 1] + 0.2 * x[:, 0] - 0.3


RULES: dict[str, RuleSpec] = {
    "linear": RuleSpec(
        "linear",
        "y = 0.8*f1 - 0.5*f2 + 0.2*f3 + 0.3 (additive, no interactions)",
        _linear,
    ),
    "interaction": RuleSpec(
        "interaction",
        "y = 1.2*f1*f2 + 0.6*f3 (requires a multiplicative interaction)",
        _interaction,
    ),
    "trigonometric": RuleSpec(
        "trigonometric",
        "y = sin(2*f1) + 0.5*f2*f3 - 0.3 (non-monotonic plus interaction)",
        _trigonometric,
    ),
    "quadratic": RuleSpec(
        "quadratic",
        "y = f1^2 - f2^2 + 0.5*f3 (symmetric, non-monotonic)",
        _quadratic,
    ),
    "changed": RuleSpec(
        "changed",
        "y = -0.7*f1 + 0.9*f2*f3 + 0.1 (post-change rule for E003)",
        _changed,
    ),
    "linear_alt": RuleSpec(
        "linear_alt",
        "y = -1.1*f1 + 0.7*f2 - 0.4*f3 + 0.2 (same additive structure, new coefficients)",
        _linear_alt,
    ),
    "interaction_alt": RuleSpec(
        "interaction_alt",
        "y = -1.1*f1*f2 + 0.4*f3 - 0.2 (same interaction structure, new signs/coefficients)",
        _interaction_alt,
    ),
    "partial_interaction": RuleSpec(
        "partial_interaction",
        "y = 1.2*f1*f2 + 0.2*f1 - 0.3 (interaction plus an additive term)",
        _partial_interaction,
    ),
}


@dataclass(frozen=True)
class SyntheticWorldConfig:
    rule: str = "linear"
    n_features: int = 3
    noise_std: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; choose from {sorted(RULES)}")
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


class SyntheticWorld(Environment):
    """Deterministic generator for the Genesis synthetic world."""

    GENERATOR_VERSION = "synthetic_world.v1"

    def __init__(
        self,
        config: SyntheticWorldConfig | None = None,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.config = config or SyntheticWorldConfig()
        names = tuple(feature_names) if feature_names else tuple(
            f"f{i + 1}" for i in range(self.config.n_features)
        )
        if len(names) != self.config.n_features:
            raise ValueError("feature_names length must match n_features")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        self.feature_names = names

    @property
    def rule(self) -> RuleSpec:
        return RULES[self.config.rule]

    def with_rule(self, rule: str) -> "SyntheticWorld":
        """Return a new world with the same surface features but a new rule."""

        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}")
        return SyntheticWorld(
            replace(self.config, rule=rule), feature_names=self.feature_names
        )

    def _rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(self.config.seed if seed is None else seed)

    def sample_features(self, n: int, seed: int | None = None) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be non-negative")
        return self._rng(seed).uniform(-1.0, 1.0, size=(n, self.config.n_features))

    def outcomes_for(
        self,
        features: np.ndarray,
        *,
        rule: str | None = None,
        noise_std: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Apply a hidden rule (and optional noise) to a feature matrix."""

        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.config.n_features:
            raise ValueError("features must be 2-D with n_features columns")
        rule_spec = RULES[rule or self.config.rule]
        outcomes = rule_spec.fn(features)
        sigma = self.config.noise_std if noise_std is None else noise_std
        if sigma < 0:
            raise ValueError("noise_std must be non-negative")
        if sigma > 0:
            outcomes = outcomes + self._rng(seed).normal(0.0, sigma, size=outcomes.shape)
        return outcomes

    def generate(self, n: int, seed: int | None = None) -> Dataset:
        rng_seed = self.config.seed if seed is None else seed
        features = self.sample_features(n, seed=rng_seed)
        outcomes = self.outcomes_for(features, seed=rng_seed + 1)
        return Dataset(features, outcomes, self.feature_names)

    def generate_split(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """IID train/validation/test split with disjoint indices."""

        total = n_train + n_val + n_test
        dataset = self.generate(total, seed=seed)
        permutation = self._rng((self.config.seed if seed is None else seed) + 101).permutation(total)
        train_idx = permutation[:n_train]
        val_idx = permutation[n_train : n_train + n_val]
        test_idx = permutation[n_train + n_val :]
        return (
            self._subset(dataset, train_idx),
            self._subset(dataset, val_idx),
            self._subset(dataset, test_idx),
        )

    @staticmethod
    def _subset(dataset: Dataset, indices: np.ndarray) -> Dataset:
        return Dataset(
            dataset.features[indices], dataset.outcomes[indices], dataset.feature_names
        )

    def generate_compositional_split(
        self,
        n_train: int,
        n_test: int,
        *,
        seed: int | None = None,
        held_out_combinations: Iterable[tuple[int, int]] = ((1, 1),),
        max_draws_factor: int = 200,
    ) -> tuple[Dataset, Dataset]:
        """Split by sign(f1), sign(f2) combinations.

        Held-out combinations appear only in the test set, so a model that only
        memorises regions seen during training cannot succeed by interpolation
        inside those quadrants. Outcomes are still generated by the same rule.
        """

        held = {tuple(int(bit) for bit in combo) for combo in held_out_combinations}
        if not held:
            raise ValueError("held_out_combinations must not be empty")
        if any(len(combo) != 2 or any(bit not in (0, 1) for bit in combo) for combo in held):
            raise ValueError("combinations must be pairs of 0/1 sign bits")

        seed_value = self.config.seed if seed is None else seed
        feature_rng = np.random.default_rng(seed_value)
        max_draws = max(n_train + n_test, 1) * max_draws_factor

        train_rows: list[np.ndarray] = []
        test_rows: list[np.ndarray] = []
        draws = 0
        while (len(train_rows) < n_train or len(test_rows) < n_test) and draws < max_draws:
            batch = feature_rng.uniform(-1.0, 1.0, size=(256, self.config.n_features))
            signs = (batch[:, 0] >= 0).astype(int) * 2 + (batch[:, 1] >= 0).astype(int)
            combo_ids = [(int(bits // 2), int(bits % 2)) for bits in signs]
            for row, combo in zip(batch, combo_ids, strict=True):
                if combo in held:
                    if len(test_rows) < n_test:
                        test_rows.append(row)
                elif len(train_rows) < n_train:
                    train_rows.append(row)
            draws += len(batch)

        if len(train_rows) < n_train or len(test_rows) < n_test:
            raise RuntimeError(
                "could not draw enough compositional samples; "
                "reduce requested sizes or held-out combinations"
            )

        train_features = np.vstack(train_rows)
        test_features = np.vstack(test_rows)
        train_outcomes = self.outcomes_for(train_features, seed=seed_value + 1)
        test_outcomes = self.outcomes_for(test_features, seed=seed_value + 2)
        return (
            Dataset(train_features, train_outcomes, self.feature_names),
            Dataset(test_features, test_outcomes, self.feature_names),
        )

    def permute_features(self, features: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
        """Permute feature columns (a renamed-feature robustness probe)."""

        features = np.asarray(features, dtype=float)
        permutation = list(permutation)
        if sorted(permutation) != list(range(self.config.n_features)):
            raise ValueError("permutation must be a permutation of feature indices")
        return features[:, permutation]

    def to_observations(
        self,
        dataset: Dataset,
        *,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> list[Observation]:
        seed_value = self.config.seed if seed is None else seed
        return observations_from_arrays(
            dataset.features,
            dataset.outcomes,
            feature_names=dataset.feature_names,
            provenance=Provenance.synthetic(
                f"{self.GENERATOR_VERSION}:{self.config.rule}", seed_value
            ),
            episode_id=episode_id,
        )


def available_rules() -> list[str]:
    return sorted(RULES)
