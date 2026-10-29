import numpy as np
import pytest

from genesis.environments.synthetic_world import (
    RULES,
    SyntheticWorld,
    SyntheticWorldConfig,
)


def test_generation_is_deterministic():
    world = SyntheticWorld(SyntheticWorldConfig(rule="interaction", seed=11))
    first = world.generate(64, seed=11)
    second = world.generate(64, seed=11)
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.outcomes, second.outcomes)


def test_different_seeds_differ():
    world = SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=1))
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        world.generate(32, seed=1).features,
        world.generate(32, seed=2).features,
    )


def test_split_is_disjoint_and_sized():
    world = SyntheticWorld(SyntheticWorldConfig(seed=3))
    train, val, test = world.generate_split(60, 20, 20, seed=3)
    assert (len(train), len(val), len(test)) == (60, 20, 20)
    combined = np.vstack([train.features, val.features, test.features])
    assert combined.shape[0] == 100
    for i in range(combined.shape[0]):
        for j in range(i + 1, combined.shape[0]):
            assert not np.array_equal(combined[i], combined[j])


def test_compositional_split_holds_out_sign_combinations():
    world = SyntheticWorld(SyntheticWorldConfig(rule="interaction", seed=5))
    train, test = world.generate_compositional_split(
        120, 60, seed=5, held_out_combinations=[(1, 1)]
    )
    train_combos = {
        (int(f1 >= 0), int(f2 >= 0)) for f1, f2, _ in train.features
    }
    test_combos = {(int(f1 >= 0), int(f2 >= 0)) for f1, f2, _ in test.features}
    assert (1, 1) not in train_combos
    assert test_combos == {(1, 1)}


def test_noise_is_controlled_and_reproducible():
    clean = SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=9))
    noisy = SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=9, noise_std=0.2))
    dataset_clean = clean.generate(200, seed=9)
    dataset_noisy = noisy.generate(200, seed=9)
    np.testing.assert_allclose(
        dataset_clean.features, dataset_noisy.features, rtol=0, atol=0
    )
    assert not np.allclose(dataset_clean.outcomes, dataset_noisy.outcomes)
    noise = dataset_noisy.outcomes - RULES["linear"].fn(dataset_noisy.features)
    assert 0.12 < float(np.std(noise)) < 0.30
    repeated = noisy.generate(200, seed=9)
    np.testing.assert_array_equal(dataset_noisy.outcomes, repeated.outcomes)


def test_with_rule_changes_outcomes_not_inputs():
    stable = SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=4))
    changed = stable.with_rule("changed")
    dataset = stable.generate(100, seed=4)
    changed_dataset = changed.generate(100, seed=4)
    np.testing.assert_array_equal(dataset.features, changed_dataset.features)
    changed_outcomes = changed.outcomes_for(dataset.features)
    assert not np.allclose(dataset.outcomes, changed_outcomes)


def test_feature_permutation_validation():
    world = SyntheticWorld(SyntheticWorldConfig(seed=1))
    dataset = world.generate(10, seed=1)
    permuted = world.permute_features(dataset.features, [2, 1, 0])
    np.testing.assert_array_equal(permuted[:, 0], dataset.features[:, 2])
    with pytest.raises(ValueError):
        world.permute_features(dataset.features, [0, 0, 1])
