import numpy as np
import pytest

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig
from genesis.representation.checkpoint import checkpoints_equivalent, load_checkpoint
from genesis.representation.interface import FeatureSpec
from genesis.representation.latent_state import LatentState
from genesis.representation.versioning import VersionCompatibilityError, ensure_supported_checkpoint


def test_latent_state_target_round_trip_and_validation():
    state = LatentState(
        spec=FeatureSpec(("f1", "f2")),
        state_dict={},
        encoder_state={"feature_names": ["f1", "f2"]},
        target_mean=2.0,
        target_std=0.5,
    )
    original = np.array([1.0, 2.0, 3.0])
    normalized = state.normalize_target(original, state.target_mean, state.target_std)
    restored = state.denormalize_target(normalized, state.target_mean, state.target_std)
    np.testing.assert_allclose(original, restored)
    with pytest.raises(ValueError):
        LatentState(
            spec=FeatureSpec(("f1",)),
            state_dict={},
            encoder_state={"feature_names": ["f1"]},
            target_std=0.0,
        )


def test_unsupported_checkpoint_version_is_rejected():
    with pytest.raises(VersionCompatibilityError):
        ensure_supported_checkpoint({"format": "genesis.checkpoint", "schema_version": 999})
    with pytest.raises(VersionCompatibilityError):
        ensure_supported_checkpoint({"format": "other", "schema_version": 1})


def test_checkpoint_equivalence_helper(tmp_path):
    world = SyntheticWorld(SyntheticWorldConfig(seed=0))
    dataset = world.generate(40, seed=0)
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=5, hidden_sizes=[4], seed=0),
        )
    )
    learner.fit(world.to_observations(dataset, seed=0))
    first = learner.save(tmp_path / "a", checkpoint_id="a")
    second = learner.save(tmp_path / "b", checkpoint_id="b")
    left = load_checkpoint(tmp_path / "a" / f"{first.checkpoint_id}.pt")
    right = load_checkpoint(tmp_path / "b" / f"{second.checkpoint_id}.pt")
    assert checkpoints_equivalent(left, right)
    right.state_dict[next(iter(right.state_dict))].add_(0.1)
    assert not checkpoints_equivalent(left, right)
