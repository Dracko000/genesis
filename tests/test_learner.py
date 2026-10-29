import numpy as np
import pytest

from genesis.environments.base import Dataset
from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.evaluation.benchmarks import MeanBaseline
from genesis.evaluation.datasets import assert_no_target_in_features
from genesis.evaluation.metrics import mean_squared_error
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig


def _world(seed: int = 0):
    return SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=seed))


def _train_learner(seed: int = 0, epochs: int = 200) -> tuple[Learner, Dataset]:
    world = _world(seed)
    dataset = world.generate(300, seed=seed)
    records = world.to_observations(dataset, seed=seed)
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=epochs, hidden_sizes=[16, 16], seed=seed),
        )
    )
    learner.fit(records)
    return learner, dataset


def test_learner_reduces_error_below_mean_baseline():
    learner, dataset = _train_learner()
    predictions = learner.predict(learner.encoder.transform_matrix(dataset.features))
    neural_mse = mean_squared_error(dataset.outcomes, predictions)
    baseline = MeanBaseline().fit(dataset.features, dataset.outcomes)
    mean_mse = mean_squared_error(dataset.outcomes, baseline.predict(dataset.features))
    assert neural_mse < mean_mse * 0.5


def test_identical_configs_are_deterministic():
    first, dataset = _train_learner(epochs=40)
    second, _ = _train_learner(epochs=40)
    a = first.predict(first.encoder.transform_matrix(dataset.features))
    b = second.predict(second.encoder.transform_matrix(dataset.features))
    np.testing.assert_array_equal(a, b)


def test_encoder_is_fitted_on_train_only():
    world = _world()
    train, _, test = world.generate_split(100, 20, 20, seed=0)
    train_records = world.to_observations(train, seed=0)
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=3, hidden_sizes=[8], seed=0),
        )
    )
    learner.fit(train_records)
    np.testing.assert_allclose(learner.encoder.mean, train.features.mean(axis=0))
    combined = np.vstack([train.features, test.features])
    assert not np.allclose(learner.encoder.mean, combined.mean(axis=0))


def test_targets_never_enter_feature_contract():
    world = _world()
    dataset = world.generate(100, seed=0)
    records = world.to_observations(dataset, seed=0)
    assert all("outcome" not in record.features for record in records)
    assert_no_target_in_features(dataset.features, dataset.outcomes)

    with pytest.raises(ValueError):
        assert_no_target_in_features(
            np.hstack([dataset.features, dataset.outcomes.reshape(-1, 1)]),
            dataset.outcomes,
        )


def test_extra_unlisted_feature_does_not_change_predictions():
    world = _world()
    dataset = world.generate(60, seed=0)
    records = world.to_observations(dataset, seed=0)
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=20, hidden_sizes=[8], seed=0),
        )
    )
    learner.fit(records)
    baseline = learner.predict(learner.encoder.transform(records))

    contaminated = [
        record.model_copy(
            update={"features": {**record.features, "leaked_target": record.outcome or 0.0}}
        )
        for record in records
    ]
    contaminated_predictions = learner.predict(learner.encoder.transform(contaminated))
    np.testing.assert_array_equal(baseline, contaminated_predictions)


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    learner, dataset = _train_learner(epochs=50)
    metadata = learner.save(tmp_path, metrics={"mse": 0.1}, tags=["test"])
    assert metadata.checkpoint_id
    restored, loaded_metadata = Learner.from_checkpoint(
        tmp_path / f"{metadata.checkpoint_id}.pt"
    )
    assert loaded_metadata.checkpoint_id == metadata.checkpoint_id
    original = learner.predict(learner.encoder.transform_matrix(dataset.features))
    reloaded = restored.predict(restored.encoder.transform_matrix(dataset.features))
    np.testing.assert_allclose(original, reloaded, atol=1e-6, rtol=0)


def test_checkpoint_refuses_silent_overwrite(tmp_path):
    learner, _ = _train_learner(epochs=5)
    metadata = learner.save(tmp_path)
    with pytest.raises(FileExistsError):
        learner.save(tmp_path, checkpoint_id=metadata.checkpoint_id)


def test_warm_start_requires_fitted_learner():
    world = _world()
    dataset = world.generate(20, seed=0)
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=2, hidden_sizes=[4], seed=0),
        )
    )
    with pytest.raises(RuntimeError):
        learner.fit(world.to_observations(dataset), warm_start=True)
