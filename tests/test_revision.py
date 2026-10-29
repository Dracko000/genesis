import numpy as np

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.evaluation.metrics import mean_squared_error
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig
from genesis.learning.revision import RevisionController


def _learner(rule: str, seed: int = 0) -> tuple[Learner, SyntheticWorld]:
    world = SyntheticWorld(SyntheticWorldConfig(rule=rule, seed=seed))
    dataset = world.generate(300, seed=seed)
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=200, hidden_sizes=[16, 16], seed=seed),
        )
    )
    learner.fit(world.to_observations(dataset, seed=seed))
    return learner, world


def test_isolated_anomaly_is_rejected_and_state_unchanged():
    learner, world = _learner("linear")
    validation_world = SyntheticWorld(SyntheticWorldConfig(rule="changed", seed=2))
    validation = validation_world.to_observations(
        validation_world.generate(100, seed=2), seed=2
    )
    reference = learner.predict(learner.encoder.transform(validation)).copy()

    anomaly = validation[0].model_copy(update={"outcome": 10_000.0, "id": "anomaly"})
    result = RevisionController(epochs=20).controlled_update(
        learner, [anomaly], validation
    )
    assert not result.accepted
    after = learner.predict(learner.encoder.transform(validation))
    np.testing.assert_allclose(reference, after, atol=1e-8)


def test_beneficial_update_is_accepted():
    learner, old_world = _learner("linear")
    new_world = SyntheticWorld(SyntheticWorldConfig(rule="changed", seed=3))
    new_train = new_world.to_observations(new_world.generate(300, seed=3), seed=3)
    new_val = new_world.to_observations(new_world.generate(150, seed=4), seed=4)

    before_predictions = learner.predict(learner.encoder.transform(new_val))
    before = mean_squared_error(
        [record.outcome for record in new_val], before_predictions
    )
    result = RevisionController(epochs=100).controlled_update(
        learner, new_train, new_val
    )
    after_predictions = learner.predict(learner.encoder.transform(new_val))
    after = mean_squared_error([record.outcome for record in new_val], after_predictions)

    assert result.accepted
    assert after < before


def test_revision_requires_fitted_learner():
    world = SyntheticWorld(SyntheticWorldConfig(seed=0))
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=1, hidden_sizes=[4], seed=0),
        )
    )
    records = world.to_observations(world.generate(10, seed=0), seed=0)
    try:
        RevisionController().controlled_update(learner, records, records)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unfitted learner")


def test_replay_buffer_reservoir_and_consolidation():
    from genesis.learning.consolidation import ReplayBuffer, consolidate

    world = SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=5))
    records = world.to_observations(world.generate(200, seed=5), seed=5)

    buffer = ReplayBuffer(capacity=50, seed=1)
    buffer.add(records)
    assert len(buffer) == 50
    first_sample = buffer.sample(10)
    second_sample = ReplayBuffer(capacity=50, seed=1)
    second_sample.add(records)
    assert [record.id for record in first_sample] == [
        record.id for record in second_sample.sample(10)
    ]

    learner, _ = _learner("linear")
    new_world = SyntheticWorld(SyntheticWorldConfig(rule="changed", seed=6))
    new_records = new_world.to_observations(new_world.generate(40, seed=6), seed=6)
    summary = consolidate(learner, new_records, buffer, replay_fraction=1 / 3)
    assert summary["n_samples"] == 60
