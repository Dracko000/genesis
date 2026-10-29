import pytest

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.experience.provenance import Provenance
from genesis.experience.schema import Observation
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig
from genesis.storage.sqlite_repository import SQLiteRepository


def _observation(seed: int = 0) -> Observation:
    return Observation(
        features={"f1": 0.1 * seed, "f2": -0.2},
        outcome=1.0,
        provenance=Provenance(source="test"),
    )


def test_repository_round_trip_experiences(tmp_path):
    repo = SQLiteRepository(tmp_path / "test.sqlite3")
    record = repo.add_experience(_observation())
    assert repo.get_experience(record.id) == record
    assert len(repo.list_experiences()) == 1
    with pytest.raises(ValueError):
        repo.add_experience(record)


def test_repository_experiment_and_metrics(tmp_path):
    repo = SQLiteRepository(tmp_path / "test.sqlite3")
    repo.create_experiment(
        experiment_id="exp-1", name="E001", status="running", config={"a": 1}, seed=0
    )
    repo.add_metric(experiment_id="exp-1", name="mse", value=0.5)
    repo.update_experiment("exp-1", status="completed", result={"ok": True})
    record = repo.get_experiment("exp-1")
    assert record is not None
    assert record.status == "completed"
    assert record.result == {"ok": True}
    assert repo.list_metrics("exp-1")[0].value == 0.5
    assert not hasattr(repo, "delete_experiment")


def test_repository_metric_requires_experiment(tmp_path):
    repo = SQLiteRepository(tmp_path / "test.sqlite3")
    with pytest.raises(KeyError):
        repo.add_metric(experiment_id="missing", name="mse", value=1.0)


def test_repository_checkpoints_are_append_only(tmp_path):
    world = SyntheticWorld(SyntheticWorldConfig(seed=0))
    dataset = world.generate(30, seed=0)
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=3, hidden_sizes=[4], seed=0),
        )
    )
    learner.fit(world.to_observations(dataset, seed=0))
    metadata = learner.checkpoint_metadata(metrics={"mse": 0.1})
    repo = SQLiteRepository(tmp_path / "test.sqlite3")
    record = repo.add_checkpoint(path=str(tmp_path / "ckpt.pt"), metadata=metadata)
    assert repo.get_checkpoint(record.checkpoint_id) is not None
    with pytest.raises(ValueError):
        repo.add_checkpoint(path=str(tmp_path / "ckpt.pt"), metadata=metadata)


def test_repository_logs_and_revisions(tmp_path):
    repo = SQLiteRepository(tmp_path / "test.sqlite3")
    repo.add_log(level="INFO", message="hello", payload={"x": 1})
    revision_id = repo.add_revision(experiment_id=None, payload={"accepted": True})
    assert repo.list_logs()[0].message == "hello"
    assert repo.list_revisions()[0]["id"] == revision_id
