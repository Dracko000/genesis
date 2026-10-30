import json

import pytest

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.evaluation.retention_experiments import (
    E003RetentionConfig,
    run_e003_retention,
)
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig
from genesis.learning.retention import (
    EvidenceConfig,
    EvidenceValidator,
    RetentionPolicy,
    RetentionRevisionController,
)
from genesis.learning.revision import RevisionController


def _fitted(rule: str = "linear", seed: int = 0, n: int = 300) -> tuple[Learner, SyntheticWorld]:
    world = SyntheticWorld(SyntheticWorldConfig(rule=rule, seed=seed))
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=150, hidden_sizes=[16, 16], seed=seed),
        )
    )
    learner.fit(world.to_observations(world.generate(n, seed=seed), seed=seed))
    return learner, world


def test_evidence_filter_rejects_injected_outlier():
    learner, world = _fitted()
    records = world.to_observations(world.generate(50, seed=3), seed=3)
    outlier = records[0].model_copy(update={"outcome": 10_000.0, "id": "outlier"})
    report = EvidenceValidator().validate(learner, [*records, outlier])
    assert report.n_total == 51
    assert report.n_accepted == 50
    assert [entry["id"] for entry in report.rejected] == ["outlier"]


def test_evidence_rejects_batches_below_minimum_size():
    learner, world = _fitted()
    record = world.to_observations(world.generate(1, seed=4), seed=4)[0]
    report = EvidenceValidator(EvidenceConfig(min_batch_size=3)).validate(learner, [record])
    assert not report.sufficient
    assert report.n_accepted == 0
    assert "min_batch_size" in report.reasons[0]


def test_dual_gate_rejects_forgetting_and_replay_is_accepted():
    config = E003RetentionConfig(
        seeds=[0],
        n_train=300,
        n_new=100,
        n_new_val=80,
        n_test=100,
        adaptation_epochs=[50],
        neural=NeuralConfig(epochs=200, hidden_sizes=[16, 16], seed=0),
    )
    result = run_e003_retention(config)
    curve = {
        entry["condition"]: entry for entry in result["per_seed"][0]["curve"]
    }

    dual = curve["dual_gate"]
    assert not dual["accepted"]
    assert dual["candidate_old_validation_mse"] > dual["retention_budget"]
    assert dual["old_task_mse"] < dual["retention_budget"]

    replay = curve["dual_gate_replay"]
    assert replay["accepted"]
    assert replay["new_task_mse"] < result["per_seed"][0]["new_task_before"]["mse"]
    assert replay["old_task_mse"] <= replay["retention_budget"]
    assert result["criterion"]["met"]


def test_controller_history_is_structured_and_serialisable():
    learner, world = _fitted()
    old_val = world.to_observations(world.generate(40, seed=5), seed=5)
    new_world = SyntheticWorld(SyntheticWorldConfig(rule="changed", seed=6))
    new_train = new_world.to_observations(new_world.generate(60, seed=6), seed=6)
    controller = RetentionRevisionController(
        RetentionPolicy(epochs=5), EvidenceConfig(min_batch_size=3)
    )
    result = controller.controlled_update(learner, new_train, new_train, old_val)
    assert controller.history
    payload = json.dumps(controller.history)
    assert '"accepted"' in payload
    assert result.to_dict()["policy"]["epochs"] == 5


def test_legacy_revision_controller_still_available():
    learner, world = _fitted()
    validation = world.to_observations(world.generate(40, seed=8), seed=8)
    result = RevisionController(epochs=2).controlled_update(
        learner, validation, validation
    )
    assert isinstance(result.accepted, bool)


def test_dual_gate_requires_both_validation_sets():
    learner, world = _fitted()
    records = world.to_observations(world.generate(40, seed=9), seed=9)
    with pytest.raises(ValueError):
        RetentionRevisionController().controlled_update(learner, records, records, [])


def test_retention_result_reports_candidate_and_restored_state():
    learner, world = _fitted()
    old_val = world.to_observations(world.generate(40, seed=10), seed=10)
    new_world = SyntheticWorld(SyntheticWorldConfig(rule="changed", seed=11))
    new_train = new_world.to_observations(new_world.generate(40, seed=11), seed=11)
    result = RetentionRevisionController(RetentionPolicy(epochs=1)).controlled_update(
        learner, new_train, new_train, old_val
    )
    payload = result.to_dict()
    assert "candidate" not in payload
    assert {"before_new", "after_new", "before_old", "after_old"} <= set(payload)
    if not result.accepted:
        assert result.state_restored
