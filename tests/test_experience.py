import pytest
from pydantic import ValidationError

from genesis.experience.provenance import Provenance
from genesis.experience.schema import ExperienceKind, Observation


def test_observation_accepts_finite_values():
    record = Observation(
        features={"f1": 0.5, "f2": -1.0},
        outcome=2.0,
        action=0.1,
        provenance=Provenance(source="test"),
    )
    assert record.has_target
    assert record.feature_vector(["f1", "f2"]) == [0.5, -1.0]


def test_unlabeled_observation_is_not_training_signal():
    record = Observation(features={"f1": 0.5})
    assert not record.has_target


def test_inference_requires_inferred_provenance():
    with pytest.raises(ValidationError):
        Observation(
            features={"f1": 0.5},
            kind=ExperienceKind.INFERENCE,
            provenance=Provenance(source="test", inferred=False),
        )


def test_inferred_provenance_cannot_be_observation():
    with pytest.raises(ValidationError):
        Observation(
            features={"f1": 0.5},
            provenance=Provenance(source="test", inferred=True),
        )


def test_non_finite_feature_is_rejected():
    with pytest.raises(ValidationError):
        Observation(features={"f1": float("nan")})


def test_missing_feature_vector_raises():
    record = Observation(features={"f1": 0.5})
    with pytest.raises(KeyError):
        record.feature_vector(["f1", "f2"])


def test_prediction_record_keeps_error_auditable():
    from genesis.learning.prediction import make_prediction_record

    record = make_prediction_record(
        1.5, actual=1.0, observation_id="obs-1", checkpoint_id="ckpt-1"
    )
    assert record.error == pytest.approx(0.5)
    assert record.checkpoint_id == "ckpt-1"
    assert make_prediction_record(2.0).error is None
