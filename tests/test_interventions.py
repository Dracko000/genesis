import numpy as np

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.evaluation.intervention_experiments import (
    InterventionConfig,
    run_intervention_probe,
)
from genesis.learning.interventions import (
    ablation_effect,
    linear_cka,
    matched_random_units,
    top_units,
)
from genesis.learning.learner import Learner, LearnerConfig
from genesis.learning.neural import NeuralConfig


def _learner() -> tuple[Learner, SyntheticWorld]:
    world = SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=0))
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=40, hidden_sizes=[8, 6], seed=0),
        )
    )
    learner.fit(world.to_observations(world.generate(80, seed=0), seed=0))
    return learner, world


def test_hidden_activations_have_configured_widths():
    learner, world = _learner()
    features = world.generate(20, seed=1).features
    activations = learner.predictor.hidden_activations(features)
    assert [activation.shape for activation in activations] == [(20, 8), (20, 6)]


def test_ablation_is_reverted_after_context():
    learner, world = _learner()
    features = world.generate(20, seed=1).features
    before = learner.predictor.predict(features)
    with learner.predictor.ablated_units(0, [0, 1]):
        ablated = learner.predictor.predict(features)
    after = learner.predictor.predict(features)
    assert not np.allclose(before, ablated)
    np.testing.assert_allclose(before, after, atol=1e-8)


def test_cka_is_one_for_identical_representations_and_low_for_independent():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    y = rng.normal(size=(200, 5))
    assert linear_cka(x, x) == 1.0
    assert abs(linear_cka(x, y)) < 0.2


def test_unit_selection_helpers():
    profile = np.array([0.1, 0.9, 0.3, 0.7])
    assert top_units(profile, 2) == [1, 3]
    units = matched_random_units(10, 3, seed=1)
    assert len(set(units)) == 3
    assert units == sorted(units)


def test_ablation_effect_reports_before_and_after():
    learner, world = _learner()
    dataset = world.generate(30, seed=2)
    features = learner.encoder.transform(
        world.to_observations(dataset, seed=2)
    )
    effect = ablation_effect(learner, features, dataset.outcomes, 0, [0])
    assert set(effect) >= {"mse_before", "mse_after", "mse_delta"}
    assert effect["mse_delta"] == effect["mse_after"] - effect["mse_before"]


def test_intervention_probe_runs_and_reports_layers():
    config = InterventionConfig(
        seeds=[0],
        n_source=120,
        n_target=60,
        n_target_test=60,
        source_epochs=40,
        target_epochs=10,
        neural=NeuralConfig(epochs=40, hidden_sizes=[8, 8], seed=0),
    )
    result = run_intervention_probe(config)
    assert set(result["summary"]) == {"0", "1"}
    assert "cka_source_target" in result["summary"]["0"]
    assert "met" in result["criterion"]
    assert len(result["per_seed"][0]["layers"]) == 2
