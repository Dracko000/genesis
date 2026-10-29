import numpy as np

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig
from genesis.learning.symbolic import SymbolicConfig, SymbolicRuleLearner


def test_symbolic_learner_recovers_linear_terms():
    world = SyntheticWorld(SyntheticWorldConfig(rule="linear", seed=0))
    dataset = world.generate(300, seed=0)
    learner = SymbolicRuleLearner().fit(dataset.features, dataset.outcomes)
    predictions = learner.predict(dataset.features)
    mse = float(np.mean((predictions - dataset.outcomes) ** 2))
    assert mse < 1e-6
    terms = {entry["term"] for entry in learner.expression()}
    assert {"1", "f1", "f2", "f3"}.issubset(terms)


def test_symbolic_expression_is_inspectable_and_deterministic():
    world = SyntheticWorld(SyntheticWorldConfig(rule="interaction", seed=1))
    dataset = world.generate(300, seed=1)
    first = SymbolicRuleLearner(SymbolicConfig(coefficient_threshold=1e-3)).fit(
        dataset.features, dataset.outcomes
    )
    second = SymbolicRuleLearner(SymbolicConfig(coefficient_threshold=1e-3)).fit(
        dataset.features, dataset.outcomes
    )
    assert first.expression() == second.expression()
    terms = {entry["term"] for entry in first.expression()}
    assert "f1*f2" in terms
