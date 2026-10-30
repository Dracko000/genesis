from genesis.environments.synthetic_world import RULES
from genesis.evaluation.transfer_experiments import (
    E002DiagnosticsConfig,
    TransferPair,
    run_e002_diagnostics,
)
from genesis.learning.neural import NeuralConfig


def _tiny() -> NeuralConfig:
    return NeuralConfig(epochs=20, hidden_sizes=[8], seed=0)


def test_related_rule_pairs_exist():
    assert {"linear_alt", "interaction_alt", "partial_interaction"} <= set(RULES)


def test_transfer_diagnostics_are_reproducible_and_report_controls():
    config = E002DiagnosticsConfig(
        pairs=[
            TransferPair(
                label="additive_related",
                source_rule="linear",
                target_rule="linear_alt",
                related=True,
            )
        ],
        seeds=[0],
        n_source=120,
        n_target=60,
        n_target_test=80,
        source_epochs=40,
        target_epochs=10,
        full_target_epochs=40,
        neural=_tiny(),
    )
    first = run_e002_diagnostics(config)
    second = run_e002_diagnostics(config)
    assert first["summary"] == second["summary"]
    summary = first["summary"]["additive_related"]
    assert {
        "transfer_small_budget",
        "fresh_small_budget",
        "fresh_full_budget",
        "unrelated_prior_small_budget",
        "transfer_source_retention",
    } <= set(summary["metrics"])
    assert first["fairness_controls"]["same_target_epochs"] is True
    assert first["criterion"]["related_pairs"] == ["additive_related"]
    assert isinstance(first["criterion"]["met"], bool)
