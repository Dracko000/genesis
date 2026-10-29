from genesis.evaluation.experiments import (
    E001Config,
    E003Config,
    HybridAblationConfig,
    run_e001,
    run_e003,
    run_hybrid_ablation,
)
from genesis.learning.neural import NeuralConfig


def _tiny_neural() -> NeuralConfig:
    return NeuralConfig(epochs=10, hidden_sizes=[8], seed=0)


def test_e001_is_reproducible_and_reports_all_baselines():
    config = E001Config(
        seeds=[0],
        n_train=60,
        n_test=40,
        sample_sizes=[30],
        noise_levels=[0.0],
        run_noise_study=False,
        neural=_tiny_neural(),
    )
    first = run_e001(config)
    second = run_e001(config)
    assert first["summary"] == second["summary"]
    assert set(first["summary"]) >= {
        "neural",
        "nearest_neighbour",
        "linear_ridge",
        "mean",
        "oracle",
    }
    assert "met" in first["criterion"]
    assert first["summary"]["oracle"]["mse"]["mean"] == 0.0


def test_e003_protocol_contains_revision_and_anomaly_checks():
    config = E003Config(
        seeds=[0],
        n_train=60,
        n_new=30,
        n_new_val=20,
        n_test=40,
        adaptation_epochs=[5],
        neural=_tiny_neural(),
    )
    result = run_e003(config)
    entry = result["per_seed"][0]
    assert "old_task_before" in entry
    assert "single_anomaly" in entry
    assert "adaptation_curve" in entry
    assert "old_task_deterioration" in result["summary"]


def test_hybrid_ablation_is_reproducible():
    config = HybridAblationConfig(
        seeds=[0], n_train=80, n_val=40, n_test=60, neural=_tiny_neural()
    )
    first = run_hybrid_ablation(config)
    second = run_hybrid_ablation(config)
    assert first["summary"] == second["summary"]
    assert "hybrid_selected" in first["summary"]
    assert first["criterion"]["selected_counts"]
