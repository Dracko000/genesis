"""End-to-end smoke demo: generate, train, checkpoint, restore, predict.

This is a fast, real (not mocked) workflow using the synthetic world. It is
useful for checking an installation without launching the API.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis.environments.synthetic_world import SyntheticWorld, SyntheticWorldConfig  # noqa: E402
from genesis.evaluation.metrics import regression_report  # noqa: E402
from genesis.learning.learner import Learner, LearnerConfig  # noqa: E402
from genesis.learning.neural import NeuralConfig  # noqa: E402


def main() -> int:
    world = SyntheticWorld(SyntheticWorldConfig(rule="interaction", seed=7))
    train, val, test = world.generate_split(400, 100, 200, seed=7)
    learner = Learner(
        LearnerConfig(
            feature_names=list(world.feature_names),
            neural=NeuralConfig(epochs=250, hidden_sizes=[32, 32], seed=7),
        )
    )
    learner.fit(
        world.to_observations(train, seed=7),
        validation_records=world.to_observations(val, seed=8),
    )
    predictions = learner.predict(
        learner.encoder.transform(world.to_observations(test, seed=9))
    )
    metrics = regression_report(test.outcomes, predictions)
    print("test metrics:", metrics)

    metadata = learner.save(ROOT / "checkpoints", metrics={"mse": metrics["mse"]})
    restored, _ = Learner.from_checkpoint(ROOT / "checkpoints" / f"{metadata.checkpoint_id}.pt")
    restored_predictions = restored.predict(
        restored.encoder.transform(world.to_observations(test, seed=9))
    )
    max_difference = float(abs(predictions - restored_predictions).max())
    print("checkpoint:", metadata.checkpoint_id, "max prediction difference:", max_difference)
    assert max_difference <= 1e-6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
