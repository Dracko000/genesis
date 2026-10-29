"""The high-level learner that binds an encoder to a neural predictor."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from genesis.experience.observation import select_labeled
from genesis.experience.schema import Observation
from genesis.learning.neural import NeuralConfig, NeuralPredictor
from genesis.learning.prediction import PredictionRecord, make_prediction_record
from genesis.representation.checkpoint import (
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from genesis.representation.interface import FeatureSpec, NumericFeatureEncoder


class LearnerConfig(BaseModel):
    """Explicit, serialisable learner contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_names: list[str] = Field(min_length=1)
    include_action: bool = False
    neural: NeuralConfig = Field(default_factory=NeuralConfig)

    @field_validator("feature_names")
    @classmethod
    def _validate_names(cls, names: list[str]) -> list[str]:
        if any(not name.strip() for name in names):
            raise ValueError("feature names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        return names


class Learner:
    """Trainable learner with checkpoint save/restore.

    The encoder is fitted only on the training records passed to :meth:`fit`.
    The predictor never reads targets from its input arrays: targets are passed
    as a separate vector.
    """

    def __init__(self, config: LearnerConfig) -> None:
        self.config = config
        self.spec = FeatureSpec(tuple(config.feature_names), config.include_action)
        self.encoder = NumericFeatureEncoder(self.spec)
        self.predictor = NeuralPredictor(self.spec.input_dim, config.neural)
        self.checkpoint_id: str | None = None
        self._training_summary: dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return self.predictor.fitted

    @property
    def training_summary(self) -> dict[str, Any]:
        return dict(self._training_summary)

    def _labeled(self, records: Iterable[Observation]) -> list[Observation]:
        labeled = select_labeled(records)
        if not labeled:
            raise ValueError("no labeled observations available for training")
        return labeled

    def _transform(self, records: Sequence[Observation]) -> np.ndarray:
        if not self.encoder._fitted:  # noqa: SLF001 - same module contract
            raise RuntimeError("encoder must be fitted before transforming")
        return self.encoder.transform(records)

    def fit(
        self,
        records: Iterable[Observation],
        *,
        validation_records: Iterable[Observation] | None = None,
        warm_start: bool = False,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Fit encoder and predictor on labeled training records.

        With ``warm_start=True`` the already-fitted encoder is reused and the
        existing predictor weights are updated instead of reinitialised.
        """

        train = self._labeled(records)
        if warm_start:
            if not self.fitted:
                raise RuntimeError("warm_start requires a fitted learner")
            features = self.encoder.transform(train)
        else:
            features = self.encoder.fit_transform(train)
        outcomes = np.asarray([record.outcome for record in train], dtype=float)

        validation = None
        if validation_records is not None:
            val = self._labeled(validation_records)
            validation = (
                self.encoder.transform(val),
                np.asarray([record.outcome for record in val], dtype=float),
            )

        history = self.predictor.fit(
            features,
            outcomes,
            validation=validation,
            reset_parameters=not warm_start,
            epochs=epochs,
        )
        self._training_summary = {
            "n_train": len(train),
            "n_validation": 0 if validation is None else len(validation[1]),
            "warm_start": warm_start,
            "history": {
                "final_train_loss": history.get("final_train_loss"),
                "final_val_loss": history.get("final_val_loss"),
            },
        }
        return history

    def predict_value(self, record: Observation) -> float:
        return float(self.predict(self._transform([record]))[0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predictor.predict(np.asarray(features, dtype=float))

    def predict_records(
        self,
        records: Sequence[Observation],
        *,
        checkpoint_id: str | None = None,
    ) -> list[PredictionRecord]:
        """Predict and retain per-record audit information."""

        predictions = self.predict(self._transform(records))
        return [
            make_prediction_record(
                float(prediction),
                actual=record.outcome,
                observation_id=record.id,
                checkpoint_id=checkpoint_id or self.checkpoint_id,
                features=dict(record.features),
            )
            for record, prediction in zip(records, predictions, strict=True)
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "predictor": self.predictor.snapshot(),
            "encoder": self.encoder.state(),
            "training_summary": dict(self._training_summary),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.predictor.restore(snapshot["predictor"])
        self.encoder = NumericFeatureEncoder.from_state(snapshot["encoder"])
        self._training_summary = dict(snapshot.get("training_summary", {}))

    def checkpoint_metadata(
        self,
        *,
        metrics: dict[str, float] | None = None,
        tags: Sequence[str] | None = None,
    ) -> CheckpointMetadata:
        if not self.fitted:
            raise RuntimeError("cannot checkpoint an unfitted learner")
        return CheckpointMetadata(
            seed=self.config.neural.seed,
            model_configuration=self.config.model_dump(mode="json"),
            feature_names=list(self.spec.feature_names),
            include_action=self.spec.include_action,
            encoder_state=self.encoder.state(),
            target_mean=self.predictor.target_mean,
            target_std=self.predictor.target_std,
            training_summary=self._training_summary,
            metrics=metrics or {},
            tags=list(tags or []),
        )

    def save(
        self,
        directory: str | Path,
        *,
        metrics: dict[str, float] | None = None,
        tags: Sequence[str] | None = None,
        checkpoint_id: str | None = None,
    ) -> CheckpointMetadata:
        metadata = self.checkpoint_metadata(metrics=metrics, tags=tags)
        if checkpoint_id is not None:
            metadata.checkpoint_id = checkpoint_id
        target = Path(directory) / f"{metadata.checkpoint_id}.pt"
        save_checkpoint(target, state_dict=self.predictor.state_dict(), metadata=metadata)
        self.checkpoint_id = metadata.checkpoint_id
        return metadata

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> tuple["Learner", CheckpointMetadata]:
        loaded = load_checkpoint(path)
        metadata = loaded.metadata
        config = LearnerConfig.model_validate(metadata.model_configuration)
        learner = cls(config)
        learner.encoder = NumericFeatureEncoder.from_state(metadata.encoder_state)
        learner.predictor.target_mean = metadata.target_mean
        learner.predictor.target_std = metadata.target_std
        learner.predictor.load_state_dict(loaded.state_dict)
        learner._training_summary = dict(metadata.training_summary)
        learner.checkpoint_id = metadata.checkpoint_id
        return learner, metadata

    def describe(self) -> dict[str, Any]:
        return {
            "spec": {
                "feature_names": list(self.spec.feature_names),
                "include_action": self.spec.include_action,
                "input_dim": self.spec.input_dim,
            },
            "neural": self.config.neural.model_dump(mode="json"),
            "fitted": self.fitted,
            "checkpoint_id": self.checkpoint_id,
            "training_summary": self._training_summary,
        }
