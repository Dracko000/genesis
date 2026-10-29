"""Minimal PyTorch learner that predicts scalar outcomes from features.

This is the initial neural baseline. It does not invent architectures: it
adapts the parameters of a configurable MLP. Initialisation uses PyTorch's
documented default ``Linear`` initialisation; the random seed is explicit and
stored with every result. Targets are standardised using training statistics
only, and predictions are de-standardised before being returned.
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import nn


class NeuralConfig(BaseModel):
    """Configuration for :class:`NeuralPredictor`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_sizes: list[int] = Field(default_factory=lambda: [32, 32])
    learning_rate: float = Field(default=0.01, gt=0)
    epochs: int = Field(default=200, ge=1)
    batch_size: int = Field(default=32, ge=1)
    weight_decay: float = Field(default=1e-4, ge=0)
    seed: int = Field(default=7, ge=0)
    activation: Literal["relu", "tanh"] = "relu"
    standardize_target: bool = True

    @field_validator("hidden_sizes")
    @classmethod
    def _validate_hidden(cls, sizes: list[int]) -> list[int]:
        if any(size < 1 for size in sizes):
            raise ValueError("hidden sizes must be >= 1")
        return sizes


class NeuralPredictor:
    """A small MLP regressor with deterministic training."""

    def __init__(self, input_dim: int, config: NeuralConfig | None = None) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        self.input_dim = input_dim
        self.config = config or NeuralConfig()
        self._fitted = False
        self.target_mean = 0.0
        self.target_std = 1.0
        self._model = self._build()
        self._history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _build(self) -> nn.Module:
        torch.manual_seed(self.config.seed)
        activation: type[nn.Module] = nn.ReLU if self.config.activation == "relu" else nn.Tanh
        layers: list[nn.Module] = []
        previous = self.input_dim
        for size in self.config.hidden_sizes:
            layers.append(nn.Linear(previous, size))
            layers.append(activation())
            previous = size
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _as_arrays(self, features: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float).reshape(-1)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must be shape (n, {self.input_dim})")
        if features.shape[0] != outcomes.shape[0]:
            raise ValueError("features and outcomes must have the same number of rows")
        if features.shape[0] == 0:
            raise ValueError("cannot fit on zero samples")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(outcomes)):
            raise ValueError("training data must be finite")
        return features, outcomes

    def _forward(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            output = self._model(tensor).reshape(-1).numpy()
        return output

    def fit(
        self,
        features: np.ndarray,
        outcomes: np.ndarray,
        *,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        reset_parameters: bool = True,
        epochs: int | None = None,
    ) -> dict[str, object]:
        """Train on the provided arrays.

        ``reset_parameters=False`` warm-starts from the current weights (used by
        transfer experiments) while keeping the source target statistics.
        """

        features, outcomes = self._as_arrays(features, outcomes)
        n_epochs = self.config.epochs if epochs is None else epochs
        if n_epochs < 1:
            raise ValueError("epochs must be >= 1")

        if reset_parameters:
            self._model = self._build()
            self.target_mean = (
                float(outcomes.mean()) if self.config.standardize_target else 0.0
            )
            raw_std = float(outcomes.std())
            self.target_std = (
                raw_std if (self.config.standardize_target and raw_std > 1e-8) else 1.0
            )
        elif not self._fitted:
            raise ValueError("warm-start requires a previously fitted predictor")
        targets = (outcomes - self.target_mean) / self.target_std

        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        x_tensor = torch.as_tensor(features, dtype=torch.float32)
        y_tensor = torch.as_tensor(targets, dtype=torch.float32)
        n_samples = features.shape[0]
        batch_size = min(self.config.batch_size, n_samples)

        val_arrays = None
        if validation is not None:
            val_features, val_outcomes = validation
            val_arrays = self._as_arrays(val_features, val_outcomes)
            val_targets = (val_arrays[1] - self.target_mean) / self.target_std
            val_tensor = torch.as_tensor(val_arrays[0], dtype=torch.float32)
            val_target_tensor = torch.as_tensor(val_targets, dtype=torch.float32)

        train_losses: list[float] = []
        val_losses: list[float] = []
        self._model.train()
        for _epoch in range(n_epochs):
            permutation = torch.randperm(n_samples, generator=generator)
            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                indices = permutation[start : start + batch_size]
                batch_x = x_tensor[indices]
                batch_y = y_tensor[indices]
                optimizer.zero_grad(set_to_none=True)
                prediction = self._model(batch_x).reshape(-1)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * indices.shape[0]
            train_losses.append(epoch_loss / n_samples)
            if val_arrays is not None:
                self._model.eval()
                with torch.no_grad():
                    val_prediction = self._model(val_tensor).reshape(-1)
                    val_losses.append(float(loss_fn(val_prediction, val_target_tensor).item()))
                self._model.train()

        self._model.eval()
        self._fitted = True
        self._history = {"train_loss": train_losses, "val_loss": val_losses}
        return {
            "n_samples": n_samples,
            "epochs": n_epochs,
            "seed": self.config.seed,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("NeuralPredictor must be fitted before prediction")
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.input_dim:
            raise ValueError(f"features must have {self.input_dim} columns")
        normalized = self._forward(features)
        return normalized * self.target_std + self.target_mean

    @property
    def history(self) -> dict[str, list[float]]:
        return copy.deepcopy(self._history)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self._model.state_dict().items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._fitted = True

    def snapshot(self) -> dict[str, object]:
        """Capture parameters and target statistics for rollback."""

        return {
            "state_dict": self.state_dict(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "fitted": self._fitted,
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        state_dict = snapshot.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("invalid snapshot")
        self._model.load_state_dict(state_dict)
        self.target_mean = float(snapshot.get("target_mean", 0.0))
        self.target_std = float(snapshot.get("target_std", 1.0))
        self._fitted = bool(snapshot.get("fitted", True))
        self._model.eval()
