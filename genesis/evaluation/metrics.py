"""Regression metrics and seed aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("metrics require finite values")
    return true, pred


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _arrays(y_true, y_pred)
    denominator = float(np.sum((true - true.mean()) ** 2))
    if denominator < 1e-12:
        return 0.0
    return float(1.0 - np.sum((true - pred) ** 2) / denominator)


def max_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _arrays(y_true, y_pred)
    return float(np.max(np.abs(true - pred)))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true, pred = _arrays(y_true, y_pred)
    return {
        "mse": mean_squared_error(true, pred),
        "rmse": root_mean_squared_error(true, pred),
        "mae": mean_absolute_error(true, pred),
        "r2": r2_score(true, pred),
        "max_abs_error": max_absolute_error(true, pred),
        "n": float(true.size),
    }


def summarize(values: Iterable[float]) -> dict[str, float]:
    """Return mean/std/min/max across seeds (sample std where possible)."""

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarise an empty sequence")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": float(array.size),
    }
