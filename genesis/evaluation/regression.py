"""Regression checks for previously learned tasks.

A retention check compares a reference metric captured earlier against the
current metric. Thresholds must be chosen *before* deployment and recorded with
the check; this module never invents a threshold from observed results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegressionCheck:
    """A named tolerance for metric degradation."""

    name: str
    reference_metric: float
    max_deterioration: float

    def __post_init__(self) -> None:
        if self.max_deterioration < 0:
            raise ValueError("max_deterioration must be non-negative")

    def evaluate(self, current_metric: float) -> "RegressionResult":
        deterioration = current_metric - self.reference_metric
        passed = deterioration <= self.max_deterioration
        return RegressionResult(
            name=self.name,
            reference=self.reference_metric,
            current=current_metric,
            deterioration=deterioration,
            max_deterioration=self.max_deterioration,
            passed=passed,
        )


@dataclass(frozen=True)
class RegressionResult:
    name: str
    reference: float
    current: float
    deterioration: float
    max_deterioration: float
    passed: bool

    def message(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: reference={self.reference:.4f} "
            f"current={self.current:.4f} deterioration={self.deterioration:.4f} "
            f"limit={self.max_deterioration:.4f}"
        )


def assert_no_regression(result: RegressionResult) -> None:
    if not result.passed:
        raise AssertionError(result.message())
