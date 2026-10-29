import pytest

from genesis.evaluation.regression import RegressionCheck, assert_no_regression


def test_regression_check_passes_within_tolerance():
    check = RegressionCheck("old_task", reference_metric=0.10, max_deterioration=0.02)
    result = check.evaluate(0.115)
    assert result.passed
    assert_no_regression(result)


def test_regression_check_fails_when_deterioration_exceeds_limit():
    check = RegressionCheck("old_task", reference_metric=0.10, max_deterioration=0.02)
    result = check.evaluate(0.20)
    assert not result.passed
    with pytest.raises(AssertionError):
        assert_no_regression(result)


def test_regression_check_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        RegressionCheck("old_task", reference_metric=0.1, max_deterioration=-1)
