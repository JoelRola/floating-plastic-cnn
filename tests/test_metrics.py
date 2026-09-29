import pytest

from floating_plastic.metrics import accuracy, balanced_accuracy, severity_mae, sensitivity, specificity


def test_binary_metrics():
    truth, predicted = [1, 1, 0, 0], [1, 0, 0, 0]
    assert accuracy(truth, predicted) == 0.75
    assert sensitivity(truth, predicted) == 0.5
    assert specificity(truth, predicted) == 1.0
    assert balanced_accuracy(truth, predicted) == 0.75


def test_specificity_requires_negative_examples():
    with pytest.raises(ValueError, match="no negative"):
        specificity([1], [1])


def test_severity_mae_uses_percentage_points():
    assert severity_mae([10, 40], [15, 30]) == 7.5
