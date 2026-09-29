import pytest

from floating_plastic.metrics import (
    accuracy,
    balanced_accuracy,
    classification_metrics,
    severity_mae,
    severity_regression_metrics,
    sensitivity,
    specificity,
    threshold_predictions,
)


def test_binary_metrics():
    truth, predicted = [1, 1, 0, 0], [1, 0, 0, 0]
    assert accuracy(truth, predicted) == 0.75
    assert sensitivity(truth, predicted) == 0.5
    assert specificity(truth, predicted) == 1.0
    assert balanced_accuracy(truth, predicted) == 0.75


def test_specificity_requires_negative_examples():
    with pytest.raises(ValueError, match="negative true label"):
        specificity([1], [1])


def test_severity_mae_uses_percentage_points():
    assert severity_mae([10, 40], [15, 30]) == 7.5


def test_metrics_include_confusion_counts_and_regression_sample_count():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["balanced_accuracy"] == 0.5
    assert severity_regression_metrics([0, 100], [0, 50]) == {
        "sample_count": 2,
        "mae_percentage_points": 25,
        "rmse_percentage_points": pytest.approx(35.3553390593),
    }


def test_prediction_thresholding_is_configurable():
    assert threshold_predictions([0.49, 0.5, 0.8], 0.5) == [0, 1, 1]
    assert threshold_predictions([0.49, 0.5, 0.8], 0.75) == [0, 0, 1]


def test_classification_accuracy_can_be_reported_for_single_class():
    assert accuracy([1, 1], [1, 0]) == 0.5
