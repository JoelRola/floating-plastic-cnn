"""Dependency-light classification and coverage metrics."""

import math


def _binary_inputs(y_true, y_pred):
    true, pred = list(y_true), list(y_pred)
    if not true or len(true) != len(pred):
        raise ValueError("labels must be non-empty and have equal lengths")
    if any(v not in (0, 1, False, True) for v in true + pred):
        raise ValueError("binary labels and predictions must contain only 0 and 1")
    return [int(v) for v in true], [int(v) for v in pred]


def accuracy(y_true, y_pred):
    true, pred = _binary_inputs(y_true, y_pred)
    return sum(a == b for a, b in zip(true, pred)) / len(true)


def sensitivity(y_true, y_pred):
    true, pred = _binary_inputs(y_true, y_pred)
    positives = sum(v == 1 for v in true)
    if positives == 0:
        raise ValueError("sensitivity is undefined when y_true has no positive samples")
    return sum(a == b == 1 for a, b in zip(true, pred)) / positives


def specificity(y_true, y_pred):
    true, pred = _binary_inputs(y_true, y_pred)
    negatives = sum(v == 0 for v in true)
    if negatives == 0:
        raise ValueError("specificity is undefined when y_true has no negative samples")
    return sum(a == b == 0 for a, b in zip(true, pred)) / negatives


def balanced_accuracy(y_true, y_pred):
    return (sensitivity(y_true, y_pred) + specificity(y_true, y_pred)) / 2


def severity_mae(y_true_percentage_points, y_pred_percentage_points):
    """Mean absolute error where coverage labels/predictions are 0–100 values."""
    true = list(y_true_percentage_points)
    pred = list(y_pred_percentage_points)
    if not true or len(true) != len(pred):
        raise ValueError("coverage arrays must be non-empty and have equal lengths")
    try:
        pairs = [(float(a), float(b)) for a, b in zip(true, pred)]
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage values must be finite numbers") from exc
    if any(not math.isfinite(a) or not math.isfinite(b) for a, b in pairs):
        raise ValueError("coverage values must be finite numbers")
    return sum(abs(a - b) for a, b in pairs) / len(pairs)
