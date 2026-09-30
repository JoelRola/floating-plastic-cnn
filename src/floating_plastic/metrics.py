"""Dependency-light classification and severity regression metrics."""

import math


def _binary_inputs(y_true, y_pred):
    true, pred = list(y_true), list(y_pred)
    if not true or len(true) != len(pred):
        raise ValueError("labels must be non-empty and have equal lengths")
    if any(v not in (0, 1, False, True) for v in true + pred):
        raise ValueError("binary labels and predictions must contain only 0 and 1")
    return [int(v) for v in true], [int(v) for v in pred]


def threshold_predictions(probabilities, threshold=0.5):
    """Convert finite probabilities into binary predictions at ``threshold``."""
    if not math.isfinite(float(threshold)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    values = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return [int(value >= threshold) for value in values]


def confusion_counts(y_true, y_pred):
    true, pred = _binary_inputs(y_true, y_pred)
    tn = sum(a == 0 and b == 0 for a, b in zip(true, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(true, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(true, pred))
    tp = sum(a == 1 and b == 1 for a, b in zip(true, pred))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def classification_metrics(y_true, y_pred):
    """Return accuracy, recall, specificity, balanced accuracy, precision, F1 and TN/FP/FN/TP."""
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = (counts[key] for key in ("tn", "fp", "fn", "tp"))
    n = tn + fp + fn + tp
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError("sensitivity and specificity require both classes in y_true")
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "sample_count": n,
        "accuracy": (tp + tn) / n,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": counts,
    }


def _coverage_pairs(y_true_percentage_points, y_pred_percentage_points):
    true = list(y_true_percentage_points)
    pred = list(y_pred_percentage_points)
    if not true or len(true) != len(pred):
        raise ValueError("coverage arrays must be non-empty and have equal lengths")
    try:
        pairs = [(float(a), float(b)) for a, b in zip(true, pred)]
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage values must be finite numbers") from exc
    if any(
        not math.isfinite(a) or not math.isfinite(b) or not 0 <= a <= 100 or not 0 <= b <= 100
        for a, b in pairs
    ):
        raise ValueError("coverage values must be finite and in percentage points [0, 100]")
    return pairs


def severity_regression_metrics(y_true_percentage_points, y_pred_percentage_points):
    """Calculate sample count, MAE, and RMSE in 0–100 percentage points."""
    pairs = _coverage_pairs(y_true_percentage_points, y_pred_percentage_points)
    errors = [pred - true for true, pred in pairs]
    return {
        "sample_count": len(pairs),
        "mae_percentage_points": sum(abs(error) for error in errors) / len(errors),
        "rmse_percentage_points": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def accuracy(y_true, y_pred):
    counts = confusion_counts(y_true, y_pred)
    return (counts["tn"] + counts["tp"]) / sum(counts.values())


def sensitivity(y_true, y_pred):
    counts = confusion_counts(y_true, y_pred)
    denominator = counts["tp"] + counts["fn"]
    if not denominator:
        raise ValueError("sensitivity requires at least one positive true label")
    return counts["tp"] / denominator


def specificity(y_true, y_pred):
    counts = confusion_counts(y_true, y_pred)
    denominator = counts["tn"] + counts["fp"]
    if not denominator:
        raise ValueError("specificity requires at least one negative true label")
    return counts["tn"] / denominator


def balanced_accuracy(y_true, y_pred):
    return classification_metrics(y_true, y_pred)["balanced_accuracy"]


def severity_mae(y_true_percentage_points, y_pred_percentage_points):
    return severity_regression_metrics(y_true_percentage_points, y_pred_percentage_points)[
        "mae_percentage_points"
    ]


def positive_only_metrics(probabilities, threshold=0.5):
    """Report positive recall and score distribution without negative-class claims."""
    values = [float(value) for value in probabilities]
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("probabilities must be a non-empty sequence in [0, 1]")
    predictions = threshold_predictions(values, threshold)
    ordered = sorted(values)

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "sample_count": len(values),
        "positive_recall": sum(predictions) / len(predictions),
        "predicted_positive_count": sum(predictions),
        "threshold": float(threshold),
        "predicted_probability_distribution": {
            "min": ordered[0], "q25": quantile(0.25), "median": quantile(0.5),
            "mean": sum(ordered) / len(ordered), "q75": quantile(0.75), "max": ordered[-1],
        },
    }


def multidomain_evaluation(flopwd_true, flopwd_probabilities, flopwd_severity_true,
                           flopwd_severity_predicted, ugv_waste_present_probabilities,
                           threshold=0.5):
    """Return domain-separated scores plus descriptive positive-recall gap."""
    flo_pred = threshold_predictions(flopwd_probabilities, threshold)
    flo_class = classification_metrics(flopwd_true, flo_pred)
    flo_positive_recall = flo_class["sensitivity"]
    flo = {
        "classification": flo_class,
        "severity_all": severity_regression_metrics(flopwd_severity_true, flopwd_severity_predicted),
        "positive_recall": flo_positive_recall,
    }
    ugv = positive_only_metrics(ugv_waste_present_probabilities, threshold)
    return {
        "flopwd": flo,
        "ugv_annotated_waste_present": ugv,
        "domain_recall_gap": {
            "name": "domain recall gap",
            "absolute_difference": abs(flo_positive_recall - ugv["positive_recall"]),
            "interpretation": (
                "descriptive difference between FloPWD plastic-positive recall and UGV annotated-waste "
                "recall; targets and domains differ, so this is not accuracy loss"
            ),
        },
    }
