"""Evaluate Model B and Control A on fixed held-out FloPWD/UGV partitions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.data import heterogeneous_flopwd_record, heterogeneous_ugv_record, load_flopwd, load_ugv
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.metrics import (
    classification_metrics, multidomain_evaluation, positive_only_metrics,
    severity_regression_metrics, threshold_predictions,
)
from floating_plastic.model import custom_objects
from floating_plastic.pipeline import make_heterogeneous_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _predict(model, records, batch_size, image_size):
    import tensorflow as tf
    dataset = make_heterogeneous_tf_dataset(records, batch_size, image_size)
    probabilities, severity, class_targets, severity_targets = [], [], [], []
    for images, targets in dataset:
        output = model(images, training=False)
        c = targets["classification"].numpy()
        s = targets["severity"].numpy()
        if not np.all(c[:, 1] == 1):
            raise ValueError("evaluation records must have an available classification target")
        probabilities.extend(np.asarray(output["classification"]).reshape(-1).tolist())
        severity.extend(np.asarray(output["severity"]).reshape(-1).tolist())
        class_targets.extend(c[:, 0].astype(int).tolist())
        severity_targets.extend(s[:, 0].tolist())
    return np.asarray(class_targets), np.asarray(probabilities), np.asarray(severity_targets), np.asarray(severity)


def _write_flo_csv(path, records, truth, probabilities, severity_true, severity_pred, threshold):
    predictions = threshold_predictions(probabilities, threshold)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "filename", "true_class", "predicted_probability", "predicted_class",
            "true_severity", "predicted_severity", "severity_absolute_error",
        ])
        writer.writeheader()
        for record, y, probability, pred, true_sev, pred_sev in zip(
            records, truth, probabilities, predictions, severity_true, severity_pred
        ):
            writer.writerow({"filename": record.filename, "true_class": int(y),
                             "predicted_probability": float(probability), "predicted_class": pred,
                             "true_severity": float(true_sev), "predicted_severity": float(pred_sev),
                             "severity_absolute_error": abs(float(pred_sev) - float(true_sev))})


def _write_ugv_csv(path, records, probabilities, threshold):
    predictions = threshold_predictions(probabilities, threshold)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "true_annotated_waste_present",
                                                    "predicted_probability", "predicted_positive"])
        writer.writeheader()
        for record, probability, predicted in zip(records, probabilities, predictions):
            writer.writerow({"filename": record.filename, "true_annotated_waste_present": 1,
                             "predicted_probability": float(probability), "predicted_positive": predicted})


def _save_plots(output, control_ugv, model_ugv, flo_counts, severity_true, severity_pred,
                control_recall, model_recall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tn, fp, fn, tp = (flo_counts[key] for key in ("tn", "fp", "fn", "tp"))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
    for row, values in enumerate(((tn, fp), (fn, tp))):
        for col, value in enumerate(values):
            ax.text(col, row, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Absent", "Present"],
           yticklabels=["Absent", "Present"], xlabel="Predicted", ylabel="True",
           title="Model B FloPWD test confusion matrix")
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(severity_true, severity_pred, s=18, alpha=0.55)
    ax.plot([0, 100], [0, 100], "k--", linewidth=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="True coverage (percentage points)",
           ylabel="Predicted coverage (percentage points)", title="Model B FloPWD test coverage")
    fig.tight_layout(); fig.savefig(output / "severity_scatter.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(severity_true, severity_pred - severity_true, s=18, alpha=0.55)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 100), xlabel="True coverage (percentage points)",
           ylabel="Prediction residual (percentage points)", title="Model B FloPWD test residuals")
    fig.tight_layout(); fig.savefig(output / "severity_residuals.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(control_ugv, bins=20, alpha=0.55, label="Control A")
    ax.hist(model_ugv, bins=20, alpha=0.55, label="Model B")
    ax.set(xlim=(0, 1), xlabel="Predicted annotated-waste probability", ylabel="Image count",
           title="UGV grouped test probability distributions")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "ugv_probability_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Control A", "Model B"], [control_recall, model_recall], color=["#7589a8", "#2f8061"])
    ax.set(ylim=(0, 1), ylabel="Positive recall", title="UGV annotated-waste positive recall")
    fig.tight_layout(); fig.savefig(output / "ugv_positive_recall_comparison.png", dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--control-model", type=Path, default=Path("runs/flopwd_original_seed42/model.keras"))
    parser.add_argument("--model-b", type=Path, default=Path("runs/multidomain_seed42/model.keras"))
    parser.add_argument("--control-metrics", type=Path, default=Path("runs/flopwd_original_seed42/evaluation/metrics.json"))
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/multidomain_seed42/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Evaluation directory must be absent or empty: {args.output_dir}")

    import tensorflow as tf
    with args.spec.open(encoding="utf-8") as stream:
        import yaml
        spec = yaml.safe_load(stream)
    flo_manifest_path = Path(spec["shared_protocol"]["split_manifests"]["flopwd"])
    ugv_manifest_path = Path(spec["shared_protocol"]["split_manifests"]["ugv"])
    flo_records = load_flopwd(args.data_dir)
    ugv_records = load_ugv(args.ugv_dir)
    flo_manifest = load_split_manifest(flo_manifest_path, [record.filename for record in flo_records])
    ugv_manifest = load_ugv_grouped_split_manifest(ugv_manifest_path, ugv_records)
    flo_parts = {split: records_for_split(flo_records, flo_manifest, split)
                 for split in ("train", "validation", "test")}
    ugv_parts = {split: records_for_ugv_split(ugv_records, ugv_manifest, split)
                 for split in ("train", "validation", "test")}
    flo_test = [heterogeneous_flopwd_record(record) for record in flo_parts["test"]]
    ugv_test = [heterogeneous_ugv_record(record) for record in ugv_parts["test"]]
    test_names = {"flopwd": {r.filename for r in flo_parts["test"]},
                  "ugv": {r.filename for r in ugv_parts["test"]}}
    if test_names["flopwd"] & {r.filename for r in flo_parts["train"] + flo_parts["validation"]}:
        raise ValueError("FloPWD test filenames overlap train/validation")
    if test_names["ugv"] & {r.filename for r in ugv_parts["train"] + ugv_parts["validation"]}:
        raise ValueError("UGV grouped test filenames overlap train/validation")
    if any(heterogeneous_ugv_record(record).severity_target_available for record in ugv_parts["test"]):
        raise ValueError("UGV test records must not have severity labels")

    control = tf.keras.models.load_model(args.control_model, compile=False, custom_objects=custom_objects())
    model_b = tf.keras.models.load_model(args.model_b, compile=False, custom_objects=custom_objects())
    flo_true, flo_prob, sev_true, sev_pred = _predict(model_b, flo_test, args.batch_size, (224, 224))
    ugv_true_a, ugv_prob_a, _, _ = _predict(control, ugv_test, args.batch_size, (224, 224))
    ugv_true_b, ugv_prob_b, _, ugv_pred_unused = _predict(model_b, ugv_test, args.batch_size, (224, 224))
    if len(flo_test) != 300 or len(ugv_test) != 536:
        raise ValueError(f"unexpected test counts: FloPWD={len(flo_test)}, UGV={len(ugv_test)}")
    if not np.all(ugv_true_a == 1) or not np.all(ugv_true_b == 1):
        raise ValueError("UGV test target must be positive-only annotated waste")
    if len(set(record.filename for record in flo_test)) != len(flo_test) or len(set(record.filename for record in ugv_test)) != len(ugv_test):
        raise ValueError("duplicate test filenames")
    if np.any((flo_prob < 0) | (flo_prob > 1)) or np.any((ugv_prob_a < 0) | (ugv_prob_a > 1)) or np.any((ugv_prob_b < 0) | (ugv_prob_b > 1)):
        raise ValueError("predicted probability outside [0, 1]")
    if np.any((sev_pred < 0) | (sev_pred > 100)):
        raise ValueError("severity prediction outside [0, 100]")

    threshold = float(spec["shared_protocol"]["classification_threshold"])
    flo_pred = threshold_predictions(flo_prob, threshold)
    flo_class = classification_metrics(flo_true, flo_pred)
    flo_severity = severity_regression_metrics(sev_true, sev_pred)
    positive = flo_true == 1
    flo_severity_positive = severity_regression_metrics(sev_true[positive], sev_pred[positive])
    control_ugv = positive_only_metrics(ugv_prob_a, threshold)
    model_ugv = positive_only_metrics(ugv_prob_b, threshold)
    control_metrics = json.loads(args.control_metrics.read_text(encoding="utf-8"))
    control_flo = control_metrics["classification"]
    comparison = {
        "flopwd": {
            "control_a": {**control_flo,
                          "severity_mae_all_percentage_points": control_metrics["severity_all_test_images"]["mae_percentage_points"],
                          "severity_mae_positive_percentage_points": control_metrics["severity_positive_test_images"]["mae_percentage_points"]},
            "model_b": {**flo_class,
                        "severity_mae_all_percentage_points": flo_severity["mae_percentage_points"],
                        "severity_rmse_all_percentage_points": flo_severity["rmse_percentage_points"],
                        "severity_mae_positive_percentage_points": flo_severity_positive["mae_percentage_points"],
                        "severity_rmse_positive_percentage_points": flo_severity_positive["rmse_percentage_points"]},
        },
        "ugv_annotated_waste_positive_recall": {"control_a": control_ugv, "model_b": model_ugv},
    }
    for metric, a_value, b_value in (
        ("accuracy", control_flo["accuracy"], flo_class["accuracy"]),
        ("sensitivity", control_flo["sensitivity"], flo_class["sensitivity"]),
        ("specificity", control_flo["specificity"], flo_class["specificity"]),
        ("severity_mae_all_percentage_points", control_metrics["severity_all_test_images"]["mae_percentage_points"], flo_severity["mae_percentage_points"]),
        ("severity_mae_positive_percentage_points", control_metrics["severity_positive_test_images"]["mae_percentage_points"], flo_severity_positive["mae_percentage_points"]),
    ):
        comparison["flopwd"][f"{metric}_change_b_minus_a"] = b_value - a_value
    comparison["ugv_annotated_waste_positive_recall"]["change_b_minus_a"] = (
        model_ugv["positive_recall"] - control_ugv["positive_recall"])
    gaps = {
        "control_a": abs(control_flo["sensitivity"] - control_ugv["positive_recall"]),
        "model_b": abs(flo_class["sensitivity"] - model_ugv["positive_recall"]),
        "name": "domain recall gap",
        "interpretation": "UGV target is broader annotated waste; this is a transfer diagnostic, not a matched-label benchmark.",
    }
    eval_obj = {
        "scope": "untouched test partitions evaluated after frozen matrix and completed Model B training",
        "threshold": threshold,
        "floPWD": {"classification": flo_class, "severity_all": flo_severity,
                   "severity_positive": flo_severity_positive,
                   "severity_prediction_bounds": {"min": float(sev_pred.min()), "max": float(sev_pred.max()),
                                                   "within_0_100": True}},
        "ugv_control_a": control_ugv,
        "ugv_model_b": model_ugv,
        "primary_comparison": comparison,
        "domain_recall_gap": gaps,
        "integrity": {"flopwd_test_rows": len(flo_test), "ugv_test_rows": len(ugv_test),
                      "flo_test_unique_filenames": len(test_names["flopwd"]),
                      "ugv_test_unique_filenames": len(test_names["ugv"]),
                      "probabilities_within_0_1": True, "severity_within_0_100": True,
                      "ugv_severity_labels": 0},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_flo_csv(args.output_dir / "flopwd_predictions.csv", flo_test, flo_true, flo_prob,
                   sev_true, sev_pred, threshold)
    _write_ugv_csv(args.output_dir / "ugv_control_a_predictions.csv", ugv_test, ugv_prob_a, threshold)
    _write_ugv_csv(args.output_dir / "ugv_model_b_predictions.csv", ugv_test, ugv_prob_b, threshold)
    _write_json(args.output_dir / "metrics.json", eval_obj)
    _save_plots(args.output_dir, ugv_prob_a, ugv_prob_b, flo_class["confusion_matrix"],
                sev_true, sev_pred, control_ugv["positive_recall"], model_ugv["positive_recall"])
    print(json.dumps(eval_obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
