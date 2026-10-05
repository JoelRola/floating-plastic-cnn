"""Evaluate the final frozen Model D checkpoint once on frozen test partitions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.data import (
    heterogeneous_flopwd_record, heterogeneous_ugv_record, load_flopwd, load_ugv,
)
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.metrics import (
    classification_metrics, positive_only_metrics, severity_regression_metrics,
    threshold_predictions,
)
from floating_plastic.model import custom_objects
from floating_plastic.pipeline import make_heterogeneous_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _predict_once(model, records, batch_size):
    dataset = make_heterogeneous_tf_dataset(records, batch_size, (224, 224))
    truth, probability, severity_truth, severity_prediction = [], [], [], []
    for images, targets in dataset:
        output = model(images, training=False)
        classes = targets["classification"].numpy()
        severities = targets["severity"].numpy()
        if not np.all(classes[:, 1] == 1):
            raise ValueError("test rows must have available classification targets")
        probability.extend(np.asarray(output["classification"]).reshape(-1).tolist())
        truth.extend(classes[:, 0].astype(int).tolist())
        severity_prediction.extend(np.asarray(output["severity"]).reshape(-1).tolist())
        severity_truth.extend(severities[:, 0].tolist())
    return tuple(np.asarray(values) for values in
                 (truth, probability, severity_truth, severity_prediction))


def _severity_summary(truth, prediction, class_truth):
    errors = np.abs(prediction - truth)
    positive = np.asarray(class_truth, dtype=int) == 1
    if prediction.max() <= 0.1:
        pattern = "near-zero collapse"
    elif prediction.std() <= 0.1:
        pattern = "near-constant collapse"
    elif severity_regression_metrics(truth, prediction)["mae_percentage_points"] <= 2.169630868338546:
        pattern = "functional (MAE at or below Control A)"
    elif severity_regression_metrics(truth, prediction)["mae_percentage_points"] < 5.472192791637813:
        pattern = "partial recovery relative to Model C"
    else:
        pattern = "another pattern (variable output without MAE recovery versus C)"
    return {
        **severity_regression_metrics(truth, prediction),
        "median_absolute_error_percentage_points": float(np.median(errors)),
        "p90_absolute_error_percentage_points": float(np.quantile(errors, 0.90)),
        "maximum_absolute_error_percentage_points": float(errors.max()),
        "mean_true_severity_percentage_points": float(truth.mean()),
        "mean_predicted_severity_percentage_points": float(prediction.mean()),
        "median_predicted_severity_percentage_points": float(np.median(prediction)),
        "predicted_severity_standard_deviation_percentage_points": float(prediction.std()),
        "minimum_predicted_severity_percentage_points": float(prediction.min()),
        "maximum_predicted_severity_percentage_points": float(prediction.max()),
        "positive_images": {"sample_count": int(positive.sum()),
                            **severity_regression_metrics(truth[positive], prediction[positive])},
        "head_pattern": pattern,
        "largest_absolute_errors": [
            {"index": int(i), "true_severity": float(truth[i]),
             "predicted_severity": float(prediction[i]), "absolute_error": float(errors[i])}
            for i in np.argsort(errors)[::-1][:10]
        ],
    }


def _write_predictions(path, records, truth, probabilities, predictions, severity_truth=None,
                       severity_prediction=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "true_class", "predicted_probability", "predicted_class"]
    if severity_truth is not None:
        fields += ["true_severity", "predicted_severity", "severity_absolute_error"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (record, target, probability, prediction) in enumerate(
                zip(records, truth, probabilities, predictions)):
            row = {"filename": record.filename, "true_class": int(target),
                   "predicted_probability": float(probability), "predicted_class": int(prediction)}
            if severity_truth is not None:
                row.update(true_severity=float(severity_truth[index]),
                           predicted_severity=float(severity_prediction[index]),
                           severity_absolute_error=abs(float(severity_prediction[index] - severity_truth[index])))
            writer.writerow(row)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _figures(out, flo_truth, flo_probability, flo_prediction, severity_truth, severity_prediction,
             models, ugv_models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = models["D"]["classification"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]]
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row][column]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Absent", "Present"],
           yticklabels=["Absent", "Present"], xlabel="Predicted", ylabel="True",
           title="Model D FloPWD test confusion matrix")
    fig.tight_layout(); fig.savefig(out / "d_flopwd_confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(severity_truth, severity_prediction, s=18, alpha=0.55)
    ax.plot([0, 100], [0, 100], "k--", linewidth=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="True coverage (pp)",
           ylabel="Predicted coverage (pp)", title="Model D FloPWD test severity")
    fig.tight_layout(); fig.savefig(out / "d_flopwd_severity_scatter.png", dpi=160); plt.close(fig)

    residual = severity_prediction - severity_truth
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(severity_truth, residual, s=18, alpha=0.55)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 100), xlabel="True coverage (pp)", ylabel="Prediction residual (pp)",
           title="Model D FloPWD test severity residuals")
    fig.tight_layout(); fig.savefig(out / "d_flopwd_severity_residuals.png", dpi=160); plt.close(fig)

    names = ["C", "D"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2); width = 0.24
    for offset, key, label in ((-width, "accuracy", "Accuracy"), (0, "sensitivity", "Sensitivity"),
                               (width, "specificity", "Specificity")):
        ax.bar(x + offset, [models[name]["classification"][key] for name in names], width, label=label)
    ax.set(xticks=x, xticklabels=["Model C", "Model D"], ylim=(0, 1), ylabel="Score",
           title="FloPWD classification: C versus D")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "cd_classification.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in (("C", "#bc7652"), ("D", "#2f8061")):
        item = models[name]["classification"]
        ax.scatter(item["specificity"], item["sensitivity"], color=color, s=70, label=f"Model {name}")
        ax.annotate(name, (item["specificity"], item["sensitivity"]), xytext=(5, 4),
                    textcoords="offset points")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Specificity", ylabel="Sensitivity",
           title="FloPWD specificity versus sensitivity")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "cd_specificity_sensitivity.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2); width = 0.34
    ax.bar(x - width / 2, [models[n]["severity_mae_all"] for n in names], width, label="All images")
    ax.bar(x + width / 2, [models[n]["severity_mae_positive"] for n in names], width, label="Positive images")
    ax.set(xticks=x, xticklabels=["Model C", "Model D"], ylabel="MAE (pp)", title="Severity MAE: C versus D")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "cd_severity_mae.png", dpi=160); plt.close(fig)

    names = ["A", "C", "D"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3); width = 0.34
    ax.bar(x - width / 2, [models[n]["severity_mae_all"] for n in names], width, label="All images")
    ax.bar(x + width / 2, [models[n]["severity_mae_positive"] for n in names], width, label="Positive images")
    ax.set(xticks=x, xticklabels=[f"Model {n}" for n in names], ylabel="MAE (pp)", title="A/C/D severity MAE")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "acd_severity_mae.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["C", "D"], [ugv_models[n]["positive_recall"] for n in ("C", "D")],
           color=["#bc7652", "#2f8061"])
    ax.set(ylim=(0, 1.02), xlabel="Model", ylabel="UGV annotated-waste positive recall",
           title="UGV grouped test recall: C versus D")
    fig.tight_layout(); fig.savefig(out / "cd_ugv_recall.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in zip(("A", "C", "D"), ("#7589a8", "#bc7652", "#2f8061")):
        values = np.asarray(models[name]["severity_test_predictions"], dtype=float)
        ax.hist(values, bins=20, range=(0, 100), alpha=0.4, label=f"Model {name}", color=color)
    ax.set(xlim=(0, 100), xlabel="Predicted severity (pp)", ylabel="Test images",
           title="FloPWD test severity prediction distributions")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "acd_severity_prediction_distribution.png", dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("runs/task_decoupled_multidomain_seed42/model.keras"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/task_decoupled_multidomain_seed42"))
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/task_decoupled_multidomain_seed42/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Evaluation output directory must be absent or empty: {args.output_dir}")

    import tensorflow as tf
    import yaml

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    d_spec, shared = spec["matrix"]["D"], spec["shared_protocol"]
    if (d_spec["name"] != "task_decoupled_multidomain" or d_spec["status"] != "frozen_not_run" or
            d_spec["architecture_variant"] != "task_decoupled" or
            d_spec["domain_sampling"] != {"flopwd": 0.5, "ugv": 0.5} or
            d_spec["flopwd_class_balance"]["negative_to_positive"] != "1:1" or
            int(d_spec["total_optimizer_updates"]) != 660):
        raise ValueError("Model D frozen experiment specification changed")
    threshold = float(d_spec["classification_threshold"])
    protocol = shared
    flo_manifest_path = Path(protocol["split_manifests"]["flopwd"])
    ugv_manifest_path = Path(protocol["split_manifests"]["ugv"])
    if (_sha(flo_manifest_path) != protocol["split_manifests"]["flopwd_sha256"] or
            _sha(ugv_manifest_path) != protocol["split_manifests"]["ugv_sha256"]):
        raise ValueError("split manifest SHA does not match the frozen matrix")

    flo_records = load_flopwd(args.data_dir)
    ugv_records = load_ugv(args.ugv_dir)
    flo_manifest = load_split_manifest(flo_manifest_path, [row.filename for row in flo_records])
    ugv_manifest = load_ugv_grouped_split_manifest(ugv_manifest_path, ugv_records)
    flo_parts = {part: records_for_split(flo_records, flo_manifest, part)
                 for part in ("train", "validation", "test")}
    ugv_parts = {part: records_for_ugv_split(ugv_records, ugv_manifest, part)
                 for part in ("train", "validation", "test")}
    flo_test = [heterogeneous_flopwd_record(row) for row in flo_parts["test"]]
    ugv_test = [heterogeneous_ugv_record(row) for row in ugv_parts["test"]]
    flo_test_names = [row.filename for row in flo_test]
    ugv_test_names = [row.filename for row in ugv_test]
    if len(flo_test) != 300 or len(set(flo_test_names)) != 300:
        raise ValueError("FloPWD test partition must have exactly 300 unique rows")
    if len(ugv_test) != 536 or len(set(ugv_test_names)) != 536:
        raise ValueError("UGV grouped test partition must have exactly 536 unique rows")
    if (set(flo_test_names) & ({r.filename for r in flo_parts["train"]} |
                               {r.filename for r in flo_parts["validation"]}) or
            set(ugv_test_names) & ({r.filename for r in ugv_parts["train"]} |
                                   {r.filename for r in ugv_parts["validation"]})):
        raise ValueError("test filenames overlap training or validation rows")
    groups = {part: set(ugv_manifest["group_splits"][part])
              for part in ("train", "validation", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"),
                                               ("validation", "test"))):
        raise ValueError("UGV canonical source groups overlap partitions")
    if any(row.severity_target_available for row in ugv_test):
        raise ValueError("UGV severity must remain unavailable")

    model = tf.keras.models.load_model(args.model, compile=True, custom_objects=custom_objects())
    if model.name != "floating_plastic_resnet50_multitask_task_decoupled":
        raise ValueError("loaded checkpoint is not task-decoupled Model D")
    model.get_layer("classification_tower_dense_1")
    model.get_layer("severity_tower_dense_1")
    backbone = model.get_layer("resnet50")
    if backbone.trainable_variables or {id(v) for v in backbone.weights} & {id(v) for v in model.trainable_variables}:
        raise ValueError("a ResNet weight is included in trainable variables")
    backbone.trainable = False

    # This is the sole inference pass over each frozen test partition.
    flo_truth, flo_probability, severity_truth, severity_prediction = _predict_once(
        model, flo_test, args.batch_size)
    ugv_truth, ugv_probability, ugv_severity_truth, _unused_ugv_severity = _predict_once(
        model, ugv_test, args.batch_size)
    del model
    if not np.all(ugv_truth == 1) or len(ugv_severity_truth) != 536:
        raise ValueError("UGV test target contract mismatch")
    if np.any(~np.isnan(ugv_severity_truth)):
        raise ValueError("UGV predictions contain fabricated severity labels")
    if (np.any(~np.isfinite(flo_probability)) or np.any((flo_probability < 0) | (flo_probability > 1)) or
            np.any(~np.isfinite(ugv_probability)) or np.any((ugv_probability < 0) | (ugv_probability > 1))):
        raise ValueError("classification probabilities must be finite and in [0,1]")
    if (np.any(~np.isfinite(severity_prediction)) or np.any((severity_prediction < 0) |
                                                            (severity_prediction > 100))):
        raise ValueError("severity predictions must be finite and in [0,100]")
    flo_prediction = np.asarray(threshold_predictions(flo_probability, threshold), dtype=int)
    ugv_prediction = np.asarray(threshold_predictions(ugv_probability, threshold), dtype=int)
    flo_classification = classification_metrics(flo_truth, flo_prediction.tolist())
    severity = _severity_summary(severity_truth, severity_prediction, flo_truth)
    ugv_positive = positive_only_metrics(ugv_probability, threshold)

    a_metrics_path = Path("runs/flopwd_original_seed42/evaluation/metrics.json")
    c_metrics_path = Path("runs/multidomain_class_balanced_seed42/evaluation/flopwd/metrics.json")
    c_ugv_path = Path("runs/multidomain_class_balanced_seed42/evaluation/ugv/metrics.json")
    c_comparison_path = Path("runs/multidomain_class_balanced_seed42/evaluation/comparison.json")
    a_metrics = json.loads(a_metrics_path.read_text(encoding="utf-8"))
    c_metrics = json.loads(c_metrics_path.read_text(encoding="utf-8"))
    c_ugv_metrics = json.loads(c_ugv_path.read_text(encoding="utf-8"))
    c_comparison = json.loads(c_comparison_path.read_text(encoding="utf-8"))
    a_cross = c_comparison["models"]["A"]
    c_cross = c_comparison["models"]["C"]
    a_prediction_rows = _read_csv(a_metrics_path.parent / "predictions.csv")
    c_prediction_rows = _read_csv(c_metrics_path.parent / "predictions.csv")
    if len(a_prediction_rows) != 300 or len(c_prediction_rows) != 300:
        raise ValueError("A/C comparison prediction artifacts must each contain 300 rows")
    figure_models = {
        "A": {**a_cross, "severity_test_predictions": [
            float(row["predicted_severity"]) for row in a_prediction_rows]},
        "C": {**c_cross, "severity_test_predictions": [
            float(row["predicted_severity"]) for row in c_prediction_rows]},
    }
    d_cross = {"classification": flo_classification,
               "severity_mae_all": severity["mae_percentage_points"],
               "severity_mae_positive": severity["positive_images"]["mae_percentage_points"],
               "ugv_positive_recall": ugv_positive["positive_recall"],
               "ugv_probability_mean": ugv_positive["predicted_probability_distribution"]["mean"]}
    comparison_models = {"A": a_cross, "C": c_cross, "D": d_cross}
    figure_models["D"] = {**d_cross, "severity_test_predictions": severity_prediction.tolist()}
    deltas = {
        key: float(d_cross["classification"][key] - c_cross["classification"][key])
        for key in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1")
    }
    deltas.update({
        "fp": d_cross["classification"]["confusion_matrix"]["fp"] - c_cross["classification"]["confusion_matrix"]["fp"],
        "fn": d_cross["classification"]["confusion_matrix"]["fn"] - c_cross["classification"]["confusion_matrix"]["fn"],
        "severity_mae_all_percentage_points": d_cross["severity_mae_all"] - c_cross["severity_mae_all"],
        "severity_mae_positive_percentage_points": d_cross["severity_mae_positive"] - c_cross["severity_mae_positive"],
        "ugv_annotated_waste_positive_recall": d_cross["ugv_positive_recall"] - c_cross["ugv_positive_recall"],
    })

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_predictions(out / "flopwd" / "predictions.csv", flo_test, flo_truth, flo_probability,
                       flo_prediction, severity_truth, severity_prediction)
    _write_predictions(out / "ugv" / "predictions.csv", ugv_test, ugv_truth, ugv_probability, ugv_prediction)
    _write_json(out / "flopwd" / "metrics.json", {
        "sample_count": 300, "threshold": threshold, "classification": flo_classification,
        "mean_classification_probability": float(flo_probability.mean()),
        "positive_predictions": int(flo_prediction.sum()),
        "negative_predictions": int(len(flo_prediction) - flo_prediction.sum()),
        "severity": severity,
        "integrity": {"unique_test_rows": 300, "confusion_total": sum(flo_classification["confusion_matrix"].values()),
                      "probabilities_in_bounds": True, "severity_in_bounds": True}})
    _write_json(out / "ugv" / "metrics.json", {
        "sample_count": 536, "target": "annotated-waste positive", "threshold": threshold,
        "annotated_waste_positive_recall": ugv_positive["positive_recall"],
        "mean_predicted_probability": ugv_positive["predicted_probability_distribution"]["mean"],
        "median_predicted_probability": ugv_positive["predicted_probability_distribution"]["median"],
        "standard_deviation": ugv_positive["predicted_probability_distribution"]["standard_deviation"],
        "minimum_probability": ugv_positive["predicted_probability_distribution"]["min"],
        "maximum_probability": ugv_positive["predicted_probability_distribution"]["max"],
        "below_threshold_count": int(np.sum(ugv_prediction == 0)), "severity_labels": 0})
    _figures(out, flo_truth, flo_probability, flo_prediction, severity_truth, severity_prediction,
             figure_models, {"C": {"positive_recall": c_ugv_metrics["annotated_waste_positive_recall"]},
                                 "D": {"positive_recall": ugv_positive["positive_recall"]}})
    _write_json(out / "comparison.json", {
        "threshold": threshold, "models": comparison_models, "d_minus_c": deltas,
        "ugv_target_note": "FloPWD plastic-positive and UGV annotated-waste targets are not perfectly semantically identical; recall gap is diagnostic only.",
    })
    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    _write_json(out / "integrity.json", {
        "flopwd_unique_test_rows": 300, "ugv_unique_grouped_test_rows": 536,
        "no_train_test_overlap": True, "no_validation_test_overlap": True,
        "ugv_canonical_source_groups_disjoint": True, "classification_probabilities_in_0_1": True,
        "severity_predictions_in_0_100": True, "ugv_severity_values": 0,
        "confusion_total": sum(flo_classification["confusion_matrix"].values()),
        "training_rows_in_test_predictions": 0, "validation_rows_in_test_predictions": 0,
    })
    hash_files = {
        "model_keras": args.model, "flopwd_metrics_json": out / "flopwd" / "metrics.json",
        "flopwd_predictions_csv": out / "flopwd" / "predictions.csv",
        "ugv_metrics_json": out / "ugv" / "metrics.json",
        "ugv_predictions_csv": out / "ugv" / "predictions.csv",
        "flopwd_split_manifest": flo_manifest_path, "ugv_grouped_split_manifest": ugv_manifest_path,
        "frozen_matrix_specification": args.spec,
    }
    _write_json(out / "sha256.json", {name: _sha(path) for name, path in hash_files.items()})
    print(json.dumps({"training_git_sha": metadata["git_commit_sha"], "flopwd": flo_classification,
                      "severity": severity, "ugv_annotated_waste_positive": ugv_positive,
                      "d_minus_c": deltas, "figures": sorted(path.name for path in out.glob("*.png"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
