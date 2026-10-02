"""Evaluate frozen Model C once on the predeclared FloPWD and grouped UGV tests."""

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
from floating_plastic.metrics import classification_metrics, positive_only_metrics, severity_regression_metrics, threshold_predictions
from floating_plastic.model import custom_objects
from floating_plastic.pipeline import make_heterogeneous_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _predict(model, records, batch_size):
    dataset = make_heterogeneous_tf_dataset(records, batch_size, (224, 224))
    class_true, probabilities, severity_true, severity_pred = [], [], [], []
    for images, targets in dataset:
        output = model(images, training=False)
        classes = targets["classification"].numpy()
        severities = targets["severity"].numpy()
        if not np.all(classes[:, 1] == 1):
            raise ValueError("test records must have an available classification target")
        probabilities.extend(np.asarray(output["classification"]).reshape(-1).tolist())
        severity_pred.extend(np.asarray(output["severity"]).reshape(-1).tolist())
        class_true.extend(classes[:, 0].astype(int).tolist())
        severity_true.extend(severities[:, 0].tolist())
    return tuple(np.asarray(items) for items in (class_true, probabilities, severity_true, severity_pred))


def _write_predictions(path, records, truth, probability, predicted, severity_true=None, severity_pred=None):
    fields = ["filename", "true_class", "predicted_probability", "predicted_class"]
    if severity_true is not None:
        fields += ["true_severity", "predicted_severity", "severity_absolute_error"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (record, true, prob, pred) in enumerate(zip(records, truth, probability, predicted)):
            row = {"filename": record.filename, "true_class": int(true),
                   "predicted_probability": float(prob), "predicted_class": int(pred)}
            if severity_true is not None:
                row.update(true_severity=float(severity_true[index]),
                           predicted_severity=float(severity_pred[index]),
                           severity_absolute_error=abs(float(severity_pred[index] - severity_true[index])))
            writer.writerow(row)


def _severity_report(records, class_labels, truth, predicted):
    errors = np.abs(predicted - truth)
    order = np.argsort(errors)[::-1]
    positive_mask = np.asarray(class_labels, dtype=int) == 1
    positive_metrics = severity_regression_metrics(truth[positive_mask], predicted[positive_mask])
    pred_std = float(np.std(predicted))
    if float(predicted.max()) <= 0.1:
        pattern = "near-zero collapse"
    elif pred_std <= 0.1:
        pattern = "near-constant collapse at another value"
    elif pred_std > 0.1:
        pattern = "variable output; head remains non-collapsed (quality assessed by error metrics)"
    else:
        pattern = "another failure pattern"
    return {
        **severity_regression_metrics(truth, predicted),
        "median_absolute_error_percentage_points": float(np.median(errors)),
        "p90_absolute_error_percentage_points": float(np.quantile(errors, 0.90)),
        "maximum_absolute_error_percentage_points": float(errors.max()),
        "mean_true_severity_percentage_points": float(np.mean(truth)),
        "mean_predicted_severity_percentage_points": float(np.mean(predicted)),
        "minimum_predicted_severity_percentage_points": float(predicted.min()),
        "maximum_predicted_severity_percentage_points": float(predicted.max()),
        "predicted_severity_standard_deviation_percentage_points": pred_std,
        "positive_images": {"sample_count": int(positive_mask.sum()), **positive_metrics},
        "head_pattern": pattern,
        "collapse_rule": "near-zero if maximum <= 0.1 pp; otherwise near-constant if SD <= 0.1 pp",
        "largest_absolute_errors": [
            {"filename": records[int(i)].filename, "true_severity": float(truth[i]),
             "predicted_severity": float(predicted[i]), "absolute_error": float(errors[i])}
            for i in order[:10]
        ],
    }


def _figures(out, truth, probability, prediction, severity_true, severity_pred, ugv_probability,
             metrics_by_model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = metrics_by_model["C"]["classification"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], cmap="Blues")
    for row, vals in enumerate(((counts["tn"], counts["fp"]), (counts["fn"], counts["tp"]))):
        for col, value in enumerate(vals):
            ax.text(col, row, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Absent", "Present"],
           yticklabels=["Absent", "Present"], xlabel="Predicted", ylabel="True",
           title="Model C FloPWD test confusion matrix")
    fig.tight_layout(); fig.savefig(out / "flopwd_confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(severity_true, severity_pred, s=18, alpha=0.55)
    ax.plot([0, 100], [0, 100], "k--", linewidth=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="True coverage (pp)", ylabel="Predicted coverage (pp)",
           title="Model C FloPWD test severity")
    fig.tight_layout(); fig.savefig(out / "flopwd_severity_scatter.png", dpi=160); plt.close(fig)

    residual = severity_pred - severity_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(severity_true, residual, s=18, alpha=0.55)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 100), xlabel="True coverage (pp)", ylabel="Prediction residual (pp)",
           title="Model C FloPWD test severity residuals")
    fig.tight_layout(); fig.savefig(out / "flopwd_severity_residuals.png", dpi=160); plt.close(fig)

    names = ["A", "B", "C"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3); width = 0.25
    for offset, metric, label in ((-width, "accuracy", "Accuracy"), (0, "sensitivity", "Sensitivity"),
                                  (width, "specificity", "Specificity")):
        ax.bar(x + offset, [metrics_by_model[name]["classification"][metric] for name in names],
               width, label=label)
    ax.set(xticks=x, xticklabels=[f"Model {name}" for name in names], ylim=(0, 1), ylabel="Score",
           title="FloPWD classification comparison")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "abc_classification_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in zip(names, ("#7589a8", "#bc7652", "#2f8061")):
        classification = metrics_by_model[name]["classification"]
        ax.scatter(classification["specificity"], classification["sensitivity"], label=f"Model {name}", color=color, s=65)
        ax.annotate(name, (classification["specificity"], classification["sensitivity"]), xytext=(5, 4), textcoords="offset points")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Specificity", ylabel="Sensitivity",
           title="FloPWD specificity versus sensitivity")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "abc_specificity_sensitivity.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3); width = 0.34
    ax.bar(x - width / 2, [metrics_by_model[n]["severity_mae_all"] for n in names], width, label="All images")
    ax.bar(x + width / 2, [metrics_by_model[n]["severity_mae_positive"] for n in names], width, label="Positive images")
    ax.set(xticks=x, xticklabels=[f"Model {n}" for n in names], ylabel="MAE (pp)", title="FloPWD severity MAE")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "abc_severity_mae.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    recalls = [metrics_by_model[n]["ugv_positive_recall"] for n in names]
    ax.bar(names, recalls, color=["#7589a8", "#bc7652", "#2f8061"])
    ax.set(ylim=(0, 1.02), xlabel="Model", ylabel="UGV annotated-waste positive recall",
           title="UGV grouped test recall")
    fig.tight_layout(); fig.savefig(out / "abc_ugv_positive_recall.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in zip(names, ("#7589a8", "#bc7652", "#2f8061")):
        values = np.asarray(metrics_by_model[name]["ugv_probabilities"])
        ax.hist(values, bins=20, range=(0, 1), alpha=0.4, label=f"Model {name}", color=color)
    ax.set(xlim=(0, 1), xlabel="Predicted annotated-waste probability", ylabel="Images",
           title="UGV grouped test probability distributions")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "abc_ugv_probability_distributions.png", dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("runs/multidomain_class_balanced_seed42/model.keras"))
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/multidomain_class_balanced_seed42/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="recompute reports from saved prediction CSVs without model inference")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.reuse_predictions:
        raise SystemExit(f"Evaluation output directory must be absent or empty: {args.output_dir}")

    import tensorflow as tf
    import yaml

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    protocol = spec["shared_protocol"]
    flo_manifest_path = Path(protocol["split_manifests"]["flopwd"])
    ugv_manifest_path = Path(protocol["split_manifests"]["ugv"])
    if _sha(flo_manifest_path) != protocol["split_manifests"]["flopwd_sha256"] or _sha(ugv_manifest_path) != protocol["split_manifests"]["ugv_sha256"]:
        raise ValueError("split-manifest hash differs from frozen spec")
    flo_records, ugv_records = load_flopwd(args.data_dir), load_ugv(args.ugv_dir)
    flo_manifest = load_split_manifest(flo_manifest_path, [r.filename for r in flo_records])
    ugv_manifest = load_ugv_grouped_split_manifest(ugv_manifest_path, ugv_records)
    flo_parts = {part: records_for_split(flo_records, flo_manifest, part) for part in ("train", "validation", "test")}
    ugv_parts = {part: records_for_ugv_split(ugv_records, ugv_manifest, part) for part in ("train", "validation", "test")}
    flo_test = [heterogeneous_flopwd_record(r) for r in flo_parts["test"]]
    ugv_test = [heterogeneous_ugv_record(r) for r in ugv_parts["test"]]
    ugv_source_by_name = {r.filename: r for r in ugv_parts["test"]}
    if len(flo_test) != 300 or len(ugv_test) != 536:
        raise ValueError(f"frozen test sizes differ: FloPWD={len(flo_test)}, UGV={len(ugv_test)}")
    flo_names = {r.filename for r in flo_test}
    ugv_names = {r.filename for r in ugv_test}
    if len(flo_names) != 300 or len(ugv_names) != 536:
        raise ValueError("duplicate test rows")
    if flo_names & {r.filename for r in flo_parts["train"] + flo_parts["validation"]}:
        raise ValueError("FloPWD test rows overlap train/validation")
    if ugv_names & {r.filename for r in ugv_parts["train"] + ugv_parts["validation"]}:
        raise ValueError("UGV test rows overlap train/validation")
    group_sets = {part: set(ugv_manifest["group_splits"][part]) for part in ("train", "validation", "test")}
    if any(group_sets[a] & group_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("UGV canonical groups overlap partitions")
    if any(r.severity_target_available for r in ugv_test):
        raise ValueError("UGV severity supervision must remain unavailable")
    a_metrics = json.loads(Path("runs/flopwd_original_seed42/evaluation/metrics.json").read_text(encoding="utf-8"))
    b_metrics = json.loads(Path("runs/multidomain_seed42/evaluation/metrics.json").read_text(encoding="utf-8"))
    a_ugv_pred = np.genfromtxt("runs/multidomain_seed42/evaluation/ugv_control_a_predictions.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    b_ugv_pred = np.genfromtxt("runs/multidomain_seed42/evaluation/ugv_model_b_predictions.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")

    threshold = float(protocol["classification_threshold"])
    if args.reuse_predictions:
        with (args.output_dir / "flopwd" / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            flo_rows = list(csv.DictReader(stream))
        with (args.output_dir / "ugv" / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            ugv_rows = list(csv.DictReader(stream))
        if [row["filename"] for row in flo_rows] != [r.filename for r in flo_test] or [row["filename"] for row in ugv_rows] != [r.filename for r in ugv_test]:
            raise ValueError("saved prediction rows do not match frozen test membership/order")
        y = np.asarray([int(row["true_class"]) for row in flo_rows], dtype=int)
        prob = np.asarray([float(row["predicted_probability"]) for row in flo_rows], dtype=float)
        severity_true = np.asarray([float(row["true_severity"]) for row in flo_rows], dtype=float)
        severity_pred = np.asarray([float(row["predicted_severity"]) for row in flo_rows], dtype=float)
        ugv_truth = np.asarray([int(row["true_class"]) for row in ugv_rows], dtype=int)
        ugv_prob = np.asarray([float(row["predicted_probability"]) for row in ugv_rows], dtype=float)
        ugv_severity_true = np.full(len(ugv_rows), np.nan)
        expected_class = np.asarray([r.classification_target for r in flo_test], dtype=int)
        expected_severity = np.asarray([r.severity_target for r in flo_test], dtype=float)
        if not np.array_equal(y, expected_class) or not np.allclose(severity_true, expected_severity):
            raise ValueError("saved FloPWD labels differ from the frozen test records")
        if not np.all(ugv_truth == 1):
            raise ValueError("saved UGV targets differ from the positive-only frozen test records")
    else:
        model = tf.keras.models.load_model(args.model, compile=False, custom_objects=custom_objects())
        y, prob, severity_true, severity_pred = _predict(model, flo_test, args.batch_size)
        ugv_truth, ugv_prob, ugv_severity_true, _ = _predict(model, ugv_test, args.batch_size)
    flo_predictions = np.asarray(threshold_predictions(prob, threshold), dtype=int)
    ugv_predictions = np.asarray(threshold_predictions(ugv_prob, threshold), dtype=int)
    if len(ugv_truth) != 536 or not np.all(ugv_truth == 1):
        raise ValueError("UGV test target must be 536 positive-only annotated-waste rows")
    if np.any(~np.isfinite(prob)) or np.any(~np.isfinite(ugv_prob)) or np.any((prob < 0) | (prob > 1)) or np.any((ugv_prob < 0) | (ugv_prob > 1)):
        raise ValueError("classification probabilities must be finite and within [0,1]")
    if np.any(~np.isfinite(severity_pred)) or np.any((severity_pred < 0) | (severity_pred > 100)):
        raise ValueError("FloPWD severity predictions must be finite and within [0,100]")
    if len(ugv_severity_true) != 536 or not np.all(np.isnan(ugv_severity_true)):
        raise ValueError("UGV severity values must not be fabricated or evaluated")
    flo_class = classification_metrics(y, flo_predictions)
    sev_report = _severity_report(flo_test, y, severity_true, severity_pred)
    ugv_report = positive_only_metrics(ugv_prob, threshold)
    ugv_low = [{"filename": ugv_test[i].filename, "canonical_source_id": ugv_source_by_name[ugv_test[i].filename].canonical_source_id,
                "predicted_probability": float(ugv_prob[i])}
               for i in np.argsort(ugv_prob) if ugv_prob[i] < threshold]

    false_positive = [i for i in range(len(y)) if y[i] == 0 and flo_predictions[i] == 1]
    false_negative = [i for i in range(len(y)) if y[i] == 1 and flo_predictions[i] == 0]
    by_confidence_fp = sorted(false_positive, key=lambda i: prob[i], reverse=True)
    by_confidence_fn = sorted(false_negative, key=lambda i: prob[i])
    errors = {
        "false_positives": [{"filename": flo_test[i].filename, "probability": float(prob[i])} for i in false_positive],
        "false_negatives": [{"filename": flo_test[i].filename, "probability": float(prob[i])} for i in false_negative],
        "highest_confidence_false_positives": [{"filename": flo_test[i].filename, "probability": float(prob[i])} for i in by_confidence_fp[:10]],
        "highest_confidence_false_negatives": [{"filename": flo_test[i].filename, "probability": float(prob[i])} for i in by_confidence_fn[:10]],
        "ugv_below_threshold": ugv_low,
        "lowest_confidence_ugv_positives": [{"filename": ugv_test[i].filename,
            "canonical_source_id": ugv_source_by_name[ugv_test[i].filename].canonical_source_id, "probability": float(ugv_prob[i])}
            for i in np.argsort(ugv_prob)[:10]],
    }

    model_stats = {
        "A": {"classification": a_metrics["classification"],
              "severity_mae_all": a_metrics["severity_all_test_images"]["mae_percentage_points"],
              "severity_mae_positive": a_metrics["severity_positive_test_images"]["mae_percentage_points"],
              "ugv_positive_recall": float(np.mean(a_ugv_pred["predicted_positive"])),
              "ugv_probability_mean": float(np.mean(a_ugv_pred["predicted_probability"])),
              "ugv_probabilities": a_ugv_pred["predicted_probability"].tolist()},
        "B": {"classification": b_metrics["floPWD"]["classification"],
              "severity_mae_all": b_metrics["floPWD"]["severity_all"]["mae_percentage_points"],
              "severity_mae_positive": b_metrics["floPWD"]["severity_positive"]["mae_percentage_points"],
              "ugv_positive_recall": b_metrics["ugv_model_b"]["positive_recall"],
              "ugv_probability_mean": b_metrics["ugv_model_b"]["predicted_probability_distribution"]["mean"],
              "ugv_probabilities": b_ugv_pred["predicted_probability"].tolist()},
        "C": {"classification": flo_class,
              "severity_mae_all": sev_report["mae_percentage_points"],
              "severity_mae_positive": sev_report["positive_images"]["mae_percentage_points"],
              "ugv_positive_recall": ugv_report["positive_recall"],
              "ugv_probability_mean": ugv_report["predicted_probability_distribution"]["mean"],
              "ugv_probabilities": ugv_prob.tolist()},
    }
    for name, stats in model_stats.items():
        stats["domain_recall_gap"] = abs(stats["classification"]["sensitivity"] - stats["ugv_positive_recall"])
    comparison = {name: {key: value for key, value in stats.items() if key != "ugv_probabilities"}
                  for name, stats in model_stats.items()}
    output = args.output_dir
    (output / "flopwd").mkdir(parents=True, exist_ok=True)
    (output / "ugv").mkdir(parents=True, exist_ok=True)
    _write_predictions(output / "flopwd" / "predictions.csv", flo_test, y, prob, flo_predictions,
                       severity_true, severity_pred)
    _write_predictions(output / "ugv" / "predictions.csv", ugv_test, ugv_truth, ugv_prob, ugv_predictions)
    _write_json(output / "flopwd" / "metrics.json", {
        "sample_count": len(flo_test), "threshold": threshold, "classification": flo_class,
        "mean_classification_probability": float(np.mean(prob)), "positive_predictions": int(flo_predictions.sum()),
        "negative_predictions": int(len(flo_predictions) - flo_predictions.sum()), "severity": sev_report,
        "integrity": {"unique_test_rows": len(flo_names), "confusion_total": sum(flo_class["confusion_matrix"].values()),
                      "probabilities_in_bounds": True, "severity_in_bounds": True}})
    _write_json(output / "ugv" / "metrics.json", {
        "sample_count": len(ugv_test), "target": "annotated-waste positive", "threshold": threshold,
        "annotated_waste_positive_recall": ugv_report["positive_recall"],
        "mean_predicted_probability": ugv_report["predicted_probability_distribution"]["mean"],
        "median_predicted_probability": ugv_report["predicted_probability_distribution"]["median"],
        "standard_deviation": ugv_report["predicted_probability_distribution"]["standard_deviation"],
        "minimum_probability": ugv_report["predicted_probability_distribution"]["min"],
        "maximum_probability": ugv_report["predicted_probability_distribution"]["max"],
        "positive_predictions": ugv_report["predicted_positive_count"], "below_threshold": ugv_low,
        "severity_labels": 0})
    _figures(output, y, prob, flo_predictions, severity_true, severity_pred, ugv_prob, model_stats)
    _write_json(output / "error_analysis.json", errors)
    _write_json(output / "comparison.json", {
        "threshold": threshold, "models": comparison,
        "domain_recall_gap_interpretation": "Transfer diagnostic; FloPWD plastic and UGV annotated-waste targets are not perfectly semantically identical.",
        "hypothesis_1_class_prior_shift": {
            "specificity_a": model_stats["A"]["classification"]["specificity"],
            "specificity_b": model_stats["B"]["classification"]["specificity"],
            "specificity_c": flo_class["specificity"],
            "ugv_positive_recall_c": ugv_report["positive_recall"]},
        "hypothesis_2_severity_negative_transfer": {
            "severity_mae_a": model_stats["A"]["severity_mae_all"],
            "severity_mae_b": model_stats["B"]["severity_mae_all"],
            "severity_mae_c": sev_report["mae_percentage_points"],
            "model_c_head_pattern": sev_report["head_pattern"]}})
    _write_json(output / "metrics.json", {
        "experiment_type": "multidomain_class_balanced", "threshold": threshold,
        "flopwd": {"sample_count": len(flo_test), "classification": flo_class, "severity": sev_report},
        "ugv_annotated_waste_positive": {"sample_count": len(ugv_test), **ugv_report},
        "domain_recall_gap": {name: value["domain_recall_gap"] for name, value in model_stats.items()},
        "integrity": {"flopwd_unique_rows": len(flo_names), "ugv_unique_rows": len(ugv_names),
                      "ugv_canonical_groups_disjoint": True, "ugv_severity_labels": 0,
                      "training_or_validation_rows_in_test": 0}})
    hash_files = {
        "model_keras": args.model,
        "flopwd_metrics_json": output / "flopwd" / "metrics.json",
        "flopwd_predictions_csv": output / "flopwd" / "predictions.csv",
        "ugv_metrics_json": output / "ugv" / "metrics.json",
        "ugv_predictions_csv": output / "ugv" / "predictions.csv",
        "flopwd_split_manifest": flo_manifest_path, "ugv_grouped_split_manifest": ugv_manifest_path,
        "frozen_matrix_specification": args.spec,
    }
    _write_json(output / "sha256.json", {name: _sha(path) for name, path in hash_files.items()})
    print(json.dumps({"floPWD": flo_class, "severity": sev_report, "ugv": ugv_report,
                      "domain_recall_gaps": {name: value["domain_recall_gap"] for name, value in model_stats.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
