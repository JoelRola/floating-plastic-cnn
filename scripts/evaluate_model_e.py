"""Evaluate the frozen Model E checkpoint once on the fixed test partitions."""

import csv
import hashlib
import json
from pathlib import Path
import re
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
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _predict(model, records, batch_size, image_size):
    import tensorflow as tf
    dataset = make_heterogeneous_tf_dataset(records, batch_size, image_size)
    classes, probabilities, severity_true, severity_pred = [], [], [], []
    for images, targets in dataset:
        output = model(images, training=False)
        class_target = targets["classification"].numpy()
        severity_target = targets["severity"].numpy()
        if not np.all(class_target[:, 1] == 1):
            raise ValueError("test records must have classification labels")
        classes.extend(class_target[:, 0].astype(int).tolist())
        probabilities.extend(np.asarray(output["classification"]).reshape(-1).tolist())
        severity_true.extend(severity_target[:, 0].tolist())
        severity_pred.extend(np.asarray(output["severity"]).reshape(-1).tolist())
    return tuple(np.asarray(values) for values in (classes, probabilities, severity_true, severity_pred))


def _plot_comparisons(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["model"] for row in rows]
    sensitivity = [row["sensitivity"] for row in rows]
    specificity = [row["specificity"] for row in rows]
    prevalence = [row["effective_positive_rate"] for row in rows]
    balanced = [row["balanced_accuracy"] for row in rows]
    ugv = [row["ugv_positive_recall"] for row in rows]
    severity = [row["severity_mae"] for row in rows]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(specificity, sensitivity, "o-", color="#405f91")
    for row in rows:
        ax.annotate(row["model"], (row["specificity"], row["sensitivity"]), xytext=(5, 5), textcoords="offset points")
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="FloPWD specificity", ylabel="FloPWD sensitivity",
           title="B/C/E sensitivity vs specificity")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "bce_sensitivity_specificity.png", dpi=160); plt.close(fig)

    for values, label, filename in ((specificity, "Specificity", "bce_prevalence_specificity.png"),
                                    (sensitivity, "Sensitivity", "bce_prevalence_sensitivity.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(prevalence, values, "o-", color="#a55a3b")
        for row in rows:
            ax.annotate(row["model"], (row["effective_positive_rate"], row[label.lower()]),
                        xytext=(5, 5), textcoords="offset points")
        ax.set(xlim=(0.60, 0.90), ylim=(0, 1.02), xlabel="Effective training positive rate", ylabel=label,
               title=f"Effective positive rate vs {label.lower()}")
        ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / filename, dpi=160); plt.close(fig)

    panels = ((balanced, "Balanced accuracy", "bce_balanced_accuracy.png"),
              (ugv, "UGV annotated-waste positive recall", "bce_ugv_positive_recall.png"),
              (severity, "FloPWD severity MAE (pp)", "bce_severity_mae.png"))
    for values, ylabel, filename in panels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, values, color=["#7589a8", "#567ca2", "#2f8061"])
        ax.set(ylabel=ylabel, title=ylabel); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(output / filename, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(rows)); width = 0.36
    ax.bar(x - width / 2, [row["fp"] for row in rows], width, label="FP", color="#c26b54")
    ax.bar(x + width / 2, [row["fn"] for row in rows], width, label="FN", color="#567ca2")
    ax.set(xticks=x, xticklabels=names, ylabel="Count", title="FloPWD false positives and false negatives")
    ax.legend(); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(output / "bce_false_positive_negative_counts.png", dpi=160); plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/multidomain_2to1_seed42"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    eval_dir = args.run_dir / "evaluation"
    if eval_dir.exists() and any(eval_dir.iterdir()):
        raise SystemExit(f"Evaluation directory must be absent or empty: {eval_dir}")

    import tensorflow as tf
    import yaml
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    shared = spec["shared_protocol"]
    threshold = float(shared["classification_threshold"])
    flo_manifest_path = Path(shared["split_manifests"]["flopwd"])
    ugv_manifest_path = Path(shared["split_manifests"]["ugv"])
    flo_records = load_flopwd(args.data_dir)
    ugv_records = load_ugv(args.ugv_dir)
    flo_manifest = load_split_manifest(flo_manifest_path, [r.filename for r in flo_records])
    ugv_manifest = load_ugv_grouped_split_manifest(ugv_manifest_path, ugv_records)
    flo_parts = {key: records_for_split(flo_records, flo_manifest, key) for key in ("train", "validation", "test")}
    ugv_parts = {key: records_for_ugv_split(ugv_records, ugv_manifest, key) for key in ("train", "validation", "test")}
    flo_test = [heterogeneous_flopwd_record(r) for r in flo_parts["test"]]
    ugv_test = [heterogeneous_ugv_record(r) for r in ugv_parts["test"]]
    if len(flo_test) != 300 or len(ugv_test) != 536:
        raise ValueError(f"unexpected test row counts: FloPWD={len(flo_test)}, UGV={len(ugv_test)}")
    if len({r.filename for r in flo_test}) != 300 or len({r.filename for r in ugv_test}) != 536:
        raise ValueError("duplicate test rows")
    if {r.filename for r in flo_test} & {r.filename for r in flo_parts["train"] + flo_parts["validation"]}:
        raise ValueError("FloPWD test rows overlap train/validation")
    if {r.canonical_source_id for r in ugv_test} & {
        r.canonical_source_id for r in ugv_parts["train"] + ugv_parts["validation"]
    }:
        raise ValueError("UGV canonical source groups overlap train/validation")
    if any(r.severity_target_available or r.severity_target is not None for r in ugv_test):
        raise ValueError("UGV test data must not have severity values")

    # This fresh-process load is also the required serialized-model reload.
    model = tf.keras.models.load_model(args.run_dir / "model.keras", compile=False,
                                       custom_objects=custom_objects())
    flo_true, flo_prob, sev_true, sev_pred = _predict(model, flo_test, args.batch_size, (224, 224))
    ugv_true, ugv_prob, _, _ = _predict(model, ugv_test, args.batch_size, (224, 224))
    if not np.all(ugv_true == 1):
        raise ValueError("UGV test target is not positive-only annotated waste")
    if not np.isfinite(flo_prob).all() or not np.isfinite(ugv_prob).all() or not np.isfinite(sev_pred).all():
        raise ValueError("non-finite prediction")
    if np.any((flo_prob < 0) | (flo_prob > 1)) or np.any((ugv_prob < 0) | (ugv_prob > 1)):
        raise ValueError("classification probability outside [0, 1]")
    if np.any((sev_pred < 0) | (sev_pred > 100)):
        raise ValueError("severity prediction outside [0, 100]")

    flo_pred = np.asarray(threshold_predictions(flo_prob, threshold), dtype=int)
    flo_class = classification_metrics(flo_true, flo_pred)
    flo_sev = severity_regression_metrics(sev_true, sev_pred)
    positive = flo_true == 1
    flo_sev_positive = severity_regression_metrics(sev_true[positive], sev_pred[positive])
    absolute_errors = np.abs(sev_pred - sev_true)
    ordered_errors = np.sort(absolute_errors)
    severity_sd = float(np.std(sev_pred))
    severity_mean = float(np.mean(sev_pred))
    severity_max = float(np.max(sev_pred))
    if severity_max <= 0.1:
        severity_pattern = "near-zero collapse"
    elif severity_sd <= 0.1:
        severity_pattern = "near-constant collapse"
    elif flo_sev["mae_percentage_points"] > float(np.mean(sev_true)):
        severity_pattern = "other failure mode: prediction error exceeds mean true coverage"
    else:
        severity_pattern = "functional"
    ugv_metrics = positive_only_metrics(ugv_prob, threshold)
    fp_indices = np.flatnonzero((flo_true == 0) & (flo_pred == 1))
    fn_indices = np.flatnonzero((flo_true == 1) & (flo_pred == 0))
    fp_rows = sorted(fp_indices.tolist(), key=lambda i: flo_prob[i], reverse=True)
    fn_rows = sorted(fn_indices.tolist(), key=lambda i: flo_prob[i])
    ugv_below = np.flatnonzero(ugv_prob < threshold)
    ugv_low = np.argsort(ugv_prob)[:10]

    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "flopwd").mkdir()
    (eval_dir / "ugv").mkdir()
    with (eval_dir / "flopwd" / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "true_class", "predicted_probability", "predicted_class",
                                                     "true_severity", "predicted_severity", "severity_absolute_error"])
        writer.writeheader()
        for i, record in enumerate(flo_test):
            writer.writerow({"filename": record.filename, "true_class": int(flo_true[i]),
                             "predicted_probability": float(flo_prob[i]), "predicted_class": int(flo_pred[i]),
                             "true_severity": float(sev_true[i]), "predicted_severity": float(sev_pred[i]),
                             "severity_absolute_error": float(absolute_errors[i])})
    with (eval_dir / "ugv" / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "canonical_source_id", "true_annotated_waste_present",
                                                     "predicted_probability", "predicted_positive"])
        writer.writeheader()
        for i, record in enumerate(ugv_test):
            writer.writerow({"filename": record.filename, "canonical_source_id": record.canonical_source_id,
                             "true_annotated_waste_present": 1, "predicted_probability": float(ugv_prob[i]),
                             "predicted_positive": int(ugv_prob[i] >= threshold)})

    model_rows = []
    for letter, run in (("B", Path("runs/multidomain_seed42")),
                        ("C", Path("runs/multidomain_class_balanced_seed42")),
                        ("E", args.run_dir)):
        metrics_path = (run / "evaluation" / "metrics.json" if letter == "B" else
                        run / "evaluation" / "flopwd" / "metrics.json")
        ugv_path = (run / "evaluation" / "metrics.json" if letter == "B" else
                    run / "evaluation" / "ugv" / "metrics.json")
        metadata_path = run / "metadata.json"
        if letter == "E":
            flo = {"classification": flo_class,
                   "severity_all_test_images": {"mae_percentage_points": flo_sev["mae_percentage_points"]}}
            ugv = {"annotated_waste_positive_recall": ugv_metrics["positive_recall"]}
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            positive_rate = meta["observed_overall_positive_rate"]
        else:
            previous_eval = json.loads(metrics_path.read_text(encoding="utf-8"))
            if letter == "B":
                flo = previous_eval["floPWD"]
                ugv = previous_eval["ugv_model_b"]
            else:
                flo = previous_eval
                ugv = json.loads(ugv_path.read_text(encoding="utf-8"))
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            positive_rate = (0.5 * (1038 / 1402) + 0.5 if letter == "B" else
                             meta.get("observed_overall_positive_rate", meta.get("overall_classification_positive_rate")))
        cl = flo.get("classification", flo)
        cm = cl.get("confusion_matrix", {})
        ugv_recall = ugv.get("annotated_waste_positive_recall", ugv.get("positive_recall"))
        severity_block = flo.get("severity", flo.get("severity_all_test_images", flo.get("severity_all", {})))
        severity_mae = severity_block.get("mae_percentage_points")
        model_rows.append({"model": letter, "effective_positive_rate": float(positive_rate),
                           "accuracy": float(cl["accuracy"]), "balanced_accuracy": float(cl["balanced_accuracy"]),
                           "sensitivity": float(cl["sensitivity"]), "specificity": float(cl["specificity"]),
                           "precision": float(cl["precision"]), "f1": float(cl["f1"]),
                           "fp": int(cm["fp"]), "fn": int(cm["fn"]), "severity_mae": float(severity_mae),
                           "ugv_positive_recall": float(ugv_recall)})
    _plot_comparisons(eval_dir, model_rows)

    metrics = {
        "experiment": "E multidomain_2to1",
        "threshold": threshold,
        "flopwd": {"classification": flo_class,
                   "positive_predictions": int(flo_pred.sum()),
                   "negative_predictions": int(len(flo_pred) - flo_pred.sum()),
                   "mean_probability": float(np.mean(flo_prob)),
                   "severity_all": {**flo_sev,
                                    "median_absolute_error_percentage_points": float(np.median(absolute_errors)),
                                    "p90_absolute_error_percentage_points": float(np.quantile(absolute_errors, 0.9)),
                                    "maximum_absolute_error_percentage_points": float(np.max(absolute_errors)),
                                    "mean_true_percentage_points": float(np.mean(sev_true)),
                                    "mean_predicted_percentage_points": severity_mean,
                                    "minimum_predicted_percentage_points": float(np.min(sev_pred)),
                                    "maximum_predicted_percentage_points": severity_max,
                                    "prediction_standard_deviation_percentage_points": severity_sd,
                                    "head_pattern": severity_pattern},
                   "severity_positive_images": flo_sev_positive,
                   "error_analysis": {
                       "false_positives": [{"filename": flo_test[i].filename, "probability": float(flo_prob[i])} for i in fp_indices],
                       "highest_confidence_false_positives": [{"filename": flo_test[i].filename, "probability": float(flo_prob[i])} for i in fp_rows[:10]],
                       "false_negatives": [{"filename": flo_test[i].filename, "probability": float(flo_prob[i])} for i in fn_indices],
                       "highest_confidence_false_negatives": [{"filename": flo_test[i].filename, "probability": float(flo_prob[i])} for i in fn_rows[:10]],
                       "largest_severity_errors": [{"filename": flo_test[i].filename, "true": float(sev_true[i]),
                                                    "predicted": float(sev_pred[i]), "absolute_error": float(absolute_errors[i])}
                                                   for i in np.argsort(absolute_errors)[::-1][:10]],
                       "severity_error_median": float(np.median(absolute_errors)),
                       "severity_error_p90": float(np.quantile(absolute_errors, 0.9)),
                       "severity_error_max": float(np.max(absolute_errors)),
                   }},
        "ugv_annotated_waste_positive": {
            "sample_count": len(ugv_test), "positive_recall": ugv_metrics["positive_recall"],
            "mean_probability": float(np.mean(ugv_prob)), "median_probability": float(np.median(ugv_prob)),
            "standard_deviation": float(np.std(ugv_prob)), "minimum_probability": float(np.min(ugv_prob)),
            "maximum_probability": float(np.max(ugv_prob)),
            "below_threshold": [{"filename": ugv_test[i].filename, "canonical_source_id": ugv_test[i].canonical_source_id,
                                 "probability": float(ugv_prob[i])} for i in ugv_below],
            "lowest_confidence_positives": [{"filename": ugv_test[i].filename, "canonical_source_id": ugv_test[i].canonical_source_id,
                                             "probability": float(ugv_prob[i])} for i in ugv_low],
            "severity_labels": 0,
        },
        "comparison_bce": model_rows,
        "domain_recall_gap": abs(flo_class["sensitivity"] - ugv_metrics["positive_recall"]),
        "integrity": {"flopwd_test_rows": len(flo_test), "flopwd_unique_rows": len({r.filename for r in flo_test}),
                      "ugv_test_rows": len(ugv_test), "ugv_unique_rows": len({r.filename for r in ugv_test}),
                      "confusion_total": sum(flo_class["confusion_matrix"].values()),
                      "probabilities_within_0_1": True, "severity_within_0_100": True,
                      "train_test_leakage": False, "validation_test_leakage": False,
                      "ugv_group_overlap": False, "ugv_severity_labels": 0,
                      "model_loaded_in_fresh_evaluation_process": True},
    }
    _write_json(eval_dir / "metrics.json", metrics)

    comparison = {row["model"]: row for row in model_rows}
    b, c, e = (comparison[key] for key in "BCE")
    metrics["prior_ablation"] = {
        "specificity_sequence": [b["specificity"], c["specificity"], e["specificity"]],
        "sensitivity_sequence": [b["sensitivity"], c["sensitivity"], e["sensitivity"]],
        "fp_sequence": [b["fp"], c["fp"], e["fp"]],
        "fn_sequence": [b["fn"], c["fn"], e["fn"]],
        "balanced_accuracy_sequence": [b["balanced_accuracy"], c["balanced_accuracy"], e["balanced_accuracy"]],
        "ugv_recall_sequence": [b["ugv_positive_recall"], c["ugv_positive_recall"], e["ugv_positive_recall"]],
        "interpretation": "Descriptive fixed-threshold comparison across the observed training class priors.",
    }
    metrics["c_vs_e_deltas"] = {key: e[key] - c[key] for key in
                                  ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "fp", "fn", "f1",
                                   "ugv_positive_recall", "severity_mae")}
    _write_json(eval_dir / "metrics.json", metrics)
    with (eval_dir / "bce_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(model_rows[0]))
        writer.writeheader(); writer.writerows(model_rows)

    # Provenance hashes are saved outside metrics so the two metric/prediction
    # hashes are stable and can be independently checked.
    run_metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    run_metadata["model_reload_verified"] = {"fresh_process": True, "evaluation_process": "scripts/evaluate_model_e.py"}
    run_metadata["test_evaluation_provenance"] = {"pass_count_per_domain": 1, "threshold": threshold,
                                                  "final_epoch_model": True}
    _write_json(args.run_dir / "metadata.json", run_metadata)
    hash_paths = {
        "model.keras": args.run_dir / "model.keras",
        "flopwd_metrics.json": eval_dir / "flopwd" / "metrics.json",
        "flopwd_predictions.csv": eval_dir / "flopwd" / "predictions.csv",
        "ugv_metrics.json": eval_dir / "ugv" / "metrics.json",
        "ugv_predictions.csv": eval_dir / "ugv" / "predictions.csv",
        "flopwd_split_manifest": flo_manifest_path,
        "ugv_grouped_split_manifest": ugv_manifest_path,
        "frozen_experiment_matrix": args.spec,
    }
    # Required separate domain metrics are also exported in the requested layout.
    _write_json(eval_dir / "flopwd" / "metrics.json", {"classification": flo_class,
        "mean_classification_probability": float(np.mean(flo_prob)),
        "positive_predictions": int(flo_pred.sum()), "negative_predictions": int(len(flo_pred)-flo_pred.sum()),
        "severity": metrics["flopwd"]["severity_all"], "severity_positive_images": flo_sev_positive,
        "threshold": threshold, "integrity": {"unique_test_rows": 300, "confusion_total": 300,
        "probabilities_in_bounds": True, "severity_in_bounds": True}})
    _write_json(eval_dir / "ugv" / "metrics.json", {"sample_count": len(ugv_test),
        "annotated_waste_positive_recall": ugv_metrics["positive_recall"],
        "mean_predicted_probability": float(np.mean(ugv_prob)), "median_predicted_probability": float(np.median(ugv_prob)),
        "standard_deviation": float(np.std(ugv_prob)), "minimum_probability": float(np.min(ugv_prob)),
        "maximum_probability": float(np.max(ugv_prob)), "below_threshold": metrics["ugv_annotated_waste_positive"]["below_threshold"],
        "severity_labels": 0, "target": "annotated-waste positive", "threshold": threshold})
    hash_paths["flopwd_metrics.json"] = eval_dir / "flopwd" / "metrics.json"
    hash_paths["ugv_metrics.json"] = eval_dir / "ugv" / "metrics.json"
    _write_json(eval_dir / "sha256.json", {key: _sha(path) for key, path in hash_paths.items()})

    # Catch accidental personal identifiers in generated text artifacts.
    text_files = [args.run_dir / "metadata.json", args.run_dir / "config.json", args.run_dir / "history.json",
                  eval_dir / "metrics.json", eval_dir / "flopwd" / "metrics.json", eval_dir / "ugv" / "metrics.json"]
    text_blob = "\n".join(path.read_text(encoding="utf-8") for path in text_files)
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text_blob):
        raise ValueError("generated run metadata unexpectedly contains an email address")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
