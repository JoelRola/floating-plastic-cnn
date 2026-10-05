"""Evaluate the final Model F checkpoint once on frozen FloPWD/UGV test sets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.data import (heterogeneous_flopwd_record, heterogeneous_ugv_record,
                                   load_flopwd, load_ugv)
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.metrics import (classification_metrics, positive_only_metrics,
                                      severity_regression_metrics, threshold_predictions)
from floating_plastic.model import custom_objects
from floating_plastic.pipeline import make_heterogeneous_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def predict(model, records, batch_size, include_severity):
    truth, probability, severity_truth, severity_prediction = [], [], [], []
    for images, targets in make_heterogeneous_tf_dataset(records, batch_size, (224, 224)):
        output = model(images, training=False)
        cls = targets["classification"].numpy()
        if not np.all(cls[:, 1] == 1):
            raise ValueError("test classification labels unavailable")
        truth.extend(cls[:, 0].astype(int).tolist())
        probability.extend(np.asarray(output["classification"]).reshape(-1).tolist())
        if include_severity:
            sev = targets["severity"].numpy()
            if not np.all(sev[:, 1] == 1):
                raise ValueError("FloPWD test severity labels unavailable")
            severity_truth.extend(sev[:, 0].tolist())
            severity_prediction.extend(np.asarray(output["severity"]).reshape(-1).tolist())
    return tuple(np.asarray(x) for x in (truth, probability, severity_truth, severity_prediction))


def write_predictions(path, records, truth, prob, pred, severity_truth=None, severity_pred=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "true_class", "predicted_probability", "predicted_class"]
    if severity_truth is not None:
        fields += ["true_severity", "predicted_severity", "severity_absolute_error"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, (record, y, p, yhat) in enumerate(zip(records, truth, prob, pred)):
            row = {"filename": record.filename, "true_class": int(y),
                   "predicted_probability": float(p), "predicted_class": int(yhat)}
            if severity_truth is not None:
                row.update(true_severity=float(severity_truth[i]), predicted_severity=float(severity_pred[i]),
                           severity_absolute_error=float(abs(severity_pred[i] - severity_truth[i])))
            writer.writerow(row)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def severity_metrics(y, pred, class_y):
    errors = np.abs(pred - y)
    positive = class_y == 1
    all_stats = severity_regression_metrics(y.tolist(), pred.tolist())
    positive_stats = severity_regression_metrics(y[positive].tolist(), pred[positive].tolist())
    if pred.max() <= .1:
        pattern = "near-zero collapse"
    elif pred.std() <= .1:
        pattern = "near-constant collapse"
    elif all_stats["mae_percentage_points"] <= 2.169630868338546:
        pattern = "functional"
    elif all_stats["mae_percentage_points"] < 5.472192791637813:
        pattern = "partial recovery"
    else:
        pattern = "other (no regression recovery versus C)"
    return {
        **all_stats,
        "median_absolute_error_percentage_points": float(np.median(errors)),
        "p90_absolute_error_percentage_points": float(np.quantile(errors, .90)),
        "maximum_absolute_error_percentage_points": float(errors.max()),
        "positive_images": {"sample_count": int(positive.sum()), **positive_stats},
        "mean_true_severity_percentage_points": float(y.mean()),
        "mean_predicted_severity_percentage_points": float(pred.mean()),
        "median_predicted_severity_percentage_points": float(np.median(pred)),
        "predicted_severity_standard_deviation_percentage_points": float(pred.std()),
        "minimum_predicted_severity_percentage_points": float(pred.min()),
        "maximum_predicted_severity_percentage_points": float(pred.max()),
        "head_pattern": pattern,
        "largest_absolute_errors": [
            {"filename_index": int(i), "true_severity": float(y[i]), "predicted_severity": float(pred[i]),
             "absolute_error": float(errors[i])} for i in np.argsort(errors)[::-1][:10]],
    }


def make_figures(out, flo, severity, comparison, ugv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls = flo["classification"]
    cm = cls["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    ax.imshow(matrix, cmap="Blues")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(matrix[r][c]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Absent", "Present"],
           yticklabels=["Absent", "Present"], xlabel="Predicted", ylabel="True",
           title="Model F FloPWD test confusion matrix")
    fig.tight_layout(); fig.savefig(out / "f_confusion_matrix.png", dpi=160); plt.close(fig)

    rows = read_csv(out / "flopwd" / "predictions.csv")
    y = np.asarray([float(r["true_severity"]) for r in rows])
    yp = np.asarray([float(r["predicted_severity"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, yp, s=18, alpha=.55); ax.plot([0, 100], [0, 100], "k--", linewidth=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="True coverage (pp)", ylabel="Predicted coverage (pp)",
           title="Model F FloPWD test severity")
    fig.tight_layout(); fig.savefig(out / "f_severity_scatter.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(y, yp - y, s=18, alpha=.55)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 100), xlabel="True coverage (pp)", ylabel="Prediction residual (pp)",
           title="Model F FloPWD test severity residuals")
    fig.tight_layout(); fig.savefig(out / "f_severity_residuals.png", dpi=160); plt.close(fig)

    models = comparison["models"]
    names = ["A", "C", "D", "F"]
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(4); width = .2
    for offset, metric, label in ((-1.5*width, "accuracy", "Accuracy"), (-.5*width, "sensitivity", "Sensitivity"),
                                  (.5*width, "specificity", "Specificity"), (1.5*width, "balanced_accuracy", "Balanced accuracy")):
        ax.bar(x + offset, [models[n]["classification"][metric] for n in names], width, label=label)
    ax.set(xticks=x, xticklabels=[f"Model {n}" for n in names], ylim=(0, 1), ylabel="Score",
           title="FloPWD classification: A/C/D/F")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(); fig.savefig(out / "acdf_classification.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(4); width = .34
    ax.bar(x-width/2, [models[n]["severity_mae_all"] for n in names], width, label="All images")
    ax.bar(x+width/2, [models[n]["severity_mae_positive"] for n in names], width, label="Positive images")
    ax.set(xticks=x, xticklabels=[f"Model {n}" for n in names], ylabel="MAE (pp)", title="Severity MAE: A/C/D/F")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "acdf_severity_mae.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["D", "F"], [models[n]["severity_mae_all"] for n in ("D", "F")], color=["#2f8061", "#7563a6"])
    ax.set(ylabel="All-image severity MAE (pp)", title="Severity MAE: D versus F")
    fig.tight_layout(); fig.savefig(out / "df_severity_mae.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["D", "F"], [ugv[n]["annotated_waste_positive_recall"] for n in ("D", "F")], color=["#2f8061", "#7563a6"])
    ax.set(ylim=(0, 1.02), ylabel="UGV annotated-waste positive recall", title="UGV recall: D versus F")
    fig.tight_layout(); fig.savefig(out / "df_ugv_recall.png", dpi=160); plt.close(fig)

    d_rows = read_csv(Path("runs/task_decoupled_multidomain_seed42/evaluation/flopwd/predictions.csv"))
    d_pred = np.asarray([float(r["predicted_severity"]) for r in d_rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d_pred, bins=30, range=(0, 100), alpha=.55, label="D", color="#2f8061")
    ax.hist(yp, bins=30, range=(0, 100), alpha=.55, label="F", color="#7563a6")
    ax.set(xlim=(0, 100), xlabel="Predicted severity (pp)", ylabel="Test images",
           title="D/F test severity prediction distributions")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "df_severity_prediction_distribution.png", dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/task_decoupled_original_severity_prior_seed42"))
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    out = args.run_dir / "evaluation"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing a second Model F test evaluation: {out}")
    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    if (metadata["benchmark_status"] != "full" or metadata["run_status"] != "complete" or
            metadata["total_optimizer_updates"] != 660 or metadata["optimizer_iterations"] != 660):
        raise ValueError("Model F full training did not complete the frozen budget")
    if sha(args.spec) != metadata["frozen_matrix_sha256"]:
        raise ValueError("frozen matrix hash changed")

    import tensorflow as tf
    import yaml
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    fspec = spec["matrix"]["F"]
    if fspec["status"] != "frozen_not_run" or fspec["architecture_variant"] != "task_decoupled":
        raise ValueError("frozen Model F spec changed")
    fm_path = Path(spec["shared_protocol"]["split_manifests"]["flopwd"])
    um_path = Path(spec["shared_protocol"]["split_manifests"]["ugv"])
    if sha(fm_path) != metadata["split_sha256"]["flopwd"] or sha(um_path) != metadata["split_sha256"]["ugv"]:
        raise ValueError("split hashes differ from training provenance")
    flo_all, ugv_all = load_flopwd(args.data_dir), load_ugv(args.ugv_dir)
    fm = load_split_manifest(fm_path, [r.filename for r in flo_all])
    um = load_ugv_grouped_split_manifest(um_path, ugv_all)
    flo_parts = {s: records_for_split(flo_all, fm, s) for s in ("train", "validation", "test")}
    ugv_parts = {s: records_for_ugv_split(ugv_all, um, s) for s in ("train", "validation", "test")}
    flo_test = [heterogeneous_flopwd_record(r) for r in flo_parts["test"]]
    ugv_test = [heterogeneous_ugv_record(r) for r in ugv_parts["test"]]
    if len(flo_test) != 300 or len({r.filename for r in flo_test}) != 300:
        raise ValueError("FloPWD test must contain 300 unique records")
    if len(ugv_test) != 536 or len({r.filename for r in ugv_test}) != 536:
        raise ValueError("UGV grouped test must contain 536 unique records")
    if any(r.severity_target_available for r in ugv_test):
        raise ValueError("UGV severity labels must be unavailable")
    if set(r.filename for r in flo_test) & (set(r.filename for r in flo_parts["train"]) | set(r.filename for r in flo_parts["validation"])):
        raise ValueError("FloPWD test leakage")
    if set(r.filename for r in ugv_test) & (set(r.filename for r in ugv_parts["train"]) | set(r.filename for r in ugv_parts["validation"])):
        raise ValueError("UGV test leakage")
    groups = {s: set(um["group_splits"][s]) for s in ("train", "validation", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("UGV canonical source groups overlap")

    # This script runs in a fresh Python process. Verify final F serialization and old model compatibility
    # using training-only examples before the sole test inference pass.
    model_path = args.run_dir / "model.keras"
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects(), compile=True)
    backbone = model.get_layer("resnet50")
    if model.name != "floating_plastic_resnet50_multitask_task_decoupled":
        raise ValueError("final model architecture failed reload")
    if not isinstance(model.loss, dict) or {
            name: getattr(loss, "task", None) for name, loss in model.loss.items()} != {
                "classification": "binary_crossentropy", "severity": "mae"}:
        raise ValueError("masked losses failed fresh-process reload")
    if int(model.optimizer.iterations.numpy()) != 660:
        raise ValueError("reloaded optimizer iteration is not 660")
    if len(backbone.trainable_variables) or {id(v) for v in backbone.weights} & {id(v) for v in model.trainable_variables}:
        raise ValueError("backbone weights are trainable after reload")
    # Keras 2.15 can reload the nested ResNet Model wrapper with trainable=True
    # while all child ResNet variables remain excluded from optimization.
    # Reassert the frozen wrapper before inference and verify the contract.
    backbone.trainable = False
    if backbone.trainable or backbone.trainable_variables:
        raise ValueError("reloaded ResNet50 could not be explicitly kept frozen")
    train_probe = [heterogeneous_flopwd_record(flo_parts["train"][0]),
                   heterogeneous_ugv_record(ugv_parts["train"][0])]
    reload_predictions = []
    for row in train_probe:
        image = next(iter(make_heterogeneous_tf_dataset([row], 1, (224, 224))))[0]
        output = model(image, training=False)
        reload_predictions.append({"domain": row.source_domain,
                                   "classification": float(np.asarray(output["classification"]).reshape(-1)[0]),
                                   "severity": float(np.asarray(output["severity"]).reshape(-1)[0])})
    if any(not np.isfinite(x["classification"]) or not 0 <= x["classification"] <= 1 or
           not np.isfinite(x["severity"]) or not 0 <= x["severity"] <= 100 for x in reload_predictions):
        raise ValueError("reloaded inference returned invalid values")
    del model
    tf.keras.backend.clear_session()
    compatible = {}
    for key, path in {
        "A": Path("runs/flopwd_original_seed42/model.keras"),
        "B": Path("runs/multidomain_seed42/model.keras"),
        "C": Path("runs/multidomain_class_balanced_seed42/model.keras"),
        "D": Path("runs/task_decoupled_multidomain_seed42/model.keras"),
        "E": Path("runs/multidomain_2to1_seed42/model.keras"),
    }.items():
        old = tf.keras.models.load_model(path, custom_objects=custom_objects(), compile=False)
        compatible[key] = {"loaded": True, "name": old.name}
        del old
        tf.keras.backend.clear_session()
    write_json(args.run_dir / "reload_verification.json", {
        "fresh_process": True, "model": "Model F final epoch", "optimizer_iterations": 660,
        "architecture_variant": "task_decoupled", "backbone_frozen": True,
        "classification_loss": "masked BCE", "severity_loss": "masked MAE",
        "training_only_inferences": reload_predictions, "legacy_shared_checkpoint_loads": compatible})

    # The only test inference pass for Model F.
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects(), compile=False)
    flo_y, flo_p, sev_y, sev_p = predict(model, flo_test, args.batch_size, True)
    ugv_y, ugv_p, _, _ = predict(model, ugv_test, args.batch_size, False)
    del model
    if len(flo_y) != 300 or len(np.unique([r.filename for r in flo_test])) != 300:
        raise ValueError("FloPWD inference row integrity failure")
    if len(ugv_y) != 536 or not np.all(ugv_y == 1):
        raise ValueError("UGV grouped test must contain 536 positive annotated-waste rows")
    if (not np.isfinite(flo_p).all() or not np.isfinite(ugv_p).all() or
            np.any((flo_p < 0) | (flo_p > 1)) or np.any((ugv_p < 0) | (ugv_p > 1))):
        raise ValueError("probability bounds failure")
    if not np.isfinite(sev_p).all() or np.any((sev_p < 0) | (sev_p > 100)):
        raise ValueError("severity bounds failure")
    flo_hat, ugv_hat = np.asarray(threshold_predictions(flo_p.tolist(), .5)), np.asarray(threshold_predictions(ugv_p.tolist(), .5))
    cls = classification_metrics(flo_y.tolist(), flo_hat.tolist())
    sev = severity_metrics(sev_y, sev_p, flo_y)
    ugv_summary = positive_only_metrics(ugv_p.tolist(), .5)
    ugv = {"sample_count": 536, "target": "annotated-waste positive",
           "annotated_waste_positive_recall": ugv_summary["positive_recall"],
           "mean_predicted_probability": ugv_summary["predicted_probability_distribution"]["mean"],
           "median_predicted_probability": ugv_summary["predicted_probability_distribution"]["median"],
           "standard_deviation": ugv_summary["predicted_probability_distribution"]["standard_deviation"],
           "minimum_probability": ugv_summary["predicted_probability_distribution"]["min"],
           "maximum_probability": ugv_summary["predicted_probability_distribution"]["max"],
           "below_threshold_count": int(np.sum(ugv_hat == 0)), "severity_labels": 0}
    ugv_canonical_by_filename = {row.filename: row.canonical_source_id for row in ugv_parts["test"]}
    out.mkdir(parents=True, exist_ok=True)
    write_predictions(out / "flopwd" / "predictions.csv", flo_test, flo_y, flo_p, flo_hat, sev_y, sev_p)
    write_predictions(out / "ugv" / "predictions.csv", ugv_test, ugv_y, ugv_p, ugv_hat)
    flo_metrics = {"sample_count": 300, "threshold": .5, "classification": cls,
                   "mean_classification_probability": float(flo_p.mean()),
                   "positive_predictions": int(flo_hat.sum()), "negative_predictions": int(300-flo_hat.sum()),
                   "severity": sev}
    write_json(out / "flopwd" / "metrics.json", flo_metrics)
    write_json(out / "ugv" / "metrics.json", ugv)

    prior = json.loads(Path("runs/task_decoupled_multidomain_seed42/evaluation/comparison.json").read_text())
    models = {k: prior["models"][k] for k in ("A", "C", "D")}
    f_recall = cls["sensitivity"]
    models["F"] = {"classification": cls, "severity_mae_all": sev["mae_percentage_points"],
                   "severity_mae_positive": sev["positive_images"]["mae_percentage_points"],
                   "ugv_positive_recall": ugv["annotated_waste_positive_recall"],
                   "ugv_probability_mean": ugv["mean_predicted_probability"],
                   "domain_recall_gap": abs(f_recall - ugv["annotated_waste_positive_recall"])}
    d = models["D"]
    delta = {metric: cls[metric] - d["classification"][metric]
             for metric in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1")}
    delta.update({"fp": cls["confusion_matrix"]["fp"] - d["classification"]["confusion_matrix"]["fp"],
                  "fn": cls["confusion_matrix"]["fn"] - d["classification"]["confusion_matrix"]["fn"],
                  "severity_mae_all": sev["mae_percentage_points"] - d["severity_mae_all"],
                  "severity_mae_positive": sev["positive_images"]["mae_percentage_points"] - d["severity_mae_positive"],
                  "ugv_recall": ugv["annotated_waste_positive_recall"] - d["ugv_positive_recall"]})
    write_json(out / "comparison.json", {"models": models, "f_minus_d": delta, "threshold": .5,
        "domain_recall_gap_note": "FloPWD plastic-positive and UGV annotated-waste-positive targets are not semantically identical; this is diagnostic only."})
    errors = np.abs(sev_y - sev_p)
    fp_idx = np.where((flo_y == 0) & (flo_hat == 1))[0]
    fn_idx = np.where((flo_y == 1) & (flo_hat == 0))[0]
    error_analysis = {
        "false_positives": [{"filename": flo_test[i].filename, "probability": float(flo_p[i])} for i in fp_idx],
        "false_negatives": [{"filename": flo_test[i].filename, "probability": float(flo_p[i])} for i in fn_idx],
        "highest_confidence_false_positives": [{"filename": flo_test[i].filename, "probability": float(flo_p[i])}
            for i in sorted(fp_idx, key=lambda j: flo_p[j], reverse=True)[:10]],
        "highest_confidence_false_negatives": [{"filename": flo_test[i].filename, "probability": float(flo_p[i])}
            for i in sorted(fn_idx, key=lambda j: flo_p[j])[:10]],
        "largest_severity_errors": [{"filename": flo_test[i].filename, "true": float(sev_y[i]),
            "predicted": float(sev_p[i]), "absolute_error": float(errors[i])} for i in np.argsort(errors)[::-1][:10]],
        "severity_error_median": float(np.median(errors)), "severity_error_p90": float(np.quantile(errors,.9)),
        "severity_error_max": float(errors.max()),
        "ugv_below_threshold": [{"filename": ugv_test[i].filename,
                                 "canonical_source_id": ugv_canonical_by_filename[ugv_test[i].filename],
                                 "probability": float(ugv_p[i])} for i in np.where(ugv_hat == 0)[0]],
        "lowest_confidence_ugv_positives": [{"filename": ugv_test[i].filename,
            "canonical_source_id": ugv_canonical_by_filename[ugv_test[i].filename], "probability": float(ugv_p[i])}
            for i in np.argsort(ugv_p)[:10]],
    }
    write_json(out / "error_analysis.json", error_analysis)
    integrity = {"flopwd_unique_test_rows": 300, "ugv_unique_grouped_test_rows": 536,
        "no_train_test_overlap": True, "no_validation_test_overlap": True, "ugv_canonical_source_groups_disjoint": True,
        "probabilities_in_0_1": True, "severity_predictions_in_0_100": True, "ugv_severity_values": 0,
        "confusion_total": sum(cls["confusion_matrix"].values()), "training_rows_in_test_predictions": 0,
        "validation_rows_in_test_predictions": 0, "runs_ignored": True}
    if integrity["confusion_total"] != 300:
        raise ValueError("FloPWD confusion total does not equal 300")
    write_json(out / "integrity.json", integrity)
    make_figures(out, flo_metrics, sev, {"models": models}, {"D": {"annotated_waste_positive_recall": d["ugv_positive_recall"]},
                                                             "F": ugv})
    hashes = {"model_keras": sha(model_path), "flopwd_metrics_json": sha(out/"flopwd"/"metrics.json"),
        "flopwd_predictions_csv": sha(out/"flopwd"/"predictions.csv"),
        "ugv_metrics_json": sha(out/"ugv"/"metrics.json"), "ugv_predictions_csv": sha(out/"ugv"/"predictions.csv"),
        "flopwd_split_manifest": sha(fm_path), "ugv_grouped_split_manifest": sha(um_path),
        "frozen_matrix_specification": sha(args.spec)}
    write_json(out / "sha256.json", hashes)
    provenance = {"training_git_sha": metadata["training_git_sha"],
        "evaluation_git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                              text=True, check=True).stdout.strip(),
        "frozen_matrix_sha256": sha(args.spec), "test_evaluations": {"FloPWD": 1, "UGV": 1}}
    write_json(out / "provenance.json", provenance)
    print(json.dumps({"flo_classification": cls, "severity": sev, "ugv": ugv,
        "f_minus_d": delta, "hashes": hashes, "reload": reload_predictions,
        "figures": sorted(p.name for p in out.glob("*.png"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
