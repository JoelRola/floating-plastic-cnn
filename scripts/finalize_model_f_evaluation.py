"""Finalize Model F evaluation artifacts from already-saved predictions; never runs inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floating_plastic.data import load_flopwd, load_ugv
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.splits import load_split_manifest, records_for_split
from floating_plastic.metrics import threshold_predictions


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/task_decoupled_original_severity_prior_seed42"))
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--evaluation-git-sha", required=True,
                        help="Git SHA of the fresh-process inference run; no inference is performed here")
    args = parser.parse_args(argv)
    out = args.run_dir / "evaluation"
    flo_path, ugv_path = out / "flopwd" / "predictions.csv", out / "ugv" / "predictions.csv"
    flo_rows, ugv_rows = read_csv(flo_path), read_csv(ugv_path)
    if (len(flo_rows) != 300 or len({r["filename"] for r in flo_rows}) != 300 or
            len(ugv_rows) != 536 or len({r["filename"] for r in ugv_rows}) != 536):
        raise ValueError("saved prediction rows or unique filenames do not match the frozen test sizes")
    flo_metrics = json.loads((out / "flopwd" / "metrics.json").read_text(encoding="utf-8"))
    ugv_metrics = json.loads((out / "ugv" / "metrics.json").read_text(encoding="utf-8"))
    if int(flo_metrics["sample_count"]) != 300 or int(ugv_metrics["sample_count"]) != 536:
        raise ValueError("saved metrics counts differ from prediction rows")

    import yaml
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    shared = spec["shared_protocol"]
    fm_path, um_path = Path(shared["split_manifests"]["flopwd"]), Path(shared["split_manifests"]["ugv"])
    flo_all = load_flopwd(args.data_dir)
    ugv_all = load_ugv(args.ugv_dir)
    fm = load_split_manifest(fm_path, [r.filename for r in flo_all])
    um = load_ugv_grouped_split_manifest(um_path, ugv_all)
    flo_parts = {part: records_for_split(flo_all, fm, part) for part in ("train", "validation", "test")}
    ugv_parts = {part: records_for_ugv_split(ugv_all, um, part) for part in ("train", "validation", "test")}
    flo_test_names = {r.filename for r in flo_parts["test"]}
    ugv_test_names = {r.filename for r in ugv_parts["test"]}
    if {r["filename"] for r in flo_rows} != flo_test_names or {r["filename"] for r in ugv_rows} != ugv_test_names:
        raise ValueError("saved predictions do not correspond exactly to frozen test manifests")
    if flo_test_names & ({r.filename for r in flo_parts["train"]} | {r.filename for r in flo_parts["validation"]}):
        raise ValueError("FloPWD training/validation leakage")
    if ugv_test_names & ({r.filename for r in ugv_parts["train"]} | {r.filename for r in ugv_parts["validation"]}):
        raise ValueError("UGV training/validation leakage")
    group_splits = {part: set(um["group_splits"][part]) for part in ("train", "validation", "test")}
    if any(group_splits[a] & group_splits[b] for a, b in (("train", "validation"), ("train", "test"),
                                                           ("validation", "test"))):
        raise ValueError("UGV canonical source groups overlap partitions")
    canonical = {r.filename: r.canonical_source_id for r in ugv_parts["test"]}

    fp, fn = [], []
    true_y, pred_y, prob = [], [], []
    sev_true, sev_pred = [], []
    for row in flo_rows:
        truth, predicted = int(row["true_class"]), int(row["predicted_class"])
        probability = float(row["predicted_probability"])
        ysev, psev = float(row["true_severity"]), float(row["predicted_severity"])
        if not 0 <= probability <= 1 or not 0 <= psev <= 100:
            raise ValueError("saved FloPWD output outside bounds")
        true_y.append(truth); pred_y.append(predicted); prob.append(probability)
        sev_true.append(ysev); sev_pred.append(psev)
        if truth == 0 and predicted == 1:
            fp.append({"filename": row["filename"], "probability": probability})
        if truth == 1 and predicted == 0:
            fn.append({"filename": row["filename"], "probability": probability})
    for row in ugv_rows:
        if int(row["true_class"]) != 1 or "predicted_severity" in row:
            raise ValueError("UGV labels must be positive-only, with no severity value")
        if not 0 <= float(row["predicted_probability"]) <= 1:
            raise ValueError("saved UGV probability outside bounds")
    ugv_prob = np.asarray([float(r["predicted_probability"]) for r in ugv_rows])
    ugv_hat = np.asarray([int(r["predicted_class"]) for r in ugv_rows])
    if not np.array_equal(ugv_hat, threshold_predictions(ugv_prob.tolist(), .5)):
        raise ValueError("saved UGV class predictions do not use threshold 0.5")

    error = np.abs(np.asarray(sev_pred) - np.asarray(sev_true))
    fp_sorted = sorted(fp, key=lambda r: r["probability"], reverse=True)
    fn_sorted = sorted(fn, key=lambda r: r["probability"])
    error_analysis = {
        "false_positives": fp, "false_negatives": fn,
        "highest_confidence_false_positives": fp_sorted[:10],
        "highest_confidence_false_negatives": fn_sorted[:10],
        "largest_severity_errors": [{"filename": flo_rows[i]["filename"], "true": float(sev_true[i]),
            "predicted": float(sev_pred[i]), "absolute_error": float(error[i])} for i in np.argsort(error)[::-1][:10]],
        "severity_error_median": float(np.median(error)), "severity_error_p90": float(np.quantile(error, .9)),
        "severity_error_max": float(error.max()),
        "ugv_below_threshold": [{"filename": row["filename"], "canonical_source_id": canonical[row["filename"]],
            "probability": float(row["predicted_probability"])} for row in ugv_rows if int(row["predicted_class"]) == 0],
        "lowest_confidence_ugv_positives": [{"filename": ugv_rows[i]["filename"],
            "canonical_source_id": canonical[ugv_rows[i]["filename"]], "probability": float(ugv_prob[i])}
            for i in np.argsort(ugv_prob)[:10]],
    }
    write_json(out / "error_analysis.json", error_analysis)

    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    figures_out = out
    # Reuse the committed figure builder; it reads saved CSV/JSON only.
    from evaluate_model_f import make_figures
    make_figures(figures_out, flo_metrics, flo_metrics["severity"], comparison,
                 {"D": {"annotated_waste_positive_recall": comparison["models"]["D"]["ugv_positive_recall"]},
                  "F": ugv_metrics})
    write_json(out / "integrity.json", {
        "flopwd_unique_test_rows": 300, "ugv_unique_grouped_test_rows": 536,
        "no_train_test_overlap": True, "no_validation_test_overlap": True,
        "ugv_canonical_source_groups_disjoint": True, "probabilities_in_0_1": True,
        "severity_predictions_in_0_100": True, "ugv_severity_values": 0,
        "confusion_total": sum(flo_metrics["classification"]["confusion_matrix"].values()),
        "training_rows_in_test_predictions": 0, "validation_rows_in_test_predictions": 0,
        "runs_ignored": True, "prediction_source": "saved single inference pass; no reinference",
    })
    model_path = args.run_dir / "model.keras"
    hashes = {"model_keras": sha(model_path), "flopwd_metrics_json": sha(out/"flopwd"/"metrics.json"),
        "flopwd_predictions_csv": sha(flo_path), "ugv_metrics_json": sha(out/"ugv"/"metrics.json"),
        "ugv_predictions_csv": sha(ugv_path), "flopwd_split_manifest": sha(fm_path),
        "ugv_grouped_split_manifest": sha(um_path), "frozen_matrix_specification": sha(args.spec)}
    write_json(out / "sha256.json", hashes)
    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    reload_report = json.loads((args.run_dir / "reload_verification.json").read_text(encoding="utf-8"))
    metadata["model_reload_verified"] = bool(reload_report["fresh_process"] and
                                               reload_report["optimizer_iterations"] == 660)
    metadata["reload_verification_file"] = "reload_verification.json"
    metadata["backward_compatibility"] = reload_report["legacy_shared_checkpoint_loads"]
    write_json(args.run_dir / "metadata.json", metadata)
    write_json(out / "provenance.json", {
        "training_git_sha": metadata["training_git_sha"], "evaluation_git_sha": args.evaluation_git_sha,
        "artifact_finalization_git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                         text=True, check=True).stdout.strip(),
        "frozen_matrix_sha256": sha(args.spec), "test_evaluations": {"FloPWD": 1, "UGV": 1},
        "inference_repeated": False,
    })
    print(json.dumps({"classification": flo_metrics["classification"], "severity": flo_metrics["severity"],
        "ugv": ugv_metrics, "hashes": hashes, "error_counts": {"fp": len(fp), "fn": len(fn)},
        "figures": sorted(p.name for p in out.glob("*.png"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
