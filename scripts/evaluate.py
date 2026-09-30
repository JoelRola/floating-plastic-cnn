"""Evaluate a saved model on the FloPWD TEST partition only."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.config import load_config
from floating_plastic.data import load_flopwd
from floating_plastic.metrics import classification_metrics, severity_regression_metrics, threshold_predictions
from floating_plastic.model import custom_objects
from floating_plastic.pipeline import make_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split


def _save_figures(output_dir, counts, y_true_severity, y_pred_severity):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required to save evaluation figures") from exc

    tn, fp, fn, tp = (counts[key] for key in ("tn", "fp", "fn", "tp"))
    matrix = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center", fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Plastic absent", "Plastic present"],
           yticklabels=["Plastic absent", "Plastic present"], xlabel="Predicted class", ylabel="True class",
           title="FloPWD test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true_severity, y_pred_severity, alpha=0.55, s=18)
    ax.plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="True severity (percentage points)",
           ylabel="Predicted severity (percentage points)", title="FloPWD test severity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "severity_scatter.png", dpi=160)
    plt.close(fig)

    residuals = [pred - true for true, pred in zip(y_true_severity, y_pred_severity)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true_severity, residuals, alpha=0.55, s=18)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 100), xlabel="True severity (percentage points)",
           ylabel="Prediction residual (percentage points)", title="FloPWD test severity residuals")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "severity_residuals.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=Path("experiments/splits/flopwd_seed42.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    threshold = args.threshold if args.threshold is not None else float(config["evaluation"]["classification_threshold"])
    if batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for evaluation; install compatible runtime packages from requirements.txt") from exc

    records = load_flopwd(args.data_dir)
    manifest = load_split_manifest(args.split_manifest, expected_filenames=[r.filename for r in records])
    test_records = records_for_split(records, manifest, "test")
    dataset = make_tf_dataset(
        test_records, batch_size, tuple(config["data"]["image_size"]), training=False,
        seed=int(config["training"]["seed"]),
    )
    model = tf.keras.models.load_model(
        args.model, compile=False, custom_objects=custom_objects()
    )
    outputs = model.predict(dataset, verbose=0)
    if not isinstance(outputs, dict) or not {"classification", "severity"}.issubset(outputs):
        raise ValueError("model must return named 'classification' and 'severity' outputs")
    probabilities = [float(value) for value in outputs["classification"].reshape(-1)]
    predicted_classes = threshold_predictions(probabilities, threshold)
    predicted_severity = [float(value) for value in outputs["severity"].reshape(-1)]
    true_classes = [record.binary_label for record in test_records]
    true_severity = [record.severity_percent for record in test_records]
    class_metrics = classification_metrics(true_classes, predicted_classes)
    all_severity_metrics = severity_regression_metrics(true_severity, predicted_severity)
    positive_indices = [index for index, label in enumerate(true_classes) if label == 1]
    positive_severity_metrics = (
        severity_regression_metrics(
            [true_severity[index] for index in positive_indices],
            [predicted_severity[index] for index in positive_indices],
        )
        if positive_indices
        else None
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["filename", "true_class", "predicted_probability", "predicted_class",
                        "true_severity", "predicted_severity", "severity_absolute_error"],
        )
        writer.writeheader()
        for record, probability, predicted_class, severity in zip(
            test_records, probabilities, predicted_classes, predicted_severity
        ):
            writer.writerow({
                "filename": record.filename,
                "true_class": record.binary_label,
                "predicted_probability": probability,
                "predicted_class": predicted_class,
                "true_severity": record.severity_percent,
                "predicted_severity": severity,
                "severity_absolute_error": abs(record.severity_percent - severity),
            })

    training_metadata_path = args.model.parent / "metadata.json"
    training_metadata = json.loads(training_metadata_path.read_text(encoding="utf-8")) if training_metadata_path.is_file() else {}
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "FloPWD",
        "evaluation_split": "test_only",
        "sample_count": len(test_records),
        "classification_threshold": threshold,
        "classification": class_metrics,
        "severity_all_test_images": all_severity_metrics,
        "severity_positive_test_images": positive_severity_metrics,
        "split_manifest": args.split_manifest.name,
        "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        "benchmark_scope": training_metadata.get("benchmark_scope", "scope_unknown_model_metadata_unavailable"),
        "debug_train_sample_limit": training_metadata.get("debug_train_sample_limit"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _save_figures(output_dir, class_metrics["confusion_matrix"], true_severity, predicted_severity)
    print(f"Evaluated {len(test_records)} FloPWD test records; artifacts saved to {output_dir}")
    if result["debug_train_sample_limit"] is not None:
        print("WARNING: model metadata marks this as a debug-subset run, not a full benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
