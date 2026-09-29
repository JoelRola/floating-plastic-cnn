"""Train the two-head ResNet50 model on the FloPWD train split."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.config import apply_overrides, load_config
from floating_plastic.data import load_flopwd
from floating_plastic.model import create_model
from floating_plastic.pipeline import make_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split


def _portable_reference(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="FloPWD root containing CSVs and data directories")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    backbone = parser.add_mutually_exclusive_group()
    backbone.add_argument("--backbone-trainable", dest="backbone_trainable", action="store_true")
    backbone.add_argument("--freeze-backbone", dest="backbone_trainable", action="store_false")
    parser.set_defaults(backbone_trainable=None)
    parser.add_argument("--max-train-samples", type=int, help="Debug-only subset; resulting run is marked non-benchmark")
    args = parser.parse_args(argv)
    if args.max_train_samples is not None and args.max_train_samples < 1:
        parser.error("--max-train-samples must be positive")

    config = load_config(args.config)
    config = apply_overrides(
        config,
        **{
            "training.epochs": args.epochs,
            "training.batch_size": args.batch_size,
            "training.seed": args.seed,
            "training.learning_rate": args.learning_rate,
            "model.backbone_trainable": args.backbone_trainable,
            "training.max_train_samples": args.max_train_samples,
        },
    )
    split_path = args.split_manifest or Path(config["data"]["split_manifest"])
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for training; install compatible runtime packages from requirements.txt") from exc

    seed = int(config["training"]["seed"])
    tf.keras.utils.set_random_seed(seed)
    deterministic = bool(config["reproducibility"].get("deterministic_tensorflow_ops", False))
    if deterministic:
        tf.config.experimental.enable_op_determinism()

    audit_records = load_flopwd(args.data_dir)
    manifest = load_split_manifest(split_path, expected_filenames=[r.filename for r in audit_records])
    split_records = {
        split: records_for_split(audit_records, manifest, split)
        for split in ("train", "validation", "test")
    }
    full_train_count = len(split_records["train"])
    limit = config["training"].get("max_train_samples")
    if limit is not None:
        if int(limit) < 1:
            parser.error("training.max_train_samples must be positive or null")
        split_records["train"] = split_records["train"][: int(limit)]
    if not split_records["train"]:
        raise SystemExit("training split is empty after applying the optional debug limit")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory already contains files; choose a new directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = tuple(config["data"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    train_ds = make_tf_dataset(split_records["train"], batch_size, image_size, training=True, seed=seed)
    validation_ds = make_tf_dataset(split_records["validation"], batch_size, image_size, training=False, seed=seed)
    model_config = config["model"]
    model = create_model(
        input_shape=(*image_size, 3),
        weights=model_config["weights"],
        backbone_trainable=model_config["backbone_trainable"],
        dense_units=model_config["dense_units"],
        dropout_rates=model_config["dropout_rates"],
    )
    loss_config = config["losses"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["training"]["learning_rate"])),
        loss={"classification": loss_config["classification"], "severity": loss_config["severity"]},
        loss_weights={
            "classification": float(loss_config["classification_weight"]),
            "severity": float(loss_config["severity_weight"]),
        },
        metrics={"classification": [tf.keras.metrics.BinaryAccuracy(name="accuracy")]},
    )
    fit = model.fit(train_ds, validation_data=validation_ds, epochs=int(config["training"]["epochs"]), verbose=2)
    model.save(output_dir / "model.keras")
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "history.json", {key: [float(v) for v in values] for key, values in fit.history.items()})

    counts = {
        split: {
            "sample_count": len(records),
            "class_counts": dict(Counter("positive" if r.binary_label else "negative" for r in records)),
        }
        for split, records in split_records.items()
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "seed": seed,
        "deterministic_tensorflow_ops_requested": deterministic,
        "split_manifest": _portable_reference(split_path),
        "split_manifest_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),
        "dataset": "FloPWD",
        "dataset_record_count": len(audit_records),
        "split_counts": counts,
        "full_train_split_count": full_train_count,
        "positive_zero_severity_count": sum(r.binary_label == 1 and r.severity_percent == 0 for r in audit_records),
        "preprocessing": "keras.applications.resnet.preprocess_input embedded in model; RGB resized pixels [0, 255]",
        "architecture": {
            "backbone": "ResNet50",
            "weights": model_config["weights"],
            "backbone_trainable": bool(model_config["backbone_trainable"]),
            "input_shape": [*image_size, 3],
            "severity_output": "sigmoid scaled to 0-100 percentage points",
        },
        "debug_train_sample_limit": limit,
        "benchmark_scope": "debug_subset_not_full_benchmark" if limit is not None else "full_manifest_training",
    }
    _write_json(output_dir / "metadata.json", metadata)
    print(f"Saved experiment artifacts to {output_dir}")
    if limit is not None:
        print("DEBUG RUN: model trained on a limited training subset; not a full benchmark run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
