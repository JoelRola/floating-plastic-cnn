"""Train frozen-spec multi-domain Model B with matched optimizer-step budget."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.config import load_config
from floating_plastic.data import (
    heterogeneous_flopwd_record, heterogeneous_ugv_record, load_flopwd, load_ugv,
)
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.losses import make_tensorflow_masked_loss
from floating_plastic.metrics import positive_only_metrics, threshold_predictions
from floating_plastic.model import create_model
from floating_plastic.pipeline import make_heterogeneous_tf_dataset, make_multidomain_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_config(spec, config):
    shared = spec["shared_protocol"]
    expected = {
        "training.seed": (config["training"]["seed"], shared["seed"]),
        "training.batch_size": (config["training"]["batch_size"], shared["batch_size"]),
        "training.learning_rate": (config["training"]["learning_rate"], shared["learning_rate"]),
        "model.backbone_trainable": (config["model"]["backbone_trainable"], shared["architecture"]["backbone_trainable"]),
        "model.weights": (config["model"]["weights"], shared["architecture"]["weights"]),
        "model.dense_units": (config["model"]["dense_units"], shared["architecture"]["dense_units"]),
        "model.dropout_rates": (config["model"]["dropout_rates"], shared["architecture"]["dropout_rates"]),
        "evaluation.classification_threshold": (
            config["evaluation"]["classification_threshold"], shared["classification_threshold"]),
        "losses.classification_weight": (config["losses"]["classification_weight"], shared["loss_weights"]["classification"]),
        "losses.severity_weight": (config["losses"]["severity_weight"], shared["loss_weights"]["severity"]),
        "losses.classification": (config["losses"]["classification"], "binary_crossentropy"),
        "losses.severity": (config["losses"]["severity"], "mae"),
    }
    mismatches = {key: values for key, values in expected.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"default runtime config differs from frozen experiment matrix: {mismatches}")


def _evaluate_domain(model, dataset, domain, classification_loss, severity_loss, threshold):
    class_true, probabilities, severity_true, severity_pred = [], [], [], []
    weighted_class_loss = weighted_severity_loss = 0.0
    class_count = severity_count = 0
    for images, targets in dataset:
        outputs = model(images, training=False)
        c = targets["classification"]
        s = targets["severity"]
        c_mask = c[:, 1] > 0
        s_mask = s[:, 1] > 0
        n_class = int(c_mask.numpy().sum())
        n_severity = int(s_mask.numpy().sum())
        lc = float(classification_loss(c, outputs["classification"]).numpy())
        ls = float(severity_loss(s, outputs["severity"]).numpy())
        if not np.isfinite(lc) or not np.isfinite(ls):
            raise FloatingPointError(f"non-finite validation loss for {domain}")
        weighted_class_loss += lc * n_class
        weighted_severity_loss += ls * n_severity
        class_count += n_class
        severity_count += n_severity
        if n_class:
            class_true.extend(c[:, 0].numpy()[c_mask.numpy()].astype(int).tolist())
            probabilities.extend(outputs["classification"].numpy().reshape(-1)[c_mask.numpy()].tolist())
        if n_severity:
            severity_true.extend(s[:, 0].numpy()[s_mask.numpy()].tolist())
            severity_pred.extend(outputs["severity"].numpy().reshape(-1)[s_mask.numpy()].tolist())
    report = {"sample_count": len(class_true), "classification_loss": weighted_class_loss / class_count}
    if domain == "flopwd":
        report["classification_accuracy"] = float(
            np.mean(np.asarray(threshold_predictions(probabilities, threshold)) == class_true))
        report["severity_available_count"] = severity_count
        report["severity_mae_percentage_points"] = weighted_severity_loss / severity_count
    else:
        if not class_true or any(value != 1 for value in class_true):
            raise ValueError("UGV validation target must be positive-only annotated-waste records")
        report["annotated_waste_positive_recall"] = positive_only_metrics(probabilities, threshold)["positive_recall"]
        ordered = np.sort(np.asarray(probabilities, dtype=float))
        report["predicted_probability_distribution"] = {
            "mean": float(ordered.mean()), "median": float(np.median(ordered)),
            "standard_deviation": float(ordered.std()), "min": float(ordered.min()),
            "max": float(ordered.max()),
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="FloPWD root")
    parser.add_argument("--ugv-dir", type=Path, required=True, help="Complete UGV v13 export root")
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/multidomain_seed42"))
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted run from its last epoch checkpoint")
    args = parser.parse_args(argv)
    if args.resume:
        if not args.output_dir.is_dir() or not (args.output_dir / "metadata.json").is_file():
            raise SystemExit(f"Cannot resume without run metadata: {args.output_dir}")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory must be absent or empty: {args.output_dir}")

    import tensorflow as tf

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if spec["compute_budget_strategy"] != "matched_optimizer_steps":
        raise ValueError("expected matched_optimizer_steps in frozen spec")
    config = load_config(args.config)
    _validate_config(spec, config)
    shared = spec["shared_protocol"]
    budget = shared["training_budget"]
    epochs, steps_per_epoch = int(budget["epochs"]), int(budget["steps_per_epoch"])
    total_steps = epochs * steps_per_epoch
    if total_steps != int(budget["total_optimizer_steps"]):
        raise ValueError("frozen optimizer-step budget is internally inconsistent")
    tf.keras.utils.set_random_seed(int(shared["seed"]))
    if config["reproducibility"].get("deterministic_tensorflow_ops", False):
        tf.config.experimental.enable_op_determinism()

    flo_records = load_flopwd(args.data_dir)
    ugv_records = load_ugv(args.ugv_dir)
    flo_manifest_path = Path(shared["split_manifests"]["flopwd"])
    ugv_manifest_path = Path(shared["split_manifests"]["ugv"])
    if _sha(flo_manifest_path) != shared["split_manifests"]["flopwd_sha256"] or _sha(ugv_manifest_path) != shared["split_manifests"]["ugv_sha256"]:
        raise ValueError("split manifest hash differs from the frozen specification")
    flo_manifest = load_split_manifest(flo_manifest_path, [r.filename for r in flo_records])
    ugv_manifest = load_ugv_grouped_split_manifest(ugv_manifest_path, ugv_records)
    flo_parts = {name: records_for_split(flo_records, flo_manifest, name)
                 for name in ("train", "validation", "test")}
    ugv_parts = {name: records_for_ugv_split(ugv_records, ugv_manifest, name)
                 for name in ("train", "validation", "test")}
    flo_train = [heterogeneous_flopwd_record(r) for r in flo_parts["train"]]
    ugv_train = [heterogeneous_ugv_record(r) for r in ugv_parts["train"]]
    flo_validation = [heterogeneous_flopwd_record(r) for r in flo_parts["validation"]]
    ugv_validation = [heterogeneous_ugv_record(r) for r in ugv_parts["validation"]]
    if len(flo_train) != spec["control_provenance"]["train_records"]:
        raise ValueError("FloPWD train count does not match Control A provenance")
    if any(not record.classification_target_available for record in ugv_train + ugv_validation):
        raise ValueError("unexpected unavailable label in the complete non-empty UGV export")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config = config["model"]
    previous_metadata = json.loads((args.output_dir / "metadata.json").read_text(encoding="utf-8")) if args.resume else None
    previous_history = json.loads((args.output_dir / "history.json").read_text(encoding="utf-8")) if args.resume else None
    completed_before = int(previous_metadata["completed_epochs"]) if args.resume else 0
    if args.resume:
        if previous_metadata.get("benchmark_status") != "training":
            raise ValueError("only an interrupted training run can be resumed")
        if previous_metadata.get("frozen_spec") != {"filename": args.spec.name, "sha256": _sha(args.spec)}:
            raise ValueError("run frozen specification differs from the requested specification")
        if previous_metadata.get("completed_epochs") != len(previous_history.get("epoch", [])):
            raise ValueError("metadata/history epoch counts disagree; refusing to resume")
        if not 0 < completed_before < epochs:
            raise ValueError(f"resume requires 0 < completed_epochs < {epochs}, got {completed_before}")

    # Weights are restored from the full checkpoint, so avoid another ImageNet
    # cache/network dependency when resuming. The checkpoint also stores Adam slots.
    model = create_model(input_shape=(*config["data"]["image_size"], 3),
                         weights=None if args.resume else "imagenet",
                         backbone_trainable=False, dense_units=model_config["dense_units"],
                         dropout_rates=model_config["dropout_rates"])
    class_loss = make_tensorflow_masked_loss("binary_crossentropy")
    severity_loss = make_tensorflow_masked_loss("mae")
    class_weight = float(shared["loss_weights"]["classification"])
    severity_weight = float(shared["loss_weights"]["severity"])
    learning_rate = float(shared["learning_rate"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"classification": class_loss, "severity": severity_loss},
        loss_weights={"classification": class_weight, "severity": severity_weight},
    )
    if args.resume:
        # Create optimizer slots before loading; Keras 2.15 otherwise defers
        # slot restoration and the iteration counter can silently reset.
        model.optimizer.build(model.trainable_variables)
        model.load_weights(args.output_dir / "checkpoint.weights.h5")
        expected_iteration = completed_before * steps_per_epoch
        actual_iteration = int(model.optimizer.iterations.numpy())
        if actual_iteration != expected_iteration:
            raise ValueError(f"checkpoint optimizer iteration {actual_iteration} != expected {expected_iteration}")
    train_data, computed_steps = make_multidomain_tf_dataset(
        {"flopwd": flo_train, "ugv": ugv_train}, int(shared["batch_size"]),
        tuple(config["data"]["image_size"]), training=True, seed=int(shared["seed"]),
        strategy="balanced", include_domain_id=True,
    )
    # Do not use the combined natural epoch length: the optimizer-step budget is
    # fixed by Control A, so each epoch intentionally samples only part of UGV.
    natural_combined_steps = computed_steps
    flo_val_ds = make_heterogeneous_tf_dataset(flo_validation, int(shared["batch_size"]),
                                               tuple(config["data"]["image_size"]))
    ugv_val_ds = make_heterogeneous_tf_dataset(ugv_validation, int(shared["batch_size"]),
                                               tuple(config["data"]["image_size"]))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    metadata = previous_metadata if args.resume else {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "benchmark_status": "training",
        "experiment_type": "multidomain", "balance_ratio": "original",
        "git_commit_sha": commit, "seed": int(shared["seed"]), "epochs": epochs,
        "steps_per_epoch": steps_per_epoch, "total_optimizer_steps": total_steps,
        "natural_combined_steps_per_epoch": int(natural_combined_steps),
        "compute_budget_strategy": "matched_optimizer_steps", "batch_size": int(shared["batch_size"]),
        "optimizer": "Adam", "learning_rate": learning_rate,
        "domain_sampling": {"strategy": "balanced", "probability": {"flopwd": 0.5, "ugv": 0.5}},
        "class_balancing": "none", "classification_threshold": float(shared["classification_threshold"]),
        "loss_functions": {"classification": "masked_binary_crossentropy", "severity": "masked_mae"},
        "loss_weights": {"classification": class_weight, "severity": severity_weight},
        "backbone_trainable": False,
        "architecture": {"backbone": "ResNet50", "weights": "imagenet", "include_top": False,
                         "input_shape": [224, 224, 3], "dense_units": model_config["dense_units"],
                         "dropout_rates": model_config["dropout_rates"],
                         "heads": ["binary classification", "bounded image-area severity"]},
        "preprocessing": "keras.applications.resnet.preprocess_input embedded in model",
        "tensorflow_version": tf.__version__, "keras_version": importlib.metadata.version("keras"),
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "split_manifests": {
            "flopwd": {"filename": flo_manifest_path.name, "sha256": _sha(flo_manifest_path)},
            "ugv": {"filename": ugv_manifest_path.name, "sha256": _sha(ugv_manifest_path)},
        },
        "frozen_spec": {"filename": args.spec.name, "sha256": _sha(args.spec)},
        "dataset_counts": {
            "flopwd": {split: len(rows) for split, rows in flo_parts.items()},
            "ugv": {split: len(rows) for split, rows in ugv_parts.items()},
        },
        "train_class_counts": {
            "flopwd": dict(Counter("positive" if r.classification_target else "negative" for r in flo_train)),
            "ugv_annotated_waste_positive": sum(r.classification_target_available and r.classification_target == 1 for r in ugv_train),
        },
        "positive_zero_severity_count": sum(r.binary_label == 1 and r.severity_percent == 0 for r in flo_records),
        "source_absolute_paths_saved": False, "completed_epochs": 0,
    }
    if args.resume:
        resume_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        metadata["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["resume_git_commit_sha"] = resume_commit
        metadata["resume_from_epoch"] = completed_before
        metadata["resume_optimizer_iteration"] = int(model.optimizer.iterations.numpy())
        metadata["input_stream_replayed_batches"] = completed_before * steps_per_epoch
        metadata["checkpoint_restored_optimizer_state"] = True
    _write(args.output_dir / "config.json", {"experiment_profile": "multidomain", "frozen_spec": args.spec.as_posix(),
                                                "protocol": shared, "domain_sampling": metadata["domain_sampling"],
                                                "class_balancing": "none"})
    _write(args.output_dir / "metadata.json", metadata)
    history = previous_history if args.resume else {"epoch": [], "train": [], "validation_flopwd": [], "validation_ugv": [],
               "examples_consumed_per_domain": []}
    _write(args.output_dir / "history.json", history)
    train_iter = iter(train_data)
    replay_batches = completed_before * steps_per_epoch
    for _ in range(replay_batches):
        next(train_iter)
    consumed = Counter(previous_metadata.get("examples_consumed", {})) if args.resume else Counter()
    start = time.monotonic()
    for epoch in range(completed_before + 1, epochs + 1):
        sums = Counter()
        for _step in range(steps_per_epoch):
            images, targets, domain_ids = next(train_iter)
            ids = [item.decode("utf-8") for item in domain_ids.numpy()]
            consumed.update(ids)
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                lc = class_loss(targets["classification"], outputs["classification"])
                ls = severity_loss(targets["severity"], outputs["severity"])
                total = class_weight * lc + severity_weight * ls
            gradients = tape.gradient(total, model.trainable_variables)
            pairs = [(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables)
                     if gradient is not None]
            if not pairs or not bool(tf.math.is_finite(total).numpy()):
                raise FloatingPointError(f"invalid loss/gradient at epoch={epoch}")
            if any(not bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()) for gradient, _ in pairs):
                raise FloatingPointError(f"non-finite gradient at epoch={epoch}")
            model.optimizer.apply_gradients(pairs)
            sums["classification_loss"] += float(lc.numpy())
            sums["severity_loss"] += float(ls.numpy())
            sums["total_loss"] += float(total.numpy())
        epoch_train = {key: value / steps_per_epoch for key, value in sums.items()}
        val_flo = _evaluate_domain(model, flo_val_ds, "flopwd", class_loss, severity_loss,
                                   float(shared["classification_threshold"]))
        val_ugv = _evaluate_domain(model, ugv_val_ds, "ugv", class_loss, severity_loss,
                                   float(shared["classification_threshold"]))
        history["epoch"].append(epoch)
        history["train"].append(epoch_train)
        history["validation_flopwd"].append(val_flo)
        history["validation_ugv"].append(val_ugv)
        history["examples_consumed_per_domain"].append(dict(consumed))
        _write(args.output_dir / "history.json", history)
        metadata["completed_epochs"] = epoch
        metadata["examples_consumed"] = dict(consumed)
        metadata["observed_domain_proportions"] = {key: value / sum(consumed.values()) for key, value in consumed.items()}
        metadata["elapsed_seconds"] = float(previous_metadata.get("elapsed_seconds", 0.0) if args.resume else 0.0) + time.monotonic() - start
        _write(args.output_dir / "metadata.json", metadata)
        model.save_weights(args.output_dir / "checkpoint.weights.h5")
        print(f"Epoch {epoch}/{epochs} train={epoch_train} val_flopwd={val_flo} val_ugv={val_ugv}", flush=True)

    duration = float(previous_metadata.get("elapsed_seconds", 0.0) if args.resume else 0.0) + time.monotonic() - start
    if int(model.optimizer.iterations.numpy()) != total_steps:
        raise AssertionError(f"optimizer iteration count {int(model.optimizer.iterations.numpy())} != {total_steps}")
    if sum(consumed.values()) != total_steps * int(shared["batch_size"]):
        raise AssertionError("consumed example total differs from the fixed full-batch optimizer budget")
    if consumed.get("flopwd", 0) == 0 or consumed.get("ugv", 0) == 0:
        raise AssertionError("both domains must contribute examples")
    model.save(args.output_dir / "model.keras")
    checkpoint = args.output_dir / "checkpoint.weights.h5"
    if checkpoint.exists():
        checkpoint.unlink()
    metadata.update({
        "benchmark_status": "full", "elapsed_seconds": duration,
        "completed_epochs": epochs, "examples_consumed": dict(consumed),
        "observed_domain_proportions": {key: value / sum(consumed.values()) for key, value in consumed.items()},
        "optimizer_steps_completed": epochs * steps_per_epoch,
        "ugv_severity_available_examples": 0,
        "model_file": "model.keras",
    })
    _write(args.output_dir / "metadata.json", metadata)
    print(f"Completed Model B in {duration:.1f}s; consumed {dict(consumed)}; saved {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
