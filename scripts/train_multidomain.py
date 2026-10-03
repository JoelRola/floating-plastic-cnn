"""Train a frozen-spec multi-domain experiment with a matched optimizer-step budget."""

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
from floating_plastic.training import masked_multitask_train_step


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
    parser.add_argument("--experiment-type", choices=("multidomain", "multidomain_class_balanced",
                                                        "multidomain_2to1", "multidomain_task_decoupled"),
                        default="multidomain")
    parser.add_argument("--flopwd-negative-to-positive", type=int, default=None,
                        help="balance only FloPWD training records; required for Models C/D/E")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted run from its last epoch checkpoint")
    args = parser.parse_args(argv)
    if args.resume:
        if not args.output_dir.is_dir() or not (args.output_dir / "metadata.json").is_file():
            raise SystemExit(f"Cannot resume without run metadata: {args.output_dir}")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory must be absent or empty: {args.output_dir}")

    expected_balance = {"multidomain": None, "multidomain_class_balanced": 1,
                        "multidomain_2to1": 2, "multidomain_task_decoupled": 1}[args.experiment_type]
    architecture_variant = ("task_decoupled" if args.experiment_type == "multidomain_task_decoupled"
                            else "shared_tower")
    if args.flopwd_negative_to_positive != expected_balance:
        raise ValueError(f"{args.experiment_type} requires frozen FloPWD negative:positive ratio "
                         f"{expected_balance!r}; got {args.flopwd_negative_to_positive!r}")
    is_class_balanced = expected_balance is not None

    import tensorflow as tf

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if spec["compute_budget_strategy"] != "matched_optimizer_steps":
        raise ValueError("expected matched_optimizer_steps in frozen spec")
    frozen_c = spec["matrix"]["C"]
    if (frozen_c["experiment_type"] != "multidomain_class_balanced" or
            frozen_c["domain_sampling"] != "balanced" or
            frozen_c["flopwd_class_balance"] != "1:1 negative:positive within FloPWD train only" or
            frozen_c["expected_domain_probability"] != {"flopwd": 0.5, "ugv": 0.5}):
        raise ValueError("Model C protocol does not match the frozen experiment matrix")
    if args.experiment_type in {"multidomain_2to1", "multidomain_task_decoupled"}:
        matrix_key = "E" if args.experiment_type == "multidomain_2to1" else "D"
        frozen = spec["matrix"][matrix_key]
        expected_name = "multidomain_2to1" if matrix_key == "E" else "task_decoupled_multidomain"
        expected_ratio = "2:1" if matrix_key == "E" else "1:1"
        if (frozen["name"] != expected_name or frozen["status"] != "frozen_not_run" or
                frozen["experiment_type"] != args.experiment_type or
                frozen["architecture_variant"] != architecture_variant or
                frozen["domain_sampling"] != {"flopwd": 0.5, "ugv": 0.5} or
                frozen["flopwd_class_balance"]["negative_to_positive"] != expected_ratio or
                frozen["optimizer"] != "adam" or
                int(frozen["seed"]) != int(spec["shared_protocol"]["seed"]) or
                int(frozen["epochs"]) != int(spec["shared_protocol"]["training_budget"]["epochs"]) or
                int(frozen["steps_per_epoch"]) != int(spec["shared_protocol"]["training_budget"]["steps_per_epoch"]) or
                int(frozen["total_optimizer_updates"]) != int(spec["shared_protocol"]["training_budget"]["total_optimizer_steps"]) or
                int(frozen["batch_size"]) != int(spec["shared_protocol"]["batch_size"]) or
                float(frozen["learning_rate"]) != float(spec["shared_protocol"]["learning_rate"]) or
                float(frozen["classification_threshold"]) != float(spec["shared_protocol"]["classification_threshold"]) or
                float(frozen["classification_loss_weight"]) != float(spec["shared_protocol"]["loss_weights"]["classification"]) or
                float(frozen["severity_loss_weight"]) != float(spec["shared_protocol"]["loss_weights"]["severity"]) or
                frozen["backbone"] != "resnet50_imagenet_frozen" or
                frozen["preprocessing"] != "same_as_models_b_c" or
                frozen["split_manifests"] != {"flopwd": spec["shared_protocol"]["split_manifests"]["flopwd"],
                                              "ugv": spec["shared_protocol"]["split_manifests"]["ugv"]}):
            raise ValueError(f"Model {matrix_key} protocol does not match the frozen experiment matrix")
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
        if previous_metadata.get("experiment_type") != args.experiment_type:
            raise ValueError("run experiment type differs from requested experiment type")
        if previous_metadata.get("architecture_variant", architecture_variant) != architecture_variant:
            raise ValueError("run architecture variant differs from requested experiment type")
        if previous_metadata.get("flopwd_negative_to_positive") != args.flopwd_negative_to_positive:
            raise ValueError("run class-balance ratio differs from requested ratio")
        if previous_metadata.get("completed_epochs") != len(previous_history.get("epoch", [])):
            raise ValueError("metadata/history epoch counts disagree; refusing to resume")
        if not 0 < completed_before < epochs:
            raise ValueError(f"resume requires 0 < completed_epochs < {epochs}, got {completed_before}")

    # Weights are restored from the full checkpoint when resuming. For a fresh
    # run, use the repository's cached ImageNet file directly so Keras does not
    # try to create its default cache under a protected user profile.
    imagenet_weights = (Path(__file__).resolve().parents[1] / ".keras-cache" / ".keras" /
                        "models" / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    if not args.resume and not imagenet_weights.is_file():
        raise FileNotFoundError("cached ImageNet ResNet50 weights are unavailable")
    model = create_model(input_shape=(*config["data"]["image_size"], 3),
                         weights=None if args.resume else str(imagenet_weights),
                         backbone_trainable=False, dense_units=model_config["dense_units"],
                         dropout_rates=model_config["dropout_rates"],
                         architecture_variant=architecture_variant)
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
        strategy="balanced", flopwd_negative_to_positive=args.flopwd_negative_to_positive,
        include_domain_id=True,
    )
    # Do not use the combined natural epoch length: the optimizer-step budget is
    # fixed by Control A, so each epoch intentionally samples only part of UGV.
    natural_combined_steps = computed_steps
    flo_val_ds = make_heterogeneous_tf_dataset(flo_validation, int(shared["batch_size"]),
                                               tuple(config["data"]["image_size"]))
    ugv_val_ds = make_heterogeneous_tf_dataset(ugv_validation, int(shared["batch_size"]),
                                               tuple(config["data"]["image_size"]))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    trainer_sha = _sha(Path(__file__))
    worktree_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip())
    metadata = previous_metadata if args.resume else {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "benchmark_status": "training",
        "experiment_type": args.experiment_type,
        "balance_ratio": ({1: "1:1 negative:positive", 2: "2:1 negative:positive"}.get(expected_balance)
                          if is_class_balanced else "original"),
        "architecture_variant": architecture_variant,
        "git_commit_sha": commit, "seed": int(shared["seed"]), "epochs": epochs,
        "experiment_matrix_sha256": _sha(args.spec),
        "matrix_sha256": _sha(args.spec),
        "trainer_sha256": trainer_sha, "worktree_dirty_at_training": worktree_dirty,
        "steps_per_epoch": steps_per_epoch, "total_optimizer_steps": total_steps,
        "natural_combined_steps_per_epoch": int(natural_combined_steps),
        "compute_budget_strategy": "matched_optimizer_steps", "batch_size": int(shared["batch_size"]),
        "optimizer": "Adam", "learning_rate": learning_rate,
        "domain_sampling": {"strategy": "balanced", "probability": {"flopwd": 0.5, "ugv": 0.5}},
        "class_balancing": (f"FloPWD train only, {expected_balance}:1 negative:positive with replacement"
                            if is_class_balanced else "none"),
        "flopwd_class_balance": (f"{expected_balance}:1 negative:positive" if is_class_balanced else "original"),
        "expected_class_prior": {"flopwd_positive_rate": (1.0 / (expected_balance + 1)
                                                               if is_class_balanced else None),
                                  "overall_positive_rate": (0.5 * (1.0 / (expected_balance + 1)) + 0.5
                                                            if is_class_balanced else None),
                                  "severity_supervision_fraction": 0.5},
        "flopwd_negative_to_positive": args.flopwd_negative_to_positive,
        "classification_threshold": float(shared["classification_threshold"]),
        "loss_functions": {"classification": "masked_binary_crossentropy", "severity": "masked_mae"},
        "loss_weights": {"classification": class_weight, "severity": severity_weight},
        "backbone_trainable": False,
        "architecture": {"backbone": "ResNet50", "weights": "imagenet", "include_top": False,
                         "input_shape": [224, 224, 3], "dense_units": model_config["dense_units"],
                         "dropout_rates": model_config["dropout_rates"],
                         "architecture_variant": architecture_variant,
                         "heads": ["binary classification", "bounded image-area severity"]},
        "imagenet_weights_sha256": _sha(imagenet_weights),
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
        metadata["resume_trainer_sha256"] = _sha(Path(__file__))
        metadata["resume_from_epoch"] = completed_before
        metadata["resume_optimizer_iteration"] = int(model.optimizer.iterations.numpy())
        metadata["input_stream_replayed_batches"] = completed_before * steps_per_epoch
        metadata["checkpoint_restored_optimizer_state"] = True
        metadata.setdefault("interruption_events", []).append({
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_epochs_at_interruption": completed_before,
            "reason": "interrupted at an epoch checkpoint to correct the audit invariant: sampled class counts may fluctuate around the frozen 1:1 distribution",
        })
    _write(args.output_dir / "config.json", {"experiment_profile": args.experiment_type, "frozen_spec": args.spec.as_posix(),
                                                "architecture_variant": architecture_variant,
                                                "protocol": shared, "domain_sampling": metadata["domain_sampling"],
                                                "class_balancing": metadata["class_balancing"],
                                                "flopwd_negative_to_positive": args.flopwd_negative_to_positive})
    _write(args.output_dir / "metadata.json", metadata)
    history = previous_history if args.resume else {"epoch": [], "train": [], "validation_flopwd": [], "validation_ugv": [],
               "examples_consumed_per_domain": []}
    _write(args.output_dir / "history.json", history)
    train_iter = iter(train_data)
    replay_batches = completed_before * steps_per_epoch
    for _ in range(replay_batches):
        next(train_iter)
    consumed = Counter(previous_metadata.get("examples_consumed", {})) if args.resume else Counter()
    sample_consumption = Counter(previous_metadata.get("sample_consumption", {})) if args.resume else Counter()
    consumption_history = (previous_history.get("sample_consumption_per_epoch", [])
                          if args.resume else [])
    start = time.monotonic()
    for epoch in range(completed_before + 1, epochs + 1):
        sums = Counter()
        epoch_consumption = Counter()
        for _step in range(steps_per_epoch):
            images, targets, domain_ids = next(train_iter)
            ids = [item.decode("utf-8") for item in domain_ids.numpy()]
            consumed.update(ids)
            class_values = targets["classification"].numpy()
            severity_values = targets["severity"].numpy()
            for domain in ("flopwd", "ugv"):
                domain_mask = np.asarray(ids) == domain
                class_available = class_values[:, 1] > 0
                severity_available = severity_values[:, 1] > 0
                epoch_consumption[f"{domain}_examples"] += int(domain_mask.sum())
                epoch_consumption[f"{domain}_classification_positive"] += int(
                    np.sum(domain_mask & class_available & (class_values[:, 0] == 1)))
                epoch_consumption[f"{domain}_classification_negative"] += int(
                    np.sum(domain_mask & class_available & (class_values[:, 0] == 0)))
                epoch_consumption[f"{domain}_severity_supervised"] += int(
                    np.sum(domain_mask & severity_available))
                epoch_consumption[f"{domain}_severity_unsupervised"] += int(
                    np.sum(domain_mask & ~severity_available))
            step_values = masked_multitask_train_step(
                model, model.optimizer, images, targets, class_loss, severity_loss,
                class_weight, severity_weight)
            sums["classification_loss"] += step_values["classification_loss"]
            sums["severity_loss"] += step_values["severity_loss"]
            sums["total_loss"] += step_values["total_loss"]
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
        sample_consumption.update(epoch_consumption)
        consumption_history.append(dict(epoch_consumption))
        history["sample_consumption_per_epoch"] = consumption_history
        _write(args.output_dir / "history.json", history)
        metadata["completed_epochs"] = epoch
        metadata["examples_consumed"] = dict(consumed)
        metadata["observed_domain_proportions"] = {key: value / sum(consumed.values()) for key, value in consumed.items()}
        metadata["sample_consumption"] = dict(sample_consumption)
        metadata["observed_flopwd_positive_rate"] = (
            sample_consumption["flopwd_classification_positive"] /
            (sample_consumption["flopwd_classification_positive"] +
             sample_consumption["flopwd_classification_negative"]))
        metadata["observed_flopwd_negative_to_positive_ratio"] = (
            sample_consumption["flopwd_classification_negative"] /
            sample_consumption["flopwd_classification_positive"])
        metadata["observed_overall_positive_rate"] = (
            (sample_consumption["flopwd_classification_positive"] +
             sample_consumption["ugv_classification_positive"]) /
            (sample_consumption["flopwd_examples"] + sample_consumption["ugv_examples"]))
        metadata["observed_severity_supervision_fraction"] = (
            (sample_consumption["flopwd_severity_supervised"] + sample_consumption["ugv_severity_supervised"]) /
            (sample_consumption["flopwd_examples"] + sample_consumption["ugv_examples"]))
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
    if sample_consumption["ugv_severity_supervised"] != 0:
        raise AssertionError("UGV must not contribute severity labels")
    if sample_consumption["flopwd_severity_supervised"] != consumed.get("flopwd", 0):
        raise AssertionError("every sampled FloPWD example must contribute a severity label")
    if is_class_balanced and not all(sample_consumption[key] > 0 for key in (
            "flopwd_classification_positive", "flopwd_classification_negative")):
        raise AssertionError("both FloPWD classes must be consumed")
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
        "overall_classification_positive_samples": (
            sample_consumption["flopwd_classification_positive"] +
            sample_consumption["ugv_classification_positive"]),
        "overall_classification_negative_samples": sample_consumption["flopwd_classification_negative"],
        "severity_supervised_examples": sample_consumption["flopwd_severity_supervised"],
        "severity_unsupervised_examples": sample_consumption["flopwd_severity_unsupervised"] +
                                          sample_consumption["ugv_severity_unsupervised"],
        "observed_flopwd_negative_to_positive_ratio": (
            sample_consumption["flopwd_classification_negative"] /
            sample_consumption["flopwd_classification_positive"]),
        "observed_flopwd_positive_rate": (
            sample_consumption["flopwd_classification_positive"] /
            (sample_consumption["flopwd_classification_positive"] +
             sample_consumption["flopwd_classification_negative"])),
        "observed_overall_positive_rate": (
            (sample_consumption["flopwd_classification_positive"] +
             sample_consumption["ugv_classification_positive"]) /
            (sample_consumption["flopwd_examples"] + sample_consumption["ugv_examples"])),
        "observed_severity_supervision_fraction": (
            (sample_consumption["flopwd_severity_supervised"] + sample_consumption["ugv_severity_supervised"]) /
            (sample_consumption["flopwd_examples"] + sample_consumption["ugv_examples"])),
        "model_file": "model.keras",
    })
    _write(args.output_dir / "metadata.json", metadata)
    print(f"Completed {args.experiment_type} in {duration:.1f}s; consumed {dict(consumed)}; saved {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
