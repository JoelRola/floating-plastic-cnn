"""Train the frozen 660-update Model F two-stream benchmark."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.config import load_config
from floating_plastic.data import (heterogeneous_flopwd_record, heterogeneous_ugv_record,
                                   load_flopwd, load_ugv)
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.losses import make_tensorflow_masked_loss
from floating_plastic.model import create_model, custom_objects
from floating_plastic.pipeline import (decode_resize_rgb, make_heterogeneous_tf_dataset,
                                       make_multidomain_tf_dataset)
from floating_plastic.samplers import OriginalPriorCyclicSampler, severity_distribution
from floating_plastic.splits import load_split_manifest, records_for_split
from floating_plastic.training import task_decoupled_two_stream_train_step


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8")


def safe_save_model(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".pending.keras")
    model.save(temporary)
    os.replace(temporary, path)


def model_summary(model):
    return {"total": int(model.count_params()),
            "trainable": int(sum(np.prod(v.shape) for v in model.trainable_variables)),
            "non_trainable": int(sum(np.prod(v.shape) for v in model.non_trainable_variables))}


def severity_head_summary(model, images, targets):
    import tensorflow as tf
    pred = np.asarray(model(images, training=False)["severity"]).reshape(-1)
    packed = targets["severity"].numpy()
    mask = packed[:, 1] > 0
    errors = np.abs(pred[mask] - packed[mask, 0])
    layer = model.get_layer("severity_fraction")
    feature_model = tf.keras.Model(model.input, layer.input)
    features = feature_model(images, training=False)
    logits = tf.linalg.matmul(features, layer.kernel) + layer.bias
    return {"mae": float(errors.mean()), "prediction_mean": float(pred.mean()),
            "prediction_median": float(np.median(pred)), "prediction_sd": float(pred.std()),
            "prediction_min": float(pred.min()), "prediction_max": float(pred.max()),
            "logit_mean": float(tf.reduce_mean(logits).numpy()),
            "final_layer_bias": np.asarray(layer.bias).reshape(-1).astype(float).tolist(),
            "final_layer_kernel_norm": float(tf.linalg.global_norm([layer.kernel]).numpy())}


def validate_epoch(model, records, batch_size, threshold, domain):
    from floating_plastic.metrics import classification_metrics, severity_regression_metrics, threshold_predictions
    dataset = make_heterogeneous_tf_dataset(records, batch_size, (224, 224))
    classes, probs, true_severity, predicted_severity = [], [], [], []
    for images, targets in dataset:
        output = model(images, training=False)
        cls = targets["classification"].numpy()
        if not np.all(cls[:, 1] == 1):
            raise ValueError(f"{domain} validation classification labels unavailable")
        classes.extend(cls[:, 0].astype(int).tolist())
        probs.extend(np.asarray(output["classification"]).reshape(-1).tolist())
        if domain == "flopwd":
            sev = targets["severity"].numpy()
            true_severity.extend(sev[:, 0].tolist())
            predicted_severity.extend(np.asarray(output["severity"]).reshape(-1).tolist())
    classes, probs = np.asarray(classes), np.asarray(probs)
    cls_metrics = classification_metrics(classes, threshold_predictions(probs, threshold))
    result = {"sample_count": int(len(classes)), "classification": cls_metrics,
              "mean_probability": float(probs.mean()), "median_probability": float(np.median(probs)),
              "probability_sd": float(probs.std()), "minimum_probability": float(probs.min()),
              "maximum_probability": float(probs.max()),
              "positive_recall": float(np.sum((probs >= threshold) & (classes == 1)) / max(1, np.sum(classes == 1)))}
    if domain == "flopwd":
        true_severity, predicted_severity = np.asarray(true_severity), np.asarray(predicted_severity)
        result["severity"] = severity_regression_metrics(true_severity, predicted_severity)
        result["severity_prediction"] = {"mean": float(predicted_severity.mean()),
                                         "median": float(np.median(predicted_severity)),
                                         "sd": float(predicted_severity.std()),
                                         "min": float(predicted_severity.min()),
                                         "max": float(predicted_severity.max())}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/task_decoupled_original_severity_prior_seed42"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    expected_matrix_sha = "ffb75c06dd39684ee1f5fa635ec839ba778593176edd6ddad688980175692e85"
    if sha256(args.spec) != expected_matrix_sha:
        raise ValueError("frozen Model F matrix hash mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory exists; refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(args.output_dir.glob("resume_state_*.json"))
    if args.resume and not checkpoints:
        raise FileNotFoundError("resume requires a completed epoch state checkpoint")
    if not args.resume and checkpoints:
        raise FileExistsError("fresh Model F output contains an existing checkpoint")

    import tensorflow as tf
    from floating_plastic.pipeline import pack_masked_targets

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    frozen = spec["matrix"]["F"]
    shared = spec["shared_protocol"]
    if frozen["status"] != "frozen_not_run" or frozen["seed"] != 42:
        raise ValueError("frozen Model F identity changed")
    config = load_config(args.config)
    if (config["training"]["seed"] != 42 or config["training"]["batch_size"] != 32 or
            config["training"]["learning_rate"] != .001 or config["model"]["backbone_trainable"] or
            config["model"]["weights"] != "imagenet" or
            config["model"]["dense_units"] != shared["architecture"]["dense_units"] or
            config["model"]["dropout_rates"] != shared["architecture"]["dropout_rates"] or
            config["losses"]["classification"] != "binary_crossentropy" or
            config["losses"]["severity"] != "mae" or
            config["losses"]["classification_weight"] != 1.0 or
            config["losses"]["severity_weight"] != .5 or
            config["evaluation"]["classification_threshold"] != .5):
        raise ValueError("runtime config differs from frozen Model F protocol")
    tf.keras.utils.set_random_seed(42)
    if config["reproducibility"].get("deterministic_tensorflow_ops", False):
        tf.config.experimental.enable_op_determinism()

    flo_all, ugv_all = load_flopwd(args.data_dir), load_ugv(args.ugv_dir)
    flo_manifest_path, ugv_manifest_path = Path(shared["split_manifests"]["flopwd"]), Path(shared["split_manifests"]["ugv"])
    hashes = {"flopwd": sha256(flo_manifest_path), "ugv": sha256(ugv_manifest_path), "matrix": sha256(args.spec)}
    if (hashes["flopwd"] != shared["split_manifests"]["flopwd_sha256"] or
            hashes["ugv"] != shared["split_manifests"]["ugv_sha256"]):
        raise ValueError("frozen split manifest hash mismatch")
    fm = load_split_manifest(flo_manifest_path, [r.filename for r in flo_all])
    um = load_ugv_grouped_split_manifest(ugv_manifest_path, ugv_all)
    flo_train = [heterogeneous_flopwd_record(r) for r in records_for_split(flo_all, fm, "train")]
    flo_val = [heterogeneous_flopwd_record(r) for r in records_for_split(flo_all, fm, "validation")]
    ugv_train = [heterogeneous_ugv_record(r) for r in records_for_ugv_split(ugv_all, um, "train")]
    ugv_val = [heterogeneous_ugv_record(r) for r in records_for_ugv_split(ugv_all, um, "validation")]
    if len(flo_train) != 1402 or not flo_val or not ugv_train or not ugv_val:
        raise ValueError("unexpected frozen training/validation partition size")
    if not all(r.severity_target_available for r in flo_train) or any(r.severity_target_available for r in ugv_train):
        raise ValueError("severity supervision source contract violated")

    d_meta_path = Path("runs/task_decoupled_multidomain_seed42/metadata.json")
    d_meta = json.loads(d_meta_path.read_text(encoding="utf-8"))
    target_updates = 660
    target_severity_count = int(d_meta["examples_consumed"]["flopwd"])
    sampler_check = OriginalPriorCyclicSampler(flo_train, seed=42)
    planned_draws = sampler_check.next_records(target_severity_count)
    planned_distribution = severity_distribution(planned_draws)
    class_dataset, _ = make_multidomain_tf_dataset(
        {"flopwd": flo_train, "ugv": ugv_train}, 32, tuple(config["data"]["image_size"]),
        training=True, seed=42, strategy="balanced", flopwd_negative_to_positive=1,
        include_domain_id=True)
    class_iterator = iter(class_dataset)
    severity_sampler = OriginalPriorCyclicSampler(flo_train, seed=42)
    class_loss, severity_loss = make_tensorflow_masked_loss("binary_crossentropy"), make_tensorflow_masked_loss("mae")
    history, counts = [], Counter()
    severity_values_consumed = []
    completed = 0
    resume_events = []
    prior_elapsed_seconds = 0.0
    original_run_started_utc = datetime.now(timezone.utc).isoformat()
    if args.resume:
        state_path = checkpoints[-1]
        suffix = state_path.stem.removeprefix("resume_state_")
        checkpoint_path = args.output_dir / f"resume_{suffix}.keras"
        if not checkpoint_path.is_file():
            raise FileNotFoundError("latest resume state has no matching model checkpoint")
        model = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects(), compile=True)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["matrix_sha256"] != expected_matrix_sha or state["split_hashes"] != {k: hashes[k] for k in ("flopwd", "ugv")}:
            raise ValueError("resume provenance does not match frozen inputs")
        completed = int(state["completed_updates"])
        if completed % 44:
            raise ValueError("checkpoint must be at an epoch boundary")
        if int(model.optimizer.iterations.numpy()) != completed:
            raise ValueError("optimizer iteration does not match checkpoint update count")
        severity_sampler.load_state_dict(state["severity_sampler_state"])
        history, counts = state["history"], Counter(state["counts"])
        severity_values_consumed = state["severity_values_consumed"]
        resume_events = state.get("resume_events", [])
        prior_elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
        original_run_started_utc = state.get("run_started_utc", original_run_started_utc)
        resume_events.append({"time_utc": datetime.now(timezone.utc).isoformat(), "resumed_from_update": completed})
        for _ in range(completed):
            next(class_iterator)
    else:
        cache = (Path(__file__).resolve().parents[1] / ".keras-cache" / ".keras" / "models" /
                 "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
        if not cache.is_file() or sha256(cache) != d_meta["imagenet_weights_sha256"]:
            raise FileNotFoundError("verified cached ImageNet ResNet50 weights not available")
        model = create_model(input_shape=(224, 224, 3), weights=str(cache), backbone_trainable=False,
                             dense_units=config["model"]["dense_units"],
                             dropout_rates=config["model"]["dropout_rates"],
                             architecture_variant="task_decoupled")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                      loss={"classification": class_loss, "severity": severity_loss},
                      loss_weights={"classification": 1.0, "severity": .5})
        # Stable canonical variable order for Keras optimizer state serialization.
        model.optimizer.build(model.trainable_variables)
    params = model_summary(model)
    if params != {"total": 24702850, "trainable": 1115138, "non_trainable": 23587712}:
        raise AssertionError(f"Model F architecture parameter mismatch: {params}")
    backbone = model.get_layer("resnet50")
    if backbone.trainable or any(id(v) in {id(t) for t in model.trainable_variables} for v in backbone.weights):
        raise AssertionError("ResNet50 backbone is not frozen")

    train_monitor = sorted(flo_val, key=lambda r: r.filename)[:32]
    monitor_ds = make_heterogeneous_tf_dataset(train_monitor, len(train_monitor), (224, 224))
    monitor_images, monitor_targets = next(iter(monitor_ds))
    started = time.monotonic()
    run_started = original_run_started_utc
    for epoch in range(completed // 44, 15):
        epoch_results = []
        epoch_class_domains, epoch_severity_targets = Counter(), []
        for _step in range(44):
            class_images, class_targets, domain_tensor = next(class_iterator)
            domains = [v.decode("utf-8") for v in domain_tensor.numpy()]
            k = domains.count("flopwd")
            if k == 0:
                raise AssertionError("classification update had no FloPWD example; cannot match K")
            selected = severity_sampler.next_records(k)
            sev_images = tf.stack([decode_resize_rgb(str(r.image_path), (224, 224)) for r in selected])
            sev_targets = {
                "classification": tf.convert_to_tensor(pack_masked_targets(
                    [r.classification_target for r in selected], [r.classification_target_available for r in selected]), dtype=tf.float32),
                "severity": tf.convert_to_tensor(pack_masked_targets(
                    [r.severity_target for r in selected], [r.severity_target_available for r in selected]), dtype=tf.float32),
            }
            if any(r.source_domain != "flopwd" for r in selected) or len(selected) != k:
                raise AssertionError("severity stream source/count contract violated")
            labels = class_targets["classification"].numpy()
            class_available = labels[:, 1] > 0
            for domain in ("flopwd", "ugv"):
                mask = np.asarray(domains) == domain
                epoch_class_domains[domain] += int(mask.sum())
                counts[f"classification_{domain}"] += int(mask.sum())
                counts[f"classification_{domain}_positive"] += int(np.sum(mask & class_available & (labels[:, 0] == 1)))
                counts[f"classification_{domain}_negative"] += int(np.sum(mask & class_available & (labels[:, 0] == 0)))
            counts["classification_total"] += len(domains)
            counts["classification_positive"] += int(np.sum(class_available & (labels[:, 0] == 1)))
            counts["severity_examples"] += k
            epoch_severity_targets.extend(float(r.severity_target) for r in selected)
            severity_values_consumed.extend(float(r.severity_target) for r in selected)
            step = task_decoupled_two_stream_train_step(
                model, model.optimizer, class_images, class_targets, sev_images, sev_targets,
                ["flopwd"] * k, class_loss, severity_loss, 1.0, .5)
            if step["severity_example_count"] != k or step["classification_tower_gradient_norm"] <= 0 or step["severity_tower_gradient_norm"] < 0:
                raise AssertionError("task gradient/example contract violated")
            if not np.isfinite([step["classification_loss"], step["severity_loss"], step["total_loss"]]).all():
                raise FloatingPointError("non-finite training loss")
            epoch_results.append(step)
        update_count = (epoch + 1) * 44
        if int(model.optimizer.iterations.numpy()) != update_count:
            raise AssertionError("optimizer update count diverged")
        val_flo = validate_epoch(model, flo_val, 32, .5, "flopwd")
        val_ugv = validate_epoch(model, ugv_val, 32, .5, "ugv")
        head = severity_head_summary(model, monitor_images, monitor_targets)
        history.append({
            "epoch": epoch + 1, "optimizer_updates": update_count,
            "training": {
                "classification_loss": float(np.mean([r["classification_loss"] for r in epoch_results])),
                "severity_loss": float(np.mean([r["severity_loss"] for r in epoch_results])),
                "classification_tower_gradient_norm": float(np.mean([r["classification_tower_gradient_norm"] for r in epoch_results])),
                "severity_tower_gradient_norm": float(np.mean([r["severity_tower_gradient_norm"] for r in epoch_results])),
                "classification_domain_count": dict(epoch_class_domains),
                "severity_examples": len(epoch_severity_targets),
                "severity_distribution": severity_distribution([
                    type("Observed", (), {"severity_target": x}) for x in epoch_severity_targets]),
            },
            "flopwd_validation": val_flo, "ugv_validation": val_ugv,
            "severity_head_monitor": head,
        })
        state = {
            "matrix_sha256": expected_matrix_sha, "split_hashes": {k: hashes[k] for k in ("flopwd", "ugv")},
            "completed_updates": update_count, "severity_sampler_state": severity_sampler.state_dict(),
            "history": history, "counts": dict(counts), "severity_values_consumed": severity_values_consumed,
            "resume_events": resume_events, "elapsed_seconds": prior_elapsed_seconds + time.monotonic() - started,
            "run_started_utc": run_started,
        }
        suffix = f"{update_count:04d}"
        safe_save_model(model, args.output_dir / f"resume_{suffix}.keras")
        write_json(args.output_dir / f"resume_state_{suffix}.json", state)
        write_json(args.output_dir / "history.json", {"epochs": history})
        print(json.dumps({"epoch": epoch + 1, "updates": update_count,
                          "train": history[-1]["training"], "validation_flo": val_flo,
                          "validation_ugv": val_ugv, "severity_monitor": head}, allow_nan=False), flush=True)

    elapsed = time.monotonic() - started
    if int(model.optimizer.iterations.numpy()) != target_updates or counts["severity_examples"] != target_severity_count:
        raise AssertionError(f"final budget mismatch updates={model.optimizer.iterations.numpy()} severity={counts['severity_examples']} target={target_severity_count}")
    actual_distribution = severity_distribution([
        type("Observed", (), {"severity_target": x}) for x in severity_values_consumed])
    if not .20 <= counts["classification_flopwd"] / counts["classification_total"] <= .80:
        raise AssertionError("realized domain balance is implausibly far from frozen 50/50")
    if actual_distribution["sample_count"] != target_severity_count:
        raise AssertionError("severity stream draw count mismatch")
    safe_save_model(model, args.output_dir / "model.keras")
    metadata = {
        "benchmark_status": "full", "run_status": "complete", "experiment_type": frozen["experiment_type"],
        "architecture_variant": "task_decoupled", "classification_sampling": "balanced_multidomain_1to1_flopwd",
        "severity_sampling": "original_flopwd_prior", "compute_budget_strategy": "matched_optimizer_steps_and_severity_count",
        "seed": 42, "epochs_completed": 15, "steps_per_epoch": 44, "total_optimizer_updates": 660,
        "optimizer_iterations": int(model.optimizer.iterations.numpy()), "training_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "frozen_matrix_sha256": hashes["matrix"], "split_sha256": {k: hashes[k] for k in ("flopwd", "ugv")},
        "parameter_counts": params, "optimizer": "Adam", "learning_rate": .001,
        "losses": {"classification": "masked BCE", "severity": "masked MAE percentage points"},
        "loss_weights": {"classification": 1.0, "severity": .5}, "severity_output": "sigmoid * 100",
        "classification_threshold": .5, "backbone_frozen": True,
        "classification_consumption": dict(counts),
        "classification_domain_proportions": {d: counts[f"classification_{d}"] / counts["classification_total"] for d in ("flopwd", "ugv")},
        "classification_effective_positive_rate": counts["classification_positive"] / counts["classification_total"],
        "severity_consumption": {**actual_distribution, "ugv_severity_count": 0,
                                 "matches_model_d_supervision_count": True},
        "severity_distribution_reference": {"planned_full_run": planned_distribution,
            "original_flopwd_train": severity_distribution(flo_train),
            "model_d_balanced": d_meta["severity_target_distribution_context"]["flopwd_seed42_1to1_balanced_sampler_cycle"]},
        "duration_seconds": prior_elapsed_seconds + elapsed, "run_started_utc": run_started,
        "resume_events": resume_events, "test_set_used_during_training": False,
        "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__,
                    "keras": importlib.metadata.version("keras"), "numpy": np.__version__},
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "resolved_config.json", {"frozen_spec": frozen, "shared_protocol": shared,
                                                           "runtime_config": config})
    print(json.dumps({"run_status": "complete", "metadata": metadata}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
