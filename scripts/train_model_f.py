"""Run the fixed eight-update real-data Model F engineering smoke test."""

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
from floating_plastic.data import heterogeneous_flopwd_record, heterogeneous_ugv_record, load_flopwd, load_ugv
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.losses import make_tensorflow_masked_loss
from floating_plastic.model import create_model
from floating_plastic.pipeline import (
    make_heterogeneous_tf_dataset, make_multidomain_tf_dataset,
    make_original_prior_severity_stream,
)
from floating_plastic.samplers import OriginalPriorCyclicSampler, severity_distribution
from floating_plastic.splits import load_split_manifest, records_for_split
from floating_plastic.training import task_decoupled_two_stream_train_step


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _monitor_summary(model, images, targets):
    import tensorflow as tf

    outputs = model(images, training=False)
    prediction = np.asarray(outputs["severity"]).reshape(-1)
    packed = targets["severity"].numpy()
    available = packed[:, 1] > 0
    true = packed[available, 0]
    errors = np.abs(prediction[available] - true)
    layer = model.get_layer("severity_fraction")
    feature_model = tf.keras.Model(model.input, layer.input)
    features = feature_model(images, training=False)
    logits = tf.linalg.matmul(features, layer.kernel) + layer.bias
    return {
        "prediction_mean": float(prediction.mean()), "prediction_median": float(np.median(prediction)),
        "prediction_standard_deviation": float(prediction.std()), "prediction_minimum": float(prediction.min()),
        "prediction_maximum": float(prediction.max()), "severity_mae": float(errors.mean()),
        "output_logit_mean": float(tf.reduce_mean(logits).numpy()),
    }


def _validate_spec(spec):
    shared = spec["shared_protocol"]
    frozen = spec["matrix"]["F"]
    checks = (
        frozen["name"] == "task_decoupled_original_severity_prior",
        frozen["status"] == "frozen_not_run",
        frozen["experiment_type"] == "multidomain_task_decoupled_severity_prior",
        frozen["architecture_variant"] == "task_decoupled",
        frozen["classification_stream"]["sources"] == ["flopwd", "ugv"],
        frozen["classification_stream"]["domain_sampling"] == {"flopwd": 0.5, "ugv": 0.5},
        frozen["classification_stream"]["flopwd_class_balance"]["negative_to_positive"] == "1:1",
        frozen["classification_stream"]["batch_size"] == 32,
        frozen["severity_stream"]["sampling"] == "original_unbalanced_distribution",
        frozen["severity_stream"]["independent_of_classification_stream"] is True,
        frozen["severity_stream"]["examples_per_update"] ==
        "equal_to_flopwd_count_in_classification_batch",
        frozen["seed"] == 42 and frozen["epochs"] == 15 and frozen["steps_per_epoch"] == 44,
        frozen["total_optimizer_updates"] == 660 and frozen["optimizer"] == "adam",
        float(frozen["learning_rate"]) == 0.001,
        float(frozen["classification_loss_weight"]) == 1.0,
        float(frozen["severity_loss_weight"]) == 0.5,
        float(frozen["classification_stream"]["threshold"]) == 0.5,
        frozen["severity_output"] == "unchanged_sigmoid_scaled_0_to_100",
        frozen["backbone"] == "resnet50_imagenet_frozen",
        frozen["split_manifests"]["flopwd"] == shared["split_manifests"]["flopwd"],
        frozen["split_manifests"]["ugv"] == shared["split_manifests"]["ugv"],
    )
    if not all(checks):
        raise ValueError("Model F frozen specification changed or is internally inconsistent")
    d_spec = spec["matrix"]["D"]
    if (d_spec["architecture_variant"] != frozen["architecture_variant"] or
            d_spec["domain_sampling"] != frozen["classification_stream"]["domain_sampling"] or
            d_spec["flopwd_class_balance"] != frozen["classification_stream"]["flopwd_class_balance"]):
        raise ValueError("F classification/architecture protocol differs from D")
    return shared, frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/debug_model_f_seed42"))
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)
    if args.steps != 8:
        raise ValueError("this phase provides a fixed eight-update debug runner only")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"debug output directory must be absent or empty: {args.output_dir}")

    import tensorflow as tf

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    shared, frozen = _validate_spec(spec)
    config = load_config(args.config)
    # Match D's frozen runtime architecture/training values.
    if (config["training"]["seed"] != 42 or config["training"]["batch_size"] != 32 or
            config["training"]["learning_rate"] != 0.001 or
            config["model"]["backbone_trainable"] is not False or
            config["model"]["weights"] != "imagenet" or
            config["model"]["dense_units"] != shared["architecture"]["dense_units"] or
            config["model"]["dropout_rates"] != shared["architecture"]["dropout_rates"] or
            config["losses"]["classification_weight"] != 1.0 or
            config["losses"]["severity_weight"] != 0.5 or
            config["losses"]["classification"] != "binary_crossentropy" or
            config["losses"]["severity"] != "mae" or
            config["evaluation"]["classification_threshold"] != 0.5):
        raise ValueError("runtime config does not match frozen Model D/F values")
    tf.keras.utils.set_random_seed(42)
    if config["reproducibility"].get("deterministic_tensorflow_ops", False):
        tf.config.experimental.enable_op_determinism()

    flo_all = load_flopwd(args.data_dir)
    ugv_all = load_ugv(args.ugv_dir)
    flo_manifest_path = Path(shared["split_manifests"]["flopwd"])
    ugv_manifest_path = Path(shared["split_manifests"]["ugv"])
    if (_sha(flo_manifest_path) != shared["split_manifests"]["flopwd_sha256"] or
            _sha(ugv_manifest_path) != shared["split_manifests"]["ugv_sha256"]):
        raise ValueError("split manifest hashes differ from the frozen matrix")
    flo_manifest = load_split_manifest(flo_manifest_path, [row.filename for row in flo_all])
    ugv_manifest = load_ugv_grouped_split_manifest(ugv_manifest_path, ugv_all)
    flo_train = [heterogeneous_flopwd_record(row) for row in records_for_split(flo_all, flo_manifest, "train")]
    flo_validation = [heterogeneous_flopwd_record(row)
                      for row in records_for_split(flo_all, flo_manifest, "validation")]
    ugv_train = [heterogeneous_ugv_record(row) for row in records_for_ugv_split(ugv_all, ugv_manifest, "train")]
    if len(flo_train) != 1402 or not flo_validation or not ugv_train:
        raise ValueError("unexpected frozen FloPWD/UGV train or validation counts")
    if any(not record.classification_target_available for record in ugv_train):
        raise ValueError("UGV training records must have eligible annotated-waste labels")

    # The original-prior simulation matches Model D's observed FloPWD severity
    # supervision count, assuming the identical seeded classification stream.
    d_reference = json.loads(Path("runs/task_decoupled_multidomain_seed42/metadata.json").read_text())
    planned_severity_count = int(d_reference["examples_consumed"]["flopwd"])
    planned_sampler = OriginalPriorCyclicSampler(flo_train, seed=42)
    planned_distribution = severity_distribution(planned_sampler.next_records(planned_severity_count))
    original_distribution = severity_distribution(flo_train)
    balanced_distribution = d_reference["severity_target_distribution_context"][
        "flopwd_seed42_1to1_balanced_sampler_cycle"]

    image_size = tuple(config["data"]["image_size"])
    class_dataset, _natural_steps = make_multidomain_tf_dataset(
        {"flopwd": flo_train, "ugv": ugv_train}, 32, image_size, training=True, seed=42,
        strategy="balanced", flopwd_negative_to_positive=1, include_domain_id=True)
    class_iterator = iter(class_dataset)
    severity_dataset = make_original_prior_severity_stream(flo_train, image_size, seed=42)
    severity_iterator = iter(severity_dataset)
    monitor_records = sorted(flo_validation, key=lambda row: row.filename)[:32]
    monitor_dataset = make_heterogeneous_tf_dataset(monitor_records, len(monitor_records), image_size)
    monitor_images, monitor_targets = next(iter(monitor_dataset))

    imagenet_weights = (Path(__file__).resolve().parents[1] / ".keras-cache" / ".keras" /
                        "models" / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    if not imagenet_weights.is_file() or _sha(imagenet_weights) != d_reference["imagenet_weights_sha256"]:
        raise FileNotFoundError("the verified Model D ImageNet ResNet50 weights cache is unavailable")
    model = create_model(input_shape=(224, 224, 3), weights=str(imagenet_weights), backbone_trainable=False,
                         dense_units=config["model"]["dense_units"],
                         dropout_rates=config["model"]["dropout_rates"],
                         architecture_variant="task_decoupled")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"classification": make_tensorflow_masked_loss("binary_crossentropy"),
              "severity": make_tensorflow_masked_loss("mae")},
        loss_weights={"classification": 1.0, "severity": 0.5})
    parameter_counts = {
        "total": int(model.count_params()),
        "trainable": int(sum(np.prod(variable.shape) for variable in model.trainable_variables)),
        "non_trainable": int(sum(np.prod(variable.shape) for variable in model.non_trainable_variables)),
    }
    d_counts = d_reference["parameter_counts"]
    if parameter_counts != d_counts:
        raise AssertionError(f"Model F parameter counts differ from D: {parameter_counts} != {d_counts}")
    backbone = model.get_layer("resnet50")
    if backbone.trainable or {id(v) for v in backbone.weights} & {id(v) for v in model.trainable_variables}:
        raise AssertionError("Model F ResNet50 must remain frozen")
    tower_ids = {
        "classification": {id(v) for name in ("classification_tower_dense_1", "classification_tower_dense_2",
                                                "classification") for v in model.get_layer(name).trainable_variables},
        "severity": {id(v) for name in ("severity_tower_dense_1", "severity_tower_dense_2", "severity_fraction")
                     for v in model.get_layer(name).trainable_variables},
    }
    if not tower_ids["classification"] or not tower_ids["severity"] or tower_ids["classification"] & tower_ids["severity"]:
        raise AssertionError("Model F task towers must be trainable and disjoint")
    severity_tower_layers = [model.get_layer(name) for name in
                             ("severity_tower_dense_1", "severity_tower_dense_2", "severity_fraction")]
    severity_tower_before = [weight.copy() for layer in severity_tower_layers for weight in layer.get_weights()]

    class_loss = make_tensorflow_masked_loss("binary_crossentropy")
    severity_loss = make_tensorflow_masked_loss("mae")
    before_monitor = _monitor_summary(model, monitor_images, monitor_targets)
    consumption = Counter()
    severity_records_consumed = []
    history = [{"step": 0, "phase": "before_training", "monitor": before_monitor}]
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        class_images, class_targets, domains = next(class_iterator)
        domain_ids = [value.decode("utf-8") for value in domains.numpy()]
        k = sum(domain == "flopwd" for domain in domain_ids)
        if k <= 0:
            raise AssertionError("classification batch must contain FloPWD examples for severity-count parity")
        severity_images_parts, severity_target_parts, severity_domain_ids = [], {"classification": [], "severity": []}, []
        for _ in range(k):
            image, targets, domain = next(severity_iterator)
            severity_images_parts.append(image)
            for key in severity_target_parts:
                severity_target_parts[key].append(targets[key])
            severity_domain_ids.append(domain.numpy()[0].decode("utf-8"))
        severity_images = tf.concat(severity_images_parts, axis=0)
        severity_targets = {key: tf.concat(value, axis=0) for key, value in severity_target_parts.items()}
        if any(domain != "flopwd" for domain in severity_domain_ids):
            raise AssertionError("UGV entered Model F severity stream")
        class_values = class_targets["classification"].numpy()
        class_mask = class_values[:, 1] > 0
        for domain in ("flopwd", "ugv"):
            mask = np.asarray(domain_ids) == domain
            consumption[f"classification_{domain}"] += int(mask.sum())
            consumption[f"classification_{domain}_positive"] += int(np.sum(mask & class_mask & (class_values[:, 0] == 1)))
            consumption[f"classification_{domain}_negative"] += int(np.sum(mask & class_mask & (class_values[:, 0] == 0)))
        severity_values = severity_targets["severity"].numpy()
        severity_available = severity_values[:, 1] > 0
        if not severity_available.all():
            raise AssertionError("an independent Model F severity draw lacks a real target")
        for target in severity_values[:, 0]:
            severity_records_consumed.append(float(target))
        step_result = task_decoupled_two_stream_train_step(
            model, model.optimizer, class_images, class_targets, severity_images, severity_targets,
            severity_domain_ids, class_loss, severity_loss, 1.0, 0.5)
        if step_result["severity_example_count"] != k:
            raise AssertionError("severity example count must equal K FloPWD classification examples")
        if (step_result["classification_cross_tower_gradient_norm"] != 0.0 or
                step_result["severity_cross_tower_gradient_norm"] != 0.0 or
                step_result["classification_tower_gradient_norm"] <= 0):
            raise AssertionError("Model F task-gradient isolation/participation check failed")
        consumption["classification_total"] += len(domain_ids)
        consumption["classification_positive"] += int(np.sum(class_mask & (class_values[:, 0] == 1)))
        consumption["severity_examples"] += k
        after_monitor = _monitor_summary(model, monitor_images, monitor_targets)
        history.append({"step": step, "classification_loss": step_result["classification_loss"],
                        "severity_loss": step_result["severity_loss"],
                        "severity_gradient_norm": step_result["severity_tower_gradient_norm"],
                        "classification_gradient_norm": step_result["classification_tower_gradient_norm"],
                        "monitor": after_monitor})
    elapsed = time.monotonic() - start

    if consumption["severity_examples"] != consumption["classification_flopwd"]:
        raise AssertionError("smoke severity count must equal FloPWD classification count")
    if consumption["classification_total"] != args.steps * 32:
        raise AssertionError("classification smoke stream did not consume the frozen batch size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "model.keras")
    severity_tower_after = [weight for layer in severity_tower_layers for weight in layer.get_weights()]
    severity_tower_changed = any(not np.array_equal(before, after)
                                 for before, after in zip(severity_tower_before, severity_tower_after))
    observed_severity = severity_distribution([
        type("ObservedSeverity", (), {"severity_target": value}) for value in severity_records_consumed])
    metadata = {
        "benchmark_status": "debug", "experiment_type": frozen["experiment_type"],
        "architecture_variant": "task_decoupled", "seed": 42,
        "classification_stream": frozen["classification_stream"],
        "severity_stream": frozen["severity_stream"],
        "optimizer": "Adam", "learning_rate": 0.001,
        "losses": {"classification": "binary_crossentropy", "severity": "MAE percentage points"},
        "loss_weights": {"classification": 1.0, "severity": 0.5},
        "severity_output": "sigmoid * 100; unchanged from D",
        "frozen_matrix_sha256": _sha(args.spec),
        "imagenet_weights_sha256": _sha(imagenet_weights),
        "flopwd_split_sha256": _sha(flo_manifest_path), "ugv_split_sha256": _sha(ugv_manifest_path),
        "git_commit_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                          text=True, check=True).stdout.strip(),
        "worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                                text=True, check=True).stdout.strip()),
        "parameter_counts": parameter_counts,
        "model_d_parameter_counts_equal": True,
        "severity_tower_weights_changed_during_smoke": severity_tower_changed,
        "debug_optimizer_updates": args.steps,
        "optimizer_iterations": int(model.optimizer.iterations.numpy()),
        "elapsed_seconds": elapsed,
        "classification_consumption": dict(consumption),
        "classification_domain_proportions": {
            "flopwd": consumption["classification_flopwd"] / consumption["classification_total"],
            "ugv": consumption["classification_ugv"] / consumption["classification_total"]},
        "classification_effective_positive_rate": consumption["classification_positive"] / consumption["classification_total"],
        "severity_consumption": observed_severity,
        "severity_examples_equal_classification_flopwd_count": True,
        "ugv_severity_examples": 0,
        "planned_full_run_severity_draw_count": planned_severity_count,
        "planned_full_run_severity_distribution": planned_distribution,
        "original_flopwd_severity_distribution": original_distribution,
        "model_d_balanced_cycle_severity_distribution": balanced_distribution,
        "expected_full_run_domain_sampling_stream_identical_to_d": True,
        "compute_note": frozen["compute_note"],
        "source_absolute_paths_saved": False,
        "test_set_used": False,
        "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__,
                    "keras": importlib.metadata.version("keras"), "numpy": np.__version__},
    }
    _write(args.output_dir / "metadata.json", metadata)
    _write(args.output_dir / "history.json", {"step_history": history})
    print(json.dumps({"metadata": metadata, "step_history": history}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
