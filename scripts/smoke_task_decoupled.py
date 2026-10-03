"""Smoke-test frozen Model D architecture and gradient isolation on real train data."""

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

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.config import load_config
from floating_plastic.data import heterogeneous_flopwd_record, heterogeneous_ugv_record, load_flopwd, load_ugv
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.losses import make_tensorflow_masked_loss
from floating_plastic.model import create_model
from floating_plastic.pipeline import make_heterogeneous_tf_dataset, make_multidomain_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split
from floating_plastic.training import masked_multitask_train_step


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _layers(model, prefix):
    names = ([f"{prefix}_dense_1", f"{prefix}_dense_2", "classification"]
             if prefix == "classification_tower" else
             [f"{prefix}_dense_1", f"{prefix}_dense_2", "severity_fraction"])
    return [model.get_layer(name) for name in names]


def _weights(layers):
    return [value.copy() for layer in layers for value in layer.get_weights()]


def _unchanged(before, after):
    return len(before) == len(after) and all(np.array_equal(left, right) for left, right in zip(before, after))


def _changed(before, after):
    return len(before) == len(after) and any(not np.array_equal(left, right) for left, right in zip(before, after))


def _finite_losses(model, batch, class_loss, severity_loss, class_weight, severity_weight):
    import tensorflow as tf
    images, targets = batch[:2]
    outputs = model(images, training=False)
    class_value = class_loss(targets["classification"], outputs["classification"])
    severity_value = severity_loss(targets["severity"], outputs["severity"])
    total = class_weight * class_value + severity_weight * severity_value
    values = {"classification": float(class_value.numpy()),
              "severity": float(severity_value.numpy()), "total": float(total.numpy())}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite masked loss in real data smoke batch")
    return values


def _severity_gradient_norm(model, batch, severity_loss, severity_weight):
    import tensorflow as tf
    images, targets = batch[:2]
    severity_variables = [variable for layer in _layers(model, "severity_tower")
                          for variable in layer.trainable_variables]
    with tf.GradientTape() as tape:
        outputs = model(images, training=False)
        value = float(severity_weight) * severity_loss(targets["severity"], outputs["severity"])
    gradients = tape.gradient(value, severity_variables)
    present = [gradient for gradient in gradients if gradient is not None]
    if not present:
        raise AssertionError("FloPWD severity loss produced no severity-tower gradients")
    norm = float(tf.linalg.global_norm(present).numpy())
    if not np.isfinite(norm) or norm <= 0:
        raise AssertionError(f"expected non-zero finite severity-tower gradient norm, got {norm}")
    return {"weighted_severity_loss": float(value.numpy()), "global_norm": norm,
            "gradient_tensor_count": len(present), "severity_trainable_variable_count": len(severity_variables)}


def _predict(model, dataset):
    predictions, names = [], []
    for images, _targets, domains in dataset:
        output = model(images, training=False)
        values = np.asarray(output["severity"], dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
            raise AssertionError("severity output is not finite and bounded in [0, 100]")
        predictions.extend(values.tolist())
        names.extend(value.decode("utf-8") for value in domains.numpy())
    return predictions, names


def _verify_frozen_d(spec, spec_path):
    d = spec["matrix"]["D"]
    shared = spec["shared_protocol"]
    expected = {
        "name": "task_decoupled_multidomain", "status": "frozen_not_run",
        "experiment_type": "multidomain_task_decoupled", "architecture_variant": "task_decoupled",
        "sources": ["flopwd", "ugv"], "domain_sampling": {"flopwd": 0.5, "ugv": 0.5},
        "seed": 42, "epochs": 15, "steps_per_epoch": 44, "total_optimizer_updates": 660,
        "batch_size": 32, "optimizer": "adam", "learning_rate": 0.001,
        "backbone": "resnet50_imagenet_frozen", "classification_threshold": 0.5,
        "classification_loss_weight": 1.0, "severity_loss_weight": 0.5,
        "preprocessing": "same_as_models_b_c",
    }
    mismatches = {key: (d.get(key), value) for key, value in expected.items() if d.get(key) != value}
    if d.get("flopwd_class_balance", {}).get("negative_to_positive") != "1:1":
        mismatches["flopwd_class_balance"] = (d.get("flopwd_class_balance"), "1:1")
    if d.get("split_manifests") != {"flopwd": shared["split_manifests"]["flopwd"],
                                     "ugv": shared["split_manifests"]["ugv"]}:
        mismatches["split_manifests"] = (d.get("split_manifests"), shared["split_manifests"])
    if shared["seed"] != 42 or shared["batch_size"] != 32 or shared["learning_rate"] != 0.001:
        mismatches["shared_protocol"] = (shared, "seed 42, batch 32, Adam 0.001")
    if shared["training_budget"] != {"steps_per_epoch": 44, "epochs": 15, "total_optimizer_steps": 660}:
        mismatches["training_budget"] = (shared["training_budget"], "15 x 44 = 660")
    if shared["loss_weights"] != {"classification": 1.0, "severity": 0.5} or shared["classification_threshold"] != 0.5:
        mismatches["loss_or_threshold"] = (shared["loss_weights"], "1.0 / 0.5; threshold .5")
    if mismatches:
        raise ValueError(f"frozen Model D specification mismatch: {mismatches}")
    return d, shared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flopwd-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/debug_task_decoupled_seed42"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size != 16 or args.steps != 8:
        parser.error("the frozen engineering smoke is fixed at batch size 16 and 8 steps (128 examples)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty debug run directory: {args.output_dir}")

    import tensorflow as tf

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    frozen_d, shared = _verify_frozen_d(spec, args.spec)
    config = load_config(args.config)
    if (config["model"]["backbone_trainable"] is not False or
            config["model"]["dense_units"] != shared["architecture"]["dense_units"] or
            config["model"]["dropout_rates"] != shared["architecture"]["dropout_rates"] or
            config["training"]["learning_rate"] != frozen_d["learning_rate"] or
            config["losses"]["classification_weight"] != frozen_d["classification_loss_weight"] or
            config["losses"]["severity_weight"] != frozen_d["severity_loss_weight"] or
            config["evaluation"]["classification_threshold"] != frozen_d["classification_threshold"]):
        raise ValueError("runtime config does not match frozen Model D shared protocol")
    tf.keras.utils.set_random_seed(int(frozen_d["seed"]))
    if config["reproducibility"].get("deterministic_tensorflow_ops", False):
        tf.config.experimental.enable_op_determinism()

    flo_records = load_flopwd(args.flopwd_dir)
    ugv_records = load_ugv(args.ugv_dir)
    flo_manifest_path = Path(frozen_d["split_manifests"]["flopwd"])
    ugv_manifest_path = Path(frozen_d["split_manifests"]["ugv"])
    if (_sha(flo_manifest_path) != shared["split_manifests"]["flopwd_sha256"] or
            _sha(ugv_manifest_path) != shared["split_manifests"]["ugv_sha256"]):
        raise ValueError("split manifest SHA does not match frozen shared protocol")
    flo_manifest = load_split_manifest(flo_manifest_path, [record.filename for record in flo_records])
    ugv_manifest = load_ugv_grouped_split_manifest(ugv_manifest_path, ugv_records)
    flo_train_source = records_for_split(flo_records, flo_manifest, "train")
    flo_val_source = records_for_split(flo_records, flo_manifest, "validation")
    ugv_train_source = records_for_ugv_split(ugv_records, ugv_manifest, "train")
    flo_train = [heterogeneous_flopwd_record(record) for record in flo_train_source]
    ugv_train = [heterogeneous_ugv_record(record) for record in ugv_train_source]
    if any(not record.classification_target_available for record in ugv_train):
        raise ValueError("all sampled UGV training records must have eligible annotated-waste labels")
    if any(record.severity_target_available for record in ugv_train):
        raise ValueError("UGV severity must remain unavailable")

    severity_targets = np.asarray([record.severity_percent for record in flo_train_source], dtype=np.float64)
    distribution = {
        "sample_count": int(len(severity_targets)), "mean": float(np.mean(severity_targets)),
        "median": float(np.median(severity_targets)), "standard_deviation": float(np.std(severity_targets)),
        "minimum": float(np.min(severity_targets)), "maximum": float(np.max(severity_targets)),
        "exactly_zero_count": int(np.count_nonzero(severity_targets == 0)),
        "exactly_zero_percentage": float(np.mean(severity_targets == 0) * 100),
        "p25": float(np.quantile(severity_targets, 0.25)), "p75": float(np.quantile(severity_targets, 0.75)),
        "p90": float(np.quantile(severity_targets, 0.90)),
        "split": "FloPWD train only",
    }

    size = tuple(config["data"]["image_size"])
    flo_batch_records = sorted(flo_train, key=lambda record: record.filename)[:args.batch_size]
    ugv_batch_records = sorted(ugv_train, key=lambda record: record.filename)[:args.batch_size]
    flo_batch = next(iter(make_heterogeneous_tf_dataset(flo_batch_records, args.batch_size, size,
                                                        include_domain_id=True)))
    ugv_batch = next(iter(make_heterogeneous_tf_dataset(ugv_batch_records, args.batch_size, size,
                                                        include_domain_id=True)))
    mixed_data, _ = make_multidomain_tf_dataset(
        {"flopwd": flo_train, "ugv": ugv_train}, args.batch_size, size,
        training=True, seed=int(frozen_d["seed"]), strategy="balanced",
        flopwd_negative_to_positive=1, include_domain_id=True)
    mixed_iterator = iter(mixed_data)
    mixed_batch = next(mixed_iterator)
    for _ in range(8):
        if set(value.decode("utf-8") for value in mixed_batch[2].numpy()) == {"flopwd", "ugv"}:
            break
        mixed_batch = next(mixed_iterator)

    # Count the unchanged shared model and the new variant with actual model graphs.
    model_config = config["model"]
    shared_model = create_model(input_shape=(*size, 3), weights=None,
                                backbone_trainable=False, dense_units=model_config["dense_units"],
                                dropout_rates=model_config["dropout_rates"], architecture_variant="shared_tower")
    shared_count = {"total": int(shared_model.count_params()),
                    "trainable": int(sum(np.prod(v.shape) for v in shared_model.trainable_variables)),
                    "non_trainable": int(sum(np.prod(v.shape) for v in shared_model.non_trainable_variables))}
    del shared_model
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(frozen_d["seed"]))

    imagenet_weights = (Path(__file__).resolve().parents[1] / ".keras-cache" / ".keras" /
                        "models" / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    if not imagenet_weights.is_file():
        raise FileNotFoundError("cached ImageNet ResNet50 weights are unavailable")
    model = create_model(input_shape=(*size, 3), weights=str(imagenet_weights),
                         backbone_trainable=False, dense_units=model_config["dense_units"],
                         dropout_rates=model_config["dropout_rates"],
                         architecture_variant="task_decoupled")
    class_loss = make_tensorflow_masked_loss("binary_crossentropy")
    severity_loss = make_tensorflow_masked_loss("mae")
    learning_rate = float(frozen_d["learning_rate"])
    class_weight = float(frozen_d["classification_loss_weight"])
    severity_weight = float(frozen_d["severity_loss_weight"])

    def compile_model():
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss={"classification": class_loss, "severity": severity_loss},
                      loss_weights={"classification": class_weight, "severity": severity_weight})

    compile_model()
    if model.get_layer("resnet50").trainable or len(model.get_layer("resnet50").trainable_variables):
        raise AssertionError("ResNet50 must remain frozen")
    model(np.zeros((1, *size, 3), dtype=np.float32), training=False)
    backbone = model.get_layer("resnet50")
    classification_layers = _layers(model, "classification_tower")
    severity_layers = _layers(model, "severity_tower")
    backbone_initial = _weights([backbone])
    initial_model_weights = [weight.copy() for weight in model.get_weights()]
    model_counts = {
        "total": int(model.count_params()),
        "trainable": int(sum(np.prod(variable.shape) for variable in model.trainable_variables)),
        "non_trainable": int(sum(np.prod(variable.shape) for variable in model.non_trainable_variables)),
    }
    if model_counts["total"] != model_counts["trainable"] + model_counts["non_trainable"]:
        raise AssertionError("parameter counts do not add up")

    class_batch_losses = _finite_losses(model, flo_batch, class_loss, severity_loss, class_weight, severity_weight)
    ugv_batch_losses = _finite_losses(model, ugv_batch, class_loss, severity_loss, class_weight, severity_weight)
    mixed_batch_losses = _finite_losses(model, mixed_batch, class_loss, severity_loss, class_weight, severity_weight)
    if ugv_batch_losses["severity"] != 0.0:
        raise AssertionError("UGV-only masked severity loss must be exactly zero")
    mixed_ids = [value.decode("utf-8") for value in mixed_batch[2].numpy()]
    mixed_severity_available = int(mixed_batch[1]["severity"][:, 1].numpy().sum())
    if mixed_severity_available != sum(value == "flopwd" for value in mixed_ids):
        raise AssertionError("only FloPWD rows in the mixed batch may supervise severity")
    if not (class_batch_losses["severity"] >= 0 and ugv_batch_losses["classification"] >= 0):
        raise AssertionError("masked loss smoke values are not valid")

    gradient_norm = _severity_gradient_norm(model, flo_batch, severity_loss, severity_weight)

    # Explicit real-data optimizer probes. They are reset before the 8-step smoke.
    ugv_class_before = _weights(classification_layers)
    ugv_severity_before = _weights(severity_layers)
    ugv_backbone_before = _weights([backbone])
    ugv_step = masked_multitask_train_step(model, model.optimizer, ugv_batch[0], ugv_batch[1],
                                           class_loss, severity_loss, class_weight, severity_weight)
    ugv_class_changed = _changed(ugv_class_before, _weights(classification_layers))
    ugv_severity_unchanged = _unchanged(ugv_severity_before, _weights(severity_layers))
    ugv_backbone_unchanged = _unchanged(ugv_backbone_before, _weights([backbone]))
    if not (ugv_class_changed and ugv_severity_unchanged and ugv_backbone_unchanged):
        raise AssertionError("UGV class-only update violated task or frozen-backbone isolation")

    flo_severity_before = _weights(severity_layers)
    flo_step = masked_multitask_train_step(model, model.optimizer, flo_batch[0], flo_batch[1],
                                           class_loss, severity_loss, class_weight, severity_weight)
    flo_severity_changed = _changed(flo_severity_before, _weights(severity_layers))
    flo_backbone_unchanged = _unchanged(backbone_initial, _weights([backbone]))
    if not (flo_severity_changed and flo_backbone_unchanged):
        raise AssertionError("FloPWD severity supervision did not update the severity tower as expected")

    # The probes are engineering checks, not part of the 8-step smoke weights.
    model.set_weights(initial_model_weights)
    compile_model()
    fixed_validation = sorted((heterogeneous_flopwd_record(record) for record in flo_val_source),
                              key=lambda record: record.filename)[:args.batch_size]
    if not fixed_validation:
        raise ValueError("FloPWD validation split has no records for the fixed prediction check")
    fixed_val_dataset = make_heterogeneous_tf_dataset(fixed_validation, len(fixed_validation), size,
                                                      include_domain_id=True)
    before_prediction, before_domains = _predict(model, fixed_val_dataset)
    validation_filenames = [record.filename for record in fixed_validation]

    train_data, _ = make_multidomain_tf_dataset(
        {"flopwd": flo_train, "ugv": ugv_train}, args.batch_size, size,
        training=True, seed=int(frozen_d["seed"]), strategy="balanced",
        flopwd_negative_to_positive=1, include_domain_id=True)
    train_iterator = iter(train_data)
    smoke_initial_severity = _weights(severity_layers)
    history, consumed = [], Counter()
    consumed_severity_supervised = consumed_severity_unsupervised = 0
    class_counts = Counter()
    for step in range(1, args.steps + 1):
        batch = next(train_iterator)
        result = masked_multitask_train_step(model, model.optimizer, batch[0], batch[1],
                                             class_loss, severity_loss, class_weight, severity_weight)
        domains = [value.decode("utf-8") for value in batch[2].numpy()]
        classes = batch[1]["classification"].numpy()
        severity = batch[1]["severity"].numpy()
        for index, domain in enumerate(domains):
            consumed[domain] += 1
            if classes[index, 1] > 0:
                class_counts[f"{domain}_{'positive' if classes[index, 0] == 1 else 'negative'}"] += 1
            if severity[index, 1] > 0:
                consumed_severity_supervised += 1
            else:
                consumed_severity_unsupervised += 1
        result["step"] = step
        history.append(result)
    smoke_severity_changed = _changed(smoke_initial_severity, _weights(severity_layers))
    if not smoke_severity_changed or not consumed_severity_supervised:
        raise AssertionError("real smoke steps did not update the severity tower from FloPWD supervision")

    after_prediction, after_domains = _predict(model, fixed_val_dataset)
    prediction_delta = np.abs(np.asarray(after_prediction) - np.asarray(before_prediction))
    if (not np.isfinite(prediction_delta).all() or not np.any(prediction_delta > 1e-9) or
            len(set(np.round(before_prediction, 9))) < 2 or len(set(np.round(after_prediction, 9))) < 2):
        raise AssertionError("severity predictions were constant or did not respond to smoke training")
    if before_domains != after_domains or len(before_prediction) != len(after_prediction):
        raise AssertionError("fixed validation prediction records changed between before/after checks")
    examples = sum(consumed.values())
    proportions = {key: value / examples for key, value in sorted(consumed.items())}
    flo_count, ugv_count = consumed["flopwd"], consumed["ugv"]
    flo_neg, flo_pos = class_counts["flopwd_negative"], class_counts["flopwd_positive"]
    if not (flo_count and ugv_count and flo_neg and flo_pos):
        raise AssertionError("smoke failed to consume both domains and FloPWD classes")
    ratio = flo_neg / flo_pos
    if not 0.5 <= ratio <= 2.0:
        raise AssertionError(f"small-batch observed FloPWD ratio is unexpectedly far from 1:1: {ratio}")
    if consumed_severity_supervised != flo_count or consumed_severity_unsupervised != ugv_count:
        raise AssertionError("severity availability must be exactly FloPWD-only in smoke consumption")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    matrix_sha = _sha(args.spec)
    config_out = {
        "experiment_type": frozen_d["experiment_type"],
        "architecture_variant": "task_decoupled",
        "benchmark_status": "debug",
        "frozen_spec": {"filename": args.spec.name, "sha256": matrix_sha},
        "protocol": {key: frozen_d[key] for key in (
            "sources", "domain_sampling", "flopwd_class_balance", "seed", "epochs",
            "steps_per_epoch", "total_optimizer_updates", "batch_size", "optimizer",
            "learning_rate", "backbone", "classification_threshold",
            "classification_loss_weight", "severity_loss_weight", "preprocessing", "split_manifests")},
        "smoke_steps": args.steps, "smoke_batch_size": args.batch_size,
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "benchmark_status": "debug",
        "experiment_type": frozen_d["experiment_type"], "architecture_variant": "task_decoupled",
        "git_commit_sha": commit, "frozen_spec": {"filename": args.spec.name, "sha256": matrix_sha},
        "python_version": platform.python_version(), "tensorflow_version": tf.__version__,
        "keras_version": importlib.metadata.version("keras"), "numpy_version": np.__version__,
        "seed": int(frozen_d["seed"]), "steps": int(args.steps), "batch_size": int(args.batch_size),
        "optimizer": "Adam", "learning_rate": learning_rate, "loss_weights": {
            "classification": class_weight, "severity": severity_weight},
        "domain_sampling": {"strategy": "balanced", "probability": {"flopwd": 0.5, "ugv": 0.5}},
        "flopwd_class_balance": "1:1 negative:positive within FloPWD train only",
        "split_manifests": {"flopwd": {"filename": flo_manifest_path.name, "sha256": _sha(flo_manifest_path)},
                            "ugv": {"filename": ugv_manifest_path.name, "sha256": _sha(ugv_manifest_path)}},
        "target_distribution": distribution,
        "parameter_counts": {"shared_tower": shared_count, "task_decoupled": model_counts},
        "gradient_checks": {
            "floPWD_severity_gradient": gradient_norm,
            "ugv_classification_step": {"classification_tower_changed": ugv_class_changed,
                                        "severity_tower_unchanged": ugv_severity_unchanged,
                                        "backbone_unchanged": ugv_backbone_unchanged,
                                        "masked_losses": ugv_step},
            "floPWD_severity_step": {"severity_tower_changed": flo_severity_changed,
                                     "backbone_unchanged": flo_backbone_unchanged,
                                     "masked_losses": flo_step},
            "probe_updates_reset_before_smoke": True,
        },
        "examples_consumed": dict(consumed), "observed_domain_proportions": proportions,
        "flopwd_class_consumption": {"negative": flo_neg, "positive": flo_pos,
                                     "negative_to_positive_ratio": ratio},
        "severity_supervised_examples": consumed_severity_supervised,
        "severity_unsupervised_examples": consumed_severity_unsupervised,
        "ugv_severity_available_examples": 0,
        "severity_tower_changed_during_smoke": smoke_severity_changed,
        "model_reload_verified": False,
        "source_absolute_paths_saved": False,
    }
    _write(args.output_dir / "config.json", config_out)
    _write(args.output_dir / "history.json", {"step": history, "total_examples": examples})
    _write(args.output_dir / "severity_target_distribution.json", distribution)
    _write(args.output_dir / "gradient_checks.json", metadata["gradient_checks"])
    _write(args.output_dir / "smoke_behavior.json", {
        "scope": "fixed small FloPWD validation batch predictions only; no performance metrics",
        "filenames": validation_filenames,
        "predictions_before": before_prediction, "predictions_after": after_prediction,
        "max_absolute_change": float(prediction_delta.max()),
        "before_standard_deviation": float(np.std(before_prediction)),
        "after_standard_deviation": float(np.std(after_prediction)),
        "finite_and_bounded_0_100": True, "not_constant_or_hard_coded": True,
        "severity_tower_changed_during_smoke": smoke_severity_changed,
    })
    _write(args.output_dir / "metadata.json", metadata)
    model.save(args.output_dir / "model.keras")
    print(json.dumps(metadata, indent=2))
    print(f"Saved Model D DEBUG smoke artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
