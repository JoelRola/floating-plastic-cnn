"""Run a tiny real-data Model B smoke check; this is never a benchmark run."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.config import load_config
from floating_plastic.data import (
    heterogeneous_flopwd_record, heterogeneous_ugv_record, load_flopwd, load_ugv,
)
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.losses import make_tensorflow_masked_loss
from floating_plastic.metrics import multidomain_evaluation
from floating_plastic.model import create_model
from floating_plastic.pipeline import (
    domain_sampling_weights, make_heterogeneous_tf_dataset, make_multidomain_tf_dataset,
)
from floating_plastic.splits import load_split_manifest, records_for_split


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_numpy(value):
    return value.numpy()


def _stats(batch):
    images, targets, domains = batch
    decoded_domains = [v.decode("utf-8") for v in _as_numpy(domains)]
    return {
        "image_shape": list(images.shape),
        "classification_target_shape": list(targets["classification"].shape),
        "classification_available_count": int(_as_numpy(targets["classification"][:, 1]).sum()),
        "severity_available_count": int(_as_numpy(targets["severity"][:, 1]).sum()),
        "source_domains": sorted(set(decoded_domains)),
        "domain_counts": dict(Counter(decoded_domains)),
    }


def _select_flo_validation(records, per_class=16):
    selected = []
    for label in (0, 1):
        candidates = [record for record in records if record.binary_label == label]
        selected.extend(candidates[:per_class])
    if len({sample.binary_label for sample in selected}) != 2:
        raise ValueError("smoke evaluation requires validation samples from both FloPWD classes")
    return selected


def _predict_batch(model, dataset):
    images_all, y_class, y_severity, probs, severity = [], [], [], [], []
    for images, targets, _domain in dataset:
        outputs = model(images, training=False)
        images_all.append(len(images))
        class_targets = _as_numpy(targets["classification"])
        severity_targets = _as_numpy(targets["severity"])
        y_class.extend(class_targets[:, 0].tolist())
        y_severity.extend(severity_targets[:, 0].tolist())
        probs.extend(_as_numpy(outputs["classification"]).reshape(-1).tolist())
        severity.extend(_as_numpy(outputs["severity"]).reshape(-1).tolist())
    return {"count": sum(images_all), "y_class": y_class, "y_severity": y_severity,
            "probabilities": probs, "severity_predictions": severity}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flopwd-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--flopwd-split", type=Path, default=Path("experiments/splits/flopwd_seed42.json"))
    parser.add_argument("--ugv-split", type=Path, default=Path("experiments/splits/ugv_grouped_seed42.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/debug_multidomain_smoke"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.batch_size < 2 or args.steps < 1:
        parser.error("--batch-size must be at least 2 and --steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    import tensorflow as tf

    config = load_config(args.config)
    profile_config = json.loads(json.dumps(config))
    with Path("configs/experiments.yaml").open(encoding="utf-8") as stream:
        import yaml
        model_b = yaml.safe_load(stream)["profiles"]["multidomain"]
    if model_b["domain_sampling"]["strategy"] != "balanced" or model_b["class_balancing"]["enabled"]:
        raise ValueError("Model B smoke must use balanced domains and no class balancing")

    tf.keras.utils.set_random_seed(args.seed)
    if config["reproducibility"].get("deterministic_tensorflow_ops", False):
        tf.config.experimental.enable_op_determinism()

    flo_records = load_flopwd(args.flopwd_dir)
    flo_manifest = load_split_manifest(args.flopwd_split, [r.filename for r in flo_records])
    ugv_records = load_ugv(args.ugv_dir)
    ugv_manifest = load_ugv_grouped_split_manifest(args.ugv_split, ugv_records)
    flo_by_split = {split: records_for_split(flo_records, flo_manifest, split)
                    for split in ("train", "validation", "test")}
    ugv_by_split = {split: records_for_ugv_split(ugv_records, ugv_manifest, split)
                    for split in ("train", "validation", "test")}
    flo_train = [heterogeneous_flopwd_record(r) for r in flo_by_split["train"]]
    ugv_train = [heterogeneous_ugv_record(r) for r in ugv_by_split["train"]]
    flo_val = [heterogeneous_flopwd_record(r) for r in _select_flo_validation(flo_by_split["validation"])]
    ugv_val = [heterogeneous_ugv_record(r) for r in ugv_by_split["validation"][:16]]
    if any(not r.classification_target_available for r in ugv_train + ugv_val):
        raise ValueError("unexpected unavailable UGV labels in selected non-empty annotated records")

    print("Resolved supervision: FloPWD=(plastic presence available, image-area coverage available); "
          "UGV=(annotated waste present available, severity unavailable).")
    print("Model B sampling: balanced domain sampling (0.5 expected probability per domain); "
          "FloPWD class balancing: none.")
    weights = domain_sampling_weights({"flopwd": len(flo_train), "ugv": len(ugv_train)}, "balanced")
    print(f"Training source records: FloPWD={len(flo_train)}, UGV={len(ugv_train)}; sampling weights={weights}")

    image_size = tuple(config["data"]["image_size"])
    flo_batch = next(iter(make_heterogeneous_tf_dataset(flo_train[:args.batch_size], args.batch_size,
                                                        image_size, include_domain_id=True)))
    ugv_batch = next(iter(make_heterogeneous_tf_dataset(ugv_train[:args.batch_size], args.batch_size,
                                                        image_size, include_domain_id=True)))
    print("Real FloPWD batch:", json.dumps(_stats(flo_batch), sort_keys=True))
    print("Real UGV batch:", json.dumps(_stats(ugv_batch), sort_keys=True))

    model_config = config["model"]
    model = create_model(input_shape=(*image_size, 3), weights=model_config["weights"],
                         backbone_trainable=model_config["backbone_trainable"],
                         dense_units=model_config["dense_units"], dropout_rates=model_config["dropout_rates"])
    class_loss = make_tensorflow_masked_loss("binary_crossentropy")
    severity_loss = make_tensorflow_masked_loss("mae")
    class_weight = float(config["losses"]["classification_weight"])
    severity_weight = float(config["losses"]["severity_weight"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["training"]["learning_rate"])),
        loss={"classification": class_loss, "severity": severity_loss},
        loss_weights={"classification": class_weight, "severity": severity_weight},
    )

    def calculate_losses(batch, training=False):
        images, targets = batch[:2]
        outputs = model(images, training=training)
        lc = class_loss(targets["classification"], outputs["classification"])
        ls = severity_loss(targets["severity"], outputs["severity"])
        total = class_weight * lc + severity_weight * ls
        values = {"classification": float(lc.numpy()), "severity": float(ls.numpy()),
                  "total": float(total.numpy())}
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError("non-finite masked loss on real batch")
        return total, values

    _, flo_losses = calculate_losses(flo_batch)
    _, ugv_losses = calculate_losses(ugv_batch)
    mixed_data, _ = make_multidomain_tf_dataset(
        {"flopwd": flo_train, "ugv": ugv_train}, args.batch_size, image_size,
        training=True, seed=args.seed, strategy="balanced", include_domain_id=True)
    mixed_batch = next(iter(mixed_data))
    _, mixed_losses = calculate_losses(mixed_batch)
    mixed_stats = _stats(mixed_batch)
    print("Masked loss smoke:", json.dumps({"flopwd": flo_losses, "ugv": ugv_losses,
                                           "mixed": mixed_losses}, sort_keys=True))
    print("Mixed real batch masks/domains:", json.dumps(mixed_stats, sort_keys=True))
    if mixed_stats["severity_available_count"] != mixed_stats["domain_counts"].get("flopwd", 0):
        raise AssertionError("mixed batch severity availability must match FloPWD-only samples")
    if ugv_losses["severity"] != 0.0:
        raise AssertionError(f"UGV unavailable severity loss must be 0, got {ugv_losses['severity']}")
    if flo_losses["severity"] <= 0 or ugv_losses["classification"] <= 0:
        raise AssertionError("expected both FloPWD losses and UGV classification loss to contribute")

    train_data, _ = make_multidomain_tf_dataset(
        {"flopwd": flo_train, "ugv": ugv_train}, args.batch_size, image_size,
        training=True, seed=args.seed, strategy="balanced", include_domain_id=True)
    optimizer = model.optimizer
    history = {"total_loss": [], "classification_loss": [], "severity_loss": []}
    consumed = Counter()
    for batch_number, batch in enumerate(train_data.take(args.steps), start=1):
        images, targets, domains = batch
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            lc = class_loss(targets["classification"], outputs["classification"])
            ls = severity_loss(targets["severity"], outputs["severity"])
            total = class_weight * lc + severity_weight * ls
        gradients = tape.gradient(total, model.trainable_variables)
        pairs = [(grad, var) for grad, var in zip(gradients, model.trainable_variables) if grad is not None]
        if not pairs or not bool(tf.math.is_finite(total).numpy()):
            raise FloatingPointError(f"invalid training step {batch_number}")
        optimizer.apply_gradients(pairs)
        ids = [value.decode("utf-8") for value in domains.numpy()]
        consumed.update(ids)
        history["total_loss"].append(float(total.numpy()))
        history["classification_loss"].append(float(lc.numpy()))
        history["severity_loss"].append(float(ls.numpy()))
    examples_seen = sum(consumed.values())
    observed = {domain: count / examples_seen for domain, count in sorted(consumed.items())}
    print(f"Debug optimizer steps: {args.steps}; examples/domain: {dict(consumed)}; observed proportions: {observed}")
    if set(consumed) != {"flopwd", "ugv"}:
        raise AssertionError("tiny run failed to sample from both domains")

    # Separate validation reports use real held-out validation records only.
    flo_val_ds = make_heterogeneous_tf_dataset(flo_val, 32, image_size, include_domain_id=True)
    ugv_val_ds = make_heterogeneous_tf_dataset(ugv_val, 16, image_size, include_domain_id=True)
    flo_result = _predict_batch(model, flo_val_ds)
    ugv_result = _predict_batch(model, ugv_val_ds)
    evaluation = multidomain_evaluation(
        [int(v) for v in flo_result["y_class"]], flo_result["probabilities"],
        flo_result["y_severity"], flo_result["severity_predictions"], ugv_result["probabilities"],
        threshold=float(config["evaluation"]["classification_threshold"]),
    )
    evaluation["scope"] = "debug validation batches only; not benchmark evaluation"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_config["experiment_profile"] = "multidomain"
    profile_config["debug"] = True
    profile_config["domain_sampling"] = model_b["domain_sampling"]
    profile_config["class_balancing"] = model_b["class_balancing"]
    _json_write(args.output_dir / "config.json", profile_config)
    _json_write(args.output_dir / "history.json", history)
    _json_write(args.output_dir / "evaluation_smoke.json", evaluation)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_status": "debug",
        "experiment_type": "multidomain",
        "domain_sampling": "balanced",
        "class_balance": "none",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "seed": int(args.seed),
        "git_commit_sha": commit,
        "split_manifests": {"flopwd": args.flopwd_split.name, "ugv": args.ugv_split.name},
        "train_source_record_counts": {"flopwd": len(flo_train), "ugv": len(ugv_train)},
        "examples_consumed": dict(consumed),
        "observed_domain_proportions": observed,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "preprocessing": "keras.applications.resnet.preprocess_input embedded in model",
        "ugv_target": "annotated waste present; positive-only eligible population",
        "ugv_severity_target": None,
        "absolute_dataset_paths_saved": False,
    }
    _json_write(args.output_dir / "metadata.json", metadata)
    model.save(args.output_dir / "model.keras")
    print(f"Saved DEBUG smoke artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
