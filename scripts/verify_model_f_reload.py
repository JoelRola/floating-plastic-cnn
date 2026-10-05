"""Reload debug Model F in a fresh process and check historical checkpoints."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.data import heterogeneous_flopwd_record, heterogeneous_ugv_record, load_flopwd, load_ugv
from floating_plastic.group_splits import load_ugv_grouped_split_manifest, records_for_ugv_split
from floating_plastic.model import custom_objects
from floating_plastic.pipeline import make_heterogeneous_tf_dataset
from floating_plastic.splits import load_split_manifest, records_for_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/debug_model_f_seed42"))
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    args = parser.parse_args(argv)
    import tensorflow as tf
    import yaml

    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    matrix_hash = hashlib.sha256(args.spec.read_bytes()).hexdigest()
    if (metadata["benchmark_status"] != "debug" or metadata["architecture_variant"] != "task_decoupled" or
            metadata["severity_stream"]["sampling"] != "original_unbalanced_distribution" or
            metadata["frozen_matrix_sha256"] != matrix_hash):
        raise AssertionError("Model F reload metadata/spec provenance mismatch")
    frozen = spec["matrix"]["F"]
    if frozen["status"] != "frozen_not_run" or metadata["experiment_type"] != frozen["experiment_type"]:
        raise AssertionError("Model F smoke does not match its frozen identity")

    shared = spec["shared_protocol"]
    flo = load_flopwd(args.data_dir)
    ugv = load_ugv(args.ugv_dir)
    fm = load_split_manifest(Path(shared["split_manifests"]["flopwd"]), [r.filename for r in flo])
    um = load_ugv_grouped_split_manifest(Path(shared["split_manifests"]["ugv"]), ugv)
    flo_train = sorted(records_for_split(flo, fm, "train"), key=lambda r: r.filename)
    ugv_train = sorted(records_for_ugv_split(ugv, um, "train"), key=lambda r: r.filename)
    records = {"flopwd": heterogeneous_flopwd_record(flo_train[0]),
               "ugv": heterogeneous_ugv_record(ugv_train[0])}
    model = tf.keras.models.load_model(args.run_dir / "model.keras", compile=True,
                                       custom_objects=custom_objects())
    if model.name != "floating_plastic_resnet50_multitask_task_decoupled":
        raise AssertionError("reloaded F architecture is not task-decoupled")
    model.get_layer("classification_tower_dense_1")
    model.get_layer("severity_tower_dense_1")
    loss_tasks = {name: getattr(loss, "task", None) for name, loss in model.loss.items()}
    if loss_tasks != {"classification": "binary_crossentropy", "severity": "mae"}:
        raise AssertionError(f"F masked losses failed to deserialize: {loss_tasks}")
    backbone = model.get_layer("resnet50")
    trainable = {id(v) for v in model.trainable_variables}
    if backbone.trainable_variables or trainable & {id(v) for v in backbone.weights}:
        raise AssertionError("reloaded F has trainable ResNet variables")
    backbone.trainable = False
    predictions = {}
    for domain, record in records.items():
        ds = make_heterogeneous_tf_dataset([record], 1, (224, 224))
        images, _ = next(iter(ds))
        output = model(images, training=False)
        probability = float(np.asarray(output["classification"]).reshape(-1)[0])
        severity = float(np.asarray(output["severity"]).reshape(-1)[0])
        if not np.isfinite(probability) or not 0 <= probability <= 1 or not np.isfinite(severity) or not 0 <= severity <= 100:
            raise AssertionError("reloaded F predictions are invalid")
        predictions[domain] = {"classification_probability": probability,
                               "severity_percentage_points": severity}
    del model
    tf.keras.backend.clear_session()

    compatibility = {}
    checkpoints = {
        "A": "runs/flopwd_original_seed42/model.keras",
        "B": "runs/multidomain_seed42/model.keras",
        "C": "runs/multidomain_class_balanced_seed42/model.keras",
        "D": "runs/task_decoupled_multidomain_seed42/model.keras",
        "E": "runs/multidomain_2to1_seed42/model.keras",
    }
    for label, checkpoint in checkpoints.items():
        existing = tf.keras.models.load_model(checkpoint, compile=True, custom_objects=custom_objects())
        expected_name = ("floating_plastic_resnet50_multitask_task_decoupled" if label == "D"
                         else "floating_plastic_resnet50_multitask")
        if existing.name != expected_name:
            raise AssertionError(f"existing Model {label} checkpoint did not load as expected")
        compatibility[label] = {"loaded": True, "model_name": existing.name}
        del existing
        tf.keras.backend.clear_session()

    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "fresh_python_process": True,
        "model_f": {"loaded": True, "architecture_variant": "task_decoupled", "losses": loss_tasks,
                    "backbone_frozen": True, "training_record_inference": predictions},
        "existing_models": compatibility, "test_set_used": False,
    }
    (args.run_dir / "reload_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata["reload_verified"] = True
    metadata["backward_compatibility"] = compatibility
    (args.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
