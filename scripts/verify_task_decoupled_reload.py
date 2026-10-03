"""Fresh-process Model D inference and A/B/C/E checkpoint compatibility check."""

import argparse
from datetime import datetime, timezone
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
    parser.add_argument("--flopwd-dir", type=Path, required=True)
    parser.add_argument("--ugv-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/debug_task_decoupled_seed42"))
    parser.add_argument("--spec", type=Path, default=Path("experiments/specs/multidomain_matrix_seed42.yaml"))
    args = parser.parse_args(argv)

    import tensorflow as tf
    import yaml
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    shared = spec["shared_protocol"]
    flo_records = load_flopwd(args.flopwd_dir)
    ugv_records = load_ugv(args.ugv_dir)
    flo_manifest = load_split_manifest(Path(shared["split_manifests"]["flopwd"]),
                                       [record.filename for record in flo_records])
    ugv_manifest = load_ugv_grouped_split_manifest(Path(shared["split_manifests"]["ugv"]), ugv_records)
    flo_train = sorted(records_for_split(flo_records, flo_manifest, "train"), key=lambda row: row.filename)
    ugv_train = sorted(records_for_ugv_split(ugv_records, ugv_manifest, "train"), key=lambda row: row.filename)
    one_flo = heterogeneous_flopwd_record(flo_train[0])
    one_ugv = heterogeneous_ugv_record(ugv_train[0])
    size = (224, 224)

    custom = custom_objects()
    d_model = tf.keras.models.load_model(args.run_dir / "model.keras", custom_objects=custom, compile=True)
    if d_model.name != "floating_plastic_resnet50_multitask_task_decoupled":
        raise AssertionError(f"reloaded Model D has unexpected name {d_model.name}")
    d_model.get_layer("classification_tower_dense_1")
    d_model.get_layer("severity_tower_dense_1")
    if not isinstance(d_model.loss, dict) or set(d_model.loss) != {"classification", "severity"}:
        raise AssertionError("reloaded Model D masked losses did not deserialize")
    loss_tasks = {name: getattr(loss, "task", None) for name, loss in d_model.loss.items()}
    if loss_tasks != {"classification": "binary_crossentropy", "severity": "mae"}:
        raise AssertionError(f"unexpected reloaded Model D loss configuration: {loss_tasks}")

    def predict_one(record):
        dataset = make_heterogeneous_tf_dataset([record], 1, size)
        images, _targets = next(iter(dataset))
        output = d_model(images, training=False)
        probability = float(np.asarray(output["classification"]).reshape(-1)[0])
        severity = float(np.asarray(output["severity"]).reshape(-1)[0])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise AssertionError("reloaded D classification inference is not finite/in [0,1]")
        if not np.isfinite(severity) or not 0 <= severity <= 100:
            raise AssertionError("reloaded D severity inference is not finite/in [0,100]")
        return {"classification_probability": probability, "severity_percentage_points": severity}

    d_predictions = {"flopwd": predict_one(one_flo), "ugv": predict_one(one_ugv)}
    backbone = d_model.get_layer("resnet50")
    backbone_weight_ids = {id(variable) for variable in backbone.weights}
    model_trainable_ids = {id(variable) for variable in d_model.trainable_variables}
    backbone_trainable_ids = {id(variable) for variable in backbone.trainable_variables}
    # Keras 2.15 reloads the nested Model wrapper with trainable=True even though
    # every ResNet weight remains excluded from the parent model's trainable
    # variables. Reassert the frozen wrapper state and validate the actual
    # optimization boundary by variable identity.
    if backbone_trainable_ids or backbone_weight_ids & model_trainable_ids:
        raise AssertionError("reloaded D exposes a ResNet50 weight as trainable")
    backbone.trainable = False
    if backbone.trainable or backbone.trainable_variables:
        raise AssertionError("reloaded D backbone could not be explicitly kept frozen")
    del d_model
    tf.keras.backend.clear_session()

    compatibility = {}
    checkpoint_paths = {
        "A": Path("runs/flopwd_original_seed42/model.keras"),
        "B": Path("runs/multidomain_seed42/model.keras"),
        "C": Path("runs/multidomain_class_balanced_seed42/model.keras"),
        "E": Path("runs/multidomain_2to1_seed42/model.keras"),
    }
    for letter, checkpoint in checkpoint_paths.items():
        model = tf.keras.models.load_model(checkpoint, custom_objects=custom, compile=True)
        if model.name != "floating_plastic_resnet50_multitask":
            raise AssertionError(f"existing Model {letter} no longer loads as shared_tower: {model.name}")
        model.get_layer("shared_dense_1")
        if any("classification_tower" in layer.name or "severity_tower" in layer.name for layer in model.layers):
            raise AssertionError(f"existing Model {letter} unexpectedly changed architecture")
        compatibility[letter] = {"loaded": True, "model_name": model.name,
                                 "losses_deserialized": bool(isinstance(model.loss, dict) and len(model.loss) == 2)}
        del model
        tf.keras.backend.clear_session()

    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_python_process": True,
        "model_d": {"loaded": True, "architecture_variant": "task_decoupled",
                    "model_name": "floating_plastic_resnet50_multitask_task_decoupled",
                    "masked_losses": loss_tasks, "backbone_frozen": True,
                    "real_training_inference": d_predictions},
        "existing_shared_tower_checkpoints": compatibility,
        "scope": "single FloPWD and single UGV training-record inference; no test-set access",
    }
    (args.run_dir / "reload_verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                          encoding="utf-8")
    metadata_path = args.run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_reload_verified"] = True
    metadata["reload_verification_file"] = "reload_verification.json"
    metadata["backward_compatibility"] = compatibility
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
