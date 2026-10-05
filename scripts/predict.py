"""Run one-image inference with a saved portfolio model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.config import load_config
from floating_plastic.metrics import threshold_predictions
from floating_plastic.model import custom_objects
from floating_plastic.pipeline import decode_resize_rgb


def resolve_severity_status(model_path, model_id=None):
    """Return only a capability recorded in the tracked result package."""
    if model_id is None:
        run_name = Path(model_path).parent.name.lower()
        model_id = {
            "flopwd_original_seed42": "A",
            "multidomain_seed42": "B",
            "multidomain_class_balanced_seed42": "C",
            "task_decoupled_multidomain_seed42": "D",
            "multidomain_2to1_seed42": "E",
            "task_decoupled_original_severity_prior_seed42": "F",
        }.get(run_name)
    if model_id is None:
        return "unverified"
    summary_path = Path("experiments/results") / f"model_{model_id.lower()}.json"
    if not summary_path.is_file():
        return "unverified"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("severity_status", "unverified")


def make_prediction_report(domain, probability, threshold, severity, severity_status):
    predicted_class = threshold_predictions([probability], threshold)[0]
    report = {
        "domain": domain,
        "classification_probability": probability,
        "classification_threshold": threshold,
        "classification": (
            "plastic detected" if predicted_class else "no plastic detected"
        ) if domain == "flopwd" else (
            "annotated waste detected" if predicted_class else "annotated waste not detected"
        ),
        "severity_status": severity_status,
    }
    if domain == "flopwd" and severity_status == "functional":
        report["estimated_flopwd_style_coverage_percent"] = severity
    elif domain == "ugv":
        report["severity_estimate"] = None
        report["severity_note"] = "UGV has no comparable severity labels."
    else:
        report["severity_estimate"] = None
        report["severity_note"] = (
            "Severity is withheld because this model's severity head is not validated for use."
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--domain", choices=("flopwd", "ugv"), default="flopwd",
                        help="Select the label semantics to use when describing the score")
    parser.add_argument("--model-id", choices=tuple("ABCDEF"),
                        help="Read model capability from experiments/results/model_<id>.json")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}")
    if not args.image.is_file():
        parser.error(f"image file does not exist: {args.image}")
    config = load_config(args.config)
    threshold = args.threshold
    if threshold is None:
        threshold = float(config["evaluation"]["classification_threshold"])
    if not 0 <= threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for prediction; install compatible runtime packages from requirements.txt") from exc

    image = decode_resize_rgb(str(args.image), tuple(config["data"]["image_size"]))
    model = tf.keras.models.load_model(
        args.model, compile=False, custom_objects=custom_objects()
    )
    outputs = model(tf.convert_to_tensor(image[None, ...]), training=False)
    probability = float(np.asarray(outputs["classification"]).reshape(-1)[0])
    severity = float(np.asarray(outputs["severity"]).reshape(-1)[0])
    capability = resolve_severity_status(args.model, args.model_id)
    result = make_prediction_report(args.domain, probability, threshold, severity, capability)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
