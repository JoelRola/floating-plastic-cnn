"""Run one-image inference with a saved FloPWD portfolio model."""

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--threshold", type=float)
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
    predicted_class = threshold_predictions([probability], threshold)[0]
    print(json.dumps({
        "plastic_probability": probability,
        "classification_threshold": threshold,
        "predicted_class": predicted_class,
        "predicted_label": "plastic" if predicted_class else "non_plastic",
        "predicted_severity_percentage_points": severity,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
