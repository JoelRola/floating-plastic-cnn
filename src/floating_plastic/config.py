"""Small YAML configuration loader and validation for experiment settings."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path):
    """Read and validate a nested experiment YAML configuration."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    required = {"data", "model", "training", "losses", "evaluation", "reproducibility"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(missing)}")
    for section in required:
        if not isinstance(config[section], dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")
    image_size = config["data"].get("image_size")
    if image_size != [224, 224]:
        raise ValueError("data.image_size must be [224, 224] for the documented ResNet50 model")
    training = config["training"]
    for key in ("epochs", "batch_size", "seed", "learning_rate"):
        if key not in training:
            raise ValueError(f"training.{key} is required")
    if training["epochs"] < 1 or training["batch_size"] < 1:
        raise ValueError("training.epochs and training.batch_size must be positive")
    if training["learning_rate"] <= 0:
        raise ValueError("training.learning_rate must be positive")
    threshold = config["evaluation"].get("classification_threshold")
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("evaluation.classification_threshold must be in [0, 1]")
    weights = config["losses"]
    for key in ("classification_weight", "severity_weight"):
        if key not in weights or weights[key] < 0:
            raise ValueError(f"losses.{key} must be present and non-negative")
    return config


def apply_overrides(config, **overrides):
    """Copy config and apply dotted-key overrides without mutating the source."""
    result = deepcopy(config)
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        keys = dotted_key.split(".")
        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Unknown configuration key: {dotted_key}")
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"Unknown configuration key: {dotted_key}")
        target[keys[-1]] = value
    return result
