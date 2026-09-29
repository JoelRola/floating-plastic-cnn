from pathlib import Path

import pytest

from floating_plastic.config import apply_overrides, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads_and_overrides_without_mutating_source():
    config = load_config(ROOT / "configs/default.yaml")
    changed = apply_overrides(config, **{"training.seed": 9})
    assert config["training"]["seed"] == 42
    assert changed["training"]["seed"] == 9


def test_invalid_image_size_fails_clearly(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data: {image_size: [128, 128]}\nmodel: {}\ntraining: {epochs: 1, batch_size: 1, seed: 1, learning_rate: 0.1}\nlosses: {classification_weight: 1, severity_weight: 1}\nevaluation: {classification_threshold: 0.5}\nreproducibility: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image_size"):
        load_config(path)
