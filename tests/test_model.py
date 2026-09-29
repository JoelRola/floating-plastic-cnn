import importlib.util

import pytest

from floating_plastic.model import create_model


def test_model_construction_is_skipped_without_tensorflow():
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is an optional model dependency and is not installed in lightweight CI")
    # Avoid ImageNet weight downloads even in environments that have TensorFlow.
    model = create_model(weights=None)
    assert model.input_shape == (None, 224, 224, 3)
    assert set(model.output_names) == {"classification", "coverage"}
