import importlib.util

import pytest

from floating_plastic.model import create_model, custom_objects


def test_model_construction_is_skipped_without_tensorflow(tmp_path):
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is an optional model dependency and is not installed in lightweight CI")
    # Avoid ImageNet weight downloads even in environments that have TensorFlow.
    model = create_model(weights=None)
    assert model.input_shape == (None, 224, 224, 3)
    assert set(model.output_names) == {"classification", "severity_percentage_points"}
    assert model.get_layer("severity_fraction").activation.__name__ == "sigmoid"

    import tensorflow as tf

    output = model(tf.zeros((1, 224, 224, 3)), training=False)
    severity = output["severity"].numpy()
    assert ((severity >= 0) & (severity <= 100)).all()
    model_path = tmp_path / "model.keras"
    model.save(model_path)
    loaded = tf.keras.models.load_model(
        model_path, compile=False, custom_objects=custom_objects()
    )
    reloaded_output = loaded(tf.zeros((1, 224, 224, 3)), training=False)
    assert set(reloaded_output) == {"classification", "severity"}
    assert reloaded_output["classification"].shape == (1, 1)
    assert reloaded_output["severity"].shape == (1, 1)
