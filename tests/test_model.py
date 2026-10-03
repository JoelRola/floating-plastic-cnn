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
    assert model.name == "floating_plastic_resnet50_multitask"
    assert model.get_layer("shared_dense_1")
    assert not any("classification_tower" in layer.name for layer in model.layers)
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


def test_task_decoupled_towers_isolate_ugv_classification_gradients(tmp_path):
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is an optional model dependency and is not installed in lightweight CI")
    import numpy as np
    import tensorflow as tf
    from floating_plastic.losses import make_tensorflow_masked_loss
    from floating_plastic.training import masked_multitask_train_step

    tf.keras.utils.set_random_seed(42)
    model = create_model(weights=None, architecture_variant="task_decoupled")
    class_layers = [model.get_layer("classification_tower_dense_1"),
                    model.get_layer("classification_tower_dense_2"),
                    model.get_layer("classification")]
    severity_layers = [model.get_layer("severity_tower_dense_1"),
                       model.get_layer("severity_tower_dense_2"),
                       model.get_layer("severity_fraction")]
    backbone = model.get_layer("resnet50")
    assert model.name.endswith("task_decoupled")
    assert backbone.trainable is False
    assert not set(v.ref() for layer in class_layers for v in layer.trainable_variables) & set(
        v.ref() for layer in severity_layers for v in layer.trainable_variables
    )
    class_loss = make_tensorflow_masked_loss("binary_crossentropy")
    severity_loss = make_tensorflow_masked_loss("mae")
    optimizer = tf.keras.optimizers.Adam(0.001)
    images = tf.random.uniform((2, 224, 224, 3), maxval=255.0)
    ugv_targets = {"classification": tf.constant([[1.0, 1.0], [1.0, 1.0]]),
                   "severity": tf.constant([[float("nan"), 0.0], [float("nan"), 0.0]])}
    class_before = [w.copy() for layer in class_layers for w in layer.get_weights()]
    severity_before = [w.copy() for layer in severity_layers for w in layer.get_weights()]
    backbone_before = [w.copy() for w in backbone.get_weights()]

    losses = masked_multitask_train_step(model, optimizer, images, ugv_targets,
                                         class_loss, severity_loss, 1.0, 0.5)
    assert np.isfinite(losses["classification_loss"])
    assert losses["severity_loss"] == 0.0
    assert losses["classification_tower_gradient_norm"] > 0
    assert losses["classification_tower_nonzero_gradient_variables"] > 0
    assert losses["severity_tower_gradient_norm"] == 0.0
    assert losses["severity_tower_nonzero_gradient_variables"] == 0
    class_after = [w for layer in class_layers for w in layer.get_weights()]
    severity_after = [w for layer in severity_layers for w in layer.get_weights()]
    assert any(not np.array_equal(before, after) for before, after in zip(class_before, class_after))
    assert all(np.array_equal(before, after) for before, after in zip(severity_before, severity_after))
    assert all(np.array_equal(before, after) for before, after in zip(backbone_before, backbone.get_weights()))

    flopwd_targets = {"classification": tf.constant([[1.0, 1.0], [0.0, 1.0]]),
                      "severity": tf.constant([[40.0, 1.0], [0.0, 1.0]])}
    severity_before = [w.copy() for layer in severity_layers for w in layer.get_weights()]
    flo_losses = masked_multitask_train_step(model, optimizer, images, flopwd_targets,
                                            class_loss, severity_loss, 1.0, 0.5)
    assert np.isfinite(flo_losses["severity_loss"])
    assert flo_losses["severity_tower_gradient_norm"] > 0
    assert flo_losses["severity_tower_nonzero_gradient_variables"] > 0
    severity_after = [w for layer in severity_layers for w in layer.get_weights()]
    assert any(not np.array_equal(before, after) for before, after in zip(severity_before, severity_after))
    assert all(np.array_equal(before, after) for before, after in zip(backbone_before, backbone.get_weights()))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss={"classification": class_loss, "severity": severity_loss},
                  loss_weights={"classification": 1.0, "severity": 0.5})
    model_path = tmp_path / "task_decoupled.keras"
    model.save(model_path)
    loaded = tf.keras.models.load_model(model_path, custom_objects=custom_objects())
    assert loaded.name.endswith("task_decoupled")
    assert loaded.get_layer("classification_tower_dense_1")
    assert loaded.get_layer("severity_tower_dense_1")
    assert len(loaded.loss) == 2
