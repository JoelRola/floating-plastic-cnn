"""TensorFlow model factory for the bounded two-head ResNet50 model."""


_PREPROCESS_LAYER = None


def _resnet_preprocess_layer(tf):
    """Return a registered layer that applies Keras' official ResNet transform.

    Wrapping the application function in one serializable layer avoids Keras
    2.15 serializing its internal TensorFlow ops as a non-portable Lambda graph.
    """
    global _PREPROCESS_LAYER
    if _PREPROCESS_LAYER is None:
        class ResNet50Preprocess(tf.keras.layers.Layer):
            def call(self, inputs):
                return tf.keras.applications.resnet.preprocess_input(inputs)

            def get_config(self):
                return super().get_config()

        ResNet50Preprocess.__name__ = "ResNet50Preprocess"
        ResNet50Preprocess.__qualname__ = "ResNet50Preprocess"
        ResNet50Preprocess.__module__ = __name__
        _PREPROCESS_LAYER = tf.keras.utils.register_keras_serializable(
            package="floating_plastic"
        )(ResNet50Preprocess)
    return _PREPROCESS_LAYER


def custom_objects():
    """Register and return custom layers required to reload saved models."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load the model") from exc
    layer = _resnet_preprocess_layer(tf)
    from .losses import make_tensorflow_masked_loss
    classification_loss = make_tensorflow_masked_loss("binary_crossentropy")
    severity_loss = make_tensorflow_masked_loss("mae")
    classification_loss_type = type(classification_loss)
    severity_loss_type = type(severity_loss)
    return {
        "ResNet50Preprocess": layer,
        "floating_plastic>ResNet50Preprocess": layer,
        classification_loss_type.__name__: classification_loss_type,
        f"floating_plastic>{classification_loss_type.__name__}": classification_loss_type,
        severity_loss_type.__name__: severity_loss_type,
        f"floating_plastic>{severity_loss_type.__name__}": severity_loss_type,
    }


def create_model(
    input_shape=(224, 224, 3),
    weights="imagenet",
    backbone_trainable=False,
    dense_units=(256, 128),
    dropout_rates=(0.3, 0.2),
):
    """Create a ResNet50 classifier and image-area severity regressor.

    Model inputs are RGB pixels in [0, 255]. The official
    ``keras.applications.resnet.preprocess_input`` is embedded in the model
    graph so training, evaluation, and single-image prediction share it.
    Severity is ``sigmoid * 100``, constraining outputs to 0–100 percentage
    points. ``weights=None`` avoids all downloads during architecture tests.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to construct the ResNet50 model") from exc

    if tuple(input_shape) != (224, 224, 3):
        raise ValueError("the documented model input shape is (224, 224, 3)")
    if len(dense_units) != len(dropout_rates):
        raise ValueError("dense_units and dropout_rates must have equal lengths")
    if any(int(units) < 1 for units in dense_units):
        raise ValueError("dense layer sizes must be positive")
    if any(not 0 <= float(rate) < 1 for rate in dropout_rates):
        raise ValueError("dropout rates must be in [0, 1)")

    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="rgb_image_0_255")
    preprocessed = _resnet_preprocess_layer(tf)(name="resnet50_preprocess")(inputs)
    backbone = tf.keras.applications.ResNet50(
        weights=weights, include_top=False, input_shape=input_shape
    )
    backbone.trainable = bool(backbone_trainable)
    x = backbone(preprocessed, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    for index, (units, dropout) in enumerate(zip(dense_units, dropout_rates), start=1):
        x = tf.keras.layers.Dense(int(units), activation="relu", name=f"shared_dense_{index}")(x)
        if dropout:
            x = tf.keras.layers.Dropout(float(dropout), name=f"shared_dropout_{index}")(x)

    classification = tf.keras.layers.Dense(
        1, activation="sigmoid", name="classification"
    )(x)
    severity_fraction = tf.keras.layers.Dense(
        1, activation="sigmoid", name="severity_fraction"
    )(x)
    severity = tf.keras.layers.Rescaling(100.0, name="severity_percentage_points")(
        severity_fraction
    )
    return tf.keras.Model(
        inputs=inputs,
        outputs={"classification": classification, "severity": severity},
        name="floating_plastic_resnet50_multitask",
    )
