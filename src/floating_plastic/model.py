"""TensorFlow model factory for the documented two-head ResNet50 concept."""


def create_model(input_shape=(224, 224, 3), weights="imagenet"):
    """Build a ResNet50 classifier and coverage regressor.

    Inputs are RGB pixels in the 0–255 range; the application-specific
    ResNet50 preprocessing is included in the graph. ``weights=None`` supports
    offline shape/structure checks without downloading pretrained weights.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to construct the ResNet50 model") from exc
    if tuple(input_shape) != (224, 224, 3):
        raise ValueError("the documented model input shape is (224, 224, 3)")
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    backbone = tf.keras.applications.ResNet50(
        weights=weights, include_top=False, input_shape=input_shape
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    classification = tf.keras.layers.Dense(1, activation="sigmoid", name="classification")(x)
    coverage = tf.keras.layers.Dense(1, activation="linear", name="coverage")(x)
    return tf.keras.Model(inputs=inputs, outputs={"classification": classification, "coverage": coverage})
