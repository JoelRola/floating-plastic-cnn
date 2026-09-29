"""TensorFlow input pipeline for manifest-selected FloPWD image records."""


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for image input pipelines") from exc
    return tf


def decode_resize_rgb(image_path, image_size=(224, 224)):
    """Read, decode, and resize RGB pixels, preserving the 0–255 range.

    ResNet50 preprocessing is embedded in the model graph through the official
    Keras ``preprocess_input`` function, ensuring one shared transform across
    train, validation, test, and prediction.
    """
    tf = _tensorflow()
    encoded = tf.io.read_file(image_path)
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, image_size, antialias=True)
    return tf.cast(image, tf.float32)


def make_tf_dataset(records, batch_size, image_size=(224, 224), training=False, seed=42):
    """Build a streaming batched dataset; only train records are shuffled."""
    tf = _tensorflow()
    if not records:
        raise ValueError("cannot build a TensorFlow dataset from zero records")
    ordered = sorted(records, key=lambda record: record.filename)
    filenames = [record.filename for record in ordered]
    if len(filenames) != len(set(filenames)):
        raise ValueError("split records must have unique filenames")
    image_paths = [str(record.image_path) for record in ordered]
    classifications = [[float(record.binary_label)] for record in ordered]
    severities = [[float(record.severity_percent)] for record in ordered]
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, {"classification": classifications, "severity": severities})
    )

    def load_example(path, targets):
        image = decode_resize_rgb(path, image_size)
        return image, targets

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(load_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(ordered), seed=int(seed), reshuffle_each_iteration=True
        )
    dataset = dataset.batch(int(batch_size), drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)
