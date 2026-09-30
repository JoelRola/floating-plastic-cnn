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
    # Antialiased interpolation can overshoot the source pixel range by a few
    # floating-point ULPs; keep the documented ResNet input contract explicit.
    image = tf.clip_by_value(image, 0.0, 255.0)
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


def domain_sampling_weights(domain_sizes, strategy="balanced"):
    """Return per-domain sample probabilities, independent of class balancing."""
    if strategy not in {"balanced", "proportional"}:
        raise ValueError("domain strategy must be 'balanced' or 'proportional'")
    names = sorted(domain_sizes)
    if not names or any(int(domain_sizes[name]) <= 0 for name in names):
        raise ValueError("every sampled domain must contain records")
    raw = ({name: 1.0 for name in names} if strategy == "balanced" else
           {name: float(domain_sizes[name]) for name in names})
    total = sum(raw.values())
    return {name: raw[name] / total for name in names}


def make_heterogeneous_tf_dataset(records, batch_size, image_size=(224, 224),
                                 training=False, seed=42, include_domain_id=False):
    """Build a deterministic finite dataset for one source domain."""
    tf = _tensorflow()
    ordered = sorted(records, key=lambda record: record.filename)
    if not ordered:
        raise ValueError("cannot build a TensorFlow dataset from zero records")
    if len({record.filename for record in ordered}) != len(ordered):
        raise ValueError("dataset records must have unique filenames")
    paths = [str(record.image_path) for record in ordered]
    targets = {
        "classification": pack_masked_targets(
            [record.classification_target for record in ordered],
            [record.classification_target_available for record in ordered]),
        "severity": pack_masked_targets(
            [record.severity_target for record in ordered],
            [record.severity_target_available for record in ordered]),
    }
    domains = [record.source_domain for record in ordered]
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets, domains))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    def load(path, labels, domain):
        image = decode_resize_rgb(path, image_size)
        if include_domain_id:
            return image, labels, domain
        return image, labels

    if training:
        dataset = dataset.shuffle(len(ordered), seed=int(seed), reshuffle_each_iteration=True)
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return dataset.batch(int(batch_size), drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def make_multidomain_tf_dataset(records_by_domain, batch_size, image_size=(224, 224),
                                training=True, seed=42, strategy="balanced",
                                flopwd_negative_to_positive=None, include_domain_id=False):
    """Sample domains independently, then stream mixed batches with task masks.

    Training datasets repeat within domain before weighted sampling. Validation
    and test should remain separate datasets per source domain and must not use
    this mixed metric path. Optional class balancing applies only to FloPWD.
    """
    tf = _tensorflow()
    if not training:
        raise ValueError("keep validation/test datasets domain-specific; build them separately")
    records_by_domain = {str(k): list(v) for k, v in records_by_domain.items() if v}
    if not records_by_domain or any(not values for values in records_by_domain.values()):
        raise ValueError("each requested domain must have records")
    if any(record.source_domain != domain for domain, records in records_by_domain.items() for record in records):
        raise ValueError("record source_domain does not match records_by_domain key")

    # Class balancing is a separate operation restricted to labeled FloPWD train records.
    if flopwd_negative_to_positive is not None:
        from .balancing import balance_indices
        if "flopwd" not in records_by_domain:
            raise ValueError("class balancing requires FloPWD records")
        labels = [record.classification_target for record in records_by_domain["flopwd"]]
        indices = balance_indices(labels, int(flopwd_negative_to_positive), seed=int(seed))
        records_by_domain["flopwd"] = [records_by_domain["flopwd"][int(i)] for i in indices]

    domain_names = sorted(records_by_domain)
    weights = domain_sampling_weights({name: len(records_by_domain[name]) for name in domain_names}, strategy)
    datasets = []
    for domain_index, domain in enumerate(domain_names):
        ordered = sorted(records_by_domain[domain], key=lambda record: record.filename)
        paths = [str(record.image_path) for record in ordered]
        class_targets = pack_masked_targets([record.classification_target for record in ordered],
                                            [record.classification_target_available for record in ordered])
        severity_targets = pack_masked_targets([record.severity_target for record in ordered],
                                               [record.severity_target_available for record in ordered])
        raw = tf.data.Dataset.from_tensor_slices(
            (paths, {"classification": class_targets, "severity": severity_targets},
             [record.source_domain for record in ordered])
        )
        options = tf.data.Options()
        options.experimental_deterministic = True
        raw = raw.with_options(options)
        raw = raw.shuffle(len(ordered), seed=int(seed) + domain_index,
                          reshuffle_each_iteration=True).repeat()
        def load(path, targets, domain_label):
            image = decode_resize_rgb(path, image_size)
            return (image, targets, domain_label) if include_domain_id else (image, targets)

        raw = raw.map(load,
                      num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        datasets.append(raw)
    mixed = tf.data.Dataset.sample_from_datasets(
        datasets, weights=[weights[name] for name in domain_names], seed=int(seed), stop_on_empty_dataset=False
    )
    steps = (sum(len(records) for records in records_by_domain.values()) + int(batch_size) - 1) // int(batch_size)
    return mixed.batch(int(batch_size), drop_remainder=False).prefetch(tf.data.AUTOTUNE), steps


def pack_masked_targets(values, available):
    from .losses import pack_masked_targets as pack
    return pack(values, available)
