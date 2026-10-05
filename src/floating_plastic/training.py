"""Shared optimizer-step logic for masked multi-domain training."""


def masked_multitask_train_step(model, optimizer, images, targets,
                                classification_loss, severity_loss,
                                classification_weight=1.0, severity_weight=0.5):
    """Apply one optimizer step using only task labels available in ``targets``.

    On an UGV-only batch the masked severity loss is exactly zero and its
    independent Model D tower is disconnected from classification gradients.
    """
    import tensorflow as tf

    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        class_value = classification_loss(targets["classification"], outputs["classification"])
        severity_value = severity_loss(targets["severity"], outputs["severity"])
        total = float(classification_weight) * class_value + float(severity_weight) * severity_value
    gradients = tape.gradient(total, model.trainable_variables)
    pairs = [(gradient, variable) for gradient, variable in zip(gradients, model.trainable_variables)
             if gradient is not None]
    if not pairs or not bool(tf.math.is_finite(total).numpy()):
        raise FloatingPointError("non-finite total loss or no trainable gradients")
    if any(not bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()) for gradient, _ in pairs):
        raise FloatingPointError("non-finite gradient")
    classification_tower_gradients = [
        gradient for gradient, variable in pairs
        if "classification_tower" in variable.name or "/classification/" in variable.name
    ]
    severity_tower_gradients = [
        gradient for gradient, variable in pairs
        if "severity_tower" in variable.name or "/severity_fraction/" in variable.name
    ]
    classification_tower_norm = (float(tf.linalg.global_norm(classification_tower_gradients).numpy())
                                 if classification_tower_gradients else 0.0)
    severity_tower_norm = (float(tf.linalg.global_norm(severity_tower_gradients).numpy())
                           if severity_tower_gradients else 0.0)
    optimizer.apply_gradients(pairs)
    return {
        "classification_loss": float(class_value.numpy()),
        "severity_loss": float(severity_value.numpy()),
        "total_loss": float(total.numpy()),
        "gradient_variable_count": len(pairs),
        "classification_tower_gradient_norm": classification_tower_norm,
        "classification_tower_nonzero_gradient_variables": sum(
            bool(tf.reduce_any(gradient != 0).numpy()) for gradient in classification_tower_gradients),
        "severity_tower_gradient_norm": severity_tower_norm,
        "severity_tower_nonzero_gradient_variables": sum(
            bool(tf.reduce_any(gradient != 0).numpy()) for gradient in severity_tower_gradients),
    }


def task_decoupled_two_stream_train_step(
        model, optimizer, classification_images, classification_targets,
        severity_images, severity_targets, severity_domain_ids,
        classification_loss, severity_loss,
        classification_weight=1.0, severity_weight=0.5):
    """Apply one update from a mixed classification batch and an independent FloPWD batch.

    The independent severity batch must contain only original-prior FloPWD
    examples. Gradients are computed separately against the two disjoint task
    towers, then applied together in one optimizer update.
    """
    import tensorflow as tf

    classification_layers = [model.get_layer("classification_tower_dense_1"),
                             model.get_layer("classification_tower_dense_2"),
                             model.get_layer("classification")]
    severity_layers = [model.get_layer("severity_tower_dense_1"),
                       model.get_layer("severity_tower_dense_2"),
                       model.get_layer("severity_fraction")]
    classification_variables = [v for layer in classification_layers for v in layer.trainable_variables]
    severity_variables = [v for layer in severity_layers for v in layer.trainable_variables]
    if (not classification_variables or not severity_variables or
            {id(v) for v in classification_variables} & {id(v) for v in severity_variables}):
        raise ValueError("two-stream updates require disjoint trainable task towers")
    if not severity_domain_ids or any(value != "flopwd" for value in severity_domain_ids):
        raise ValueError("severity stream may contain only FloPWD examples")
    severity_mask = severity_targets["severity"][:, 1] > 0
    if len(severity_domain_ids) != int(severity_mask.shape[0]) or not bool(tf.reduce_all(severity_mask).numpy()):
        raise ValueError("every independent severity example must have a real severity label")

    with tf.GradientTape(persistent=True) as tape:
        class_output = model(classification_images, training=True)["classification"]
        severity_output = model(severity_images, training=True)["severity"]
        class_value = classification_loss(classification_targets["classification"], class_output)
        severity_value = severity_loss(severity_targets["severity"], severity_output)
        weighted_class = float(classification_weight) * class_value
        weighted_severity = float(severity_weight) * severity_value
        total = weighted_class + weighted_severity

    class_gradients = tape.gradient(weighted_class, classification_variables)
    severity_gradients = tape.gradient(weighted_severity, severity_variables)
    class_cross = tape.gradient(weighted_class, severity_variables)
    severity_cross = tape.gradient(weighted_severity, classification_variables)
    del tape
    if any(gradient is not None and bool(tf.reduce_any(gradient != 0).numpy()) for gradient in class_cross):
        raise AssertionError("classification loss reached the severity tower")
    if any(gradient is not None and bool(tf.reduce_any(gradient != 0).numpy()) for gradient in severity_cross):
        raise AssertionError("severity loss reached the classification tower")
    pairs = [(g, v) for g, v in zip(class_gradients, classification_variables) if g is not None]
    pairs += [(g, v) for g, v in zip(severity_gradients, severity_variables) if g is not None]
    if not pairs or not bool(tf.math.is_finite(total).numpy()):
        raise FloatingPointError("non-finite two-stream loss or no trainable gradients")
    if any(not bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()) for gradient, _ in pairs):
        raise FloatingPointError("non-finite two-stream gradient")
    class_norm = float(tf.linalg.global_norm([g for g in class_gradients if g is not None]).numpy())
    severity_norm = float(tf.linalg.global_norm([g for g in severity_gradients if g is not None]).numpy())
    optimizer.apply_gradients(pairs)
    return {
        "classification_loss": float(class_value.numpy()),
        "severity_loss": float(severity_value.numpy()),
        "total_loss": float(total.numpy()),
        "classification_tower_gradient_norm": class_norm,
        "severity_tower_gradient_norm": severity_norm,
        "classification_cross_tower_gradient_norm": 0.0,
        "severity_cross_tower_gradient_norm": 0.0,
        "classification_severity_example_count": int(classification_images.shape[0]),
        "severity_example_count": int(severity_images.shape[0]),
    }
