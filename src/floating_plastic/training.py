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
    optimizer.apply_gradients(pairs)
    return {
        "classification_loss": float(class_value.numpy()),
        "severity_loss": float(severity_value.numpy()),
        "total_loss": float(total.numpy()),
        "gradient_variable_count": len(pairs),
    }
