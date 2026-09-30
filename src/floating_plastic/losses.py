"""Loss utilities for partially supervised multi-task records."""

import math


def pack_masked_targets(values, available):
    """Pack scalar labels and availability flags as ``[..., value, mask]``.

    Unavailable values are stored as NaN to prevent accidental use as labels;
    masked loss implementations replace them before arithmetic.
    """
    if len(values) != len(available):
        raise ValueError("values and availability must have equal lengths")
    packed = []
    for value, mask in zip(values, available):
        if mask:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("available target values must be finite")
        else:
            value = float("nan")
        packed.append([value, float(bool(mask))])
    return packed


def masked_loss_numpy(y_true_packed, y_pred, task="mae"):
    """Dependency-light masked MAE/BCE reference used by tests and audits."""
    if task not in {"mae", "binary_crossentropy"}:
        raise ValueError("task must be 'mae' or 'binary_crossentropy'")
    if len(y_true_packed) != len(y_pred) or not y_true_packed:
        raise ValueError("packed targets and predictions must be non-empty and equal length")
    losses = []
    for packed, prediction in zip(y_true_packed, y_pred):
        if len(packed) != 2:
            raise ValueError("each packed target must contain [value, availability]")
        value, available = float(packed[0]), bool(packed[1])
        if not available:
            continue
        prediction = float(prediction)
        if not math.isfinite(value) or not math.isfinite(prediction):
            raise ValueError("available targets and predictions must be finite")
        if task == "mae":
            losses.append(abs(value - prediction))
        else:
            if not 0 <= value <= 1 or not 0 <= prediction <= 1:
                raise ValueError("binary cross-entropy inputs must be in [0, 1]")
            p = min(max(prediction, 1e-7), 1 - 1e-7)
            losses.append(-(value * math.log(p) + (1 - value) * math.log(1 - p)))
    return sum(losses) / len(losses) if losses else 0.0


def make_tensorflow_masked_loss(task="mae"):
    """Create a serializable Keras loss consuming packed ``[value, mask]`` labels."""
    if task not in {"mae", "binary_crossentropy"}:
        raise ValueError("task must be 'mae' or 'binary_crossentropy'")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to construct a Keras masked loss") from exc

    class MaskedTaskLoss(tf.keras.losses.Loss):
        def __init__(self, task=task, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                     name=None):
            if task not in {"mae", "binary_crossentropy"}:
                raise ValueError("unsupported serialized masked loss task")
            self.task = task
            name = name or f"masked_{task}"
            super().__init__(reduction=reduction, name=name)

        def call(self, y_true, y_pred):
            values = tf.cast(y_true[..., 0:1], y_pred.dtype)
            mask = tf.cast(y_true[..., 1:2], y_pred.dtype)
            safe_values = tf.where(mask > 0, values, tf.zeros_like(values))
            if self.task == "mae":
                per_item = tf.abs(safe_values - y_pred)
            else:
                per_item = tf.keras.backend.binary_crossentropy(safe_values, y_pred)
            weighted = per_item * mask
            return tf.reduce_sum(weighted) / tf.maximum(tf.reduce_sum(mask), 1.0)

        def get_config(self):
            return {**super().get_config(), "task": self.task}

    MaskedTaskLoss.__name__ = f"Masked{task.title().replace('_', '')}Loss"
    MaskedTaskLoss.__qualname__ = MaskedTaskLoss.__name__
    MaskedTaskLoss.__module__ = __name__
    return tf.keras.utils.register_keras_serializable(package="floating_plastic")(MaskedTaskLoss)()
