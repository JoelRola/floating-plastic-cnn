import importlib.util
import math

import pytest

from floating_plastic.losses import make_tensorflow_masked_loss, masked_loss_numpy, pack_masked_targets
from floating_plastic.metrics import multidomain_evaluation, positive_only_metrics
from floating_plastic.pipeline import domain_sampling_weights


def test_masked_severity_and_classification_labels_have_no_loss_contribution():
    packed = pack_masked_targets([25.0, None, 50.0], [True, False, True])
    assert math.isnan(packed[1][0])
    loss_a = masked_loss_numpy(packed, [20.0, 0.0, 60.0], task="mae")
    loss_b = masked_loss_numpy(packed, [20.0, 99.0, 60.0], task="mae")
    assert loss_a == loss_b == 7.5
    unavailable = pack_masked_targets([None, None], [False, False])
    assert masked_loss_numpy(unavailable, [0.0, 100.0], task="mae") == 0.0

    class_labels = pack_masked_targets([1, None], [True, False])
    assert masked_loss_numpy(class_labels, [0.8, 0.0], task="binary_crossentropy") == pytest.approx(
        masked_loss_numpy(class_labels, [0.8, 1.0], task="binary_crossentropy")
    )


def test_tensorflow_masked_loss_ignores_unavailable_values():
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is not installed in lightweight CI")
    import tensorflow as tf

    truth = tf.constant([[float("nan"), 0.0], [25.0, 1.0]])
    loss = make_tensorflow_masked_loss("mae")
    assert float(loss(truth, tf.constant([[0.0], [20.0]]))) == pytest.approx(5.0)
    assert float(loss(truth, tf.constant([[100.0], [20.0]]))) == pytest.approx(5.0)
    no_severity = tf.constant([[float("nan"), 0.0], [float("nan"), 0.0]])
    assert float(loss(no_severity, tf.constant([[0.0], [100.0]]))) == 0.0


def test_domain_sampling_is_balanced_independently_of_dataset_size():
    assert domain_sampling_weights({"flopwd": 1402, "ugv": 2484}, "balanced") == {
        "flopwd": 0.5, "ugv": 0.5
    }
    proportional = domain_sampling_weights({"flopwd": 1, "ugv": 3}, "proportional")
    assert proportional == {"flopwd": 0.25, "ugv": 0.75}
    with pytest.raises(ValueError, match="strategy"):
        domain_sampling_weights({"flopwd": 1}, "unknown")


def test_positive_only_evaluation_omits_unsupported_negative_metrics():
    report = positive_only_metrics([0.1, 0.7, 0.9], threshold=0.5)
    assert report["sample_count"] == 3
    assert report["positive_recall"] == pytest.approx(2 / 3)
    assert report["specificity"] is None
    assert report["accuracy"] is None


def test_multidomain_evaluation_keeps_domain_metrics_separate():
    report = multidomain_evaluation(
        [0, 1, 1], [0.1, 0.6, 0.4], [0, 20, 40], [2, 18, 35], [0.7, 0.3]
    )
    assert report["flopwd"]["classification"]["specificity"] == 1.0
    assert report["ugv_annotated_waste_present"]["positive_recall"] == 0.5
    assert "accuracy degradation" in report["cross_domain_positive_recall_gap"]["interpretation"]
