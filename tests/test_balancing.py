import numpy as np
import pytest

from floating_plastic.balancing import balance_indices


def test_balance_indices_has_requested_ratio_and_is_seeded():
    labels = np.array([0, 0, 0, 1, 1])
    first = balance_indices(labels, negative_to_positive=1, seed=17)
    second = balance_indices(labels, negative_to_positive=1, seed=17)
    assert np.array_equal(first, second)
    assert (labels[first] == 0).sum() == (labels[first] == 1).sum() == 2


def test_balance_indices_supports_frozen_two_to_one_ratio():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    selected = balance_indices(labels, negative_to_positive=2, seed=42)
    assert (labels[selected] == 0).sum() == 2 * (labels[selected] == 1).sum()
    assert (labels[selected] == 1).sum() == 3


def test_balance_indices_rejects_missing_class():
    with pytest.raises(ValueError, match="each class"):
        balance_indices([1, 1])
