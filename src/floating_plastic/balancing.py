"""Deterministic class balancing utilities."""

import numpy as np


def balance_indices(binary_labels, negative_to_positive=1, seed=42):
    """Return shuffled indices with the requested negative:positive ratio.

    The positive examples are retained once; negatives are sampled with
    replacement. Both classes must be present. This operates only on training
    indices; callers must split before balancing.
    """
    labels = np.asarray(binary_labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("binary_labels must be a 1D array containing only 0 and 1")
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if not len(positives) or not len(negatives):
        raise ValueError("class balancing requires at least one sample from each class")
    if not isinstance(negative_to_positive, int) or negative_to_positive < 1:
        raise ValueError("negative_to_positive must be a positive integer")
    rng = np.random.default_rng(seed)
    selected_negatives = rng.choice(
        negatives, size=len(positives) * negative_to_positive, replace=True
    )
    indices = np.concatenate((positives, selected_negatives))
    rng.shuffle(indices)
    return indices


def balance_arrays(images, binary_labels, coverage_labels, negative_to_positive=1, seed=42):
    """Apply training-set indices consistently to images and both targets."""
    x, y_binary, y_coverage = map(np.asarray, (images, binary_labels, coverage_labels))
    if not (len(x) == len(y_binary) == len(y_coverage)):
        raise ValueError("images and label arrays must have equal lengths")
    idx = balance_indices(y_binary, negative_to_positive, seed)
    return x[idx], y_binary[idx], y_coverage[idx]
