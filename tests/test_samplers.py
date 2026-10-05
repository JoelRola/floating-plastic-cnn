from types import SimpleNamespace

import numpy as np
import pytest

from floating_plastic.balancing import balance_indices
from floating_plastic.pipeline import domain_sampling_weights
from floating_plastic.samplers import OriginalPriorCyclicSampler, severity_distribution


def _records():
    # Original distribution: 8 negatives, 2 positives, and 3 zero targets.
    severities = [0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0]
    return [SimpleNamespace(filename=f"f{i:02}.jpg", source_domain="flopwd",
                            classification_target=int(i >= 8), severity_target=value,
                            severity_target_available=True)
            for i, value in enumerate(severities)]


def test_model_f_classification_policy_keeps_balanced_domains_and_flopwd_prior():
    records = _records()
    indices = balance_indices([r.classification_target for r in records], 1, seed=42)
    selected = [records[int(i)] for i in indices]
    labels = [record.classification_target for record in selected]
    assert labels.count(0) == labels.count(1)
    assert domain_sampling_weights({"flopwd": len(selected), "ugv": 500}, "balanced") == {
        "flopwd": 0.5, "ugv": 0.5
    }


def test_original_severity_stream_is_independent_and_has_no_balancing_duplicates():
    records = _records()
    balanced = balance_indices([r.classification_target for r in records], 1, seed=42)
    assert sum(records[int(i)].classification_target == 0 for i in balanced) == sum(
        records[int(i)].classification_target == 1 for i in balanced)
    sampler = OriginalPriorCyclicSampler(records, seed=42)
    first_cycle = sampler.next_records(len(records))
    assert len({r.filename for r in first_cycle}) == len(records)
    assert sorted(r.filename for r in first_cycle) == sorted(r.filename for r in records)
    assert sum(r.classification_target == 0 for r in first_cycle) == 8
    assert sum(r.classification_target == 1 for r in first_cycle) == 2
    assert [r.filename for r in first_cycle] != [records[int(i)].filename for i in balanced]


def test_severity_sampler_returns_exact_k_and_preserves_original_target_statistics():
    records = _records()
    sampler = OriginalPriorCyclicSampler(records, seed=42)
    selected = sampler.next_records(7)
    assert len(selected) == 7
    assert all(record.source_domain == "flopwd" and record.severity_target_available for record in selected)
    complete = sampler.next_records(13)
    assert len(complete) == 13
    # Two full ten-row traversals preserve every record exactly twice.
    assert sorted(r.filename for r in selected + complete) == sorted(
        [r.filename for r in records] * 2)
    distribution = severity_distribution(selected + complete)
    expected = severity_distribution(records)
    assert distribution["mean"] == pytest.approx(expected["mean"])
    assert distribution["zero_percentage"] == expected["zero_percentage"]


def test_original_severity_sampler_is_seed_deterministic():
    records = _records()
    first = OriginalPriorCyclicSampler(records, seed=42).next_records(17)
    second = OriginalPriorCyclicSampler(records, seed=42).next_records(17)
    assert [r.filename for r in first] == [r.filename for r in second]


def test_original_severity_sampler_rejects_ugv_and_unavailable_targets():
    with pytest.raises(ValueError, match="only FloPWD"):
        OriginalPriorCyclicSampler([SimpleNamespace(filename="u", source_domain="ugv",
            severity_target_available=False, severity_target=None)], seed=42)
    unavailable = SimpleNamespace(filename="f", source_domain="flopwd",
                                  severity_target_available=False, severity_target=None)
    with pytest.raises(ValueError, match="available FloPWD"):
        OriginalPriorCyclicSampler([unavailable], seed=42)
