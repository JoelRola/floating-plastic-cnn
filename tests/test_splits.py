import json
from collections import Counter

from floating_plastic.data import FloPWDRecord
from floating_plastic.splits import load_split_manifest, make_split_manifest, write_split_manifest
from floating_plastic.data import UGVRecord
from floating_plastic.group_splits import (
    load_ugv_grouped_split_manifest, make_ugv_grouped_split_manifest,
    records_for_ugv_split, ugv_leakage_report, write_ugv_grouped_split_manifest,
)


def test_split_manifest_is_reproducible_filename_only_and_stratified(tmp_path):
    records = [
        FloPWDRecord(f"{label}-{index}.jpg", label, 0.0, tmp_path / f"{label}-{index}.jpg")
        for label in (0, 1)
        for index in range(20)
    ]
    first = make_split_manifest(records, seed=7)
    assert first == make_split_manifest(records, seed=7)
    flat = [name for names in first["splits"].values() for name in names]
    assert len(flat) == len(set(flat)) == 40
    assert set(first) == {"dataset", "seed", "ratios", "splits"}
    for names in first["splits"].values():
        assert names
    counts = {key: Counter(name.split("-")[0] for name in names) for key, names in first["splits"].items()}
    assert [counts[key]["0"] for key in ("train", "validation", "test")] == [14, 3, 3]
    assert [counts[key]["1"] for key in ("train", "validation", "test")] == [14, 3, 3]
    path = write_split_manifest(first, tmp_path / "splits" / "manifest.json")
    loaded = load_split_manifest(path, expected_filenames=[record.filename for record in records])
    assert json.loads(path.read_text()) == loaded


def test_ugv_grouped_split_is_deterministic_and_prevents_group_leakage(tmp_path):
    records = []
    for group in range(20):
        for variant in range(1 + group % 3):
            suffix = f".rf.{group * 10 + variant:016x}" if variant else ""
            name = f"source{group}{suffix}.jpg"
            records.append(UGVRecord("train", name, tmp_path / name, tmp_path / (name + ".txt"),
                                     (0,), False, f"source{group}", 1))
    first = make_ugv_grouped_split_manifest(records, seed=42)
    assert first == make_ugv_grouped_split_manifest(records, seed=42)
    seen = {}
    for split, group_ids in first["group_splits"].items():
        for group in group_ids:
            assert group not in seen
            seen[group] = split
    path = write_ugv_grouped_split_manifest(first, tmp_path / "ugv.json")
    loaded = load_ugv_grouped_split_manifest(path, records)
    assert records_for_ugv_split(records, loaded, "test")


def test_ugv_leakage_report_counts_original_partition_overlap(tmp_path):
    records = [
        UGVRecord("train", "id.rf.0123456789abcdef.jpg", tmp_path / "a", tmp_path / "b", (0,), False, "id", 1),
        UGVRecord("test", "id.rf.abcdef0123456789.jpg", tmp_path / "c", tmp_path / "d", (0,), False, "id", 1),
    ]
    report = ugv_leakage_report(records)
    assert report["cross_split_group_count"] == 1
