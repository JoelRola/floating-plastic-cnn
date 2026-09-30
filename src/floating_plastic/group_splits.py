"""Deterministic grouped splits for filename-derived UGV source IDs."""

import json
from pathlib import Path
import random

from .splits import DEFAULT_RATIOS, _split_sizes


def ugv_leakage_report(records):
    by_source = {}
    for record in records:
        by_source.setdefault(record.canonical_source_id, {}).setdefault(record.split, []).append(record.filename)
    leaked = {source: {split: sorted(names) for split, names in sorted(splits.items())}
              for source, splits in sorted(by_source.items()) if len(splits) > 1}
    return {
        "dataset": "UGV-NBWASTE",
        "grouping": "filename stem with terminal .rf.<hex hash> removed (case-insensitive)",
        "grouping_is_source_identity_proof": False,
        "exported_image_count": len(records),
        "canonical_source_group_count": len(by_source),
        "cross_split_group_count": len(leaked),
        "cross_split_groups_by_partition_pair": _partition_pair_counts(leaked),
        "examples": dict(list(leaked.items())[:10]),
    }


def _partition_pair_counts(leaked):
    pairs = {"train+validation": 0, "train+test": 0, "validation+test": 0, "train+validation+test": 0}
    for split_map in leaked.values():
        normalized = {"validation" if split == "valid" else split for split in split_map}
        key = "+".join(split for split in ("train", "validation", "test") if split in normalized)
        pairs[key] += 1
    return pairs


def make_ugv_grouped_split_manifest(records, seed=42, ratios=None):
    ratios = dict(ratios or DEFAULT_RATIOS)
    if list(ratios) != ["train", "validation", "test"] or any(v <= 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("ratios must define positive train/validation/test fractions summing to 1")
    groups = {}
    filenames = []
    for record in records:
        source_id = record.canonical_source_id or ""
        if not source_id:
            raise ValueError(f"record has no canonical source ID: {record.filename}")
        groups.setdefault(source_id, []).append(record.filename)
        filenames.append(record.filename)
    if not groups or len(filenames) != len(set(filenames)):
        raise ValueError("records must be non-empty and filenames unique")
    ordered_groups = sorted(groups)
    random.Random(int(seed)).shuffle(ordered_groups)
    sizes = _split_sizes(len(ordered_groups), ratios)
    group_splits, offset = {}, 0
    for split, size in sizes.items():
        group_splits[split] = ordered_groups[offset:offset + size]
        offset += size
    splits = {split: sorted(name for group in group_ids for name in groups[group])
              for split, group_ids in group_splits.items()}
    manifest = {
        "dataset": "UGV-NBWASTE",
        "seed": int(seed),
        "ratios": ratios,
        "grouping": "filename stem with terminal .rf.<hex hash> removed (case-insensitive)",
        "grouping_is_source_identity_proof": False,
        "group_splits": {split: sorted(ids) for split, ids in group_splits.items()},
        "splits": splits,
        "counts": {
            split: {"groups": len(group_splits[split]), "images": len(splits[split])}
            for split in ratios
        },
    }
    _validate_ugv_manifest(manifest)
    return manifest


def _validate_ugv_manifest(manifest):
    if manifest.get("dataset") != "UGV-NBWASTE":
        raise ValueError("UGV manifest dataset must be UGV-NBWASTE")
    splits, groups = manifest.get("splits"), manifest.get("group_splits")
    expected = ["train", "validation", "test"]
    if not isinstance(splits, dict) or list(splits) != expected or not isinstance(groups, dict) or list(groups) != expected:
        raise ValueError("UGV manifest must define train, validation, and test in order")
    all_images = [name for values in splits.values() for name in values]
    all_groups = [name for values in groups.values() for name in values]
    if len(all_images) != len(set(all_images)) or len(all_groups) != len(set(all_groups)):
        raise ValueError("UGV filenames and canonical source groups must each occur in exactly one split")


def write_ugv_grouped_split_manifest(manifest, path):
    _validate_ugv_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_ugv_grouped_split_manifest(path, records=None):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_ugv_manifest(manifest)
    if records is not None:
        by_name = {record.filename: record for record in records}
        expected = {name for names in manifest["splits"].values() for name in names}
        if expected != set(by_name):
            raise ValueError("UGV manifest filenames do not match the local export")
        for split, names in manifest["splits"].items():
            expected_groups = set(manifest["group_splits"][split])
            actual_groups = {by_name[name].canonical_source_id for name in names}
            if expected_groups != actual_groups:
                raise ValueError(f"UGV manifest group IDs do not match filenames in {split}")
    return manifest


def records_for_ugv_split(records, manifest, split):
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    by_name = {record.filename: record for record in records}
    names = manifest["splits"][split]
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"UGV manifest contains filenames missing from data: {missing[:5]}")
    return [by_name[name] for name in sorted(names)]
