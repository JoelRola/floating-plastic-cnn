"""Filename-only, deterministic, stratified dataset split manifests."""

import json
from pathlib import Path
import random
import zlib


DEFAULT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def _split_sizes(count, ratios):
    exact = {key: count * value for key, value in ratios.items()}
    sizes = {key: int(value) for key, value in exact.items()}
    remainder = count - sum(sizes.values())
    for key in sorted(ratios, key=lambda k: (-(exact[k] - sizes[k]), list(ratios).index(k)))[:remainder]:
        sizes[key] += 1
    if count >= len(ratios):
        for key in ratios:
            if sizes[key] == 0:
                donor = max(sizes, key=sizes.get)
                sizes[donor] -= 1
                sizes[key] += 1
    return sizes


def make_split_manifest(records, seed=42, ratios=None):
    """Stratify records by ``binary_label`` and return filename lists only.

    Ratios default to 70/15/15. A class needs at least one sample per split.
    No image data or labels are written into the manifest.
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    if list(ratios) != ["train", "validation", "test"]:
        raise ValueError("ratios must define train, validation, test in that order")
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("split ratios must be positive and sum to 1")
    grouped = {}
    all_names = []
    for record in records:
        filename = str(record.filename)
        label = int(record.binary_label)
        if label not in (0, 1):
            raise ValueError(f"Invalid binary label {label} for {filename}")
        all_names.append(filename)
        grouped.setdefault(label, []).append(filename)
    if not grouped or len(all_names) != len(set(all_names)):
        raise ValueError("records must be non-empty and filenames unique")
    splits = {key: [] for key in ratios}
    for label, filenames in sorted(grouped.items()):
        if len(filenames) < 3:
            raise ValueError(f"class {label} needs at least 3 records for stratified train/validation/test")
        rng = random.Random(seed ^ zlib.crc32(str(label).encode("ascii")))
        rng.shuffle(filenames)
        sizes = _split_sizes(len(filenames), ratios)
        offset = 0
        for split, size in sizes.items():
            splits[split].extend(filenames[offset : offset + size])
            offset += size
    for split, filenames in splits.items():
        random.Random(seed ^ zlib.crc32(split.encode("ascii"))).shuffle(filenames)
    return {
        "dataset": "FloPWD",
        "seed": int(seed),
        "ratios": ratios,
        "splits": splits,
    }


def write_split_manifest(manifest, path):
    """Write a JSON split manifest after validating filename-only contents."""
    path = Path(path)
    _validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_split_manifest(path, expected_filenames=None):
    """Load and validate a split manifest, optionally against current labels."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    if expected_filenames is not None:
        expected, actual = set(expected_filenames), {
            name for names in manifest["splits"].values() for name in names
        }
        if expected != actual:
            raise ValueError(
                f"manifest/data filename mismatch: missing={sorted(expected-actual)[:5]}, "
                f"unexpected={sorted(actual-expected)[:5]}"
            )
    return manifest


def records_for_split(records, manifest, split):
    """Resolve a manifest partition to filename-sorted dataset records.

    The manifest controls membership. Sorting produces canonical order before
    the training-only shuffle in the TensorFlow input pipeline.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    by_name = {record.filename: record for record in records}
    if len(by_name) != len(records):
        raise ValueError("dataset records contain duplicate filenames")
    filenames = manifest["splits"][split]
    missing = sorted(set(filenames) - set(by_name))
    if missing:
        raise ValueError(f"manifest contains filenames missing from data: {missing[:5]}")
    return [by_name[name] for name in sorted(filenames)]


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("dataset") != "FloPWD":
        raise ValueError("manifest dataset must be FloPWD")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or list(splits) != ["train", "validation", "test"]:
        raise ValueError("manifest must contain train, validation, and test splits")
    names = []
    for values in splits.values():
        if not isinstance(values, list) or any(not isinstance(name, str) or not name for name in values):
            raise ValueError("manifest splits must contain non-empty filenames")
        names.extend(values)
    if not names or len(names) != len(set(names)):
        raise ValueError("manifest filenames must be non-empty and appear in exactly one split")
