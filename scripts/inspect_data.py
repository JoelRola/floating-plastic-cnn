"""Inspect local dataset metadata and annotations without training or decoding pixels."""

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floating_plastic.data import inspect_flopwd, inspect_ugv
from floating_plastic.group_splits import (
    make_ugv_grouped_split_manifest, ugv_leakage_report,
    write_ugv_grouped_split_manifest,
)
from floating_plastic.splits import make_split_manifest, write_split_manifest


def _inspect_flopwd(path, image_limit, seed, manifest_path):
    audit = inspect_flopwd(path)
    print(f"Dataset: FloPWD\nRoot: {audit.root}")
    print(f"Images with both label rows: {len(audit.records)}")
    print(f"Binary classes: {audit.binary_counts}")
    print(f"Severity range (percentage points): {audit.severity_range}")
    print(f"Positive labels with zero severity: {len(audit.positive_with_zero_severity)}")
    for title, values in (
        ("duplicate binary names", audit.duplicate_binary_names),
        ("duplicate severity names", audit.duplicate_severity_names),
        ("binary rows missing severity", audit.missing_severity_labels),
        ("severity rows missing binary", audit.missing_binary_labels),
        ("CSV names missing images", audit.missing_images),
        ("images missing one or more labels", audit.unlabeled_images),
        ("expected masks missing", audit.missing_masks),
        ("unexpected masks", audit.extra_masks),
    ):
        print(f"{title}: {len(values)}")
    if not audit.valid:
        raise SystemExit("FloPWD contract validation failed; see discrepancy counts above.")
    records = audit.records[:image_limit] if image_limit else audit.records
    if image_limit:
        print(f"Records selected by explicit --image-limit: {len(records)}")
    manifest = make_split_manifest(records, seed=seed)
    sizes = {key: len(values) for key, values in manifest["splits"].items()}
    print(f"Proposed stratified split sizes (seed={seed}; 70/15/15): {sizes}")
    if manifest_path:
        write_split_manifest(manifest, manifest_path)
        print(f"Filename-only split manifest written: {manifest_path}")


def _inspect_ugv(path, seed=42, write_grouped=False, write_leakage=False):
    audit = inspect_ugv(path)
    print(f"Dataset: UGV-NBWASTE\nRoot: {audit.root}")
    print(f"Class metadata: {'found' if audit.class_names is not None else 'MISSING (class IDs not mapped)'}")
    if audit.class_names is not None:
        print(f"Class names: {audit.class_names}")
    split_records = Counter(record.split for record in audit.records)
    split_empty = Counter(record.split for record in audit.records if record.empty_annotation)
    class_counts = Counter(class_id for record in audit.records for class_id in record.class_ids)
    print(f"Images by split: {audit.image_counts}")
    print(f"Images with readable annotation files by split: {dict(split_records)}")
    print(f"OBB instances by raw class ID: {dict(sorted(class_counts.items()))}")
    print(f"OBB rows with finite coordinates outside [0, 1]: {audit.out_of_bounds_boxes}")
    print(f"Empty annotation files by split: {dict(split_empty)}")
    print(f"Images without annotation file: {len(audit.missing_annotations)}")
    print(f"Orphan annotation files: {len(audit.orphan_annotations)}")
    print(f"Unmapped/observed class IDs: {audit.missing_class_ids}")
    print(f"Source-like filename IDs occurring across splits: {len(audit.duplicate_source_ids_across_splits)}")
    if audit.duplicate_source_ids_across_splits:
        examples = list(audit.duplicate_source_ids_across_splits.items())[:5]
        print(f"Cross-split overlap examples: {examples}")
    if audit.class_names is None:
        print("Note: UGV image-level plastic labels are not inferred from raw object IDs.")
    eligible = sum(bool(record.class_ids) for record in audit.records)
    print(f"Defensible annotated-waste-present image records: {eligible}/{len(audit.records)}")
    if write_leakage:
        report = ugv_leakage_report(audit.records)
        output = Path("experiments/splits/ugv_leakage_report.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(__import__("json").dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Filename-derived split leakage report written: {output}")
    if write_grouped:
        manifest = make_ugv_grouped_split_manifest(audit.records, seed=seed)
        output = Path("experiments/splits") / f"ugv_grouped_seed{seed}.json"
        write_ugv_grouped_split_manifest(manifest, output)
        print(f"Grouped filename-only split written: {output}")
        print(f"Grouped partition counts (groups/images): {manifest['counts']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("flopwd", "ugv"), required=True)
    parser.add_argument("--path", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-limit", type=int)
    parser.add_argument(
        "--write-split-manifest",
        action="store_true",
        help="For FloPWD only, write a filename-only manifest under experiments/splits/",
    )
    parser.add_argument("--write-grouped-split", action="store_true",
                        help="For UGV only, write deterministic source-grouped manifest")
    parser.add_argument("--write-leakage-report", action="store_true",
                        help="For UGV only, write a report for the export's current partitions")
    args = parser.parse_args()
    if args.image_limit is not None and args.image_limit < 1:
        parser.error("--image-limit must be positive")
    if args.dataset == "ugv" and args.write_split_manifest:
        parser.error("--write-split-manifest is currently supported only for FloPWD")
    if args.dataset == "flopwd" and (args.write_grouped_split or args.write_leakage_report):
        parser.error("UGV split/report options are only supported for --dataset ugv")
    if args.dataset == "flopwd":
        manifest_path = (
            Path("experiments/splits") / f"flopwd_seed{args.seed}.json"
            if args.write_split_manifest
            else None
        )
        _inspect_flopwd(args.path, args.image_limit, args.seed, manifest_path)
    else:
        _inspect_ugv(args.path, args.seed, args.write_grouped_split, args.write_leakage_report)


if __name__ == "__main__":
    main()
