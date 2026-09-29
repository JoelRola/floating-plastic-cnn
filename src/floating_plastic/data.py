"""Metadata and annotation readers for FloPWD and YOLO-OBB UGV exports."""

from dataclasses import dataclass
import math
from pathlib import Path
from pathlib import PureWindowsPath
import re

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FLOPWD_BINARY_CSV = "Image_labels_Binary Classification Task.csv"
FLOPWD_SEVERITY_CSV = "Mask_foreground_percentages_Regression Task.csv"


@dataclass(frozen=True)
class FloPWDRecord:
    filename: str
    binary_label: int
    severity_percent: float
    image_path: Path


@dataclass(frozen=True)
class UGVRecord:
    split: str
    filename: str
    image_path: Path
    annotation_path: Path
    class_ids: tuple[int, ...]
    empty_annotation: bool


@dataclass
class FloPWDAudit:
    root: Path
    records: list[FloPWDRecord]
    binary_counts: dict[str, int]
    severity_range: tuple[float, float] | None
    duplicate_binary_names: list[str]
    duplicate_severity_names: list[str]
    missing_severity_labels: list[str]
    missing_binary_labels: list[str]
    missing_images: list[str]
    unlabeled_images: list[str]
    missing_masks: list[str]
    extra_masks: list[str]
    positive_with_zero_severity: list[str]

    @property
    def valid(self):
        return not any(
            (
                self.duplicate_binary_names,
                self.duplicate_severity_names,
                self.missing_severity_labels,
                self.missing_binary_labels,
                self.missing_images,
                self.unlabeled_images,
                self.missing_masks,
                self.extra_masks,
            )
        )


@dataclass
class UGVAudit:
    root: Path
    class_names: dict[int, str] | None
    records: list[UGVRecord]
    image_counts: dict[str, int]
    missing_class_ids: list[int]
    missing_annotations: list[str]
    orphan_annotations: list[str]
    duplicate_source_ids_across_splits: dict[str, list[str]]
    out_of_bounds_boxes: int


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    import csv

    if not path.is_file():
        raise FileNotFoundError(f"Required label CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(required_columns - fields)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        return list(reader)


def inspect_flopwd(root) -> FloPWDAudit:
    """Inspect FloPWD metadata and files without loading image pixels."""
    root = Path(root)
    image_dir = root / "Raw_Images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"FloPWD image directory not found: {image_dir}")
    binary_rows = _read_csv(root / FLOPWD_BINARY_CSV, {"image name", "Presence of plastic waste?"})
    severity_rows = _read_csv(root / FLOPWD_SEVERITY_CSV, {"image name", "plastic waste accumulation (in percentage)"})

    def collect(rows, path, label_column, parser):
        mapping, duplicates = {}, []
        for row_num, row in enumerate(rows, start=2):
            filename = (row.get("image name") or "").strip()
            if not filename or Path(filename).name != filename or PureWindowsPath(filename).name != filename:
                raise ValueError(f"Invalid image name in {path}:{row_num}: {filename!r}")
            try:
                value = parser((row.get(label_column) or "").strip())
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid label in {path}:{row_num} for {filename!r}") from exc
            if filename in mapping:
                duplicates.append(filename)
            mapping[filename] = value
        return mapping, sorted(set(duplicates))

    binary, duplicate_binary = collect(
        binary_rows,
        root / FLOPWD_BINARY_CSV,
        "Presence of plastic waste?",
        lambda value: {"no": 0, "yes": 1}[value.lower()],
    )

    def severity_value(value):
        severity = float(value)
        if not 0 <= severity <= 100:
            raise ValueError("severity must be in percentage points [0, 100]")
        return severity

    severity, duplicate_severity = collect(
        severity_rows,
        root / FLOPWD_SEVERITY_CSV,
        "plastic waste accumulation (in percentage)",
        severity_value,
    )

    binary_names, severity_names = set(binary), set(severity)
    image_paths = sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )
    image_names = {p.name for p in image_paths}
    masks_dir = root / "Segmentation_Masks"
    mask_names = {p.name for p in masks_dir.iterdir() if p.is_file()} if masks_dir.is_dir() else set()
    expected_masks = {f"{Path(name).stem}_mask.png" for name in binary_names | severity_names}
    common = sorted(binary_names & severity_names & image_names)
    records = [FloPWDRecord(n, binary[n], severity[n], image_dir / n) for n in common]
    values = [r.severity_percent for r in records]
    return FloPWDAudit(
        root=root,
        records=records,
        binary_counts={"no": sum(v == 0 for v in binary.values()), "yes": sum(v == 1 for v in binary.values())},
        severity_range=(min(values), max(values)) if values else None,
        duplicate_binary_names=duplicate_binary,
        duplicate_severity_names=duplicate_severity,
        missing_severity_labels=sorted(binary_names - severity_names),
        missing_binary_labels=sorted(severity_names - binary_names),
        missing_images=sorted((binary_names | severity_names) - image_names),
        unlabeled_images=sorted(image_names - (binary_names & severity_names)),
        missing_masks=sorted(expected_masks - mask_names),
        extra_masks=sorted(mask_names - expected_masks),
        positive_with_zero_severity=sorted(n for n in common if binary[n] == 1 and severity[n] == 0),
    )


def load_flopwd(root, image_limit=None) -> list[FloPWDRecord]:
    """Load validated, filename-sorted FloPWD records (coverage is 0–100%)."""
    audit = inspect_flopwd(root)
    problems = {
        "duplicate binary filenames": audit.duplicate_binary_names,
        "duplicate severity filenames": audit.duplicate_severity_names,
        "binary rows missing severity": audit.missing_severity_labels,
        "severity rows missing binary": audit.missing_binary_labels,
        "CSV filenames absent from Raw_Images": audit.missing_images,
        "images without both labels": audit.unlabeled_images,
        "expected segmentation masks absent": audit.missing_masks,
        "unexpected segmentation mask files": audit.extra_masks,
    }
    errors = [f"{name}: {values[:5]}" for name, values in problems.items() if values]
    if errors:
        raise ValueError("Invalid FloPWD dataset contract; " + "; ".join(errors))
    if image_limit is not None:
        if not isinstance(image_limit, int) or image_limit < 1:
            raise ValueError("image_limit must be a positive integer or None")
        return audit.records[:image_limit]
    return audit.records


def _parse_class_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"UGV class metadata not found: {data_yaml}. Supply the original data.yaml; "
            "class IDs will not be guessed from the paper."
        )
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"{data_yaml} must contain a YOLO 'names' mapping or list")
    names = data["names"]
    if isinstance(names, list):
        mapping = {i: str(name) for i, name in enumerate(names)}
    elif isinstance(names, dict):
        try:
            mapping = {int(i): str(name) for i, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Class IDs in {data_yaml} must be integers") from exc
    else:
        raise ValueError(f"{data_yaml} 'names' must be a list or mapping")
    if not mapping or len(mapping) != len(names) or any(not name.strip() for name in mapping.values()):
        raise ValueError(f"{data_yaml} contains duplicate IDs or empty class names")
    return mapping


def parse_obb_annotation(path, class_names=None) -> tuple[int, ...]:
    """Parse YOLO oriented boxes: class id followed by four normalized x/y pairs."""
    path = Path(path)
    classes = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 9:
            raise ValueError(f"Expected class ID and 8 OBB coordinates at {path}:{line_num}")
        try:
            class_id = int(fields[0])
            coords = [float(v) for v in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid OBB annotation at {path}:{line_num}") from exc
        if class_id < 0 or any(not math.isfinite(v) for v in coords):
            raise ValueError(f"Class IDs must be non-negative and coordinates finite at {path}:{line_num}")
        if class_names is not None and class_id not in class_names:
            raise ValueError(f"Unknown class ID {class_id} at {path}:{line_num}")
        classes.append(class_id)
    return tuple(classes)


def inspect_ugv(root) -> UGVAudit:
    """Inspect YOLO-OBB files, retaining raw IDs if class YAML is absent."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"UGV dataset directory not found: {root}")
    yaml_path = next((p for p in (root / "data.yaml", root / "data.yml") if p.is_file()), None)
    names = _parse_class_names(yaml_path) if yaml_path else None
    records, image_counts, missing_annotations, orphan_annotations = [], {}, [], []
    source_splits: dict[str, set[str]] = {}
    ids_seen: set[int] = set()
    out_of_bounds_boxes = 0
    for split in ("train", "valid", "test"):
        image_dir, label_dir = root / split / "images", root / split / "labels"
        if not image_dir.is_dir() and not label_dir.is_dir():
            continue
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"UGV {split} must contain both images/ and labels/ directories")
        images = sorted((p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name)
        image_counts[split] = len(images)
        labels = {p.stem: p for p in label_dir.glob("*.txt") if p.is_file()}
        image_stems = {p.stem for p in images}
        missing_annotations.extend(f"{split}/{name}" for name in sorted(image_stems - labels.keys()))
        orphan_annotations.extend(f"{split}/{name}.txt" for name in sorted(labels.keys() - image_stems))
        for image in images:
            label = labels.get(image.stem)
            if label is None:
                continue
            class_ids = parse_obb_annotation(label, names)
            ids_seen.update(class_ids)
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                fields = line.split()
                if fields and len(fields) == 9:
                    coords = [float(value) for value in fields[1:]]
                    out_of_bounds_boxes += int(any(value < 0 or value > 1 for value in coords))
            records.append(UGVRecord(split, image.name, image, label, class_ids, not class_ids))
            source_id = re.sub(r"\.rf\.[^.]+$", "", image.stem, flags=re.IGNORECASE)
            source_splits.setdefault(source_id, set()).add(split)
    duplicate_source_ids = {
        source_id: sorted(splits)
        for source_id, splits in sorted(source_splits.items())
        if len(splits) > 1
    }
    missing_ids = sorted(ids_seen - set(names)) if names is not None else sorted(ids_seen)
    return UGVAudit(
        root, names, records, image_counts, missing_ids, sorted(missing_annotations),
        sorted(orphan_annotations), duplicate_source_ids, out_of_bounds_boxes
    )


def load_ugv(root, require_class_metadata=True) -> list[UGVRecord]:
    """Load OBB records; class metadata is mandatory by default."""
    root = Path(root)
    if require_class_metadata:
        metadata_path = next((p for p in (root / "data.yaml", root / "data.yml") if p.is_file()), root / "data.yaml")
        _parse_class_names(metadata_path)
    audit = inspect_ugv(root)
    if audit.missing_annotations or audit.orphan_annotations or audit.missing_class_ids:
        raise ValueError(
            "Invalid UGV export: "
            f"missing annotations={audit.missing_annotations[:5]}, "
            f"orphan annotations={audit.orphan_annotations[:5]}, "
            f"unmapped class IDs={audit.missing_class_ids[:5]}"
        )
    if audit.class_names is None:
        raise ValueError("UGV class metadata is required; refusing to guess class mappings")
    return audit.records
