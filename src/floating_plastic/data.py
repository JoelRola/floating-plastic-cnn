"""Dataset readers with explicit schemas; UGV labels require an adapter."""

import csv
from pathlib import Path


def load_flopwd_labels(root, image_limit=None):
    """Read paired FloPWD CSV labels and return sorted (name, class, coverage) rows.

    Coverage is returned in percentage points (0–100), matching the CSV's
    documented percentage units. ``image_limit`` applies after deterministic
    filename sorting.
    """
    root = Path(root)
    binary_path = root / "Image_labels_Binary Classification Task.csv"
    coverage_path = root / "Mask_foreground_percentages_Regression Task.csv"
    for path in (binary_path, coverage_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required FloPWD label file not found: {path}")
    images_dir = root / "Raw_Images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Required FloPWD image directory not found: {images_dir}")
    binary, coverage = {}, {}
    with binary_path.open(newline="", encoding="utf-8-sig") as stream:
        for row_num, row in enumerate(csv.DictReader(stream), start=2):
            name = (row.get("image name") or "").strip()
            label = (row.get("Presence of plastic waste?") or "").strip().lower()
            if not name or label not in {"yes", "no"}:
                raise ValueError(f"Invalid FloPWD binary label at {binary_path}:{row_num}")
            if name in binary:
                raise ValueError(f"Duplicate image name {name!r} in {binary_path}")
            binary[name] = int(label == "yes")
    with coverage_path.open(newline="", encoding="utf-8-sig") as stream:
        for row_num, row in enumerate(csv.DictReader(stream), start=2):
            name = (row.get("image name") or "").strip()
            raw = (row.get("plastic waste accumulation (in percentage)") or "").strip()
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid FloPWD coverage at {coverage_path}:{row_num}") from exc
            if not name or not 0 <= value <= 100:
                raise ValueError(f"Invalid FloPWD coverage at {coverage_path}:{row_num}")
            if name in coverage:
                raise ValueError(f"Duplicate image name {name!r} in {coverage_path}")
            coverage[name] = value
    if binary.keys() != coverage.keys():
        missing_coverage = sorted(binary.keys() - coverage.keys())[:5]
        missing_binary = sorted(coverage.keys() - binary.keys())[:5]
        raise ValueError(
            "FloPWD label files contain different image names; "
            f"missing coverage={missing_coverage}, missing binary={missing_binary}"
        )
    missing_images = [name for name in sorted(binary) if not (images_dir / name).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"FloPWD image files are missing from {images_dir}: {missing_images[:5]}"
        )
    rows = [(name, binary[name], coverage[name]) for name in sorted(binary)]
    if image_limit is not None:
        if not isinstance(image_limit, int) or image_limit < 1:
            raise ValueError("image_limit must be a positive integer or None")
        rows = rows[:image_limit]
    return rows


def load_ugv_labels(root, schema=None):
    """Load UGV annotations only after a verified class-name schema is supplied.

    The supplied YOLO label files contain numeric class IDs, but this project
    has not verified which IDs represent plastic. Pass a mapping from integer
    class ID to binary plastic presence (0 or 1) once confirmed from the source
    dataset documentation. No mapping is guessed here.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"UGV dataset directory not found: {root}")
    if schema is None:
        raise NotImplementedError(
            "UGV class IDs have not been mapped to plastic presence. Supply a "
            "verified class-ID-to-binary-label schema before using UGV data."
        )
    raise NotImplementedError("UGV annotation parsing awaits verified dataset schema documentation")
