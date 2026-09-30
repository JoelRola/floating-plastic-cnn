import csv
from pathlib import Path

import pytest

from floating_plastic.data import (
    HeterogeneousRecord, canonical_ugv_source_id, heterogeneous_ugv_record,
    inspect_flopwd, inspect_ugv, load_flopwd, load_ugv, parse_obb_annotation,
)


def make_flopwd(root, binary_rows=None, severity_rows=None, image_names=None):
    root.mkdir(parents=True, exist_ok=True)
    images, masks = root / "Raw_Images", root / "Segmentation_Masks"
    images.mkdir()
    masks.mkdir()
    binary_rows = binary_rows or [("b.jpg", "no"), ("a.jpg", "yes")]
    severity_rows = severity_rows or [("b.jpg", "0"), ("a.jpg", "12.5")]
    image_names = image_names if image_names is not None else [row[0] for row in binary_rows]
    for name in image_names:
        (images / name).touch()
    for name, _ in severity_rows:
        (masks / f"{Path(name).stem}_mask.png").touch()
    for filename, headers, rows in (
        ("Image_labels_Binary Classification Task.csv", ["image name", "Presence of plastic waste?"], binary_rows),
        ("Mask_foreground_percentages_Regression Task.csv", ["image name", "plastic waste accumulation (in percentage)"], severity_rows),
    ):
        with (root / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)


def test_flopwd_contract_orders_records_and_preserves_percentage_points(tmp_path):
    make_flopwd(tmp_path / "flo")
    records = load_flopwd(tmp_path / "flo")
    assert [record.filename for record in records] == ["a.jpg", "b.jpg"]
    assert [record.severity_percent for record in records] == [12.5, 0.0]
    assert records[0].binary_label == 1


def test_flopwd_audit_reports_duplicate_names_and_loader_fails(tmp_path):
    root = tmp_path / "flo"
    make_flopwd(root, binary_rows=[("a.jpg", "yes"), ("a.jpg", "yes")])
    audit = inspect_flopwd(root)
    assert audit.duplicate_binary_names == ["a.jpg"]
    with pytest.raises(ValueError, match="duplicate binary"):
        load_flopwd(root)


def test_flopwd_rejects_missing_and_mismatched_labels(tmp_path):
    root = tmp_path / "flo"
    make_flopwd(root, severity_rows=[("a.jpg", "12.5")])
    audit = inspect_flopwd(root)
    assert audit.missing_severity_labels == ["b.jpg"]
    with pytest.raises(ValueError, match="binary rows missing severity"):
        load_flopwd(root)


def test_flopwd_missing_label_image_and_mask_fail_validation(tmp_path):
    root = tmp_path / "flo"
    make_flopwd(root, severity_rows=[("a.jpg", "101"), ("b.jpg", "0")])
    # Invalid percentages are rejected during parsing.
    with pytest.raises(ValueError, match="Invalid label"):
        load_flopwd(root)

    make_flopwd(root / "other", image_names=["a.jpg", "b.jpg", "extra.jpg"])
    audit = inspect_flopwd(root / "other")
    assert audit.unlabeled_images == ["extra.jpg"]
    with pytest.raises(ValueError, match="images without both labels"):
        load_flopwd(root / "other")


def test_flopwd_missing_mask_is_reported_and_rejected(tmp_path):
    root = tmp_path / "flo"
    make_flopwd(root)
    (root / "Segmentation_Masks/a_mask.png").unlink()
    audit = inspect_flopwd(root)
    assert audit.missing_masks == ["a_mask.png"]
    with pytest.raises(ValueError, match="segmentation masks absent"):
        load_flopwd(root)


def test_ugv_reads_yaml_obb_and_empty_annotations(tmp_path):
    root = tmp_path / "ugv"
    root.mkdir()
    (root / "data.yaml").write_text(
        "train: train/images\nval: valid/images\nnames:\n  0: ClassA\n  1: ClassB\n",
        encoding="utf-8",
    )
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
    (root / "train/images/object.jpg").touch()
    (root / "train/labels/object.txt").write_text("1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
    (root / "valid/images/empty.jpg").touch()
    (root / "valid/labels/empty.txt").write_text("  \n")

    audit = inspect_ugv(root)
    assert audit.class_names == {0: "ClassA", 1: "ClassB"}
    assert len(load_ugv(root)) == 2
    assert audit.records[0].class_ids == (1,)
    assert audit.records[0].annotation_count == 1
    assert audit.records[0].source_dataset == "ugv"
    assert audit.records[0].canonical_source_id == "object"
    assert audit.records[1].empty_annotation
    assert parse_obb_annotation(root / "train/labels/object.txt", audit.class_names) == (1,)


def test_ugv_requires_metadata_and_rejects_invalid_obb(tmp_path):
    root = tmp_path / "ugv"
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
    with pytest.raises(FileNotFoundError, match="data.yaml"):
        load_ugv(root)
    label = root / "train/labels/bad.txt"
    label.write_text("0 0.1 0.2\n")
    with pytest.raises(ValueError, match="8 OBB coordinates"):
        parse_obb_annotation(label)


def test_canonical_ugv_source_ids_strip_only_roboflow_hash_suffix():
    assert canonical_ugv_source_id("Scene_01.rf.0123456789abcdef.jpg") == "scene_01"
    assert canonical_ugv_source_id("Scene_01.rf.short.jpg") == "scene_01.rf.short"


def test_ugv_empty_and_filtered_annotations_are_unavailable_not_negative(tmp_path):
    from floating_plastic.data import UGVRecord
    empty = UGVRecord("valid", "empty.jpg", tmp_path / "empty.jpg", tmp_path / "empty.txt", (), True)
    mapped_out = UGVRecord("valid", "other.jpg", tmp_path / "other.jpg", tmp_path / "other.txt", (2,), False)
    waste = UGVRecord("valid", "waste.jpg", tmp_path / "waste.jpg", tmp_path / "waste.txt", (2, 4), False)
    for record in (empty, mapped_out):
        sample = heterogeneous_ugv_record(record, included_class_ids={4})
        assert sample.classification_target is None
        assert not sample.classification_target_available
        assert sample.severity_target is None and not sample.severity_target_available
    eligible = heterogeneous_ugv_record(waste, included_class_ids={4})
    assert (eligible.classification_target, eligible.classification_target_available) == (1, True)
    assert eligible.source_domain == "ugv"


def test_heterogeneous_record_validates_task_availability(tmp_path):
    with pytest.raises(ValueError, match="availability flag disagree"):
        HeterogeneousRecord("x.jpg", tmp_path / "x.jpg", None, True, None, False, "ugv")
