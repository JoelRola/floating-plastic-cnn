import json
from collections import Counter

from floating_plastic.data import FloPWDRecord
from floating_plastic.splits import load_split_manifest, make_split_manifest, write_split_manifest


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
