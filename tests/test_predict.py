from pathlib import Path

from scripts.predict import make_prediction_report, resolve_severity_status


def test_control_a_can_report_floating_coverage():
    report = make_prediction_report("flopwd", 0.97, 0.5, 12.4, "functional")
    assert report["classification"] == "plastic detected"
    assert report["estimated_flopwd_style_coverage_percent"] == 12.4


def test_collapsed_severity_is_withheld():
    report = make_prediction_report("flopwd", 0.97, 0.5, 0.0, "collapsed_not_for_use")
    assert report["severity_estimate"] is None
    assert "withheld" in report["severity_note"]


def test_ugv_output_uses_annotated_waste_semantics_and_no_severity():
    report = make_prediction_report("ugv", 0.97, 0.5, 42.0, "functional")
    assert report["classification"] == "annotated waste detected"
    assert report["severity_estimate"] is None


def test_unknown_model_capability_defaults_to_unverified():
    assert resolve_severity_status(Path("untracked/model.keras")) == "unverified"


def test_completed_model_capabilities_are_read_from_result_summaries():
    assert resolve_severity_status(Path("runs/flopwd_original_seed42/model.keras")) == "functional"
    assert resolve_severity_status(Path("runs/multidomain_class_balanced_seed42/model.keras")) == "collapsed_not_for_use"
