from pathlib import Path

import pytest

from advoice.config import ProjectPaths, load_all, paths
from advoice.cli import parser
from advoice.pipeline import _bootstrap_processed_artifacts, _config_files


def test_external_workspace_keeps_code_and_configuration_separate(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVOICE_WORKSPACE_DIR", str(tmp_path / "private"))
    p = ProjectPaths(tmp_path / "code")
    assert p.configs == tmp_path / "code/configs"
    assert p.artifacts == tmp_path / "private/artifacts"
    assert p.reports == tmp_path / "private/reports"
    assert p.runs == tmp_path / "private/runs"


def test_raw_root_and_model_configuration_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVOICE_RAW_DATA_DIR", str(tmp_path / "raw"))
    model = tmp_path / "pilot.yaml"
    model.write_text("condition_c: {deep_audio: {enabled: false}}\n")
    monkeypatch.setenv("ADVOICE_MODEL_CONFIG", str(model))
    config = load_all("NCMMSC2021_AD")
    assert Path(config["dataset"]["raw_path"]) == tmp_path / "raw/NCMMSC2021_AD"
    assert config["models"]["condition_c"]["deep_audio"]["enabled"] is False
    assert _config_files(paths(), "NCMMSC2021_AD")["models"] == model


def test_processed_cli_exposes_frozen_source():
    args = parser().parse_args(["run-processed", "--source-root", "/private/frozen"])
    assert args.source_root == Path("/private/frozen")


def test_bootstrap_fails_before_modifying_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "subject_features.csv").write_text("incomplete")
    destination = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="Required frozen inputs"):
        _bootstrap_processed_artifacts(source, destination)
    assert not destination.exists()
    with pytest.raises(ValueError, match="different directories"):
        _bootstrap_processed_artifacts(source, source)


def test_partial_batch_marks_focus_analysis_unavailable(tmp_path, monkeypatch):
    from advoice.failure_analysis import build_failure_analysis
    import json
    monkeypatch.delenv("ADVOICE_WORKSPACE_DIR", raising=False)
    report = build_failure_analysis(ProjectPaths(tmp_path))
    assert report.is_file()
    status = json.loads((report.parent / "status.json").read_text())
    assert status["status"] == "unavailable"
    assert status["missing"]
