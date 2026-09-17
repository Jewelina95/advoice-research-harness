import json

import pandas as pd
import pytest
import yaml

from advoice import evidence_research_run as runner


def test_feature_loader_excludes_labels_and_transcripts(tmp_path):
    source = tmp_path / "features.csv"
    pd.DataFrame({"subject_id": ["001"], "split": ["train"], "label": [1],
                  "text": ["private text"], "x": [2.]}).to_csv(source, index=False)
    result = runner.load_subject_features(source, {"x"}, None)
    assert set(result.columns) == {"subject_id", "split", "x"}
    assert result.subject_id.iloc[0] == "001"


def test_metadata_join_checks_split_and_cardinality(tmp_path):
    source, meta = tmp_path / "f.csv", tmp_path / "m.csv"
    pd.DataFrame({"subject_id": ["001"], "split": ["train"], "x": [2.]}).to_csv(source, index=False)
    pd.DataFrame({"subject_id": ["001"], "split": ["train"], "language": ["en"]}).to_csv(meta, index=False)
    assert runner.load_subject_features(source, {"x"}, meta).language.iloc[0] == "en"
    pd.DataFrame({"subject_id": ["001"], "split": ["test"]}).to_csv(meta, index=False)
    with pytest.raises(ValueError, match="disagreement"):
        runner.load_subject_features(source, {"x"}, meta)
    pd.DataFrame({"subject_id": ["001", "001"], "language": ["en", "en"]}).to_csv(meta, index=False)
    with pytest.raises(ValueError, match="unique"):
        runner.load_subject_features(source, {"x"}, meta)


def study_recipe(tmp_path, monkeypatch):
    source = tmp_path / "src" / "advoice" / "evidence_research_run.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic source fixture\n")
    source.with_name("evidence_research.py").write_text("# synthetic source fixture\n")
    monkeypatch.setattr(runner, "__file__", str(source))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "synthetic-test-revision\n")
    metrics, states = tmp_path / "metrics.yaml", tmp_path / "states.yaml"
    metrics.write_text(yaml.safe_dump({"metrics": [{"id": "x", "role": "qc_only"}]}))
    states.write_text(yaml.safe_dump({"states": [{"id": "S1", "metrics": ["x"], "weights": [1.]}]}))
    history, features = tmp_path / "history.csv", tmp_path / "features.csv"
    pd.DataFrame({"metric_name": ["x"], "state_id": ["S1"], "metric_definition": ["synthetic"],
                  "how_calculated": ["synthetic"], "matched_current_columns": ["x"]}).to_csv(history, index=False)
    pd.DataFrame({"subject_id": ["synthetic-private-001", "synthetic-private-002"],
                  "split": ["train", "test"], "x": [1., 2.]}).to_csv(features, index=False)
    recipe = {
        "output_dir": str(tmp_path / ".local" / "run"), "metrics_config": str(metrics),
        "states_config": str(states), "historical_dictionary": str(history),
        "datasets": [{"id": "synthetic", "features_path": str(features)}],
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path, recipe


def test_runner_keeps_aggregates_and_refuses_overwrite(tmp_path, monkeypatch):
    path, recipe = study_recipe(tmp_path, monkeypatch)
    result = runner.run_study(path)
    assert result["datasets"]["synthetic"]["n_train_records"] == 1
    output = tmp_path / ".local" / "run"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed_descriptive_analysis_only"
    assert all(len(item["sha256"]) == 64 for item in manifest["inputs"])
    assert not any("synthetic-private" in p.read_text() for p in output.rglob("*") if p.is_file())
    with pytest.raises(FileExistsError):
        runner.run_study(path)


def test_runner_restricts_private_output_location(tmp_path, monkeypatch):
    path, recipe = study_recipe(tmp_path, monkeypatch)
    recipe["output_dir"] = str(tmp_path / "public_reports")
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="Private study outputs"):
        runner.run_study(path)


def test_runner_marks_partial_outputs_failed(tmp_path, monkeypatch):
    path, recipe = study_recipe(tmp_path, monkeypatch)
    pd.DataFrame({"subject_id": ["synthetic"], "split": ["unknown"], "x": [1.]}).to_csv(recipe["datasets"][0]["features_path"], index=False)
    with pytest.raises(ValueError, match="split"):
        runner.run_study(path)
    manifest = json.loads((tmp_path / ".local/run/manifest.json").read_text())
    assert manifest["status"] == "failed_do_not_interpret_partial_outputs"
