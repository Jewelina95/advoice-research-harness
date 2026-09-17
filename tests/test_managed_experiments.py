import json
import subprocess
from pathlib import Path

import pytest
import yaml

from advoice import experiments as e
from advoice.cli import parser


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "configs/datasets").mkdir(parents=True)
    (root / "configs/models").mkdir()
    (root / "configs/models/default.yaml").write_text("labels: [HC, AD]\n")
    (root / "configs/datasets/Demo.yaml").write_text("raw_path: data/raw/Demo\n")
    source = tmp_path / "source"
    (source / "Demo").mkdir(parents=True)
    for name in e.FROZEN_INPUTS:
        (source / "Demo" / name).write_text("fixture\n")
    recipe = tmp_path / "recipe.yaml"
    value = {"schema_version": 1, "workspace": "workspace", "source_root": "source",
             "datasets": ["Demo"], "input_mode": "processed"}
    recipe.write_text(yaml.safe_dump(value))
    monkeypatch.setattr(e, "project_root", lambda: root)
    monkeypatch.setattr(e, "git_identity", lambda _: {"commit": "abc", "branch": "test", "dirty": False})
    return root, recipe, value


def outputs(directory, names):
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\nfixture" if path.suffix == ".png" else b"<html>fixture</html>")


def runner(monkeypatch, workspace, failure=None):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        if failure == "process":
            raise subprocess.CalledProcessError(1, command)
        if "aggregate-report" in command:
            snapshot = Path(kwargs["env"]["ADVOICE_WORKSPACE_DIR"])
            outputs(snapshot / "reports/latest", e.AGGREGATE_OUTPUTS)
            return
        if failure == "stale":
            return
        run_dir = workspace / "runs/new"
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts/evidence.csv").write_text("immutable fixture")
        outputs(run_dir / "reports", e.DATASET_OUTPUTS)
        if failure == "missing":
            (run_dir / "reports/assets/layer_b_summary.png").unlink()
        latest = workspace / "reports/latest_runs.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps({"Demo": {"run_id": "new", "run_dir": str(run_dir)}}))
    monkeypatch.setattr(e.subprocess, "run", run)
    return calls


def test_preflight_does_not_execute_or_create_workspace(setup, monkeypatch):
    root, recipe, _ = setup
    monkeypatch.setattr(e.subprocess, "run", lambda *a, **k: pytest.fail("No execution"))
    result = e.run_experiment(recipe, True)
    assert result["status"] == "preflight_passed"
    assert not (recipe.parent / "workspace").exists()
    assert result["recipe"]["agent_provider"] == "disabled"


@pytest.mark.parametrize("change", [
    {"datasets": ["Demo", "Demo"]}, {"datasets": ["../Demo"]}, {"datasets": ["Unknown"]},
    {"force": "false"}, {"unknown_setting": True}, {"workspace": "repo"},
    {"workspace": "source/results"}, {"mode": "quick"}, {"schema_version": True},
])
def test_invalid_recipe_rejected(setup, change):
    root, recipe, value = setup
    recipe.write_text(yaml.safe_dump(value | change))
    with pytest.raises(ValueError):
        e.load_recipe(recipe, root)


def test_missing_input_rejected_before_any_output(setup):
    root, recipe, _ = setup
    (recipe.parent / "source/Demo/segments.csv").unlink()
    with pytest.raises(ValueError, match="Missing frozen input"):
        e.run_experiment(recipe)
    assert not (recipe.parent / "workspace").exists()


def test_workspace_lock_rejects_competing_writer(tmp_path):
    with e.workspace_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"):
            with e.workspace_lock(tmp_path):
                pytest.fail("Must not acquire another writer")
    assert not (tmp_path / ".experiment.lock").exists()


def test_completion_requires_new_run_and_all_report_outputs(setup, monkeypatch):
    _, recipe, _ = setup
    workspace = recipe.parent / "workspace"
    monkeypatch.setenv("ADVOICE_MODEL_CONFIG", "/wrong/inherited.yaml")
    monkeypatch.setenv("ADVOICE_RAW_DATA_DIR", "/wrong/raw")
    calls = runner(monkeypatch, workspace)
    result = e.run_experiment(recipe)
    assert result["status"] == "completed"
    assert len(result["aggregate"]) == 4
    assert result["aggregate_datasets"] == ["Demo"]
    assert "--datasets" in calls[-1][0] and calls[-1][0][-1] == "Demo"
    assert "ADVOICE_MODEL_CONFIG" not in calls[0][1]
    assert "ADVOICE_RAW_DATA_DIR" not in calls[0][1]
    assert json.loads((workspace / "experiment_latest.json").read_text())["status"] == "completed"
    archived = Path(result["aggregate"]["system_report.html"]["path"])
    assert archived.is_relative_to(Path(result["execution_dir"]))
    (workspace / "reports/latest/system_report.html").write_text("later run")
    assert e.digest(archived) == result["aggregate"]["system_report.html"]["sha256"]
    assert (Path(result["execution_dir"]) / "aggregate/artifacts/Demo/evidence.csv").read_text() == "immutable fixture"


@pytest.mark.parametrize("failure", ["process", "missing", "stale"])
def test_failure_never_uses_old_report_as_success(setup, monkeypatch, failure):
    _, recipe, _ = setup
    workspace = recipe.parent / "workspace"
    outputs(workspace / "reports/latest", e.AGGREGATE_OUTPUTS)
    latest = workspace / "reports/latest_runs.json"
    latest.write_text(json.dumps({"Demo": {"run_id": "old", "run_dir": str(workspace / "runs/old")}}))
    calls = runner(monkeypatch, workspace, failure)
    result = e.run_experiment(recipe)
    assert result["status"] == "failed"
    assert result["aggregate"] is None
    assert result["datasets"][0]["status"] == "failed"
    assert len(calls) == 1


def test_missing_or_invalid_png_rejected(tmp_path):
    (tmp_path / "chart.png").write_text("Not a figure")
    with pytest.raises(RuntimeError, match="Invalid PNG"):
        e.checked_outputs(tmp_path, ("chart.png",))


def test_report_with_missing_local_figure_fails(tmp_path):
    report = tmp_path / "evaluation.html"
    report.write_text('<img src="assets/missing.png"><a href="#results">Results</a>')
    with pytest.raises(RuntimeError, match="Broken local report link"):
        e.checked_outputs(tmp_path, ("evaluation.html",))


def test_report_anchor_and_external_links_do_not_need_local_files(tmp_path):
    (tmp_path / "evaluation.html").write_text('<a href="#results">Results</a><a href="https://example.org">Source</a>')
    assert e.checked_outputs(tmp_path, ("evaluation.html",))


def test_cli_contract():
    args = parser().parse_args(["experiment", "--config", "local.yaml", "--check"])
    assert args.check and args.config == Path("local.yaml")
    assert parser().parse_args(["aggregate-report", "--datasets", "Demo"]).datasets == ["Demo"]


def test_raw_recipe_and_force_command(setup):
    root, recipe, _ = setup
    (recipe.parent / "raw/Demo").mkdir(parents=True)
    recipe.write_text(yaml.safe_dump({"schema_version": 1, "workspace": "workspace",
                                     "raw_data_root": "raw", "datasets": ["Demo"], "force": True}))
    value = e.load_recipe(recipe, root)
    command = e.command_for(value, "Demo")
    assert "run" in command and "--mode" in command and "--force" in command
    assert "--source-root" not in command


@pytest.mark.parametrize("name", ["artifacts", "reports", "executions", "experiment_latest.json", "reports/nested"])
def test_symlinked_outputs_rejected_before_execution(setup, name):
    _, recipe, _ = setup
    workspace = recipe.parent / "workspace"
    destination = workspace / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(recipe.parent / "source", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|Symlink"):
        e.run_experiment(recipe, True)


def test_legacy_cli_respects_managed_workspace_lock(setup, monkeypatch):
    import sys
    from advoice import cli
    _, recipe, _ = setup
    workspace = recipe.parent / "workspace"
    monkeypatch.setenv("ADVOICE_WORKSPACE_DIR", str(workspace))
    monkeypatch.delenv(e.LOCK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["advoice", "aggregate-report", "--datasets", "Demo"])
    monkeypatch.setattr(cli, "_dispatch", lambda _: pytest.fail("A second writer must not run"))
    with e.workspace_lock(workspace):
        with pytest.raises(RuntimeError, match="locked"):
            cli.main()


def test_child_with_matching_token_can_use_parent_lock(tmp_path, monkeypatch):
    with e.workspace_lock(tmp_path) as token:
        monkeypatch.setenv(e.LOCK_TOKEN_ENV, token)
        with e.workspace_lock(tmp_path) as inherited:
            assert inherited == token
        assert (tmp_path / ".experiment.lock").exists()
    assert not (tmp_path / ".experiment.lock").exists()
