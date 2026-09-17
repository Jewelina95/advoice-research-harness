"""Run a versioned recipe against private local data with checked report outputs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .config import load_yaml, project_root
from .workspace import LOCK_TOKEN_ENV, validate_output_paths, workspace_lock


DATASET_OUTPUTS = (
    "system_report.html", "evaluation_report.html",
    "assets/layer_a_summary.png", "assets/layer_b_summary.png",
)
AGGREGATE_OUTPUTS = (
    "system_report.html", "aggregate_evaluation_report.html",
    "assets/figure_A_layer_medical_standards_summary.png",
    "assets/figure_B_layer_framework_validation_summary.png",
)
FROZEN_INPUTS = (
    "subject_features.csv", "subject_transcripts.csv", "metric_evidence.csv",
    "b1_predictions.csv", "b2_predictions.csv", "recording_features.csv",
    "segments.csv", "manifest.csv",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_status(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_identity(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain"))}


def source_fingerprint(root: Path) -> str:
    inventory = {}
    for directory in ("src", "configs", "schemas", "templates", "skills"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                inventory[str(path.relative_to(root))] = digest(path)
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def load_recipe(path: Path, root: Path | None = None) -> dict:
    root = (root or project_root()).resolve()
    path = path.expanduser().resolve()
    recipe = load_yaml(path)
    allowed = {"schema_version", "workspace", "raw_data_root", "source_root", "model_config",
               "datasets", "input_mode", "mode", "agent_provider", "force"}
    if not isinstance(recipe, dict) or set(recipe) - allowed:
        raise ValueError("Recipe must be a mapping with documented keys only")
    if type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise ValueError("Recipe schema_version must be 1")
    if not isinstance(recipe.get("workspace"), str) or not recipe["workspace"].strip():
        raise ValueError("An explicit private workspace is required")
    datasets = recipe.get("datasets")
    if not isinstance(datasets, list) or not datasets or any(
        not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in datasets
    ) or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must contain unique registered dataset identifiers")
    for item in datasets:
        if not (root / "configs/datasets" / f"{item}.yaml").is_file():
            raise ValueError(f"Dataset has no versioned adapter configuration: {item}")
    for name in ("workspace", "raw_data_root", "source_root", "model_config"):
        if name in recipe:
            if not isinstance(recipe[name], str) or not recipe[name].strip():
                raise ValueError(f"Invalid path: {name}")
            target = Path(recipe[name]).expanduser()
            recipe[name] = str((path.parent / target).resolve())
    workspace = Path(recipe["workspace"])
    validate_output_paths(workspace)
    if workspace == root or workspace in root.parents or (
        workspace.is_relative_to(root) and not workspace.is_relative_to(root / ".local")
    ):
        raise ValueError("Workspace must be private: outside the repository or under .local")
    for name in ("source_root", "raw_data_root"):
        if name in recipe:
            source = Path(recipe[name])
            if not source.is_dir():
                raise ValueError(f"Input directory does not exist: {source}")
            if source == workspace or source.is_relative_to(workspace) or workspace.is_relative_to(source):
                raise ValueError("Inputs and workspace must be separate, non-nested directories")
    if "model_config" in recipe and not Path(recipe["model_config"]).is_file():
        raise ValueError("model_config does not exist")
    recipe.setdefault("input_mode", "raw")
    recipe.setdefault("mode", "full")
    recipe.setdefault("agent_provider", "disabled")
    recipe.setdefault("force", False)
    for key, choices in {"input_mode": ("raw", "processed"), "mode": ("quick", "full"),
                         "agent_provider": ("disabled", "codex_cli", "openai_api")}.items():
        if recipe[key] not in choices:
            raise ValueError(f"Invalid {key}")
    if type(recipe["force"]) is not bool:
        raise ValueError("force must be a boolean")
    if recipe["input_mode"] == "processed":
        if recipe["mode"] != "full" or "source_root" not in recipe:
            raise ValueError("Processed reuse requires source_root and mode: full")
        for item in datasets:
            for filename in FROZEN_INPUTS:
                if not (Path(recipe["source_root"]) / item / filename).is_file():
                    raise ValueError(f"Missing frozen input: {item}/{filename}")
    else:
        if "source_root" in recipe or "raw_data_root" not in recipe:
            raise ValueError("Raw runs require raw_data_root, not source_root")
        for item in datasets:
            configured = Path(load_yaml(root / "configs/datasets" / f"{item}.yaml")["raw_path"])
            target = configured if configured.is_absolute() else Path(recipe["raw_data_root"]) / configured.relative_to("data/raw")
            if not target.is_dir():
                raise ValueError(f"Raw dataset directory does not exist: {target}")
    recipe["recipe_path"] = str(path)
    recipe["recipe_sha256"] = digest(path)
    return recipe


def command_for(recipe: dict, dataset: str) -> list[str]:
    command = [sys.executable, "-u", "-m", "advoice"]
    if recipe["input_mode"] == "processed":
        command += ["run-processed", "--source-root", recipe["source_root"]]
    else:
        command += ["run", "--mode", recipe["mode"]]
    command += ["--dataset", dataset, "--agent-provider", recipe["agent_provider"]]
    return command + (["--force"] if recipe["force"] else [])


def checked_outputs(directory: Path, names: tuple[str, ...]) -> dict:
    records = {}
    for name in names:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required report output is missing or empty: {path}")
        if path.suffix == ".png":
            with path.open("rb") as stream:
                if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError(f"Invalid PNG output: {path}")
        if path.suffix == ".html":
            _ReportLinks(path).feed(path.read_text(encoding="utf-8"))
        records[name] = {"path": str(path), "sha256": digest(path)}
    return records


class _ReportLinks(HTMLParser):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key not in {"src", "href"} or not value:
                continue
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            if not (self.path.parent / unquote(parsed.path)).exists():
                raise RuntimeError(f"Broken local report link in {self.path}: {value}")


def run_experiment(recipe_path: Path, check_only: bool = False) -> dict:
    root = project_root()
    recipe = load_recipe(recipe_path, root)
    identity = git_identity(root)
    fingerprint = source_fingerprint(root)
    model_path = Path(recipe.get("model_config", root / "configs/models/default.yaml"))
    model_hash = digest(model_path)
    if check_only:
        return {"status": "preflight_passed", "git": identity, "recipe": recipe, "model_sha256": model_hash,
                "scope": "Configuration and input presence only; no model or API calls"}
    workspace = Path(recipe["workspace"])
    with workspace_lock(workspace) as token:
        execution = workspace / "executions" / uuid.uuid4().hex
        execution.mkdir(parents=True)
        status = {"status": "running", "git": identity, "source_sha256": fingerprint,
                  "model_sha256": model_hash,
                  "python": sys.version, "recipe": recipe, "execution_dir": str(execution),
                  "started_at": datetime.now(timezone.utc).isoformat(), "datasets": [], "aggregate": None}
        def save():
            write_status(execution / "status.json", status)
            write_status(workspace / "experiment_latest.json", status)
        save()
        env = dict(os.environ)
        # A recipe owns these overrides; inherited shells cannot silently change it.
        for name in ("ADVOICE_RAW_DATA_DIR", "ADVOICE_MODEL_CONFIG", "ADVOICE_WORKSPACE_DIR"):
            env.pop(name, None)
        env.update(PYTHONPATH=str(root / "src"), ADVOICE_WORKSPACE_DIR=str(workspace))
        env[LOCK_TOKEN_ENV] = token
        for key, variable in (("raw_data_root", "ADVOICE_RAW_DATA_DIR"), ("model_config", "ADVOICE_MODEL_CONFIG")):
            if key in recipe:
                env[variable] = recipe[key]
        try:
            successful = []
            for dataset in recipe["datasets"]:
                row = {"dataset": dataset, "status": "running", "command": command_for(recipe, dataset)}
                status["datasets"].append(row)
                save()
                print(f"Running {dataset}; log: {execution / (dataset + '.log')}", flush=True)
                latest = workspace / "reports/latest_runs.json"
                previous = json.loads(latest.read_text()).get(dataset) if latest.exists() else None
                try:
                    with (execution / f"{dataset}.log").open("w") as log:
                        subprocess.run(row["command"], cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    current = json.loads(latest.read_text())[dataset]
                    if current == previous:
                        raise RuntimeError("No new immutable dataset run was published")
                    run_dir = Path(current["run_dir"]).resolve()
                    if not run_dir.is_relative_to(workspace / "runs"):
                        raise RuntimeError("Published run is outside this workspace")
                    row["outputs"] = checked_outputs(run_dir / "reports", DATASET_OUTPUTS)
                    row.update(status="completed", run_id=current["run_id"], run_dir=str(run_dir))
                    successful.append(dataset)
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                save()
            if successful:
                # Aggregate only immutable runs captured by this execution, not
                # the mutable artifacts directory or a future latest pointer.
                snapshot = execution / "aggregate"
                for row in status["datasets"]:
                    if row["status"] != "completed":
                        continue
                    run_dir = Path(row["run_dir"])
                    shutil.copytree(run_dir / "artifacts", snapshot / "artifacts" / row["dataset"])
                    shutil.copytree(run_dir / "reports", snapshot / "reports/datasets" / row["dataset"] / "latest")
                command = [sys.executable, "-u", "-m", "advoice", "aggregate-report", "--datasets", *successful]
                aggregate_env = {**env, "ADVOICE_WORKSPACE_DIR": str(snapshot)}
                aggregate_env.pop(LOCK_TOKEN_ENV, None)
                with (execution / "aggregate.log").open("w") as log:
                    subprocess.run(command, cwd=root, env=aggregate_env, stdout=log, stderr=subprocess.STDOUT, check=True)
                status["aggregate"] = checked_outputs(snapshot / "reports/latest", AGGREGATE_OUTPUTS)
                status["aggregate_datasets"] = successful
                shutil.copytree(snapshot / "reports/latest", workspace / "reports/latest", dirs_exist_ok=True)
            if (source_fingerprint(root) != fingerprint or digest(model_path) != model_hash
                    or digest(Path(recipe["recipe_path"])) != recipe["recipe_sha256"]):
                raise RuntimeError("Code/configuration changed during execution; results are not a frozen-version run")
            status["status"] = "completed" if len(successful) == len(recipe["datasets"]) else "failed"
        except BaseException as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            save()
        return status
