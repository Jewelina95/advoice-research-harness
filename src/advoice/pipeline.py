from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .asr import transcribe_test_audio
from .cache import StageCache, StageResult
from .config import ProjectPaths, load_all, paths
from .data import build_manifest, ncmmsc_source_inventory
from .direct_agent import run_direct_agent
from .evaluation import run_evaluation
from .evidence import build_metric_evidence
from .features import extract_features
from .models import train_b1, train_negative_controls, train_ours
from .reporting import build_reports, publish_latest
from .report_agent import run_ours_report_agent
from .states import build_state_cards
from .utils import json_dump, now_utc, run_id, runtime_metadata, sha256_file, source_inventory


def _config_files(p: ProjectPaths, dataset_id: str) -> dict[str, Path]:
    return {
        "project": p.configs / "project.yaml",
        "dataset": p.configs / "datasets" / f"{dataset_id}.yaml",
        "metrics": p.configs / "metrics" / "audio_metrics.yaml",
        "states": p.configs / "states" / "audio_states.yaml",
        "models": p.configs / "models" / "default.yaml",
        "agents": p.configs / "agents" / "default.yaml",
        "evaluation": p.configs / "evaluation" / "default.yaml",
    }


def _source(p: ProjectPaths, name: str) -> Path:
    return p.root / "src" / "advoice" / name


def _artifact_paths(directory: Path) -> dict[str, Path]:
    names = [
        "manifest.csv",
        "dataset_audit.json",
        "recording_features.csv",
        "subject_features.csv",
        "segments.csv",
        "metric_evidence.csv",
        "cn_reference.json",
        "state_cards.csv",
        "state_wide.csv",
        "b1_predictions.csv",
        "ours_predictions.csv",
        "ours_ablations.csv",
        "branch_contributions.csv",
        "concept_interventions.csv",
        "negative_control_predictions.csv",
        "recording_transcripts.csv",
        "subject_transcripts.csv",
        "b2_predictions.csv",
        "b2_reports.csv",
        "b2_status.json",
        "b2_prompt.txt",
        "ours_reports.csv",
        "ours_report_status.json",
        "ours_report_prompt.txt",
        "layer_a_metrics.csv",
        "layer_b_checks.csv",
        "evaluation_summary.json",
    ]
    values = {Path(name).stem: directory / name for name in names}
    values.update(
        {
            "b1_model_bin": directory / "b1_model.joblib",
            "b1_model_meta": directory / "b1_model.json",
            "ours_model_bin": directory / "ours_model.joblib",
            "ours_model_meta": directory / "ours_model.json",
        }
    )
    return values


def validate_dataset(dataset_id: str) -> dict[str, Any]:
    p = paths()
    configs = load_all(dataset_id)
    dataset = configs["dataset"]
    raw = p.root / dataset["raw_path"]
    result = {
        "dataset_id": dataset_id,
        "raw_path": str(raw),
        "raw_exists": raw.exists(),
        "train_files": len(list(raw.glob(dataset["train_glob"]))) if raw.exists() else 0,
        "test_files": len(list(raw.glob(dataset["test_glob"]))) if raw.exists() else 0,
        "six_second_policy": "excluded",
    }
    result["passed"] = result["raw_exists"] and result["train_files"] > 0 and result["test_files"] > 0
    return result


def run_pipeline(
    dataset_id: str,
    mode: str,
    agent_provider: str,
    force: bool = False,
) -> Path:
    p = paths()
    configs = load_all(dataset_id)
    project_config = configs["project"]
    artifact_dir = p.artifacts / dataset_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = _artifact_paths(artifact_dir)
    cache = StageCache(artifact_dir / ".stage_cache", force=force)
    config_paths = _config_files(p, dataset_id)
    stages: list[StageResult] = []

    inventory = ncmmsc_source_inventory(p, configs["dataset"])
    stages.append(
        cache.execute(
            "01_manifest",
            [config_paths["dataset"], _source(p, "data.py"), inventory],
            [output["manifest"], output["dataset_audit"]],
            lambda: build_manifest(p, configs["dataset"], output["manifest"], output["dataset_audit"]),
        )
    )
    stages.append(
        cache.execute(
            "02_audio_features",
            [output["manifest"], _source(p, "features.py")],
            [output["recording_features"], output["subject_features"], output["segments"]],
            lambda: extract_features(
                output["manifest"],
                output["recording_features"],
                output["subject_features"],
                output["segments"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "03_metric_evidence",
            [output["subject_features"], config_paths["metrics"], _source(p, "evidence.py")],
            [output["metric_evidence"], output["cn_reference"]],
            lambda: build_metric_evidence(
                output["subject_features"], configs["metrics"], output["metric_evidence"], output["cn_reference"]
            ),
        )
    )
    stages.append(
        cache.execute(
            "04_state_cards",
            [
                output["metric_evidence"],
                output["recording_features"],
                output["segments"],
                config_paths["states"],
                _source(p, "states.py"),
            ],
            [output["state_cards"], output["state_wide"]],
            lambda: build_state_cards(
                output["metric_evidence"],
                output["recording_features"],
                output["segments"],
                configs["states"],
                output["state_cards"],
                output["state_wide"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "05_b1",
            [output["subject_features"], config_paths["models"], _source(p, "models.py")],
            [output["b1_predictions"], output["b1_model_bin"], output["b1_model_meta"]],
            lambda: train_b1(
                output["subject_features"],
                configs["models"],
                output["b1_predictions"],
                output["b1_model_bin"],
                output["b1_model_meta"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "06_ours",
            [output["subject_features"], output["state_wide"], config_paths["models"], _source(p, "models.py")],
            [
                output["ours_predictions"],
                output["ours_ablations"],
                output["branch_contributions"],
                output["concept_interventions"],
                output["ours_model_bin"],
                output["ours_model_meta"],
            ],
            lambda: train_ours(
                output["subject_features"],
                output["state_wide"],
                configs["models"],
                output["ours_predictions"],
                output["ours_ablations"],
                output["branch_contributions"],
                output["concept_interventions"],
                output["ours_model_bin"],
                output["ours_model_meta"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "07_negative_controls",
            [output["subject_features"], _source(p, "models.py")],
            [output["negative_control_predictions"]],
            lambda: train_negative_controls(output["subject_features"], output["negative_control_predictions"]),
        )
    )

    if mode == "full" and agent_provider != "disabled":
        stages.append(
            cache.execute(
                "08_asr",
                [
                    output["manifest"],
                    {key: configs["agents"].get(key) for key in ["asr_model", "asr_language", "asr_workers", "asr_cpu_threads"]},
                    _source(p, "asr.py"),
                ],
                [output["recording_transcripts"], output["subject_transcripts"]],
                lambda: transcribe_test_audio(
                    output["manifest"],
                    output["recording_transcripts"],
                    output["subject_transcripts"],
                    configs["agents"],
                ),
            )
        )
    else:
        pd.DataFrame(columns=["case_id", "subject_id", "text"]).to_csv(output["recording_transcripts"], index=False)
        pd.DataFrame(columns=["subject_id", "transcript"]).to_csv(output["subject_transcripts"], index=False)

    stages.append(
        cache.execute(
            "09_b2_agent",
            [
                output["subject_transcripts"],
                {key: configs["agents"].get(key) for key in ["model", "batch_size", "direct_agent_prompt_version", "direct_agent_instruction"]},
                _source(p, "direct_agent.py"),
                _source(p, "agent_runtime.py"),
                agent_provider,
            ],
            [output["b2_predictions"], output["b2_reports"], output["b2_status"], output["b2_prompt"]],
            lambda: run_direct_agent(
                p.root,
                output["subject_transcripts"],
                output["manifest"],
                configs["agents"],
                agent_provider if mode == "full" else "disabled",
                output["b2_predictions"],
                output["b2_reports"],
                output["b2_status"],
                output["b2_prompt"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "10_ours_report_agent",
            [
                output["ours_predictions"],
                output["state_cards"],
                {key: configs["agents"].get(key) for key in ["model", "max_report_cases", "report_agent_prompt_version", "report_agent_instruction"]},
                _source(p, "report_agent.py"),
                _source(p, "agent_runtime.py"),
                agent_provider,
            ],
            [output["ours_reports"], output["ours_report_status"], output["ours_report_prompt"]],
            lambda: run_ours_report_agent(
                p.root,
                output["ours_predictions"],
                output["state_cards"],
                configs["agents"],
                agent_provider if mode == "full" else "disabled",
                output["ours_reports"],
                output["ours_report_status"],
                output["ours_report_prompt"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "11_evaluation",
            [
                output["b1_predictions"],
                output["b2_predictions"],
                output["ours_predictions"],
                output["negative_control_predictions"],
                output["metric_evidence"],
                output["state_cards"],
                output["branch_contributions"],
                output["concept_interventions"],
                output["ours_ablations"],
                output["b2_reports"],
                output["ours_reports"],
                config_paths["evaluation"],
                _source(p, "evaluation.py"),
            ],
            [output["layer_a_metrics"], output["layer_b_checks"], output["evaluation_summary"]],
            lambda: run_evaluation(
                {"B1": output["b1_predictions"], "B2": output["b2_predictions"], "Ours": output["ours_predictions"]},
                output["negative_control_predictions"],
                output["metric_evidence"],
                output["state_cards"],
                output["branch_contributions"],
                output["concept_interventions"],
                output["ours_ablations"],
                output["b2_reports"],
                output["ours_reports"],
                configs["evaluation"],
                output["layer_a_metrics"],
                output["layer_b_checks"],
                output["evaluation_summary"],
            ),
        )
    )

    identifier = run_id()
    run_dir = p.runs / identifier
    run_artifacts = run_dir / "artifacts"
    run_reports = run_dir / "reports"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(artifact_dir, run_artifacts, ignore=shutil.ignore_patterns(".stage_cache"))
    manifest = {
        "run_id": identifier,
        "harness_version": __version__,
        "dataset_id": dataset_id,
        "mode": mode,
        "agent_provider": agent_provider,
        "created_at_utc": now_utc(),
        "runtime": runtime_metadata(p.root),
        "configs": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in config_paths.items()
        },
        "source_inventory": source_inventory(p.root),
        "stages": [asdict(stage) for stage in stages],
        "immutable_artifacts": str(run_artifacts),
    }
    json_dump(manifest, run_dir / "run_manifest.json")
    build_reports(p, configs, manifest, run_artifacts, run_reports)
    latest = publish_latest(p, run_reports)
    json_dump({"run_id": identifier, "run_dir": str(run_dir)}, p.reports / "latest_run.json")
    return latest / "index.html"


def rebuild_latest_report(dataset_id: str) -> Path:
    p = paths()
    latest_meta = json.loads((p.reports / "latest_run.json").read_text(encoding="utf-8"))
    run_dir = Path(latest_meta["run_dir"])
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run_reports = run_dir / "reports"
    build_reports(p, load_all(dataset_id), manifest, run_dir / "artifacts", run_reports)
    return publish_latest(p, run_reports) / "index.html"


def clean_cache(dataset_id: str) -> None:
    cache = paths().artifacts / dataset_id / ".stage_cache"
    if cache.exists():
        shutil.rmtree(cache)
