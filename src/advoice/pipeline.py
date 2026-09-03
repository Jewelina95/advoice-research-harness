from __future__ import annotations

import json
import re
import shutil
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .aggregate_reporting import build_aggregate_report
from .asr import prepare_analysis_transcripts
from .cache import StageCache, StageResult
from .config import ProjectPaths, load_all, paths
from .condition_c import train_condition_c
from .cognitive_agent import run_cognitive_diagnostic_agent
from .data import (
    PREPARE_TASK_MAP,
    _balanced_acquisition_group_holdout,
    build_manifest,
    dataset_source_inventory,
)
from .diagnostic_agent_report import run_diagnostic_agent_reports
from .direct_agent import run_direct_agent
from .evaluation import EVALUATION_SCHEMA_VERSION, run_evaluation
from .evidence import build_metric_evidence
from .features import aggregate_subject_features, extract_features, refresh_recording_text_metrics
from .models import train_b1, train_negative_controls, train_ours
from .reporting import build_reports, publish_latest
from .report_scoring_agent import run_report_scoring_agent
from .states import build_state_cards
from .utils import json_dump, json_load, now_utc, run_id, runtime_metadata, sha256_file, source_inventory


def _config_files(p: ProjectPaths, dataset_id: str) -> dict[str, Path]:
    return {
        "project": p.configs / "project.yaml",
        "dataset": p.configs / "datasets" / f"{dataset_id}.yaml",
        "channel": p.configs / "channels" / f"{load_all(dataset_id)['dataset'].get('channel_profile', 'audio_only')}.yaml",
        "metrics": p.configs / "metrics" / "audio_metrics.yaml",
        "states": p.configs / "states" / "audio_states.yaml",
        "models": p.configs / "models" / "default.yaml",
        "agents": p.configs / "agents" / "default.yaml",
        "evaluation": p.configs / "evaluation" / "default.yaml",
    }


def _source(p: ProjectPaths, name: str) -> Path:
    return p.root / "src" / "advoice" / name


def _diagnostic_skill_inputs(p: ProjectPaths) -> list[Path]:
    skill_dir = p.root / "skills" / "ad_evidence_diagnostic"
    skill_files = sorted(path for path in skill_dir.rglob("*") if path.is_file())
    schema_files = sorted(path for path in (p.root / "schemas").glob("*.json") if path.is_file())
    return [*skill_files, *schema_files]


def _artifact_paths(directory: Path) -> dict[str, Path]:
    names = [
        "manifest.csv",
        "analysis_manifest.csv",
        "dataset_audit.json",
        "asr_failures.csv",
        "feature_extraction_failures.csv",
        "recording_features.csv",
        "subject_features.csv",
        "segments.csv",
        "metric_evidence.csv",
        "cn_reference.json",
        "state_cards.csv",
        "state_wide.csv",
        "b1_predictions.csv",
        "legacy_c_predictions.csv",
        "legacy_c_ablations.csv",
        "legacy_c_branch_contributions.csv",
        "legacy_c_concept_interventions.csv",
        "ours_predictions.csv",
        "b3_supervised_predictions.csv",
        "condition_c_base_predictions.csv",
        "diagnostic_agent_workspaces.jsonl",
        "cognitive_agent_decisions.jsonl",
        "cognitive_agent_audit.jsonl",
        "locked_agent_workspaces.jsonl",
        "cognitive_agent_status.json",
        "cognitive_agent_prompt.txt",
        "agent_calibration_predictions.csv",
        "agent_calibration_workspaces.jsonl",
        "agent_correction_calibration.json",
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
        "report_scores.csv",
        "report_scoring_status.json",
        "report_scoring_prompt.txt",
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
            "legacy_c_model_bin": directory / "legacy_c_model.joblib",
            "legacy_c_model_meta": directory / "legacy_c_model.json",
        }
    )
    return values


def _resolved_configs(dataset_id: str) -> dict[str, Any]:
    configs = load_all(dataset_id)
    model_config = {
        **configs["models"],
        "labels": configs["dataset"]["labels"],
        "positive_class": configs["dataset"]["positive_class"],
        "state_branches": {
            state["id"]: state["branch"] for state in configs["states"].get("states", [])
        },
    }
    agent_config = {
        **configs["agents"],
        "labels": configs["dataset"]["labels"],
        "target_description": configs["dataset"]["target_description"],
        "agent_evaluation_cap": configs["dataset"].get("agent_evaluation_cap", 60),
        "diagnostic_agent_evaluation_cap": configs["dataset"].get(
            "agent_evaluation_cap", 60
        ),
    }
    evaluation_config = {
        **configs["evaluation"],
        "labels": configs["dataset"]["labels"],
        "positive_class": configs["dataset"]["positive_class"],
    }
    return {
        **configs,
        "models": model_config,
        "agents": agent_config,
        "evaluation": evaluation_config,
    }


def _publish_run(
    p: ProjectPaths,
    dataset_id: str,
    configs: dict[str, Any],
    config_paths: dict[str, Path],
    artifact_dir: Path,
    stages: list[StageResult],
    mode: str,
    agent_provider: str,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    identifier = f"{run_id()}_{dataset_id}"
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
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in config_paths.items()
        },
        "source_inventory": source_inventory(p.root),
        "stages": [asdict(stage) for stage in stages],
        "immutable_artifacts": str(run_artifacts),
        **(extra_manifest or {}),
    }
    json_dump(manifest, run_dir / "run_manifest.json")
    build_reports(p, configs, manifest, run_artifacts, run_reports)
    latest = publish_latest(p, run_reports, dataset_id)
    latest_runs_path = p.reports / "latest_runs.json"
    latest_runs = (
        json.loads(latest_runs_path.read_text(encoding="utf-8"))
        if latest_runs_path.exists()
        else {}
    )
    latest_runs[dataset_id] = {"run_id": identifier, "run_dir": str(run_dir)}
    json_dump(latest_runs, latest_runs_path)
    return latest / "index.html"


_PROCESSED_REUSE_FILES = (
    "manifest.csv",
    "analysis_manifest.csv",
    "dataset_audit.json",
    "asr_failures.csv",
    "feature_extraction_failures.csv",
    "recording_features.csv",
    "subject_features.csv",
    "segments.csv",
    "metric_evidence.csv",
    "cn_reference.json",
    "recording_transcripts.csv",
    "subject_transcripts.csv",
    "b1_predictions.csv",
    "b1_model.joblib",
    "b1_model.json",
    "negative_control_predictions.csv",
    "b2_predictions.csv",
    "b2_reports.csv",
    "b2_status.json",
    "b2_prompt.txt",
)


_LEGACY_C_FILE_MAP = {
    "ours_predictions.csv": "legacy_c_predictions.csv",
    "ours_ablations.csv": "legacy_c_ablations.csv",
    "branch_contributions.csv": "legacy_c_branch_contributions.csv",
    "concept_interventions.csv": "legacy_c_concept_interventions.csv",
    "ours_model.joblib": "legacy_c_model.joblib",
    "ours_model.json": "legacy_c_model.json",
}


def _bootstrap_processed_artifacts(
    source_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Reuse only frozen preprocessing/baselines and archive the former condition C."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    missing: list[str] = []
    for name in _PROCESSED_REUSE_FILES:
        source = source_dir / name
        destination = artifact_dir / name
        if not source.exists():
            missing.append(name)
            continue
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source),
                "destination": str(destination),
                "sha256": sha256_file(destination),
                "role": "frozen_preprocessing_or_baseline",
            }
        )
    for source_name, destination_name in _LEGACY_C_FILE_MAP.items():
        source = source_dir / source_name
        destination = artifact_dir / destination_name
        if not source.exists():
            missing.append(source_name)
            continue
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source),
                "destination": str(destination),
                "sha256": sha256_file(destination),
                "role": "archived_8_13_condition_c",
            }
        )
    required = {
        "subject_features.csv",
        "subject_transcripts.csv",
        "metric_evidence.csv",
        "b1_predictions.csv",
        "b2_predictions.csv",
    }
    absent_required = sorted(required.intersection(missing))
    if absent_required:
        raise FileNotFoundError(
            f"Processed rerun cannot start; required frozen inputs are missing: {absent_required}"
        )
    provenance = {
        "source_directory": str(source_dir),
        "created_at_utc": now_utc(),
        "input_mode": "processed_8_13_reuse",
        "copied_files": copied,
        "optional_missing_files": sorted(set(missing) - required),
        "new_condition_c_predictions_reused": False,
    }
    json_dump(provenance, artifact_dir / "processed_input_provenance.json")
    return provenance


def _enrich_processed_demographics(
    p: ProjectPaths,
    dataset_id: str,
    dataset_config: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Restore non-outcome demographics omitted by the 8.13 feature schema."""
    if dataset_id != "PREPARE_DrivenData":
        return {"applied": False, "reason": "no verified demographic source adapter"}
    raw_root = Path(str(dataset_config["raw_path"]))
    if not raw_root.is_absolute():
        raw_root = p.root / raw_root
    metadata_path = raw_root / "metadata.csv"
    if not metadata_path.exists():
        return {"applied": False, "reason": f"missing {metadata_path}"}
    demographics = pd.read_csv(metadata_path, usecols=["uid", "age", "gender"])
    demographics = demographics.rename(
        columns={"uid": "subject_id", "gender": "metadata_sex"}
    )
    demographics["subject_id"] = demographics["subject_id"].astype(str)
    updated_files: list[str] = []
    coverage: dict[str, float] = {}
    for filename in [
        "manifest.csv",
        "analysis_manifest.csv",
        "recording_features.csv",
        "subject_features.csv",
    ]:
        path = artifact_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"subject_id": str})
        frame = frame.drop(columns=["age", "metadata_sex"], errors="ignore").merge(
            demographics,
            on="subject_id",
            how="left",
            validate="many_to_one",
        )
        if "sex" not in frame:
            frame["sex"] = frame["metadata_sex"]
        else:
            frame["sex"] = frame["sex"].where(
                frame["sex"].astype(str).str.lower().isin(
                    {"female", "male", "f", "m"}
                ),
                frame["metadata_sex"],
            )
        frame = frame.drop(columns="metadata_sex")
        frame.to_csv(path, index=False)
        updated_files.append(filename)
        coverage[filename] = float(pd.to_numeric(frame["age"], errors="coerce").notna().mean())
    result = {
        "applied": True,
        "source": str(metadata_path),
        "source_sha256": sha256_file(metadata_path),
        "fields": ["age", "sex"],
        "outcome_fields_used": [],
        "updated_files": updated_files,
        "age_coverage": coverage,
    }
    return result


def _refresh_processed_semantics(
    p: ProjectPaths,
    dataset_id: str,
    dataset_config: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Refresh routing and language metrics while retaining frozen audio features."""
    updated_files: list[str] = []
    task_source: str | None = None
    if dataset_id == "PREPARE_DrivenData":
        labels_path = p.root / "references" / "speechcare" / "prepare_task_labels.csv"
        task_labels = pd.read_csv(labels_path, dtype={"uid": str}).set_index("uid")
        routed = task_labels["task"].map(PREPARE_TASK_MAP).fillna("other")
        valid = task_labels["valid"].astype(str).str.lower().eq("yes")
        excluded_development_subjects: set[str] = set()
        for filename in ["manifest.csv", "analysis_manifest.csv", "recording_features.csv"]:
            path = artifact_dir / filename
            frame = pd.read_csv(path, dtype={"subject_id": str, "case_id": str})
            frame["task_type"] = frame["subject_id"].map(routed).fillna("other")
            frame["task_type_source"] = "SpeechCARE_released_llm_task_label"
            frame["task_type_valid"] = frame["subject_id"].map(
                task_labels["valid"].astype(str)
            ).fillna("unknown")
            excluded = ~frame["split"].astype(str).str.lower().eq("test") & ~frame["subject_id"].map(valid).fillna(False)
            excluded_development_subjects.update(frame.loc[excluded, "subject_id"].astype(str))
            frame = frame.loc[~excluded].copy()
            frame.to_csv(path, index=False)
            updated_files.append(filename)
        transcript_path = artifact_dir / "subject_transcripts.csv"
        if transcript_path.exists() and excluded_development_subjects:
            transcripts = pd.read_csv(transcript_path, dtype={"subject_id": str})
            transcripts = transcripts[~transcripts["subject_id"].isin(excluded_development_subjects)]
            transcripts.to_csv(transcript_path, index=False)
            updated_files.append("subject_transcripts.csv")
        task_source = str(labels_path)

    if dataset_id == "IAEAV":
        for filename in ["manifest.csv", "analysis_manifest.csv", "recording_features.csv"]:
            path = artifact_dir / filename
            frame = pd.read_csv(path, dtype={"subject_id": str, "case_id": str})
            frame["acquisition_group"] = frame["subject_id"].str.extract(
                r"(inv\d+)", flags=re.IGNORECASE, expand=False
            ).str.lower().fillna("unknown")
            frame = _balanced_acquisition_group_holdout(frame)
            frame.to_csv(path, index=False)
            updated_files.append(filename)

    if dataset_id == "TAUKADIAL":
        raw_root = Path(str(dataset_config["raw_path"]))
        if not raw_root.is_absolute():
            raw_root = p.root / raw_root
        with tarfile.open(raw_root / "TAUKADIAL-24-train.tgz", "r:gz") as handle:
            train_metadata = pd.read_csv(
                handle.extractfile("TAUKADIAL-24/train/groundtruth.csv"),
                usecols=["tkdname", "age", "sex"],
            )
        test_metadata = pd.read_csv(
            raw_root / "meta_test.csv", sep=";", usecols=["tkdname", "age", "sex"]
        )
        demographics = pd.concat([train_metadata, test_metadata], ignore_index=True)
        demographics["case_id"] = demographics["tkdname"].map(lambda value: Path(str(value)).stem)
        demographics = demographics.drop(columns="tkdname").drop_duplicates("case_id", keep="last")
        for filename in ["manifest.csv", "analysis_manifest.csv", "recording_features.csv"]:
            path = artifact_dir / filename
            frame = pd.read_csv(path, dtype={"subject_id": str, "case_id": str})
            frame = frame.drop(columns=["age", "sex"], errors="ignore").merge(
                demographics, on="case_id", how="left", validate="one_to_one"
            )
            frame.to_csv(path, index=False)
            updated_files.append(filename)

    text_refresh = refresh_recording_text_metrics(
        artifact_dir / "recording_features.csv",
        artifact_dir / "analysis_manifest.csv",
    )
    recording = pd.read_csv(
        artifact_dir / "recording_features.csv", dtype={"subject_id": str, "case_id": str}
    )
    aggregate_subject_features(recording, artifact_dir / "subject_features.csv")
    updated_files.extend(["recording_features.csv", "subject_features.csv"])

    audit_path = artifact_dir / "dataset_audit.json"
    audit = json_load(audit_path, {})
    manifest = pd.read_csv(artifact_dir / "manifest.csv", dtype={"subject_id": str})
    audit["task_counts"] = {
        str(key): int(value)
        for key, value in manifest["task_type"].fillna("unknown").astype(str).value_counts().items()
    }
    audit["language_counts"] = text_refresh["languages"]
    if "acquisition_group" in manifest:
        counts = (
            manifest.groupby(["acquisition_group", "label"])["subject_id"]
            .nunique()
            .unstack(fill_value=0)
        )
        purities = counts.max(axis=1) / counts.sum(axis=1).clip(lower=1)
        audit["acquisition_group_label_counts"] = {
            str(group): {str(label): int(value) for label, value in row.items()}
            for group, row in counts.iterrows()
        }
        audit["acquisition_group_max_label_purity"] = float(purities.max())
        audit["capture_label_confounding_flag"] = bool(purities.max() >= 0.95)
        train_groups = set(
            manifest.loc[manifest["split"].eq("train"), "acquisition_group"].astype(str)
        )
        test_groups = set(
            manifest.loc[manifest["split"].eq("test"), "acquisition_group"].astype(str)
        )
        audit["train_acquisition_groups"] = sorted(train_groups)
        audit["test_acquisition_groups"] = sorted(test_groups)
        audit["acquisition_group_overlap_count"] = len(train_groups & test_groups)
    audit["train_recordings"] = int(manifest["split"].eq("train").sum())
    audit["test_recordings"] = int(manifest["split"].eq("test").sum())
    audit["train_subjects"] = int(
        manifest.loc[manifest["split"].eq("train"), "subject_id"].nunique()
    )
    audit["test_subjects"] = int(
        manifest.loc[manifest["split"].eq("test"), "subject_id"].nunique()
    )
    audit["label_counts_by_split"] = {
        str(split): {
            str(label): int(value)
            for label, value in group[["subject_id", "label"]]
            .drop_duplicates()["label"]
            .value_counts()
            .sort_index()
            .items()
        }
        for split, group in manifest.groupby("split")
    }
    if dataset_id == "NCMMSC2021_AD":
        analysis_paths = manifest.get("audio_path", pd.Series(dtype=str)).fillna("").astype(str)
        audit["six_second_files_in_analysis"] = int(
            analysis_paths.str.contains("AD_dataset_6s", regex=False).sum()
        )
        audit["long_recording_only_passed"] = audit["six_second_files_in_analysis"] == 0
    json_dump(audit, audit_path)
    return {
        "applied": True,
        "updated_files": sorted(set(updated_files)),
        "task_label_source": task_source,
        "task_counts": audit["task_counts"],
        "language_counts": audit["language_counts"],
        "text_metric_refresh": text_refresh,
    }


def validate_dataset(dataset_id: str) -> dict[str, Any]:
    p = paths()
    configs = load_all(dataset_id)
    dataset = configs["dataset"]
    raw = Path(dataset["raw_path"])
    if not raw.is_absolute():
        raw = p.root / raw
    inventory = dataset_source_inventory(p, dataset) if raw.exists() else []
    result = {
        "dataset_id": dataset_id,
        "raw_path": str(raw),
        "raw_exists": raw.exists(),
        "source_files": len(inventory),
        "split_policy": dataset.get("split_policy"),
        "labels": dataset.get("labels", []),
        "channel_profile": dataset.get("channel_profile"),
    }
    result["passed"] = result["raw_exists"] and result["source_files"] > 0
    return result


def run_pipeline(
    dataset_id: str,
    mode: str,
    agent_provider: str,
    force: bool = False,
) -> Path:
    p = paths()
    configs = load_all(dataset_id)
    model_config = {
        **configs["models"],
        "labels": configs["dataset"]["labels"],
        "positive_class": configs["dataset"]["positive_class"],
        "state_branches": {
            state["id"]: state["branch"] for state in configs["states"].get("states", [])
        },
    }
    agent_config = {
        **configs["agents"],
        "labels": configs["dataset"]["labels"],
        "target_description": configs["dataset"]["target_description"],
        "agent_evaluation_cap": configs["dataset"].get("agent_evaluation_cap", 60),
        "diagnostic_agent_evaluation_cap": configs["dataset"].get(
            "agent_evaluation_cap", 60
        ),
    }
    evaluation_config = {
        **configs["evaluation"],
        "labels": configs["dataset"]["labels"],
        "positive_class": configs["dataset"]["positive_class"],
    }
    configs = {
        **configs,
        "models": model_config,
        "agents": agent_config,
        "evaluation": evaluation_config,
    }
    project_config = configs["project"]
    artifact_dir = p.artifacts / dataset_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = _artifact_paths(artifact_dir)
    cache = StageCache(artifact_dir / ".stage_cache", force=force)
    config_paths = _config_files(p, dataset_id)
    stages: list[StageResult] = []

    inventory = dataset_source_inventory(p, configs["dataset"])
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
            "02_transcript_preparation",
            [
                output["manifest"],
                {
                    key: agent_config.get(key)
                    for key in ["asr_backend", "asr_model", "asr_language", "asr_workers", "asr_cpu_threads"]
                },
                _source(p, "asr.py"),
                _source(p, "transcripts.py"),
                mode,
            ],
            [
                output["analysis_manifest"],
                output["recording_transcripts"],
                output["subject_transcripts"],
                output["asr_failures"],
            ],
            lambda: prepare_analysis_transcripts(
                output["manifest"],
                output["analysis_manifest"],
                output["recording_transcripts"],
                output["subject_transcripts"],
                agent_config,
                generate_missing=mode == "full",
            ),
        )
    )
    stages.append(
        cache.execute(
            "03_audio_features",
            [output["analysis_manifest"], _source(p, "features.py"), _source(p, "transcripts.py")],
            [
                output["recording_features"],
                output["subject_features"],
                output["segments"],
                output["feature_extraction_failures"],
            ],
            lambda: extract_features(
                output["analysis_manifest"],
                output["recording_features"],
                output["subject_features"],
                output["segments"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "04_metric_evidence",
            [
                output["subject_features"],
                config_paths["metrics"],
                config_paths["channel"],
                configs["metrics"],
                {"reference_label": str(model_config["labels"][0])},
                _source(p, "evidence.py"),
            ],
            [output["metric_evidence"], output["cn_reference"]],
            lambda: build_metric_evidence(
                output["subject_features"],
                configs["metrics"],
                output["metric_evidence"],
                output["cn_reference"],
                reference_label=str(model_config["labels"][0]),
            ),
        )
    )
    stages.append(
        cache.execute(
            "05_state_cards",
            [
                output["metric_evidence"],
                output["recording_features"],
                output["segments"],
                config_paths["states"],
                config_paths["channel"],
                configs["states"],
                _source(p, "states.py"),
                "state-schema-task-routing-bounded-v4",
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
            "06_b1",
            [output["subject_features"], config_paths["models"], model_config, _source(p, "models.py")],
            [output["b1_predictions"], output["b1_model_bin"], output["b1_model_meta"]],
            lambda: train_b1(
                output["subject_features"],
                model_config,
                output["b1_predictions"],
                output["b1_model_bin"],
                output["b1_model_meta"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "07_legacy_c",
            [
                output["subject_features"],
                output["state_wide"],
                output["metric_evidence"],
                config_paths["models"],
                config_paths["states"],
                model_config,
                configs["states"],
                _source(p, "models.py"),
                _source(p, "states.py"),
                "model-schema-stable-task-selection-v4",
            ],
            [
                output["legacy_c_predictions"],
                output["legacy_c_ablations"],
                output["legacy_c_branch_contributions"],
                output["legacy_c_concept_interventions"],
                output["legacy_c_model_bin"],
                output["legacy_c_model_meta"],
            ],
            lambda: train_ours(
                output["subject_features"],
                output["state_wide"],
                output["metric_evidence"],
                configs["states"],
                model_config,
                output["legacy_c_predictions"],
                output["legacy_c_ablations"],
                output["legacy_c_branch_contributions"],
                output["legacy_c_concept_interventions"],
                output["legacy_c_model_bin"],
                output["legacy_c_model_meta"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "08_b3_single_diagnostic_agent",
            [
                output["subject_features"],
                output["subject_transcripts"],
                output["state_wide"],
                output["metric_evidence"],
                output["state_cards"],
                config_paths["models"],
                config_paths["states"],
                model_config,
                configs["states"],
                _source(p, "condition_c.py"),
                _source(p, "cognitive_prototypes.py"),
                _source(p, "evidence.py"),
                _source(p, "deep_audio_embeddings.py"),
                _source(p, "deep_embeddings.py"),
                _source(p, "dynamic_gate.py"),
                _source(p, "sequence_expert.py"),
                _source(p, "diagnostic_agent.py"),
                "condition-c-evidence-agent-v3-nested",
            ],
            [
                output["b3_supervised_predictions"],
                output["condition_c_base_predictions"],
                output["ours_ablations"],
                output["concept_interventions"],
                output["diagnostic_agent_workspaces"],
                output["branch_contributions"],
                output["ours_model_bin"],
                output["ours_model_meta"],
                output["agent_calibration_predictions"],
                output["agent_calibration_workspaces"],
            ],
            lambda: train_condition_c(
                output["subject_features"],
                output["subject_transcripts"],
                output["state_wide"],
                output["metric_evidence"],
                output["state_cards"],
                configs["states"],
                model_config,
                output["b3_supervised_predictions"],
                output["condition_c_base_predictions"],
                output["ours_ablations"],
                output["concept_interventions"],
                output["diagnostic_agent_workspaces"],
                output["branch_contributions"],
                output["ours_model_bin"],
                output["ours_model_meta"],
                output["agent_calibration_predictions"],
                output["agent_calibration_workspaces"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "09_b3_cognitive_agent_decision",
            [
                output["b3_supervised_predictions"],
                output["diagnostic_agent_workspaces"],
                output["agent_calibration_predictions"],
                output["agent_calibration_workspaces"],
                {
                    key: agent_config.get(key)
                    for key in [
                        "model",
                        "labels",
                        "diagnostic_agent_evaluation_cap",
                        "diagnostic_agent_batch_size",
                        "diagnostic_agent_workers",
                        "diagnostic_agent_correction_strength",
                        "diagnostic_agent_strength_grid",
                        "diagnostic_agent_min_calibration_cases",
                        "diagnostic_agent_min_macro_f1_gain",
                        "diagnostic_agent_auroc_noninferiority_margin",
                    ]
                },
                _source(p, "cognitive_agent.py"),
                _source(p, "diagnostic_agent.py"),
                _source(p, "agent_runtime.py"),
                *_diagnostic_skill_inputs(p),
                agent_provider,
            ],
            [
                output["ours_predictions"],
                output["cognitive_agent_decisions"],
                output["cognitive_agent_audit"],
                output["locked_agent_workspaces"],
                output["cognitive_agent_status"],
                output["cognitive_agent_prompt"],
                output["agent_correction_calibration"],
            ],
            lambda: run_cognitive_diagnostic_agent(
                p.root,
                output["b3_supervised_predictions"],
                output["diagnostic_agent_workspaces"],
                agent_config,
                agent_provider if mode == "full" else "disabled",
                output["ours_predictions"],
                output["cognitive_agent_decisions"],
                output["cognitive_agent_audit"],
                output["locked_agent_workspaces"],
                output["cognitive_agent_status"],
                output["cognitive_agent_prompt"],
                output["agent_calibration_predictions"],
                output["agent_calibration_workspaces"],
                output["agent_correction_calibration"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "10_negative_controls",
            [output["subject_features"], config_paths["models"], model_config, _source(p, "models.py")],
            [output["negative_control_predictions"]],
            lambda: train_negative_controls(output["subject_features"], output["negative_control_predictions"], model_config),
        )
    )

    stages.append(
        cache.execute(
            "11_b2_agent",
            [
                output["subject_transcripts"],
                {key: agent_config.get(key) for key in ["model", "batch_size", "batch_workers", "direct_agent_prompt_version", "direct_agent_instruction", "labels", "target_description", "agent_evaluation_cap"]},
                _source(p, "direct_agent.py"),
                _source(p, "agent_runtime.py"),
                agent_provider,
            ],
            [output["b2_predictions"], output["b2_reports"], output["b2_status"], output["b2_prompt"]],
            lambda: run_direct_agent(
                p.root,
                output["subject_transcripts"],
                output["manifest"],
                agent_config,
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
            "12_b3_diagnostic_agent_communication",
            [
                output["ours_predictions"],
                output["locked_agent_workspaces"],
                {key: agent_config.get(key) for key in ["model", "max_report_cases", "diagnostic_agent_prompt_version", "diagnostic_agent_instruction", "labels", "target_description"]},
                _source(p, "diagnostic_agent_report.py"),
                *_diagnostic_skill_inputs(p),
                _source(p, "agent_runtime.py"),
                agent_provider,
                "diagnostic-agent-communication-v1",
            ],
            [output["ours_reports"], output["ours_report_status"], output["ours_report_prompt"]],
            lambda: run_diagnostic_agent_reports(
                p.root,
                output["ours_predictions"],
                output["locked_agent_workspaces"],
                agent_config,
                agent_provider if mode == "full" else "disabled",
                output["ours_reports"],
                output["ours_report_status"],
                output["ours_report_prompt"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "13_report_scoring_agent",
            [
                output["b2_reports"],
                output["ours_reports"],
                {
                    key: agent_config.get(key)
                    for key in [
                        "model",
                        "report_scoring_prompt_version",
                        "report_scoring_instruction",
                        "report_scoring_provider",
                    ]
                },
                _source(p, "report_scoring_agent.py"),
                agent_config.get("report_scoring_provider", "disabled"),
            ],
            [output["report_scores"], output["report_scoring_status"], output["report_scoring_prompt"]],
            lambda: run_report_scoring_agent(
                p.root,
                output["b2_reports"],
                output["ours_reports"],
                agent_config,
                (
                    str(agent_config.get("report_scoring_provider", "disabled"))
                    if mode == "full"
                    else "disabled"
                ),
                output["report_scores"],
                output["report_scoring_status"],
                output["report_scoring_prompt"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "14_evaluation",
            [
                output["b1_predictions"],
                output["b2_predictions"],
                output["ours_predictions"],
                output["negative_control_predictions"],
                output["metric_evidence"],
                output["state_cards"],
                output["segments"],
                output["branch_contributions"],
                output["concept_interventions"],
                output["ours_ablations"],
                output["b2_reports"],
                output["ours_reports"],
                output["report_scores"],
                output["cognitive_agent_audit"],
                output["cognitive_agent_status"],
                config_paths["evaluation"],
                evaluation_config,
                EVALUATION_SCHEMA_VERSION,
                _source(p, "evaluation.py"),
            ],
            [output["layer_a_metrics"], output["layer_b_checks"], output["evaluation_summary"]],
            lambda: run_evaluation(
                {"B1": output["b1_predictions"], "B2": output["b2_predictions"], "Ours": output["ours_predictions"]},
                output["negative_control_predictions"],
                output["metric_evidence"],
                output["state_cards"],
                output["segments"],
                output["branch_contributions"],
                output["concept_interventions"],
                output["ours_ablations"],
                output["b2_reports"],
                output["ours_reports"],
                output["report_scores"],
                output["cognitive_agent_audit"],
                output["cognitive_agent_status"],
                evaluation_config,
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
    latest = publish_latest(p, run_reports, dataset_id)
    latest_runs_path = p.reports / "latest_runs.json"
    latest_runs = json.loads(latest_runs_path.read_text(encoding="utf-8")) if latest_runs_path.exists() else {}
    latest_runs[dataset_id] = {"run_id": identifier, "run_dir": str(run_dir)}
    json_dump(latest_runs, latest_runs_path)
    return latest / "index.html"


def rebuild_latest_report(dataset_id: str) -> Path:
    p = paths()
    latest_meta = json.loads((p.reports / "latest_runs.json").read_text(encoding="utf-8"))[dataset_id]
    run_dir = Path(latest_meta["run_dir"])
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run_reports = run_dir / "reports"
    build_reports(p, load_all(dataset_id), manifest, run_dir / "artifacts", run_reports)
    return publish_latest(p, run_reports, dataset_id) / "index.html"


def run_processed_pipeline(
    dataset_id: str,
    agent_provider: str = "disabled",
    force: bool = False,
    source_root: Path | None = None,
) -> Path:
    """Retrain 8.27 condition C from explicitly frozen processed inputs.

    This path is intentionally separate from ``run_pipeline``. It never claims
    that raw audio, ASR, feature extraction, B1, or B2 were recomputed.
    """
    p = paths()
    configs = _resolved_configs(dataset_id)
    model_config = configs["models"]
    agent_config = configs["agents"]
    evaluation_config = configs["evaluation"]
    config_paths = _config_files(p, dataset_id)
    artifact_dir = p.artifacts / dataset_id
    source_dir = (source_root or p.root / "artifacts_legacy_8_13") / dataset_id
    if not source_dir.exists():
        raise FileNotFoundError(f"No frozen processed artifacts for {dataset_id}: {source_dir}")

    provenance = _bootstrap_processed_artifacts(source_dir, artifact_dir)
    provenance["demographic_enrichment"] = _enrich_processed_demographics(
        p, dataset_id, configs["dataset"], artifact_dir
    )
    provenance["semantic_routing_refresh"] = _refresh_processed_semantics(
        p, dataset_id, configs["dataset"], artifact_dir
    )
    json_dump(provenance, artifact_dir / "processed_input_provenance.json")
    output = _artifact_paths(artifact_dir)
    cache = StageCache(artifact_dir / ".stage_cache", force=force)
    stages: list[StageResult] = []
    stages.append(
        cache.execute(
            "06_refresh_metric_evidence_from_routed_features",
            [
                output["subject_features"],
                config_paths["metrics"],
                configs["metrics"],
                {"reference_label": str(model_config["labels"][0])},
                _source(p, "evidence.py"),
                _source(p, "features.py"),
                _source(p, "transcripts.py"),
                "language-and-task-routed-evidence-v1",
            ],
            [output["metric_evidence"], output["cn_reference"]],
            lambda: build_metric_evidence(
                output["subject_features"],
                configs["metrics"],
                output["metric_evidence"],
                output["cn_reference"],
                reference_label=str(model_config["labels"][0]),
            ),
        )
    )
    stages.append(
        cache.execute(
            "07_refresh_state_cards_from_frozen_evidence",
            [
                output["metric_evidence"],
                output["recording_features"],
                output["segments"],
                config_paths["states"],
                configs["states"],
                _source(p, "states.py"),
                "state-card-trace-v3-report-permission",
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
            "08_b3_single_diagnostic_agent_processed",
            [
                output["subject_features"],
                output["subject_transcripts"],
                output["state_wide"],
                output["metric_evidence"],
                output["state_cards"],
                config_paths["models"],
                config_paths["states"],
                model_config,
                configs["states"],
                _source(p, "condition_c.py"),
                _source(p, "cognitive_prototypes.py"),
                _source(p, "evidence.py"),
                _source(p, "deep_audio_embeddings.py"),
                _source(p, "deep_embeddings.py"),
                _source(p, "dynamic_gate.py"),
                _source(p, "sequence_expert.py"),
                _source(p, "diagnostic_agent.py"),
                "condition-c-evidence-agent-v3-nested",
            ],
            [
                output["b3_supervised_predictions"],
                output["condition_c_base_predictions"],
                output["ours_ablations"],
                output["concept_interventions"],
                output["diagnostic_agent_workspaces"],
                output["branch_contributions"],
                output["ours_model_bin"],
                output["ours_model_meta"],
                output["agent_calibration_predictions"],
                output["agent_calibration_workspaces"],
            ],
            lambda: train_condition_c(
                output["subject_features"],
                output["subject_transcripts"],
                output["state_wide"],
                output["metric_evidence"],
                output["state_cards"],
                configs["states"],
                model_config,
                output["b3_supervised_predictions"],
                output["condition_c_base_predictions"],
                output["ours_ablations"],
                output["concept_interventions"],
                output["diagnostic_agent_workspaces"],
                output["branch_contributions"],
                output["ours_model_bin"],
                output["ours_model_meta"],
                output["agent_calibration_predictions"],
                output["agent_calibration_workspaces"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "09_b3_cognitive_agent_decision_processed",
            [
                output["b3_supervised_predictions"],
                output["diagnostic_agent_workspaces"],
                output["agent_calibration_predictions"],
                output["agent_calibration_workspaces"],
                {
                    key: agent_config.get(key)
                    for key in [
                        "model",
                        "labels",
                        "diagnostic_agent_evaluation_cap",
                        "diagnostic_agent_batch_size",
                        "diagnostic_agent_workers",
                        "diagnostic_agent_correction_strength",
                        "diagnostic_agent_strength_grid",
                        "diagnostic_agent_min_calibration_cases",
                        "diagnostic_agent_min_macro_f1_gain",
                        "diagnostic_agent_auroc_noninferiority_margin",
                    ]
                },
                _source(p, "cognitive_agent.py"),
                _source(p, "diagnostic_agent.py"),
                _source(p, "agent_runtime.py"),
                *_diagnostic_skill_inputs(p),
                agent_provider,
            ],
            [
                output["ours_predictions"],
                output["cognitive_agent_decisions"],
                output["cognitive_agent_audit"],
                output["locked_agent_workspaces"],
                output["cognitive_agent_status"],
                output["cognitive_agent_prompt"],
                output["agent_correction_calibration"],
            ],
            lambda: run_cognitive_diagnostic_agent(
                p.root,
                output["b3_supervised_predictions"],
                output["diagnostic_agent_workspaces"],
                agent_config,
                agent_provider,
                output["ours_predictions"],
                output["cognitive_agent_decisions"],
                output["cognitive_agent_audit"],
                output["locked_agent_workspaces"],
                output["cognitive_agent_status"],
                output["cognitive_agent_prompt"],
                output["agent_calibration_predictions"],
                output["agent_calibration_workspaces"],
                output["agent_correction_calibration"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "12_b3_diagnostic_agent_communication_processed",
            [
                output["ours_predictions"],
                output["locked_agent_workspaces"],
                {
                    key: agent_config.get(key)
                    for key in [
                        "model",
                        "max_report_cases",
                        "diagnostic_agent_prompt_version",
                        "diagnostic_agent_instruction",
                        "labels",
                        "target_description",
                    ]
                },
                _source(p, "diagnostic_agent_report.py"),
                *_diagnostic_skill_inputs(p),
                _source(p, "agent_runtime.py"),
                agent_provider,
            ],
            [output["ours_reports"], output["ours_report_status"], output["ours_report_prompt"]],
            lambda: run_diagnostic_agent_reports(
                p.root,
                output["ours_predictions"],
                output["locked_agent_workspaces"],
                agent_config,
                agent_provider,
                output["ours_reports"],
                output["ours_report_status"],
                output["ours_report_prompt"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "13_report_scoring_agent_processed",
            [
                output["b2_reports"],
                output["ours_reports"],
                {
                    key: agent_config.get(key)
                    for key in [
                        "model",
                        "report_scoring_prompt_version",
                        "report_scoring_instruction",
                        "report_scoring_provider",
                    ]
                },
                _source(p, "report_scoring_agent.py"),
                agent_config.get("report_scoring_provider", "disabled"),
            ],
            [output["report_scores"], output["report_scoring_status"], output["report_scoring_prompt"]],
            lambda: run_report_scoring_agent(
                p.root,
                output["b2_reports"],
                output["ours_reports"],
                agent_config,
                str(agent_config.get("report_scoring_provider", "disabled")),
                output["report_scores"],
                output["report_scoring_status"],
                output["report_scoring_prompt"],
            ),
        )
    )
    stages.append(
        cache.execute(
            "14_evaluation_processed",
            [
                output["b1_predictions"],
                output["b2_predictions"],
                output["ours_predictions"],
                output["negative_control_predictions"],
                output["metric_evidence"],
                output["state_cards"],
                output["segments"],
                output["branch_contributions"],
                output["concept_interventions"],
                output["ours_ablations"],
                output["b2_reports"],
                output["ours_reports"],
                output["report_scores"],
                output["cognitive_agent_audit"],
                output["cognitive_agent_status"],
                config_paths["evaluation"],
                evaluation_config,
                EVALUATION_SCHEMA_VERSION,
                _source(p, "evaluation.py"),
            ],
            [output["layer_a_metrics"], output["layer_b_checks"], output["evaluation_summary"]],
            lambda: run_evaluation(
                {
                    "B1": output["b1_predictions"],
                    "B2": output["b2_predictions"],
                    "Ours": output["ours_predictions"],
                },
                output["negative_control_predictions"],
                output["metric_evidence"],
                output["state_cards"],
                output["segments"],
                output["branch_contributions"],
                output["concept_interventions"],
                output["ours_ablations"],
                output["b2_reports"],
                output["ours_reports"],
                output["report_scores"],
                output["cognitive_agent_audit"],
                output["cognitive_agent_status"],
                evaluation_config,
                output["layer_a_metrics"],
                output["layer_b_checks"],
                output["evaluation_summary"],
            ),
        )
    )
    return _publish_run(
        p,
        dataset_id,
        configs,
        config_paths,
        artifact_dir,
        stages,
        mode="processed_rerun",
        agent_provider=agent_provider,
        extra_manifest={
            "processed_input_provenance": provenance,
            "raw_audio_recomputed": False,
            "condition_c_retrained": True,
            "b1_b2_frozen_from_8_13": True,
        },
    )


def run_all_processed_pipelines(
    agent_provider: str = "disabled",
    force: bool = False,
    dataset_ids: list[str] | None = None,
) -> Path:
    p = paths()
    project = load_all("NCMMSC2021_AD")["project"]
    selected = dataset_ids or [str(value) for value in project.get("default_datasets", [])]
    status: list[dict[str, Any]] = []
    batch_status_path = p.reports / "processed_batch_run_status.json"
    for dataset_id in selected:
        started_at = now_utc()
        try:
            report = run_processed_pipeline(dataset_id, agent_provider, force)
            status.append(
                {
                    "dataset_id": dataset_id,
                    "status": "completed",
                    "report": str(report),
                    "started_at_utc": started_at,
                    "finished_at_utc": now_utc(),
                }
            )
        except Exception as error:
            status.append(
                {
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_at_utc": started_at,
                    "finished_at_utc": now_utc(),
                }
            )
        json_dump(
            {
                "updated_at_utc": now_utc(),
                "agent_provider": agent_provider,
                "force": force,
                "selected_datasets": selected,
                "datasets": status,
            },
            batch_status_path,
        )
    aggregate = build_aggregate_report(p, selected)
    failures = [row for row in status if row["status"] == "failed"]
    if failures:
        failed_ids = ", ".join(str(row["dataset_id"]) for row in failures)
        raise RuntimeError(
            f"Full batch completed with failed datasets: {failed_ids}. "
            f"Partial aggregate report: {aggregate}"
        )
    return aggregate


def reevaluate_dataset(dataset_id: str) -> Path:
    """Recompute evaluation and reports from frozen model/agent artifacts only."""
    p = paths()
    configs = load_all(dataset_id)
    evaluation_config = {
        **configs["evaluation"],
        "labels": configs["dataset"]["labels"],
        "positive_class": configs["dataset"]["positive_class"],
    }
    configs = {**configs, "evaluation": evaluation_config}
    config_paths = _config_files(p, dataset_id)
    artifact_dir = p.artifacts / dataset_id
    output = _artifact_paths(artifact_dir)
    required = [
        output["b1_predictions"],
        output["b2_predictions"],
        output["ours_predictions"],
        output["negative_control_predictions"],
        output["metric_evidence"],
        output["state_cards"],
        output["segments"],
        output["branch_contributions"],
        output["concept_interventions"],
        output["ours_ablations"],
        output["b2_reports"],
        output["ours_reports"],
        output["report_scores"],
        output["cognitive_agent_audit"],
        output["cognitive_agent_status"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot reevaluate {dataset_id}; frozen artifacts are missing: {missing}"
        )
    cache = StageCache(
        artifact_dir / ".stage_cache",
        schema_version=EVALUATION_SCHEMA_VERSION,
    )
    stage = cache.execute(
        "12_evaluation",
        [
            *required,
            config_paths["evaluation"],
            evaluation_config,
            EVALUATION_SCHEMA_VERSION,
            _source(p, "evaluation.py"),
        ],
        [output["layer_a_metrics"], output["layer_b_checks"], output["evaluation_summary"]],
        lambda: run_evaluation(
            {
                "B1": output["b1_predictions"],
                "B2": output["b2_predictions"],
                "Ours": output["ours_predictions"],
            },
            output["negative_control_predictions"],
            output["metric_evidence"],
            output["state_cards"],
            output["segments"],
            output["branch_contributions"],
            output["concept_interventions"],
            output["ours_ablations"],
            output["b2_reports"],
            output["ours_reports"],
            output["report_scores"],
            output["cognitive_agent_audit"],
            output["cognitive_agent_status"],
            evaluation_config,
            output["layer_a_metrics"],
            output["layer_b_checks"],
            output["evaluation_summary"],
        ),
    )

    identifier = f"{run_id()}_{dataset_id}_evaluation"
    run_dir = p.runs / identifier
    run_artifacts = run_dir / "artifacts"
    run_reports = run_dir / "reports"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(artifact_dir, run_artifacts, ignore=shutil.ignore_patterns(".stage_cache"))
    b2_status = json.loads(output["b2_status"].read_text(encoding="utf-8"))
    manifest = {
        "run_id": identifier,
        "harness_version": __version__,
        "dataset_id": dataset_id,
        "mode": "evaluation_only",
        "agent_provider": b2_status.get("provider", "unknown"),
        "created_at_utc": now_utc(),
        "runtime": runtime_metadata(p.root),
        "configs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in config_paths.items()
        },
        "source_inventory": source_inventory(p.root),
        "stages": [asdict(stage)],
        "immutable_artifacts": str(run_artifacts),
        "frozen_upstream_artifacts": True,
    }
    json_dump(manifest, run_dir / "run_manifest.json")
    build_reports(p, configs, manifest, run_artifacts, run_reports)
    latest = publish_latest(p, run_reports, dataset_id)
    latest_runs_path = p.reports / "latest_runs.json"
    latest_runs = (
        json.loads(latest_runs_path.read_text(encoding="utf-8"))
        if latest_runs_path.exists()
        else {}
    )
    latest_runs[dataset_id] = {"run_id": identifier, "run_dir": str(run_dir)}
    json_dump(latest_runs, latest_runs_path)
    return latest / "index.html"


def reevaluate_all(dataset_ids: list[str] | None = None) -> Path:
    p = paths()
    selected = dataset_ids or [
        str(value)
        for value in load_all("NCMMSC2021_AD")["project"].get("default_datasets", [])
    ]
    for dataset_id in selected:
        reevaluate_dataset(dataset_id)
    return build_aggregate_report(p, selected)


def clean_cache(dataset_id: str) -> None:
    cache = paths().artifacts / dataset_id / ".stage_cache"
    if cache.exists():
        shutil.rmtree(cache)


def run_all_pipelines(
    mode: str,
    agent_provider: str,
    force: bool = False,
    dataset_ids: list[str] | None = None,
) -> Path:
    p = paths()
    project = load_all("NCMMSC2021_AD")["project"]
    selected = dataset_ids or [str(value) for value in project.get("default_datasets", [])]
    status: list[dict[str, Any]] = []
    batch_status_path = p.reports / "batch_run_status.json"
    for dataset_id in selected:
        started_at = now_utc()
        try:
            report = run_pipeline(dataset_id, mode, agent_provider, force)
            status.append({"dataset_id": dataset_id, "status": "completed", "report": str(report), "started_at_utc": started_at, "finished_at_utc": now_utc()})
        except Exception as error:
            status.append(
                {
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_at_utc": started_at,
                    "finished_at_utc": now_utc(),
                }
            )
        json_dump(
            {
                "updated_at_utc": now_utc(),
                "mode": mode,
                "agent_provider": agent_provider,
                "force": force,
                "selected_datasets": selected,
                "datasets": status,
            },
            batch_status_path,
        )
    aggregate = build_aggregate_report(p, selected)
    failures = [row for row in status if row["status"] == "failed"]
    if failures:
        failed_ids = ", ".join(str(row["dataset_id"]) for row in failures)
        raise RuntimeError(
            f"Full batch completed with failed datasets: {failed_ids}. "
            f"Partial aggregate report: {aggregate}"
        )
    return aggregate
