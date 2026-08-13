from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf

from .config import ProjectPaths
from .utils import json_dump, sha256_file


NCMMSC_NAME = re.compile(r"^(AD|MCI|HC)_([FM])_(\d+)[_-]([0-9]+)$", re.IGNORECASE)


def _resolve_raw_path(paths: ProjectPaths, configured: str) -> Path:
    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = paths.root / candidate
    return candidate.resolve()


def ncmmsc_source_inventory(paths: ProjectPaths, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    files = sorted(root.glob(dataset["train_glob"])) + sorted(root.glob(dataset["test_glob"]))
    return [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]


def _manifest_row(path: Path, split: str, dataset: dict[str, Any]) -> dict[str, Any]:
    match = NCMMSC_NAME.match(path.stem)
    if not match:
        raise ValueError(f"Unexpected NCMMSC filename: {path.name}")
    label, sex, numeric_id, recording_index = match.groups()
    label = label.upper()
    sex = sex.upper()
    subject_id = f"{label}_{sex}_{numeric_id}"
    info = sf.info(path)
    return {
        "dataset_id": dataset["dataset_id"],
        "case_id": path.stem,
        "subject_id": subject_id,
        "label": label,
        "split": split,
        "task_type": dataset["task_type"],
        "language": dataset["language"],
        "channel": dataset["channel"],
        "audio_path": str(path.resolve()),
        "transcript_path": "",
        "speaker_role_source": dataset["speaker_roles"],
        "diagnosis_source": "filename_label_from_distributed_dataset",
        "sex": sex,
        "recording_index": int(recording_index),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": float(info.frames / info.samplerate),
        "audio_sha256": sha256_file(path),
    }


def build_ncmmsc_manifest(
    paths: ProjectPaths,
    dataset: dict[str, Any],
    manifest_path: Path,
    audit_path: Path,
) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    train_files = sorted(root.glob(dataset["train_glob"]))
    test_files = sorted(root.glob(dataset["test_glob"]))
    if not train_files or not test_files:
        raise FileNotFoundError(f"NCMMSC long-audio split is incomplete under {root}")
    rows = [_manifest_row(path, "train", dataset) for path in train_files]
    rows.extend(_manifest_row(path, "test", dataset) for path in test_files)
    frame = pd.DataFrame(rows).sort_values(["split", "subject_id", "recording_index"])
    frame.to_csv(manifest_path, index=False)

    train_subjects = set(frame.loc[frame["split"].eq("train"), "subject_id"])
    test_subjects = set(frame.loc[frame["split"].eq("test"), "subject_id"])
    overlap = sorted(train_subjects & test_subjects)
    duplicate_hashes = frame[frame.duplicated("audio_sha256", keep=False)][
        ["case_id", "split", "audio_sha256"]
    ].to_dict("records")
    six_second_files = list((root / "AD_dataset_6s").rglob("*.wav"))
    label_conflicts = (
        frame.groupby(["split", "subject_id"])["label"].nunique().loc[lambda value: value > 1]
    )
    audit = {
        "dataset_id": dataset["dataset_id"],
        "analysis_input": "AD_dataset_long only",
        "six_second_files_discovered_but_excluded": len(six_second_files),
        "recordings": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "train_recordings": int(frame["split"].eq("train").sum()),
        "test_recordings": int(frame["split"].eq("test").sum()),
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "label_counts_by_split": {
            split: group["label"].value_counts().sort_index().to_dict()
            for split, group in frame.groupby("split")
        },
        "subject_overlap_count": len(overlap),
        "subject_overlap": overlap,
        "subject_label_conflict_count": int(len(label_conflicts)),
        "duplicate_audio_hash_count": len(duplicate_hashes),
        "duplicate_audio_hash_rows": duplicate_hashes,
        "all_audio_mono_16khz": bool(
            frame["sample_rate"].eq(16000).all() and frame["channels"].eq(1).all()
        ),
        "passed": not overlap and not duplicate_hashes and label_conflicts.empty,
        "warnings": [
            "Per-recording task mapping is unavailable; recording index is not interpreted as a task label.",
            "NCMMSC has no local human transcript or speaker-role annotation.",
            "The subject key is diagnosis + sex + six-digit identifier because the numeric field is not globally unique across label folders.",
            "Modeling is performed at subject level to avoid treating repeated recordings as independent patients.",
        ],
    }
    json_dump(audit, audit_path)
    if not audit["passed"]:
        raise RuntimeError(f"Dataset audit failed: {audit}")


def build_manifest(
    paths: ProjectPaths,
    dataset: dict[str, Any],
    manifest_path: Path,
    audit_path: Path,
) -> None:
    if dataset["dataset_id"] == "NCMMSC2021_AD":
        build_ncmmsc_manifest(paths, dataset, manifest_path, audit_path)
        return
    raise NotImplementedError(f"No manifest adapter for {dataset['dataset_id']}")
