from __future__ import annotations

import re
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

from .config import ProjectPaths
from .transcripts import patient_intervals_from_cha, patient_intervals_from_segmentation
from .utils import json_dump, sha256_file


PREPARE_TASK_MAP = {
    "Picture Description": "picture_description",
    "Sentence Reading": "sentence_reading",
    "Voice Assistant": "voice_assistant",
    "Semantic Verbal Fluency": "semantic_verbal_fluency",
    "Story Recall": "story_recall",
    "Personal Narrative": "personal_narrative",
    "Other": "other",
}


NCMMSC_NAME = re.compile(r"^(AD|MCI|HC)_([FM])_(\d+)[_-]([0-9]+)$", re.IGNORECASE)


def _resolve_raw_path(paths: ProjectPaths, configured: str) -> Path:
    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = paths.root / candidate
    return candidate.resolve()


def ncmmsc_source_inventory(paths: ProjectPaths, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return dataset_source_inventory(paths, dataset)


def dataset_source_inventory(paths: ProjectPaths, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    globs = dataset.get("source_globs") or [dataset.get("train_glob", ""), dataset.get("test_glob", "")]
    files: list[Path] = []
    for pattern in globs:
        if pattern:
            files.extend(root.glob(pattern))
    for configured in dataset.get("reference_files", []):
        reference = Path(configured)
        if not reference.is_absolute():
            reference = paths.root / reference
        if reference.is_file():
            files.append(reference)
    files = sorted({path.resolve() for path in files if path.is_file()})
    return [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]


def _stable_subject_split(frame: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    subjects = frame[["subject_id", "label"]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        subjects["subject_id"],
        test_size=test_size,
        random_state=seed,
        stratify=subjects["label"],
    )
    output = frame.copy()
    output["split"] = np.where(output["subject_id"].isin(set(test_ids)), "test", "train")
    return output


def _balanced_acquisition_group_holdout(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold out one complete acquisition group that contains every label."""

    labels = sorted(frame["label"].dropna().astype(str).unique())
    counts = (
        frame[["subject_id", "label", "acquisition_group"]]
        .drop_duplicates()
        .groupby(["acquisition_group", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
    )
    eligible = counts[counts.min(axis=1).gt(0)].copy()
    if eligible.empty:
        raise ValueError("No acquisition group contains every label for a grouped holdout.")
    eligible["minimum_class_count"] = eligible[labels].min(axis=1)
    eligible["imbalance"] = eligible[labels].max(axis=1) - eligible[labels].min(axis=1)
    eligible["total"] = eligible[labels].sum(axis=1)
    selected_group = str(
        eligible.sort_values(
            ["minimum_class_count", "imbalance", "total"],
            ascending=[False, True, False],
        ).index[0]
    )
    output = frame.copy()
    output["split"] = np.where(
        output["acquisition_group"].astype(str).eq(selected_group), "test", "train"
    )
    return output


def _media_info(path: Path) -> tuple[int, int, float]:
    try:
        info = sf.info(path)
        return int(info.samplerate), int(info.channels), float(info.frames / info.samplerate)
    except Exception:
        return 0, 0, float("nan")


def _generic_row(
    dataset: dict[str, Any],
    path: Path,
    subject_id: str,
    label: str,
    split: str,
    transcript_path: str = "",
    task_type: str | None = None,
    language: str | None = None,
    sex: str = "U",
    age: float | None = None,
    speaker_role_source: str | None = None,
) -> dict[str, Any]:
    sr, channels, duration = _media_info(path)
    return {
        "dataset_id": dataset["dataset_id"],
        "case_id": path.stem,
        "subject_id": str(subject_id),
        "label": label,
        "split": split,
        "task_type": task_type or dataset["task_type"],
        "language": language or dataset["language"],
        "channel": dataset["channel"],
        "audio_path": str(path.resolve()),
        "transcript_path": str(transcript_path),
        "analysis_intervals": "[]",
        "speaker_role_source": speaker_role_source or dataset.get("speaker_roles", "unavailable"),
        "role_filter_required": bool(
            dataset.get("require_patient_audio_intervals", False)
        ),
        "diagnosis_source": dataset.get("diagnosis_source", "distributed_metadata"),
        "sex": sex,
        "age": age,
        "recording_index": 1,
        "sample_rate": sr,
        "channels": channels,
        "duration_sec": duration,
        "audio_sha256": sha256_file(path),
    }


def _write_manifest_audit(
    frame: pd.DataFrame,
    dataset: dict[str, Any],
    manifest_path: Path,
    audit_path: Path,
    warnings: list[str] | None = None,
) -> None:
    frame = frame.sort_values(["split", "subject_id", "case_id"]).reset_index(drop=True)
    frame.to_csv(manifest_path, index=False)
    train_subjects = set(frame.loc[frame["split"].eq("train"), "subject_id"].astype(str))
    test_subjects = set(frame.loc[frame["split"].eq("test"), "subject_id"].astype(str))
    overlap = sorted(train_subjects & test_subjects)
    conflicts = frame.groupby("subject_id")["label"].nunique().loc[lambda value: value > 1]
    duplicate_case_ids = sorted(frame.loc[frame["case_id"].duplicated(keep=False), "case_id"].astype(str).unique())
    duplicate_hashes = frame[frame.duplicated("audio_sha256", keep=False)][
        ["case_id", "split", "audio_sha256"]
    ].to_dict("records")
    hash_split_counts = frame.groupby("audio_sha256")["split"].nunique()
    cross_split_hashes = set(hash_split_counts[hash_split_counts > 1].index)
    cross_split_duplicate_rows = frame[frame["audio_sha256"].isin(cross_split_hashes)][
        ["case_id", "split", "audio_sha256"]
    ].to_dict("records")
    acquisition_audit: dict[str, Any] = {}
    if "acquisition_group" in frame.columns:
        group_table = (
            frame[["subject_id", "label", "acquisition_group"]]
            .drop_duplicates()
            .groupby(["acquisition_group", "label"])
            .size()
            .unstack(fill_value=0)
        )
        group_totals = group_table.sum(axis=1)
        group_purity = group_table.max(axis=1).div(group_totals.where(group_totals.gt(0)))
        train_groups = set(
            frame.loc[frame["split"].eq("train"), "acquisition_group"].astype(str)
        )
        test_groups = set(
            frame.loc[frame["split"].eq("test"), "acquisition_group"].astype(str)
        )
        acquisition_audit = {
            "acquisition_group_label_counts": {
                str(group): {str(label): int(value) for label, value in row.items()}
                for group, row in group_table.iterrows()
            },
            "acquisition_group_max_label_purity": float(group_purity.max()),
            "capture_label_confounding_flag": bool(
                len(group_table) > 1 and group_purity.max() >= 0.95
            ),
            "train_acquisition_groups": sorted(train_groups),
            "test_acquisition_groups": sorted(test_groups),
            "acquisition_group_overlap_count": len(train_groups & test_groups),
        }
    task_counts = (
        frame["task_type"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
    )
    audit = {
        "dataset_id": dataset["dataset_id"],
        "channel": dataset["channel"],
        "task_type": dataset["task_type"],
        "recordings": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "train_recordings": int(frame["split"].eq("train").sum()),
        "test_recordings": int(frame["split"].eq("test").sum()),
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "label_counts_by_split": {
            split: group[["subject_id", "label"]].drop_duplicates()["label"].value_counts().sort_index().to_dict()
            for split, group in frame.groupby("split")
        },
        "task_counts": {str(key): int(value) for key, value in task_counts.items()},
        "subject_overlap_count": len(overlap),
        "subject_overlap": overlap,
        "subject_label_conflict_count": int(len(conflicts)),
        "duplicate_case_id_count": len(duplicate_case_ids),
        "duplicate_case_ids": duplicate_case_ids,
        "duplicate_audio_hash_count": len(duplicate_hashes),
        "duplicate_audio_hash_rows": duplicate_hashes,
        "cross_split_duplicate_audio_hash_count": len(cross_split_duplicate_rows),
        "cross_split_duplicate_audio_hash_rows": cross_split_duplicate_rows,
        "all_audio_mono_16khz": bool(frame["sample_rate"].eq(16000).all() and frame["channels"].eq(1).all()),
        "passed": not overlap and not duplicate_case_ids and not cross_split_duplicate_rows and conflicts.empty and bool(train_subjects) and bool(test_subjects),
        "warnings": warnings or [],
        **acquisition_audit,
    }
    json_dump(audit, audit_path)
    if not audit["passed"]:
        raise RuntimeError(f"Dataset audit failed: {audit}")


def build_iaeav_manifest(paths: ProjectPaths, dataset: dict[str, Any], manifest_path: Path, audit_path: Path) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    rows = []
    for source in sorted((root / "transcripts_json").glob("*.json")):
        item = json.loads(source.read_text(encoding="utf-8"))
        group = str(item.get("group", "")).lower()
        label = "HC" if "control" in group else "AD" if "patient" in group else ""
        audio = Path(str(item.get("audio_path", "")))
        if not label or not audio.exists():
            continue
        subject_id = str(item.get("sid") or source.stem)
        row = _generic_row(dataset, audio, subject_id, label, "pending", str(source), language="es")
        row["case_id"] = subject_id
        interviewer = re.search(r"(?:^|-)inv(\d+)(?:-|$)", subject_id.lower())
        row["acquisition_group"] = f"inv{interviewer.group(1)}" if interviewer else "unknown"
        row["analysis_intervals"] = json.dumps(item.get("pac_segments", []), ensure_ascii=False)
        rows.append(row)
    frame = _balanced_acquisition_group_holdout(pd.DataFrame(rows))
    _write_manifest_audit(
        frame,
        dataset,
        manifest_path,
        audit_path,
        [
            "No official held-out split; one complete label-balanced acquisition group is held out to prevent interviewer/capture leakage.",
            "Interviewer/acquisition identifiers are audited because they are strongly associated with diagnosis in this release.",
        ],
    )


def build_adress2020_manifest(paths: ProjectPaths, dataset: dict[str, Any], manifest_path: Path, audit_path: Path) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    base = root / dataset["managed_train_root"]
    rows = []
    for folder, label in [("cc", "HC"), ("cd", "AD")]:
        for cha in sorted((base / "transcription" / folder).glob("*.cha")):
            wav = base / "Full_wave_enhanced_audio" / folder / f"{cha.stem}.wav"
            if wav.exists():
                row = _generic_row(dataset, wav, cha.stem, label, "pending", str(cha), language="en", speaker_role_source="CHAT_PAR_INV")
                row["analysis_intervals"] = json.dumps(patient_intervals_from_cha(cha))
                rows.append(row)
    frame = _stable_subject_split(pd.DataFrame(rows), float(dataset["test_size"]), int(dataset["split_seed"]))
    _write_manifest_audit(frame, dataset, manifest_path, audit_path, ["Official challenge test labels are unavailable locally; holdout comes from official training subjects."])


def build_adresso_manifest(paths: ProjectPaths, dataset: dict[str, Any], manifest_path: Path, audit_path: Path) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    base = root / dataset["managed_train_root"]
    rows = []
    for folder, label in dataset["folder_labels"].items():
        for segmentation in sorted((base / "segmentation" / folder).glob("*.csv")):
            wav = base / "audio" / folder / f"{segmentation.stem}.wav"
            if wav.exists():
                row = _generic_row(dataset, wav, segmentation.stem, label, "pending", "", language="en", speaker_role_source="distributed_segmentation")
                row["segmentation_path"] = str(segmentation)
                row["analysis_intervals"] = json.dumps(patient_intervals_from_segmentation(segmentation))
                rows.append(row)
    frame = _stable_subject_split(pd.DataFrame(rows), float(dataset["test_size"]), int(dataset["split_seed"]))
    _write_manifest_audit(frame, dataset, manifest_path, audit_path, ["Official test labels are unavailable locally; holdout comes from official training subjects."])


def build_process2_manifest(paths: ProjectPaths, dataset: dict[str, Any], manifest_path: Path, audit_path: Path) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    metadata = pd.read_csv(root / "meta-info.csv")
    label_map = {"HC": "HC", "MCI": "MCI", "Dementia": "AD"}
    rows = []
    for item in metadata.to_dict("records"):
        subject_id = str(item["IDs"])
        label = label_map[str(item["diagnosis"])]
        split = str(item["Split"]).lower()
        for task in dataset["tasks"]:
            wav = root / subject_id / f"{subject_id}__{task}.wav"
            transcript = root / subject_id / f"{subject_id}__{task}.txt"
            if wav.exists():
                rows.append(_generic_row(dataset, wav, subject_id, label, split, str(transcript) if transcript.exists() else "", task_type=task.lower(), language="en"))
                rows[-1]["case_id"] = f"{subject_id}__{task}"
                rows[-1]["sex"] = str(item.get("gender", "U"))
    _write_manifest_audit(pd.DataFrame(rows), dataset, manifest_path, audit_path)


def _extract_archive_if_needed(archive: Path, destination: Path) -> None:
    audio_suffixes = {".wav", ".mp3", ".flac", ".m4a"}
    if destination.exists() and any(path.suffix.lower() in audio_suffixes for path in destination.rglob("*")):
        return
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
    else:
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                target = (destination / member.name).resolve()
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise RuntimeError(f"Unsafe archive member: {member.name}")
            handle.extractall(destination)


def build_prepare_manifest(paths: ProjectPaths, dataset: dict[str, Any], manifest_path: Path, audit_path: Path) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    extracted = root / "_harness_extracted"
    _extract_archive_if_needed(root / "train_audios.zip", extracted / "train")
    _extract_archive_if_needed(root / "test_audios.zip", extracted / "test")
    label_map = {"Control": "HC", "MCI": "MCI", "AD": "AD", "ProbableAD": "AD", "PPA": "AD"}
    metadata = pd.read_csv(root / "metadata.csv")
    task_path = paths.root / "references" / "speechcare" / "prepare_task_labels.csv"
    if not task_path.exists():
        raise FileNotFoundError(
            "PREPARE task routing requires references/speechcare/prepare_task_labels.csv."
        )
    task_labels = pd.read_csv(task_path, dtype={"uid": str}).set_index("uid")
    rows = []
    for item in metadata.to_dict("records"):
        split = str(item["split"]).lower()
        candidates = list((extracted / split).rglob(f"{item['uid']}.*"))
        audio = next((path for path in candidates if path.suffix.lower() in {".mp3", ".wav", ".flac"}), None)
        if audio is None:
            continue
        uid = str(item["uid"])
        released_task = str(task_labels.at[uid, "task"]).strip() if uid in task_labels.index else "Other"
        task_valid = str(task_labels.at[uid, "valid"]).strip() if uid in task_labels.index else "unknown"
        # SpeechCARE excludes invalid development records but evaluates the full
        # official test set. Mirror that protocol without inspecting test labels.
        if split != "test" and task_valid.lower() != "yes":
            continue
        rows.append(
            _generic_row(
                dataset,
                audio,
                uid,
                label_map[str(item["diagnosis"])],
                split,
                task_type=PREPARE_TASK_MAP.get(released_task, "other"),
                language=str(item.get("language", "en")),
                age=float(item["age"]) if pd.notna(item.get("age")) else None,
            )
        )
        rows[-1]["sex"] = str(item.get("gender", "U"))
        rows[-1]["corpus"] = str(item.get("corpus", ""))
        rows[-1]["task_type_source"] = "SpeechCARE_released_llm_task_label"
        rows[-1]["task_type_valid"] = task_valid
    _write_manifest_audit(
        pd.DataFrame(rows),
        dataset,
        manifest_path,
        audit_path,
        [
            "AD, probable AD and PPA are grouped into ADRD for the distributed three-class target.",
            "Task routing uses the per-UID labels released with SpeechCARE; it is no longer collapsed to one generic PREPARE task.",
            "Records marked invalid by the released SpeechCARE preprocessing table are excluded from development only; the complete official test set is retained.",
        ],
    )


def build_taukadial_manifest(paths: ProjectPaths, dataset: dict[str, Any], manifest_path: Path, audit_path: Path) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    extracted = root / "_harness_extracted"
    _extract_archive_if_needed(root / "TAUKADIAL-24-train.tgz", extracted / "train")
    _extract_archive_if_needed(root / "TAUKADIAL-24-test.tgz", extracted / "test")
    with tarfile.open(root / "TAUKADIAL-24-train.tgz", "r:gz") as handle:
        groundtruth = pd.read_csv(handle.extractfile("TAUKADIAL-24/train/groundtruth.csv"))
    test_truth = pd.read_csv(root / "testgroundtruth.csv", sep=";")
    test_metadata = pd.read_csv(root / "meta_test.csv", sep=";")
    test_truth = test_truth.merge(test_metadata, on="tkdname", how="left", validate="one_to_one")
    rows = []
    for split, truth in [("train", groundtruth), ("test", test_truth)]:
        for item in truth.to_dict("records"):
            filename = str(item["tkdname"])
            audio = next(iter((extracted / split).rglob(filename)), None)
            if audio is None:
                continue
            subject_id = filename.rsplit("-", 1)[0]
            label = "HC" if str(item["dx"]) == "NC" else "MCI"
            rows.append(
                _generic_row(
                    dataset,
                    audio,
                    subject_id,
                    label,
                    split,
                    task_type="spontaneous_speech",
                    language="zh-en",
                    age=float(item["age"]) if pd.notna(item.get("age")) else None,
                )
            )
            rows[-1]["case_id"] = Path(filename).stem
            rows[-1]["sex"] = str(item.get("sex", "U"))
            index_match = re.search(r"-(\d+)$", Path(filename).stem)
            rows[-1]["recording_index"] = int(index_match.group(1)) if index_match else 1
    _write_manifest_audit(pd.DataFrame(rows), dataset, manifest_path, audit_path)


def build_dementianet_manifest(paths: ProjectPaths, dataset: dict[str, Any], manifest_path: Path, audit_path: Path) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    source = pd.read_csv(root / "local_manifest.csv")
    rows = []
    for item in source.to_dict("records"):
        audio = Path(str(item["audio_path"]))
        if not audio.exists():
            continue
        stem = audio.stem
        subject_id = re.sub(r"(?:_\d+)+$", "", stem).lower()
        label_map = {"dementia": "AD", "nodementia": "HC"}
        source_label = str(item["group"]).strip().lower()
        if source_label not in label_map:
            raise ValueError(f"Unexpected DementiaNet label: {item['group']!r}")
        label = label_map[source_label]
        rows.append(_generic_row(dataset, audio, subject_id, label, "pending", str(item.get("txt_path", "")), language="en"))
    frame = _stable_subject_split(pd.DataFrame(rows), float(dataset["test_size"]), int(dataset["split_seed"]))
    _write_manifest_audit(frame, dataset, manifest_path, audit_path, ["Public-figure corpus is engineering-only; multiple clips from one person remain in one split."])


def build_pitt_manifest(paths: ProjectPaths, dataset: dict[str, Any], manifest_path: Path, audit_path: Path) -> None:
    root = _resolve_raw_path(paths, dataset["raw_path"])
    source = pd.read_csv(root / "talkbank_media_pairing_manifest.csv")
    source = source[source["pairing_status"].eq("paired")]
    primary_task = str(dataset.get("primary_task", "")).strip().lower()
    if primary_task:
        source = source[source["task"].astype(str).str.lower().eq(primary_task)]
    rows = []
    for item in source.to_dict("records"):
        media_paths = [Path(path) for path in str(item["all_media_paths"]).split("|") if path]
        task_token = f"/{item['group']}/{item['task']}/"
        task_media = [path for path in media_paths if task_token in str(path)]
        audio = task_media[0] if task_media else Path(str(item["preferred_media_path"]))
        transcript = Path(str(item["transcript_path"]))
        if not audio.exists() or not transcript.exists():
            continue
        source_case_id = str(item["case_id"])
        label = "HC" if str(item["group"]).lower() == "control" else "AD"
        subject_id = f"{label}_{source_case_id.split('-', 1)[0]}"
        record = _generic_row(
            dataset,
            audio,
            subject_id,
            label,
            "pending",
            str(transcript),
            task_type=str(item["task"]),
            language="en",
            speaker_role_source="CHAT_PAR_INV",
        )
        task_id = re.sub(r"[^A-Za-z0-9]+", "-", str(item["task"])).strip("-").lower()
        record["case_id"] = f"PITT-{task_id}-{source_case_id}-{record['audio_sha256'][:8]}"
        record["source_case_id"] = source_case_id
        intervals = patient_intervals_from_cha(transcript)
        record["analysis_intervals"] = json.dumps(intervals)
        record["role_filter_available"] = bool(intervals)
        rows.append(record)
    frame = pd.DataFrame(rows).drop_duplicates("audio_sha256", keep="first")
    frame = _stable_subject_split(frame, float(dataset["test_size"]), int(dataset["split_seed"]))
    _write_manifest_audit(
        frame,
        dataset,
        manifest_path,
        audit_path,
        [
            "Repeated tasks and visits are grouped by Pitt participant identifier before splitting; byte-identical media are deduplicated.",
            "The primary experiment is restricted to the configured task because non-cookie task availability is diagnosis-confounded in the distributed cohort."
            if primary_task
            else "All distributed tasks are retained; task-presence confounding must be audited.",
        ],
    )


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
        "analysis_intervals": "[]",
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
    frame["source_identity_key"] = (
        frame["sex"].astype(str)
        + "_"
        + frame["subject_id"].astype(str).str.rsplit("_", n=1).str[-1]
    )
    conflicting_identity_keys = (
        frame.groupby("source_identity_key")["label"]
        .nunique()
        .loc[lambda value: value > 1]
        .index.tolist()
    )
    excluded_identity_rows = frame[
        frame["source_identity_key"].isin(conflicting_identity_keys)
    ][["case_id", "subject_id", "label", "split", "source_identity_key"]].to_dict("records")
    if conflicting_identity_keys:
        frame = frame[~frame["source_identity_key"].isin(conflicting_identity_keys)].copy()
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
    min_duration = float(dataset.get("long_recording_min_duration_sec", 25.0))
    max_duration = float(dataset.get("long_recording_max_duration_sec", 75.0))
    duration_pass = frame["duration_sec"].between(min_duration, max_duration).all()
    identity_key = frame["source_identity_key"]
    cross_label_identity_candidates = (
        frame.assign(_identity_key=identity_key)
        .groupby("_identity_key")["label"]
        .nunique()
        .loc[lambda value: value > 1]
        .index.tolist()
    )
    audit = {
        "dataset_id": dataset["dataset_id"],
        "analysis_input": "AD_dataset_long only",
        "six_second_files_discovered_but_excluded": len(six_second_files),
        "six_second_files_in_analysis": int(
            frame["audio_path"].astype(str).str.contains("AD_dataset_6s", regex=False).sum()
        ),
        "long_recording_only_passed": bool(
            ~frame["audio_path"].astype(str).str.contains("AD_dataset_6s", regex=False).any()
        ),
        "long_duration_tolerance_sec": [min_duration, max_duration],
        "long_duration_range_passed": bool(duration_pass),
        "observed_duration_range_sec": [
            float(frame["duration_sec"].min()),
            float(frame["duration_sec"].max()),
        ],
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
        "cross_label_sex_numeric_identity_candidates": cross_label_identity_candidates,
        "excluded_cross_label_identity_rows": excluded_identity_rows,
        "duplicate_audio_hash_count": len(duplicate_hashes),
        "duplicate_audio_hash_rows": duplicate_hashes,
        "all_audio_mono_16khz": bool(
            frame["sample_rate"].eq(16000).all() and frame["channels"].eq(1).all()
        ),
        "passed": bool(
            not overlap
            and not duplicate_hashes
            and label_conflicts.empty
            and duration_pass
            and frame["sample_rate"].eq(16000).all()
            and frame["channels"].eq(1).all()
            and not frame["audio_path"].astype(str).str.contains("AD_dataset_6s", regex=False).any()
        ),
        "warnings": [
            "Per-recording task mapping is unavailable; recording index is not interpreted as a task label.",
            "NCMMSC has no local human transcript or speaker-role annotation.",
            "The subject key retains diagnosis + sex + six-digit identifier because the distributed numeric field is not globally unique across label folders; cross-label sex+numeric collisions are reported for manual provenance review and are not silently merged.",
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
    adapters = {
        "IAEAV": build_iaeav_manifest,
        "ADReSS_2020": build_adress2020_manifest,
        "ADReSSo_2021_diagnosis": build_adresso_manifest,
        "ADReSSo_2021_progression": build_adresso_manifest,
        "PROCESS_2": build_process2_manifest,
        "PREPARE_DrivenData": build_prepare_manifest,
        "TAUKADIAL": build_taukadial_manifest,
        "DementiaNet_PublicFigures": build_dementianet_manifest,
        "DementiaBank_Pitt": build_pitt_manifest,
    }
    if dataset["dataset_id"] not in adapters:
        raise NotImplementedError(f"No manifest adapter for {dataset['dataset_id']}")
    adapters[dataset["dataset_id"]](paths, dataset, manifest_path, audit_path)
