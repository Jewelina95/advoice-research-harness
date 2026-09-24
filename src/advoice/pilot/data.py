"""Private-source manifest freezing for the evidence-state pilot.

This module deliberately separates the private source index (which may contain
paths and source labels) from records allowed to enter the provider boundary.
It reads metadata and historical manifests only; it never opens audio samples.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .contracts import CLASS_ORDERS, PilotContractError, SubjectRow, safe_inference_payload


SEED = 20260923
CORE_COHORTS = {
    "PREPARE_DrivenData": (150, 30),
    "ADReSS_2020": (81, 16),
    "NCMMSC2021_AD": (120, 24),
}
STRESS_COHORTS = {"IAEAV": 14, "DementiaNet_PublicFigures": 6}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a"}


class ManifestError(ValueError):
    """The inventory cannot support a leakage-safe pilot manifest."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _pseudo(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _source_hash(path: str) -> str:
    """Hash the path token, not the media bytes; full audio hashing is out of scope."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _task_for_labels(labels: set[str]) -> tuple[str, tuple[str, ...]]:
    if labels <= {"HC", "AD"}:
        return "hc_ad", CLASS_ORDERS["hc_ad"]
    if labels <= {"HC", "MCI", "AD"}:
        return "hc_mci_ad", CLASS_ORDERS["hc_mci_ad"]
    raise ManifestError("Unsupported source label mapping.")


def _normalise_source(row: Mapping[str, Any]) -> dict[str, str]:
    required = {"dataset_id", "source_id", "subject_id", "source_group_id", "label", "language", "raw_path"}
    missing = sorted(required - set(row))
    if missing:
        raise ManifestError(f"Source inventory is missing fields: {','.join(missing)}")
    value = {key: str(row.get(key, "")) for key in (
        "dataset_id", "source_id", "subject_id", "source_group_id", "label", "language", "raw_path",
        "transcript_path", "task_id", "source_split", "source_version", "source_track", "raw_hash",
    )}
    value["raw_hash"] = value["raw_hash"] or _source_hash(value["raw_path"])
    value["task_id"] = value["task_id"] or "overall"
    value["source_version"] = value["source_version"] or "metadata_v1"
    value["source_track"] = value["source_track"] or "primary"
    value["source_split"] = value["source_split"] or "development"
    return value


def _group_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["source_group_id"])].append(row)
    subjects: list[dict[str, Any]] = []
    for (_, group_id), members in sorted(grouped.items()):
        labels = {item["label"] for item in members}
        subjects_in_group = {item["subject_id"] for item in members}
        if len(labels) != 1 or len(subjects_in_group) != 1:
            raise ManifestError("An identity group has conflicting source labels or subject identities.")
        item = min(members, key=lambda x: x["source_id"])
        subjects.append({
            "dataset_id": item["dataset_id"], "source_id": item["source_id"],
            "subject_id": item["subject_id"], "source_group_id": group_id,
            "label": item["label"], "language": item["language"],
            "task_ids": tuple(sorted({member["task_id"] for member in members})),
            "raw_hashes": tuple(sorted({member["raw_hash"] for member in members})),
            "source_version": item["source_version"], "rows": tuple(members),
        })
    return subjects


def _stratified_take(rows: Sequence[dict[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ManifestError(f"Requested {count} identities but only {len(rows)} are eligible.")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["label"], row["language"])].append(row)
    # Rare language strata are retained only when they can support deterministic sampling.
    if any(len(items) < 2 for items in strata.values()):
        strata = defaultdict(list)
        for row in rows:
            strata[(row["label"], "all")].append(row)
    allocation: dict[tuple[str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str]]] = []
    for key, items in strata.items():
        raw = count * len(items) / len(rows)
        allocation[key] = min(len(items), int(raw))
        fractions.append((raw - int(raw), key))
    remaining = count - sum(allocation.values())
    for _, key in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if not remaining:
            break
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise ManifestError("Stratified allocation could not satisfy the requested cohort size.")
    chosen: list[dict[str, Any]] = []
    for offset, key in enumerate(sorted(strata)):
        values = sorted(strata[key], key=lambda row: row["source_group_id"])
        random.Random(seed + offset).shuffle(values)
        chosen.extend(values[:allocation[key]])
    return sorted(chosen, key=lambda row: row["source_group_id"])


def _subject_row(subject: Mapping[str, Any], partition: str, fold_id: str = "unassigned") -> SubjectRow:
    task, class_order = _task_for_labels({str(subject["label"])})
    # The dataset-level task is resolved by build_pilot_manifest before this call.
    return SubjectRow(
        dataset_id=str(subject["dataset_id"]), subject_id=_pseudo("sub", str(subject["dataset_id"]), str(subject["subject_id"])),
        partition=partition, task=task, class_order=class_order,
        source_group_id=_pseudo("grp", str(subject["dataset_id"]), str(subject["source_group_id"])),
        channel="source_bound", language=str(subject["language"]) or "und", task_ids=tuple(subject["task_ids"]),
        role="participant", fold_id=fold_id,
        raw_hashes={_pseudo("asset", digest): digest for digest in subject["raw_hashes"]},
        source_version=str(subject["source_version"]),
    )


def _provider_row(row: SubjectRow, *, task: str, class_order: tuple[str, ...]) -> dict[str, Any]:
    value = row.to_dict()
    value["task"] = task
    value["class_order"] = class_order
    safe_inference_payload(value)
    return value


def _validate_identity_disjoint(rows: Sequence[dict[str, Any]]) -> None:
    by_group: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_group[str(row["source_group_id"])].add(str(row["partition"]))
        for digest in row["raw_hashes"]:
            by_hash[str(digest)].add(str(row["partition"]))
    if any(len(values) > 1 for values in by_group.values()):
        raise ManifestError("A source identity group appears in multiple pilot partitions.")
    if any(len(values) > 1 for values in by_hash.values()):
        raise ManifestError("A duplicate raw hash appears in multiple pilot partitions.")


def build_pilot_manifest(config: Mapping[str, Any], source_inventory: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Freeze provider-safe subjects, coded exclusions, and a label-free audit.

    ``source_inventory`` remains caller-owned/private. The returned manifest contains
    no source labels or local filesystem paths.
    """
    seed = int(config.get("seed", SEED))
    source = [_normalise_source(row) for row in source_inventory]
    exclusions: list[dict[str, str]] = []
    eligible: list[dict[str, str]] = []
    for row in source:
        reason = ""
        if row["source_track"] == "six_second" or "AD_dataset_6s" in row["raw_path"]:
            reason = "six_second_track"
        elif not row["raw_path"]:
            reason = "missing_raw_provenance"
        elif row["dataset_id"] == "PREPARE_DrivenData" and row["source_split"] == "invalid":
            reason = "prepare_invalid_uid"
        elif row["source_split"] not in {"development", "stress"}:
            reason = "source_cohort_excluded"
        if reason:
            exclusions.append({"dataset_id": row["dataset_id"], "source_id": _pseudo("src", row["dataset_id"], row["source_id"]), "reason": reason})
        else:
            eligible.append(row)

    grouped = _group_rows(eligible)
    outputs: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"seed": seed, "source_records": len(source), "excluded_records": len(exclusions), "cohorts": {}, "canary_gap": 12}
    for dataset_id, (total, holdout_count) in CORE_COHORTS.items():
        candidates = [row for row in grouped if row["dataset_id"] == dataset_id and row["rows"][0]["source_split"] == "development"]
        selected = _stratified_take(candidates, total, seed=seed + len(outputs))
        holdout = {row["source_group_id"] for row in _stratified_take(selected, holdout_count, seed=seed + 97)}
        labels = {str(row["label"]) for row in selected}
        task, class_order = _task_for_labels(labels)
        for subject in selected:
            partition = "holdout" if subject["source_group_id"] in holdout else "development"
            value = _provider_row(_subject_row(subject, partition), task=task, class_order=class_order)
            value["source_group_id"] = _pseudo("grp", dataset_id, str(subject["source_group_id"]))
            value["raw_hashes"] = {_pseudo("asset", digest): digest for digest in subject["raw_hashes"]}
            outputs.append(value)
        audit["cohorts"][dataset_id] = {"selected": total, "development": total - holdout_count, "holdout": holdout_count, "source_pool": len(candidates)}
    for dataset_id, count in STRESS_COHORTS.items():
        candidates = [row for row in grouped if row["dataset_id"] == dataset_id and row["rows"][0]["source_split"] == "stress"]
        selected = _stratified_take(candidates, count, seed=seed + len(outputs))
        labels = {str(row["label"]) for row in selected}
        task, class_order = _task_for_labels(labels)
        for subject in selected:
            outputs.append(_provider_row(_subject_row(subject, "stress"), task=task, class_order=class_order))
        audit["cohorts"][dataset_id] = {"selected": count, "stress": count, "source_pool": len(candidates)}
    _validate_identity_disjoint(outputs)
    outputs.sort(key=lambda row: (row["dataset_id"], row["partition"], row["subject_id"]))
    return outputs, sorted(exclusions, key=lambda row: (row["dataset_id"], row["source_id"])), audit


def make_group_folds(development_manifest: Sequence[Mapping[str, Any]], labels: Mapping[str, str], seed: int, n_splits: int = 5) -> list[dict[str, str]]:
    """Assign deterministic grouped folds and fail before training on absent classes."""
    rows = [dict(row) for row in development_manifest if row["partition"] == "development"]
    if n_splits < 2 or not rows:
        raise ManifestError("Grouped OOF requires development rows and at least two folds.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_group_id"])].append(row)
    group_labels: dict[str, str] = {}
    for group, members in groups.items():
        values = {labels.get(str(item["subject_id"]), "") for item in members}
        if len(values) != 1 or "" in values:
            raise ManifestError("Every development identity group needs exactly one private label.")
        group_labels[group] = next(iter(values))
    by_label: dict[str, list[str]] = defaultdict(list)
    for group, label in group_labels.items():
        by_label[label].append(group)
    expected_classes = {str(value) for row in rows for value in row["class_order"]}
    if set(by_label) != expected_classes:
        raise ManifestError("Missing class in an OOF fit partition.")
    if any(len(groups_for_label) < n_splits for groups_for_label in by_label.values()):
        raise ManifestError("Missing class support in an OOF fit partition.")
    assignment: dict[str, str] = {}
    for offset, label in enumerate(sorted(by_label)):
        values = sorted(by_label[label])
        random.Random(seed + offset).shuffle(values)
        for index, group in enumerate(values):
            assignment[group] = f"fold_{index % n_splits}"
    result = [{"subject_id": str(row["subject_id"]), "fold_id": assignment[str(row["source_group_id"])]} for row in rows]
    for fold in {row["fold_id"] for row in result}:
        fit_labels = {labels[row["subject_id"]] for row in result if row["fold_id"] != fold}
        if fit_labels != set(by_label):
            raise ManifestError("Missing class in an OOF fit partition.")
    return sorted(result, key=lambda row: row["subject_id"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def actual_source_inventory(raw_root: str | Path, historical_root: str | Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Map current mounted data through existing historical source boundaries.

    Historical files are used only to preserve previously audited source membership,
    never as model features or OOF predictions.
    """
    raw_root, historical_root = Path(raw_root).resolve(), Path(historical_root).resolve()
    inventory: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    for dataset_id in (*CORE_COHORTS, *STRESS_COHORTS):
        rows = _read_csv(historical_root / dataset_id / "manifest.csv")
        for row in rows:
            path = row.get("audio_path", "")
            if not path or not Path(path).exists():
                exclusions.append({"dataset_id": dataset_id, "source_id": _pseudo("src", dataset_id, row.get("case_id", "")), "reason": "missing_raw_provenance"})
                continue
            old_split = row.get("split", "")
            source_split = "development"
            if dataset_id in STRESS_COHORTS:
                source_split = "stress" if old_split == "test" else "excluded"
            elif dataset_id in {"PREPARE_DrivenData", "NCMMSC2021_AD"}:
                source_split = "development" if old_split == "train" else "excluded"
            elif dataset_id == "ADReSS_2020":
                source_split = "development" if old_split == "train" else "excluded"
            track = "six_second" if "AD_dataset_6s" in path else "long" if dataset_id == "NCMMSC2021_AD" else "primary"
            inventory.append({
                "dataset_id": dataset_id, "source_id": row.get("case_id", ""), "subject_id": row.get("subject_id", ""),
                "source_group_id": row.get("source_identity_key") or row.get("subject_id", ""), "label": row.get("label", ""),
                "language": row.get("language", "und"), "raw_path": path, "transcript_path": row.get("transcript_path", ""),
                "task_id": row.get("task_type", "overall"), "source_split": source_split, "source_track": track,
                "raw_hash": row.get("audio_sha256", ""), "source_version": "historical_manifest_boundary_v1",
            })
    # Historical manifests intentionally omit PREPARE's invalid development UIDs.
    # Re-add those metadata rows solely as coded exclusions, so the audit verifies
    # identity-level agreement rather than reporting an uncheckable aggregate.
    prepare_root = raw_root / "DementiaBank_Challenges" / "PREPARE_DrivenData"
    metadata_path = prepare_root / "metadata.csv"
    task_path = Path(__file__).resolve().parents[3] / "references" / "speechcare" / "prepare_task_labels.csv"
    if metadata_path.is_file() and task_path.is_file():
        task_rows = {row["uid"]: row for row in _read_csv(task_path)}
        known = {row["source_id"] for row in inventory if row["dataset_id"] == "PREPARE_DrivenData"}
        label_map = {"Control": "HC", "MCI": "MCI", "AD": "AD", "ProbableAD": "AD", "PPA": "AD"}
        for row in _read_csv(metadata_path):
            uid = row.get("uid", "")
            task = task_rows.get(uid, {})
            if row.get("split", "").lower() != "train" or uid in known:
                continue
            candidates = sorted((prepare_root / "_harness_extracted" / "train").glob(f"{uid}.*"))
            audio = next((path for path in candidates if path.suffix.lower() in _AUDIO_SUFFIXES), None)
            inventory.append({
                "dataset_id": "PREPARE_DrivenData", "source_id": uid, "subject_id": uid, "source_group_id": uid,
                "label": label_map.get(row.get("diagnosis", ""), "AD"), "language": row.get("language", "und"),
                "raw_path": str(audio) if audio else "", "transcript_path": "", "task_id": task.get("task", "overall"),
                "source_split": "invalid", "source_track": "primary", "raw_hash": _source_hash(str(audio) if audio else uid),
                "source_version": "prepare_task_labels_v1",
            })
    return inventory, exclusions


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_pilot_artifacts(output_dir: str | Path, source_inventory: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]], exclusions: Sequence[Mapping[str, str]], audit: Mapping[str, Any], folds: Sequence[Mapping[str, str]], labels: Mapping[str, str]) -> dict[str, Path]:
    """Write private source mappings plus safe manifests under an ignored directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_index = output / "source_index.jsonl"
    source_index.write_text("".join(_canonical(dict(row)).decode("utf-8") + "\n" for row in source_inventory), encoding="utf-8")
    safe_columns = ("schema_version", "dataset_id", "subject_id", "partition", "task", "class_order", "source_group_id", "channel", "language", "task_ids", "role", "fold_id", "raw_hashes", "source_version")
    public_rows = []
    for row in manifest:
        safe_inference_payload(row)
        public_rows.append({key: json.dumps(row[key], sort_keys=True, separators=(",", ":")) if isinstance(row[key], (dict, list, tuple)) else row[key] for key in safe_columns})
    _write_csv(output / "split_manifest.csv", public_rows, safe_columns)
    _write_csv(output / "folds.csv", folds, ("subject_id", "fold_id"))
    _write_csv(output / "labels.csv", [{"subject_id": key, "label": value} for key, value in sorted(labels.items())], ("subject_id", "label"))
    _write_csv(output / "exclusions.csv", exclusions, ("dataset_id", "source_id", "reason"))
    (output / "identity_audit.json").write_bytes(_canonical(dict(audit)))
    cache = {"raw_deterministic_metrics": "eligible_with_verified_raw_hash", "unadapted_encoder_outputs": "eligible_with_verified_source_provenance", "global_references": "ineligible", "old_state_normalization": "ineligible", "historical_in_sample_predictions": "ineligible"}
    (output / "cache_provenance.json").write_bytes(_canonical(cache))
    return {name: output / name for name in ("source_index.jsonl", "split_manifest.csv", "folds.csv", "labels.csv", "exclusions.csv", "identity_audit.json", "cache_provenance.json")}
