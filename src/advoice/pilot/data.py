"""Deterministic, leakage-safe data freezing for the evidence-state pilot."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from .contracts import CLASS_ORDERS, PilotContractError, SubjectRow, safe_inference_payload
from .labels import LabelRow, read_labels_jsonl, serialize_labels_jsonl

SEED = 20260923
CORE_COHORTS = {"PREPARE_DrivenData": (150, 30), "ADReSS_2020": (81, 16), "NCMMSC2021_AD": (120, 24)}
STRESS_COHORTS = {"IAEAV": 14, "DementiaNet_PublicFigures": 6}
PREPARE_INVALID_EXPECTED = 33
PREPARE_TASK_SOURCE_VERSION = "speechcare_bf1281d2e6e3617b3c81d16220a96e02646a570d"
PREPARE_TASK_SHA256 = "3853fc9fe79802cb396b54737e63bc3851d1362382e384b6d541fabdaba05589"
PREPARE_827_MANIFEST_SHA256 = "0fd86e2f936b2c7aa7c0c79985f469d4543c5071e31111f826b684f7a83bac05"
_AUDIO = {".wav", ".mp3", ".flac", ".m4a"}
_NCMMSC = re.compile(r"^(AD|MCI|HC)_([FM])_(\d+)[_-]([0-9]+)$", re.I)


class ManifestError(ValueError):
    """The inventory cannot support a leakage-safe pilot manifest."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pseudo(prefix: str, *parts: str) -> str:
    return f"{prefix}_{hashlib.sha256(chr(0).join(parts).encode()).hexdigest()[:24]}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sort_source(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in ("dataset_id", "source_track", "source_split", "source_id", "raw_path"))


def _normalise(row: Mapping[str, Any]) -> dict[str, str]:
    required = {"dataset_id", "source_id", "subject_id", "source_group_id", "label", "language", "raw_path"}
    if required - set(row):
        raise ManifestError(f"Source inventory is missing fields: {','.join(sorted(required - set(row)))}")
    keys = (*required, "transcript_path", "transcript_availability", "transcript_provenance",
            "task_id", "source_split", "source_version", "source_track", "raw_hash",
            "source_provenance_hash", "source_quality_flag", "quality_provenance_hash")
    value = {key: str(row.get(key, "")) for key in keys}
    value["raw_hash"] = value["raw_hash"] or hashlib.sha256(value["raw_path"].encode()).hexdigest()
    value["task_id"] = value["task_id"] or "overall"
    value["source_split"] = value["source_split"] or "development"
    value["source_version"] = value["source_version"] or "metadata_v1"
    value["source_track"] = value["source_track"] or "primary"
    value["transcript_availability"] = value["transcript_availability"] or "not_declared"
    value["transcript_provenance"] = value["transcript_provenance"] or "not_declared"
    return value


def _task(labels: set[str]) -> tuple[str, tuple[str, ...]]:
    if labels == {"HC", "AD"}:
        return "hc_ad", CLASS_ORDERS["hc_ad"]
    if labels == {"HC", "MCI", "AD"}:
        return "hc_mci_ad", CLASS_ORDERS["hc_mci_ad"]
    raise ManifestError("Unsupported or incomplete source label mapping.")


def _groups(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["source_group_id"])].append(row)
    output = []
    for (_, group_id), members in sorted(grouped.items()):
        labels, subjects, versions = ({m[key] for m in members} for key in ("label", "subject_id", "source_version"))
        if len(labels) != 1 or len(subjects) != 1 or len(versions) != 1:
            raise ManifestError("An identity group has conflicting labels, subjects, or source versions.")
        first = min(members, key=_sort_source)
        output.append({"dataset_id": first["dataset_id"], "subject_id": first["subject_id"], "source_group_id": group_id,
                       "label": first["label"], "language": first["language"], "source_version": first["source_version"],
                       "task_ids": tuple(sorted({m["task_id"] for m in members})),
                       "raw_hashes": tuple(sorted({m["raw_hash"] for m in members})), "rows": tuple(members)})
    return output


def _canary_candidates(grouped: Sequence[dict[str, Any]], source: Sequence[Mapping[str, str]],
                       analytical_groups: set[tuple[str, str]], analytical_hashes: set[str]) -> list[dict[str, Any]]:
    reserved = [row for row in source if row["source_split"] == "excluded"]
    reserved_groups = {(row["dataset_id"], row["source_group_id"]) for row in reserved}
    reserved_hashes = {row["raw_hash"] for row in reserved}
    return [row for row in grouped if row["dataset_id"] == "PREPARE_DrivenData"
            and row["rows"][0]["source_split"] == "development"
            and (row["dataset_id"], row["source_group_id"]) not in analytical_groups | reserved_groups
            and not (set(row["raw_hashes"]) & (analytical_hashes | reserved_hashes))]


def _take(rows: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ManifestError(f"Requested {count} identities but only {len(rows)} are eligible.")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["label"], row["language"])].append(row)
    if any(len(items) < 2 for items in strata.values()):
        strata = defaultdict(list)
        for row in rows:
            strata[(row["label"], "all")].append(row)
    allocation, fractions = {}, []
    for key, items in strata.items():
        raw = count * len(items) / len(rows)
        allocation[key], fractions = int(raw), [*fractions, (raw - int(raw), key)]
    remaining = count - sum(allocation.values())
    for _, key in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if remaining and allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise ManifestError("Stratified allocation could not satisfy cohort size.")
    chosen = []
    for offset, key in enumerate(sorted(strata)):
        values = sorted(strata[key], key=lambda row: row["source_group_id"])
        random.Random(seed + offset).shuffle(values)
        chosen.extend(values[:allocation[key]])
    return sorted(chosen, key=lambda row: row["source_group_id"])


def _folds(subjects: Sequence[Mapping[str, Any]], seed: int, n_splits: int) -> dict[str, str]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in subjects:
        by_label[str(row["label"])].append(str(row["source_group_id"]))
    _task(set(by_label))
    if any(len(values) < n_splits for values in by_label.values()):
        raise ManifestError("Missing class support in an OOF fit partition.")
    assignment = {}
    for offset, label in enumerate(sorted(by_label)):
        values = sorted(by_label[label])
        random.Random(seed + offset).shuffle(values)
        assignment.update({group: f"fold_{index % n_splits}" for index, group in enumerate(values)})
    for fold in set(assignment.values()):
        if {row["label"] for row in subjects if assignment[row["source_group_id"]] != fold} != set(by_label):
            raise ManifestError("Missing class in an OOF fit partition.")
    return assignment


def _subject(row: Mapping[str, Any], partition: str, task: str, order: tuple[str, ...], fold: str) -> SubjectRow:
    return SubjectRow(dataset_id=row["dataset_id"], subject_id=_pseudo("sub", row["dataset_id"], row["subject_id"]),
                      partition=partition, task=task, class_order=order,
                      source_group_id=_pseudo("grp", row["dataset_id"], row["source_group_id"]), channel="source_bound",
                      language=row["language"] or "und", task_ids=tuple(row["task_ids"]), role="participant", fold_id=fold,
                      raw_hashes={_pseudo("asset", digest): digest for digest in row["raw_hashes"]}, source_version=row["source_version"])


def _prepare_audit(config: Mapping[str, Any], source: Sequence[dict[str, str]]) -> dict[str, Any]:
    invalid = [r for r in source if r["dataset_id"] == "PREPARE_DrivenData" and r["source_split"] == "invalid"]
    expected, observed = int(config.get("prepare_invalid_expected_count", PREPARE_INVALID_EXPECTED)), len(invalid)
    versions = sorted({r["source_version"] for r in invalid})
    prepare = [r for r in source if r["dataset_id"] == "PREPARE_DrivenData"]
    no_flags = [r for r in prepare if r["source_quality_flag"].lower() == "no"]
    included_no = [r for r in no_flags if r["source_split"] != "invalid"]
    task_hashes = sorted({r["quality_provenance_hash"] for r in prepare if r["quality_provenance_hash"]})
    manifest_hashes = sorted({r["source_provenance_hash"] for r in prepare if r["source_split"] != "invalid" and r["source_provenance_hash"]})
    result = {"protocol_expected_count": expected, "historical_omission_count": observed,
              "source_no_flag_count": len(no_flags), "included_no_flag_count": len(included_no),
              "included_no_flag_ids": sorted(_pseudo("src", "PREPARE_DrivenData", r["source_id"]) for r in included_no),
              "source_versions": versions, "task_label_hashes": task_hashes,
              "historical_manifest_hashes": manifest_hashes, "resolution": None}
    if expected == observed:
        result["resolution"] = "counts_match"
        return result
    pinned = (expected == 33 and observed == 24 and len(no_flags) == 28 and len(included_no) == 4
              and task_hashes == [PREPARE_TASK_SHA256]
              and manifest_hashes == [PREPARE_827_MANIFEST_SHA256])
    if not pinned:
        raise ManifestError(f"PREPARE invalid UID provenance mismatch: expected {expected}, observed {observed}; pinned source hashes/counts do not match.")
    result["resolution"] = "resolved_by_source_version"
    return result


def build_pilot_manifest(config: Mapping[str, Any], source_inventory: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    seed, n_splits = int(config.get("seed", SEED)), int(config.get("n_splits", 5))
    source = sorted((_normalise(row) for row in source_inventory), key=_sort_source)
    prepare = _prepare_audit(config, source)
    collisions = [row for row in source if row["source_split"] == "identity_collision"]
    collision_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in collisions:
        collision_groups[(row["dataset_id"], row["source_group_id"])].append(row)
    collision_audit = []
    blocked_datasets: set[str] = set()
    for (dataset, group), rows in sorted(collision_groups.items()):
        label_count = len({row["label"] for row in rows})
        hash_count = len({row["raw_hash"] for row in rows})
        unresolved = label_count > 1 or hash_count > 1
        if unresolved:
            blocked_datasets.add(dataset)
        collision_audit.append({"dataset_id": dataset, "source_group_id": _pseudo("grp", dataset, group),
                                "record_count": len(rows), "distinct_label_count": label_count,
                                "distinct_hash_count": hash_count,
                                "source_ids": sorted(_pseudo("src", dataset, row["source_id"]) for row in rows),
                                "resolution": ("unresolved_conflicting_identity_or_recording"
                                               if unresolved else "canonical_duplicate_excluded")})
    eligible, exclusions, reasons = [], [], Counter()
    for row in source:
        reason = ("six_second_track" if row["source_track"] == "six_second" else
                  "missing_raw_provenance" if not row["raw_path"] else
                  "prepare_invalid_uid" if row["dataset_id"] == "PREPARE_DrivenData" and row["source_split"] == "invalid" else
                  "source_identity_collision" if row["source_split"] == "identity_collision" else
                  "cohort_blocked_identity_collision" if row["dataset_id"] in blocked_datasets and row["source_split"] in {"development", "stress"} else
                  "source_cohort_excluded" if row["source_split"] not in {"development", "stress"} else "")
        if reason:
            reasons[reason] += 1
            exclusions.append({"dataset_id": row["dataset_id"], "source_id": _pseudo("src", row["dataset_id"], row["source_id"]), "reason": reason})
        else:
            eligible.append(row)
    hash_groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in eligible:
        hash_groups[row["raw_hash"]].add((row["dataset_id"], row["source_group_id"]))
    if any(len(groups) > 1 for groups in hash_groups.values()):
        raise ManifestError("A duplicate recording spans source identity groups.")
    grouped, output = _groups(eligible), []
    audit = {"status": "partial" if blocked_datasets else "ready", "seed": seed, "n_splits": n_splits,
             "source_records": len(source), "blocked_cohorts": sorted(blocked_datasets),
             "collision_groups": collision_audit,
             "source_records_by_dataset_track": {f"{d}:{t}": n for (d, t), n in sorted(Counter((r["dataset_id"], r["source_track"]) for r in source).items())},
             "excluded_records": len(exclusions), "exclusion_reason_counts": dict(sorted(reasons.items())),
             "prepare_invalid_uid_provenance": prepare,
             "transcript_availability_counts": {f"{d}:{status}": n for (d, status), n in sorted(
                 Counter((r["dataset_id"], r["transcript_availability"]) for r in source).items())},
             "cohorts": {}}
    analytical_groups: set[tuple[str, str]] = set()
    analytical_hashes: set[str] = set()
    for offset, (dataset, (total, holdout_n)) in enumerate(CORE_COHORTS.items()):
        if dataset in blocked_datasets:
            audit["cohorts"][dataset] = {"status": "blocked", "reason": "unresolved_source_identity_collision",
                                           "selected": 0, "development": 0, "holdout": 0,
                                           "collision_group_count": sum(row["dataset_id"] == dataset and row["resolution"].startswith("unresolved")
                                                                        for row in collision_audit)}
            continue
        candidates = [r for r in grouped if r["dataset_id"] == dataset and r["rows"][0]["source_split"] == "development"]
        selected = _take(candidates, total, seed + offset)
        holdout = {r["source_group_id"] for r in _take(selected, holdout_n, seed + 97 + offset)}
        development = [r for r in selected if r["source_group_id"] not in holdout]
        task, order = _task({r["label"] for r in selected})
        fold_map = _folds(development, seed + offset, n_splits)
        for row in selected:
            held = row["source_group_id"] in holdout
            output.append(_subject(row, "holdout" if held else "development", task, order,
                                   "not_applicable" if held else fold_map[row["source_group_id"]]).to_dict())
            analytical_groups.add((row["dataset_id"], row["source_group_id"]))
            analytical_hashes.update(row["raw_hashes"])
        audit["cohorts"][dataset] = {"status": "ready", "reason": "", "selected": total,
                                       "development": total - holdout_n, "holdout": holdout_n,
                                       "source_pool": len(candidates), "development_fold_counts": dict(sorted(Counter(fold_map.values()).items()))}
    for offset, (dataset, count) in enumerate(STRESS_COHORTS.items()):
        if dataset in blocked_datasets:
            audit["cohorts"][dataset] = {"status": "blocked", "reason": "unresolved_source_identity_collision",
                                           "selected": 0, "stress": 0,
                                           "collision_group_count": sum(row["dataset_id"] == dataset and row["resolution"].startswith("unresolved")
                                                                        for row in collision_audit)}
            continue
        candidates = [r for r in grouped if r["dataset_id"] == dataset and r["rows"][0]["source_split"] == "stress"]
        selected = _take(candidates, count, seed + 200 + offset)
        task, order = _task({r["label"] for r in selected})
        output.extend(_subject(row, "stress", task, order, "not_applicable").to_dict() for row in selected)
        audit["cohorts"][dataset] = {"status": "ready", "reason": "", "selected": count,
                                       "stress": count, "source_pool": len(candidates)}
        for row in selected:
            analytical_groups.add((row["dataset_id"], row["source_group_id"]))
            analytical_hashes.update(row["raw_hashes"])
    if "PREPARE_DrivenData" in blocked_datasets:
        audit["cohorts"]["engineering_canary"] = {"status": "blocked", "reason": "source_cohort_blocked",
                                                       "selected": 0, "source_pool": 0, "gap": 12,
                                                       "source_dataset": "PREPARE_DrivenData"}
    else:
        prepare_candidates = [row for row in grouped if row["dataset_id"] == "PREPARE_DrivenData"
                              and row["rows"][0]["source_split"] == "development"]
        prepare_unused = _canary_candidates(grouped, source, analytical_groups, analytical_hashes)
        canary_count = min(12, len(prepare_unused))
        canaries = _take(prepare_unused, canary_count, seed + 400) if canary_count else []
        if canaries:
            task, order = _task({row["label"] for row in prepare_candidates})
            output.extend(_subject(row, "engineering_canary", task, order, "not_applicable").to_dict() for row in canaries)
        audit["cohorts"]["engineering_canary"] = {"status": "ready", "reason": "", "selected": canary_count,
                                                       "source_pool": len(prepare_unused), "gap": 12 - canary_count,
                                                       "source_dataset": "PREPARE_DrivenData"}
    groups, assets = defaultdict(set), defaultdict(set)
    for row in output:
        groups[row["source_group_id"]].add(row["partition"])
        for asset in row["raw_hashes"]:
            assets[asset].add(row["partition"])
    if any(len(v) > 1 for v in groups.values()) or any(len(v) > 1 for v in assets.values()):
        raise ManifestError("An identity or duplicate recording appears in multiple partitions.")
    return sorted(output, key=lambda r: (r["dataset_id"], r["partition"], r["subject_id"])), sorted(exclusions, key=lambda r: (r["dataset_id"], r["reason"], r["source_id"])), audit


def make_group_folds(development_manifest: Sequence[Mapping[str, Any]], labels: Mapping[str, str], seed: int, n_splits: int = 5) -> list[dict[str, str]]:
    rows = [dict(r) for r in development_manifest if r["partition"] == "development"]
    private = [{"source_group_id": r["source_group_id"], "label": labels.get(r["subject_id"], "")} for r in rows]
    if not rows or any(not r["label"] for r in private) or {r["label"] for r in private} != {x for r in rows for x in r["class_order"]}:
        raise ManifestError("Missing class in an OOF fit partition.")
    assignment = _folds(private, seed, n_splits)
    result = [{"subject_id": r["subject_id"], "fold_id": assignment[r["source_group_id"]]} for r in rows]
    if any(r["fold_id"] != assignment[r["source_group_id"]] for r in rows):
        raise ManifestError("Manifest fold IDs do not match deterministic grouped folds.")
    return sorted(result, key=lambda r: r["subject_id"])


def _ncm_identity(path: Path) -> tuple[str, str, str]:
    match = _NCMMSC.match(path.stem)
    if not match:
        token = _pseudo("raw", path.stem)
        return token, token, "UNAVAILABLE"
    label, sex, number, _ = match.groups()
    return f"{label.upper()}_{sex.upper()}_{number}", f"{sex.upper()}_{number}", label.upper()


def actual_source_inventory(raw_root: str | Path, historical_root: str | Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw, history = Path(raw_root).resolve(), Path(historical_root).resolve()
    inventory, pre_exclusions = [], []
    for dataset in (*CORE_COHORTS, *STRESS_COHORTS):
        manifest_path = history / dataset / "manifest.csv"
        provenance = _sha(manifest_path)
        for row in _read_csv(manifest_path):
            path = Path(row.get("audio_path", ""))
            if not path.is_file():
                pre_exclusions.append({"dataset_id": dataset, "source_id": _pseudo("src", dataset, row.get("case_id", "")), "reason": "missing_raw_provenance"})
                continue
            old_split = row.get("split", "")
            split = ("stress" if old_split == "test" else "excluded") if dataset in STRESS_COHORTS else ("development" if old_split == "train" else "excluded")
            inventory.append({"dataset_id": dataset, "source_id": row.get("case_id", ""), "subject_id": row.get("subject_id", ""),
                              "source_group_id": row.get("source_identity_key") or row.get("subject_id", ""), "label": row.get("label", ""),
                              "language": row.get("language", "und"), "raw_path": str(path.resolve()), "transcript_path": row.get("transcript_path", ""),
                              "transcript_availability": ("available" if row.get("transcript_path") and Path(row["transcript_path"]).is_file()
                                                          else "missing_path" if row.get("transcript_path") else "not_available"),
                              "transcript_provenance": ("historical_manifest_verified" if row.get("transcript_path") and Path(row["transcript_path"]).is_file()
                                                        else "historical_manifest_missing" if row.get("transcript_path") else "dataset_declares_no_source_transcript"),
                              "task_id": row.get("task_type", "overall"), "source_split": split,
                              "source_track": "long" if dataset == "NCMMSC2021_AD" else "primary", "raw_hash": row.get("audio_sha256", ""),
                              "source_version": "historical_manifest_boundary_v1", "source_provenance_hash": provenance,
                              "source_quality_flag": "", "quality_provenance_hash": ""})
    ncm_root = raw / "NCMMSC2021_AD"
    known = {Path(r["raw_path"]).resolve() for r in inventory if r["dataset_id"] == "NCMMSC2021_AD"}
    config_hash = _sha(Path(__file__).resolve().parents[3] / "configs/datasets/NCMMSC2021_AD.yaml")
    for track, root in (("six_second", ncm_root / "AD_dataset_6s"), ("long", ncm_root / "AD_dataset_long")):
        all_files = sorted(p.resolve() for p in root.rglob("*") if p.is_file())
        files = [p for p in all_files if p.suffix.lower() in _AUDIO]
        for path in all_files:
            if path.suffix.lower() not in _AUDIO:
                pre_exclusions.append({"dataset_id": "NCMMSC2021_AD",
                    "source_id": _pseudo("src", "NCMMSC2021_AD", str(path.relative_to(ncm_root))),
                    "reason": "filesystem_metadata_ignored"})
        for path in files:
            if path in known:
                continue
            subject, group, label = _ncm_identity(path)
            split = "identity_collision" if track == "long" and "train" in path.relative_to(ncm_root).parts else "excluded"
            inventory.append({"dataset_id": "NCMMSC2021_AD", "source_id": str(path.relative_to(ncm_root)), "subject_id": subject,
                              "source_group_id": group, "label": label, "language": "zh", "raw_path": str(path), "transcript_path": "",
                              "transcript_availability": "not_available", "transcript_provenance": "dataset_declares_generated_asr_only",
                              "task_id": "six_second_challenge" if track == "six_second" else "long_picture_description", "source_split": split,
                              "source_track": track, "raw_hash": _sha(path), "source_version": "ncmmsc_raw_tree_v1", "source_provenance_hash": config_hash})
    prepare_root = raw / "DementiaBank_Challenges/PREPARE_DrivenData"
    task_path = Path(__file__).resolve().parents[3] / "references/speechcare/prepare_task_labels.csv"
    tasks, task_hash = {r["uid"]: r for r in _read_csv(task_path)}, _sha(task_path)
    for item in inventory:
        if item["dataset_id"] == "PREPARE_DrivenData":
            item["source_quality_flag"] = tasks.get(item["source_id"], {}).get("valid", "")
            item["quality_provenance_hash"] = task_hash
    known_prepare = {r["source_id"] for r in inventory if r["dataset_id"] == "PREPARE_DrivenData"}
    label_map = {"Control": "HC", "MCI": "MCI", "AD": "AD", "ProbableAD": "AD", "PPA": "AD"}
    for row in _read_csv(prepare_root / "metadata.csv"):
        uid = row["uid"]
        if row["split"].lower() != "train" or uid in known_prepare:
            continue
        audio = next((p.resolve() for p in sorted((prepare_root / "_harness_extracted/train").glob(f"{uid}.*")) if p.suffix.lower() in _AUDIO), None)
        inventory.append({"dataset_id": "PREPARE_DrivenData", "source_id": uid, "subject_id": uid, "source_group_id": uid,
                          "label": label_map[row["diagnosis"]], "language": row.get("language", "und"), "raw_path": str(audio) if audio else "",
                          "transcript_path": "", "transcript_availability": "not_available",
                          "transcript_provenance": "dataset_declares_generated_asr_only",
                          "task_id": tasks.get(uid, {}).get("task", "overall"), "source_split": "invalid", "source_track": "primary",
                          "raw_hash": _sha(audio) if audio else hashlib.sha256(uid.encode()).hexdigest(), "source_version": PREPARE_TASK_SOURCE_VERSION,
                          "source_provenance_hash": task_hash, "source_quality_flag": tasks.get(uid, {}).get("valid", ""),
                          "quality_provenance_hash": task_hash})
    return sorted(inventory, key=_sort_source), sorted(pre_exclusions, key=lambda r: (r["dataset_id"], r["reason"], r["source_id"]))


def make_label_rows(manifest: Sequence[Mapping[str, Any]], source: Sequence[Mapping[str, Any]]) -> tuple[LabelRow, ...]:
    truth: dict[str, set[str]] = defaultdict(set)
    for row in source:
        truth[_pseudo("sub", str(row["dataset_id"]), str(row["subject_id"]))].add(str(row["label"]))
    output = []
    for row in sorted(manifest, key=lambda r: (r["dataset_id"], r["subject_id"])):
        labels = truth[row["subject_id"]]
        if len(labels) != 1:
            raise ManifestError("Selected subject does not have exactly one private label.")
        label, subject = next(iter(labels)), SubjectRow.from_mapping(row)
        output.append(LabelRow(subject=subject, source_label=label, label=label, label_mapping={label: label}, source=f"{subject.dataset_id}_frozen_source_v1"))
    return tuple(output)


def load_development_labels(path: str | Path) -> tuple[LabelRow, ...]:
    rows = read_labels_jsonl(Path(path).read_bytes().splitlines(), purpose="training")
    if any(row.subject.partition != "development" for row in rows):
        raise PilotContractError("Training label capability is restricted to development rows.")
    return rows


def load_sealed_scoring_labels(path: str | Path) -> tuple[LabelRow, ...]:
    rows = read_labels_jsonl(Path(path).read_bytes().splitlines(), purpose="scoring")
    if any(row.subject.partition not in {"holdout", "stress"} for row in rows):
        raise PilotContractError("Scoring label capability cannot load development truth.")
    return rows


def load_canary_labels(path: str | Path) -> tuple[LabelRow, ...]:
    rows = read_labels_jsonl(Path(path).read_bytes().splitlines(), purpose="training")
    if any(row.subject.partition != "engineering_canary" for row in rows):
        raise PilotContractError("Canary label capability is restricted to engineering canaries.")
    return rows


def load_cohort_status(path: str | Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(Path(path))
    required = {"cohort_id", "status", "reason", "selected"}
    if any(required - set(row) for row in rows):
        raise ManifestError("Cohort status artifact is missing required fields.")
    if len({row["cohort_id"] for row in rows}) != len(rows):
        raise ManifestError("Cohort status artifact contains duplicate cohort IDs.")
    return {row["cohort_id"]: row for row in rows}


def require_cohort_ready(path: str | Path, cohort_id: str) -> dict[str, str]:
    status = load_cohort_status(path).get(cohort_id)
    if status is None:
        raise ManifestError(f"Cohort status is unavailable for {cohort_id}.")
    if status["status"] != "ready":
        raise ManifestError(f"Cohort {cohort_id} is blocked: {status['reason']}.")
    return status


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def _write_labels(path: Path, rows: Sequence[LabelRow], purpose: str) -> None:
    ordered = sorted(rows, key=lambda r: (r.subject.dataset_id, r.subject.partition, r.subject.subject_id, r.source))
    body = serialize_labels_jsonl(ordered, purpose=purpose)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")
    path.chmod(0o600)


def write_pilot_artifacts(output_dir: str | Path, source_inventory: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]],
                          exclusions: Sequence[Mapping[str, str]], audit: Mapping[str, Any], labels: Sequence[LabelRow]) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "labels.csv").exists():
        (output / "labels.csv").unlink()
    source_path = output / "source_index.jsonl"
    source_path.write_text("".join(_canonical(dict(r)).decode() + "\n" for r in sorted(source_inventory, key=_sort_source)), encoding="utf-8")
    source_path.chmod(0o600)
    columns = ("schema_version", "dataset_id", "subject_id", "partition", "task", "class_order", "source_group_id", "channel", "language", "task_ids", "role", "fold_id", "raw_hashes", "source_version")
    ordered = sorted(manifest, key=lambda r: (r["dataset_id"], r["partition"], r["subject_id"]))
    public = []
    for row in ordered:
        safe_inference_payload(row)
        public.append({key: json.dumps(row[key], sort_keys=True, separators=(",", ":")) if isinstance(row[key], (dict, list, tuple)) else row[key] for key in columns})
    _write_csv(output / "split_manifest.csv", public, columns)
    fold_rows = [{"dataset_id": r["dataset_id"], "subject_id": r["subject_id"], "fold_id": r["fold_id"]} for r in ordered if r["partition"] == "development"]
    _write_csv(output / "folds.csv", fold_rows, ("dataset_id", "subject_id", "fold_id"))
    _write_labels(output / "development_labels.jsonl", [r for r in labels if r.subject.partition == "development"], "training")
    _write_labels(output / "sealed_holdout_labels.jsonl", [r for r in labels if r.subject.partition == "holdout"], "scoring")
    _write_labels(output / "sealed_stress_labels.jsonl", [r for r in labels if r.subject.partition == "stress"], "scoring")
    _write_labels(output / "canary_labels.jsonl", [r for r in labels if r.subject.partition == "engineering_canary"], "training")
    cohort_rows = []
    for cohort_id in (*CORE_COHORTS, *STRESS_COHORTS, "engineering_canary"):
        cohort = audit.get("cohorts", {}).get(cohort_id, {})
        cohort_rows.append({"cohort_id": cohort_id, "status": cohort.get("status", "blocked"),
                            "reason": cohort.get("reason", "missing_cohort_audit"),
                            "selected": cohort.get("selected", 0)})
    _write_csv(output / "cohort_status.csv", cohort_rows, ("cohort_id", "status", "reason", "selected"))
    _write_csv(output / "exclusions.csv", sorted(exclusions, key=lambda r: (r["dataset_id"], r["reason"], r["source_id"])), ("dataset_id", "source_id", "reason"))
    (output / "identity_audit.json").write_bytes(_canonical(dict(audit)))
    cache = {"global_references": "ineligible", "historical_in_sample_predictions": "ineligible", "old_state_normalization": "ineligible",
             "raw_deterministic_metrics": "eligible_with_verified_raw_hash", "unadapted_encoder_outputs": "eligible_with_verified_source_provenance"}
    (output / "cache_provenance.json").write_bytes(_canonical(cache))
    names = ("source_index.jsonl", "split_manifest.csv", "folds.csv", "cohort_status.csv", "development_labels.jsonl", "sealed_holdout_labels.jsonl",
             "sealed_stress_labels.jsonl", "canary_labels.jsonl", "exclusions.csv", "identity_audit.json", "cache_provenance.json")
    (output / "sha256_manifest.json").write_bytes(_canonical({"algorithm": "sha256", "files": {name: _sha(output / name) for name in sorted(names)}}))
    return {name: output / name for name in (*names, "sha256_manifest.json")}


def freeze_actual_pilot_data(*, raw_root: str | Path, historical_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    source, initial = actual_source_inventory(raw_root, historical_root)
    config: dict[str, Any] = {"seed": SEED, "prepare_invalid_expected_count": 33}
    manifest, exclusions, audit = build_pilot_manifest(config, source)
    labels = make_label_rows(manifest, source)
    combined_exclusions = [*exclusions, *initial]
    audit["partition_counts"] = dict(sorted(Counter(r["partition"] for r in manifest).items()))
    audit["initial_missing_provenance_exclusions"] = len(initial)
    audit["excluded_records"] = len(combined_exclusions)
    audit["exclusion_reason_counts"] = dict(sorted(Counter(row["reason"] for row in combined_exclusions).items()))
    return write_pilot_artifacts(output_dir, source, manifest, combined_exclusions, audit, labels)


__all__ = ["ManifestError", "actual_source_inventory", "build_pilot_manifest", "freeze_actual_pilot_data",
           "load_canary_labels", "load_cohort_status", "load_development_labels", "load_sealed_scoring_labels",
           "make_group_folds", "make_label_rows", "require_cohort_ready", "write_pilot_artifacts"]
