from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from advoice.pilot.data import ManifestError, build_pilot_manifest, make_group_folds, write_pilot_artifacts


def _row(dataset: str, subject: str, label: str, *, split: str = "development", track: str = "primary", language: str = "en", raw_hash: str | None = None):
    digest = hashlib.sha256(f"{dataset}:{subject}".encode()).hexdigest()
    return {"dataset_id": dataset, "source_id": f"case-{subject}", "subject_id": subject, "source_group_id": f"group-{subject}", "label": label, "language": language, "raw_path": f"/private/{dataset}/{subject}.wav", "task_id": "task", "source_split": split, "source_track": track, "raw_hash": raw_hash or digest, "source_version": "fixture_v1"}


def _inventory():
    rows = []
    for dataset, count, labels in (("PREPARE_DrivenData", 150, ("HC", "MCI", "AD")), ("ADReSS_2020", 81, ("HC", "AD")), ("NCMMSC2021_AD", 120, ("HC", "MCI", "AD"))):
        rows += [_row(dataset, f"{dataset}-{index}", labels[index % len(labels)], language="en" if index % 4 else "es") for index in range(count)]
    rows += [_row("IAEAV", f"iae-{index}", ("HC", "AD")[index % 2], split="stress", language="es") for index in range(14)]
    rows += [_row("DementiaNet_PublicFigures", f"dn-{index}", ("HC", "AD")[index % 2], split="stress") for index in range(6)]
    return rows


def test_manifest_is_deterministic_and_provider_safe():
    left = build_pilot_manifest({"seed": 20260923}, _inventory())
    right = build_pilot_manifest({"seed": 20260923}, list(reversed(_inventory())))
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    manifest, _, audit = left
    body = json.dumps(manifest)
    assert "/private/" not in body and '"label"' not in body
    assert audit["cohorts"]["PREPARE_DrivenData"]["development"] == 120
    assert audit["cohorts"]["NCMMSC2021_AD"]["holdout"] == 24


def test_duplicate_hash_across_partitions_is_rejected():
    rows = _inventory()
    rows[0]["raw_hash"] = rows[1]["raw_hash"]
    with pytest.raises(ManifestError, match="duplicate raw hash"):
        build_pilot_manifest({"seed": 20260923}, rows)


def test_six_second_track_is_excluded_without_touching_long_track():
    rows = _inventory() + [_row("NCMMSC2021_AD", "short", "HC", track="six_second")]
    manifest, exclusions, _ = build_pilot_manifest({}, rows)
    assert any(row["reason"] == "six_second_track" for row in exclusions)
    assert len([row for row in manifest if row["dataset_id"] == "NCMMSC2021_AD"]) == 120


def test_source_cohort_mismatch_is_explicit():
    rows = _inventory()
    rows = [row for row in rows if row["dataset_id"] != "ADReSS_2020"]
    with pytest.raises(ManifestError, match="Requested 81 identities"):
        build_pilot_manifest({}, rows)


def test_group_folds_fail_before_training_when_class_is_absent():
    manifest, _, _ = build_pilot_manifest({}, _inventory())
    development = [row for row in manifest if row["dataset_id"] == "ADReSS_2020" and row["partition"] == "development"]
    labels = {row["subject_id"]: "HC" for row in development}
    with pytest.raises(ManifestError, match="Missing class"):
        make_group_folds(development, labels, 1)


def test_artifacts_keep_paths_and_labels_out_of_provider_safe_files(tmp_path: Path):
    source = _inventory()
    manifest, exclusions, audit = build_pilot_manifest({}, source)
    labels = {row["subject_id"]: "HC" for row in manifest if row["partition"] == "development"}
    # Supply valid class labels only to the ADReSS fold fixture.
    development = [row for row in manifest if row["dataset_id"] == "ADReSS_2020" and row["partition"] == "development"]
    labels.update({row["subject_id"]: ("HC" if index % 2 else "AD") for index, row in enumerate(development)})
    folds = make_group_folds(development, labels, 7)
    paths = write_pilot_artifacts(tmp_path, source, manifest, exclusions, audit, folds, labels)
    safe_text = paths["split_manifest.csv"].read_text() + paths["folds.csv"].read_text()
    assert "/private/" not in safe_text and "label" not in safe_text
    assert "/private/" in paths["source_index.jsonl"].read_text()
