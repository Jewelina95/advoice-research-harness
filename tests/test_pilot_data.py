from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
import advoice.pilot.data as pilot_data
from advoice.pilot.contracts import PilotContractError
from advoice.pilot.data import (ManifestError, PREPARE_827_MANIFEST_SHA256, PREPARE_TASK_SHA256,
    PREPARE_TASK_SOURCE_VERSION,
    build_pilot_manifest, load_canary_labels, load_development_labels, load_sealed_scoring_labels,
    make_group_folds, make_label_rows, write_pilot_artifacts)


def _row(dataset, subject, label, *, split="development", track="primary", raw_hash=None):
    return {"dataset_id": dataset, "source_id": f"case-{subject}", "subject_id": subject,
        "source_group_id": f"group-{subject}", "label": label, "language": "en",
        "raw_path": f"/private/{dataset}/{subject}.wav", "task_id": "task",
        "transcript_path": "", "transcript_availability": "not_available",
        "transcript_provenance": "fixture_no_transcript",
        "source_split": split, "source_track": track,
        "raw_hash": raw_hash or hashlib.sha256(f"{dataset}:{subject}".encode()).hexdigest(),
        "source_version": "fixture_v1", "source_provenance_hash": "f" * 64}


def _inventory():
    rows = []
    for dataset, count, labels in (("PREPARE_DrivenData", 162, ("HC", "MCI", "AD")),
            ("ADReSS_2020", 81, ("HC", "AD")), ("NCMMSC2021_AD", 120, ("HC", "MCI", "AD"))):
        rows.extend(_row(dataset, f"{dataset}-{i}", labels[i % len(labels)]) for i in range(count))
    rows.extend(_row("IAEAV", f"iae-{i}", ("HC", "AD")[i % 2], split="stress") for i in range(14))
    rows.extend(_row("DementiaNet_PublicFigures", f"dn-{i}", ("HC", "AD")[i % 2], split="stress") for i in range(6))
    return rows


def _config(**changes):
    return {"seed": 20260923, "prepare_invalid_expected_count": 0, **changes}


def test_manifest_deterministic_safe_and_folds_are_integrated():
    left = build_pilot_manifest(_config(), _inventory())
    right = build_pilot_manifest(_config(), reversed(_inventory()))
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    manifest = left[0]
    assert "/private/" not in json.dumps(manifest) and '"label"' not in json.dumps(manifest)
    assert all(r["fold_id"].startswith("fold_") for r in manifest if r["partition"] == "development")
    assert all(r["fold_id"] == "not_applicable" for r in manifest if r["partition"] != "development")
    canaries = [r for r in manifest if r["partition"] == "engineering_canary"]
    assert len(canaries) == 12
    analytical = [r for r in manifest if r["partition"] != "engineering_canary"]
    assert not ({r["source_group_id"] for r in canaries} & {r["source_group_id"] for r in analytical})
    assert not ({asset for r in canaries for asset in r["raw_hashes"]} & {asset for r in analytical for asset in r["raw_hashes"]})


def test_canaries_exclude_reserved_official_test_groups_and_hashes():
    rows = _inventory()
    first_manifest = build_pilot_manifest(_config(), rows)[0]
    reserved_hash = next(iter(next(r for r in first_manifest if r["partition"] == "engineering_canary")["raw_hashes"].values()))
    rows.append(_row("PREPARE_DrivenData", "reserved-test", "HC", split="excluded", raw_hash=reserved_hash))

    manifest, _, audit = build_pilot_manifest(_config(), rows)

    canaries = [row for row in manifest if row["partition"] == "engineering_canary"]
    assert len(canaries) == 11
    assert reserved_hash not in {digest for row in canaries for digest in row["raw_hashes"].values()}
    assert audit["cohorts"]["engineering_canary"]["gap"] == 1


def test_every_six_second_source_is_audited_and_excluded():
    rows = _inventory() + [_row("NCMMSC2021_AD", f"six-{i}", "HC", split="excluded", track="six_second") for i in range(7)]
    manifest, exclusions, audit = build_pilot_manifest(_config(), rows)
    assert audit["source_records_by_dataset_track"]["NCMMSC2021_AD:six_second"] == 7
    assert audit["exclusion_reason_counts"]["six_second_track"] == 7
    assert sum(r["reason"] == "six_second_track" for r in exclusions) == 7
    assert len([r for r in manifest if r["dataset_id"] == "NCMMSC2021_AD"]) == 120


def test_prepare_mismatch_requires_exact_pinned_source_provenance():
    rows = _inventory() + [_row("PREPARE_DrivenData", f"invalid-{i}", "HC", split="invalid") for i in range(24)]
    for row in rows[-24:]: row["source_version"] = PREPARE_TASK_SOURCE_VERSION
    with pytest.raises(ManifestError, match="pinned source hashes/counts do not match"):
        build_pilot_manifest({}, rows)
    prepare = [row for row in rows if row["dataset_id"] == "PREPARE_DrivenData"]
    for row in prepare:
        row["source_provenance_hash"] = PREPARE_827_MANIFEST_SHA256
        row["quality_provenance_hash"] = PREPARE_TASK_SHA256
        row["source_quality_flag"] = "Yes"
    for row in prepare[:4] + rows[-24:]:
        row["source_quality_flag"] = "No"
    _, exclusions, audit = build_pilot_manifest({}, rows)
    provenance = audit["prepare_invalid_uid_provenance"]
    assert provenance["resolution"] == "resolved_by_source_version"
    assert provenance["source_no_flag_count"] == 28
    assert provenance["included_no_flag_count"] == 4
    assert sum(r["reason"] == "prepare_invalid_uid" for r in exclusions) == 24


def test_duplicate_recording_and_missing_class_fail_before_training():
    rows = _inventory(); rows[0]["raw_hash"] = rows[1]["raw_hash"]
    with pytest.raises(ManifestError, match="duplicate recording"):
        build_pilot_manifest(_config(), rows)
    manifest = build_pilot_manifest(_config(), _inventory())[0]
    dev = [r for r in manifest if r["dataset_id"] == "ADReSS_2020" and r["partition"] == "development"]
    with pytest.raises(ManifestError, match="Missing class"):
        make_group_folds(dev, {r["subject_id"]: "HC" for r in dev}, 1)


def test_unresolved_identity_collision_blocks_affected_cohort():
    rows = _inventory()
    rows.extend([_row("NCMMSC2021_AD", "collision-ad", "AD", split="identity_collision"),
                 _row("NCMMSC2021_AD", "collision-mci", "MCI", split="identity_collision")])
    rows[-1]["source_group_id"] = rows[-2]["source_group_id"] = "same-source-identity"
    with pytest.raises(ManifestError, match="collisions block"):
        build_pilot_manifest(_config(), rows)


def test_blocked_inventory_removes_stale_freeze_and_hashes_audit(tmp_path: Path):
    rows = _inventory()
    rows.extend([_row("NCMMSC2021_AD", "collision-ad", "AD", split="identity_collision"),
                 _row("NCMMSC2021_AD", "collision-mci", "MCI", split="identity_collision")])
    rows[-1]["source_group_id"] = rows[-2]["source_group_id"] = "same-source-identity"
    (tmp_path / "split_manifest.csv").write_text("stale\n")
    initial = [{"dataset_id": "NCMMSC2021_AD", "source_id": "metadata", "reason": "filesystem_metadata_ignored"}]

    pilot_data._write_blocked_inventory(tmp_path, rows, initial, _config())

    assert not (tmp_path / "split_manifest.csv").exists()
    audit = json.loads((tmp_path / "identity_audit.json").read_text())
    assert audit["status"] == "blocked"
    assert audit["collision_groups"][0]["distinct_label_count"] == 2
    assert audit["collision_groups"][0]["distinct_hash_count"] == 2
    assert audit["exclusion_reason_counts"]["filesystem_metadata_ignored"] == 1
    assert audit["engineering_canary_reservation"]["status"] == "not_frozen_due_to_block"
    hashes = json.loads((tmp_path / "sha256_manifest.json").read_text())["files"]
    assert set(hashes) == {"source_index.jsonl", "exclusions.csv", "identity_audit.json", "cache_provenance.json"}
    assert all(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())


def test_typed_labels_are_partitioned_and_capability_checked(tmp_path: Path):
    source = _inventory(); manifest, exclusions, audit = build_pilot_manifest(_config(), source)
    paths = write_pilot_artifacts(tmp_path, source, manifest, exclusions, audit, make_label_rows(manifest, source))
    assert not (tmp_path / "labels.csv").exists()
    assert len(load_development_labels(paths["development_labels.jsonl"])) == 281
    assert len(load_sealed_scoring_labels(paths["sealed_holdout_labels.jsonl"])) == 70
    assert len(load_sealed_scoring_labels(paths["sealed_stress_labels.jsonl"])) == 20
    assert len(load_canary_labels(paths["canary_labels.jsonl"])) == 12
    with pytest.raises(PilotContractError): load_development_labels(paths["sealed_holdout_labels.jsonl"])
    with pytest.raises(PilotContractError): load_sealed_scoring_labels(paths["development_labels.jsonl"])
    assert audit["transcript_availability_counts"]["PREPARE_DrivenData:not_available"] == 162


def test_artifacts_are_canonical_and_all_payloads_are_hashed(tmp_path: Path):
    source = _inventory(); manifest, exclusions, audit = build_pilot_manifest(_config(), source)
    labels = make_label_rows(manifest, source)
    first = write_pilot_artifacts(tmp_path / "a", source, manifest, exclusions, audit, labels)
    second = write_pilot_artifacts(tmp_path / "b", list(reversed(source)), list(reversed(manifest)), list(reversed(exclusions)), audit, list(reversed(labels)))
    hashes = json.loads(first["sha256_manifest.json"].read_text())["files"]
    assert set(hashes) == set(first) - {"sha256_manifest.json"}
    for name, digest in hashes.items():
        assert hashlib.sha256(first[name].read_bytes()).hexdigest() == digest
        assert first[name].read_bytes() == second[name].read_bytes()
    safe = first["split_manifest.csv"].read_text() + first["folds.csv"].read_text()
    assert "/private/" not in safe and "label" not in safe and "unassigned" not in safe
