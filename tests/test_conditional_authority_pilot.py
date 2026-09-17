from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from advoice.conditional_authority_pilot import (
    FrozenAuthorityArtifactError,
    load_frozen_authority_dataset,
)


def _write_artifact(tmp_path, *, reliability: object = "0.75", duplicate_state: bool = False) -> None:
    manifest = pd.DataFrame([
        {"dataset_id": "fixture", "subject_id": "s-train", "label": "HC", "split": "train", "task_type": "Cookie", "language": "en", "channel": "speech", "case_id": "case-train"},
        {"dataset_id": "fixture", "subject_id": "s-test", "label": "AD", "split": "test", "task_type": "Cookie", "language": "zh", "channel": "speech", "case_id": "case-test"},
    ])
    rows = [
        {"subject_id": "s-train", "label": "HC", "split": "train", "metric_id": "pause_rate", "metric_instance_id": "pause_rate", "state_id": "S01", "task_scope": "cookie", "value": "1.5", "direction": "1", "reference_median": "1.0", "reference_scale": "0.5", "reliability": reliability, "missing": "False", "confound_tags": '["noise"]', "report_permission": "False", "inference_permission": "False"},
        {"subject_id": "s-test", "label": "AD", "split": "test", "metric_id": "pause_rate", "metric_instance_id": "pause_rate", "state_id": "S01", "task_scope": "cookie", "value": "2.5", "direction": "1", "reference_median": "1.0", "reference_scale": "0.5", "reliability": reliability, "missing": "false", "confound_tags": '[]', "report_permission": "true", "inference_permission": "true"},
    ]
    state_wide = pd.DataFrame([
        {"subject_id": "s-train", "label": "HC", "split": "train", "state_S01": "0.2", "rel_S01": "0.75"},
        {"subject_id": "s-test", "label": "AD", "split": "test", "state_S01": "1.5", "rel_S01": "0.75"},
    ])
    if duplicate_state:
        state_wide = pd.concat([state_wide, state_wide.iloc[[0]]], ignore_index=True)
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    pd.DataFrame(rows).to_csv(tmp_path / "metric_evidence.csv", index=False)
    pd.DataFrame([
        {"subject_id": "s-train", "task_type": "Cookie", "segment_id": "seg-train", "audio_asset_id": "asset-train"},
        {"subject_id": "s-test", "task_type": "Cookie", "segment_id": "seg-test", "audio_asset_id": "asset-test"},
    ]).to_csv(tmp_path / "segments.csv", index=False)
    state_wide.to_csv(tmp_path / "state_wide.csv", index=False)


def test_valid_legacy_artifact_converts_to_typed_evidence(tmp_path) -> None:
    _write_artifact(tmp_path)

    dataset = load_frozen_authority_dataset(tmp_path)

    evidence = dataset.evidence_for_subject("s-test")[0]
    assert evidence.evidence_id == (
        "metric:dataset=fixture:session=case-test:case=s-test:subject=s-test:"
        "task=cookie:metric=pause_rate:state=S01"
    )
    assert evidence.state_id == "S01"
    assert evidence.task_id == "cookie"
    assert evidence.case_id == "s-test"
    assert evidence.session_id == "case-test"
    assert evidence.consumed_by_supervised is True
    assert evidence.permissions.inference is True
    assert evidence.permissions.report is True
    assert evidence.provenance.source_segment_ids == ("seg-test",)
    assert evidence.provenance.source_asset_id == "asset-test"
    assert evidence.reliability_components.source == pytest.approx(0.75)
    assert evidence.reference.reference_label == "unknown"
    assert evidence.reference.sample_size == 0
    assert evidence.reference.artifact_hash == ""
    assert dataset.subject_metadata["s-test"]["language"] == ("zh",)
    assert dataset.subject_metadata["s-test"]["channel"] == ("speech",)


def test_false_csv_strings_are_not_truthy(tmp_path) -> None:
    _write_artifact(tmp_path)

    dataset = load_frozen_authority_dataset(tmp_path)

    evidence = dataset.evidence_for_subject("s-train")[0]
    assert evidence.permissions.inference is False
    assert evidence.permissions.report is False
    assert evidence.observable is True


def test_evidence_ids_are_invariant_to_legacy_csv_row_order(tmp_path) -> None:
    _write_artifact(tmp_path)
    before = load_frozen_authority_dataset(tmp_path)
    expected = sorted(item.evidence_id for item in before.evidence)

    evidence = pd.read_csv(tmp_path / "metric_evidence.csv", dtype=str, keep_default_na=False)
    evidence.sample(frac=1.0, random_state=17).to_csv(tmp_path / "metric_evidence.csv", index=False)
    after = load_frozen_authority_dataset(tmp_path)

    assert sorted(item.evidence_id for item in after.evidence) == expected


def test_missing_reliability_fails_closed(tmp_path) -> None:
    _write_artifact(tmp_path)
    evidence = pd.read_csv(tmp_path / "metric_evidence.csv")
    evidence = evidence.drop(columns=["reliability"])
    evidence.to_csv(tmp_path / "metric_evidence.csv", index=False)

    with pytest.raises(FrozenAuthorityArtifactError, match="reliability"):
        load_frozen_authority_dataset(tmp_path)


def test_duplicate_state_subject_ids_are_rejected(tmp_path) -> None:
    _write_artifact(tmp_path, duplicate_state=True)

    with pytest.raises(FrozenAuthorityArtifactError, match="duplicate subject IDs"):
        load_frozen_authority_dataset(tmp_path)


def test_test_label_feature_is_rejected(tmp_path) -> None:
    _write_artifact(tmp_path)
    state_wide = pd.read_csv(tmp_path / "state_wide.csv")
    state_wide["test_label"] = state_wide["label"]
    state_wide.to_csv(tmp_path / "state_wide.csv", index=False)

    with pytest.raises(FrozenAuthorityArtifactError, match="label leakage"):
        load_frozen_authority_dataset(tmp_path)


def test_missing_segment_or_asset_provenance_removes_report_permission(tmp_path) -> None:
    _write_artifact(tmp_path)
    pd.DataFrame(columns=["case_id", "segment_id"]).to_csv(tmp_path / "segments.csv", index=False)

    dataset = load_frozen_authority_dataset(tmp_path)

    evidence = dataset.evidence_for_subject("s-test")[0]
    assert evidence.permissions.inference is True
    assert evidence.permissions.report is False
    assert evidence.provenance.source_segment_ids == ()
    assert evidence.provenance.source_asset_id == ""


def test_recording_segments_are_subject_bound_but_retain_original_case(tmp_path) -> None:
    _write_artifact(tmp_path)
    manifest = pd.read_csv(tmp_path / "manifest.csv", dtype=str, keep_default_na=False)
    extra = manifest.iloc[[1]].copy()
    extra["case_id"] = "case-test-2"
    extra["task_type"] = "Recall"
    manifest = pd.concat([manifest, extra], ignore_index=True)
    manifest.to_csv(tmp_path / "manifest.csv", index=False)

    segments = pd.read_csv(tmp_path / "segments.csv", dtype=str, keep_default_na=False)
    extra_segment = segments.iloc[[1]].copy()
    extra_segment["case_id"] = "case-test-2"
    extra_segment["segment_id"] = "seg-test-2"
    extra_segment["audio_asset_id"] = "asset-test-2"
    segments = pd.concat([segments, extra_segment], ignore_index=True)
    segments.to_csv(tmp_path / "segments.csv", index=False)

    evidence = pd.read_csv(tmp_path / "metric_evidence.csv", dtype=str, keep_default_na=False)
    evidence.loc[evidence["subject_id"] == "s-test", "task_scope"] = "overall"
    evidence.to_csv(tmp_path / "metric_evidence.csv", index=False)

    dataset = load_frozen_authority_dataset(tmp_path)
    normalized = dataset.segments
    test_segments = normalized[normalized["subject_id"] == "s-test"]
    assert set(test_segments["case_id"]) == {"s-test"}
    assert set(test_segments["original_recording_case_id"]) == {"case-test", "case-test-2"}
    test_evidence = dataset.evidence_for_subject("s-test")[0]
    assert test_evidence.session_id == "recording-set:case-test|case-test-2"
    assert test_evidence.case_id == "s-test"
    assert test_evidence.permissions.report is False


def test_explicit_false_consumption_is_rejected_for_state_driving_evidence(tmp_path) -> None:
    _write_artifact(tmp_path)
    evidence = pd.read_csv(tmp_path / "metric_evidence.csv", dtype=str, keep_default_na=False)
    evidence["evidence_role"] = "clinical_support"
    evidence["consumed_by_supervised"] = "False"
    evidence.to_csv(tmp_path / "metric_evidence.csv", index=False)

    with pytest.raises(FrozenAuthorityArtifactError, match="consumed_by_supervised=true"):
        load_frozen_authority_dataset(tmp_path)


@pytest.mark.parametrize("dataset_name", ["ADReSS_2020", "IAEAV", "PROCESS_2"])
def test_real_frozen_dataset_loader_regression(dataset_name: str) -> None:
    root = os.environ.get("ADVOICE_ARTIFACT_ROOT")
    if not root:
        pytest.skip("Set ADVOICE_ARTIFACT_ROOT to run real frozen-artifact regression tests.")
    artifact_dir = Path(root) / dataset_name
    if not artifact_dir.is_dir():
        pytest.skip(f"Frozen artifact not available: {artifact_dir}")

    dataset = load_frozen_authority_dataset(artifact_dir)

    assert dataset.subject_labels
    assert dataset.subject_labels.keys() == dataset.subject_splits.keys()
    assert dataset.evidence
    assert all("dataset=" in item.evidence_id for item in dataset.evidence)
    assert all("session=" in item.evidence_id for item in dataset.evidence)
    assert all("case=" in item.evidence_id for item in dataset.evidence)
    assert all(item.case_id == item.subject_id for item in dataset.evidence)
    assert "original_recording_case_id" in dataset.segments.columns
