from __future__ import annotations

import pandas as pd
import pytest

from advoice.conditional_authority_pilot import (
    FrozenAuthorityArtifactError,
    load_frozen_authority_dataset,
)


def _write_artifact(tmp_path, *, reliability: object = "0.75", duplicate_state: bool = False) -> None:
    manifest = pd.DataFrame([
        {"subject_id": "s-train", "label": "HC", "split": "train", "task_type": "Cookie", "language": "en", "channel": "speech", "case_id": "case-train"},
        {"subject_id": "s-test", "label": "AD", "split": "test", "task_type": "Cookie", "language": "zh", "channel": "speech", "case_id": "case-test"},
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
    assert evidence.evidence_id == "metric:s-test:pause_rate:S01:cookie"
    assert evidence.state_id == "S01"
    assert evidence.task_id == "cookie"
    assert evidence.permissions.inference is True
    assert evidence.permissions.report is True
    assert evidence.provenance.source_segment_ids == ("seg-test",)
    assert evidence.provenance.source_asset_id == "asset-test"
    assert evidence.reliability_components.source == pytest.approx(0.75)
    assert dataset.subject_metadata["s-test"]["language"] == ("zh",)
    assert dataset.subject_metadata["s-test"]["channel"] == ("speech",)


def test_false_csv_strings_are_not_truthy(tmp_path) -> None:
    _write_artifact(tmp_path)

    dataset = load_frozen_authority_dataset(tmp_path)

    evidence = dataset.evidence_for_subject("s-train")[0]
    assert evidence.permissions.inference is False
    assert evidence.permissions.report is False
    assert evidence.observable is True


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
