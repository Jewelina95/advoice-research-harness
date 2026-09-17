from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from advoice.module_a import EXPLANATION_PACKET_VERSION, TaskConditionedStatisticalExpert


@pytest.fixture
def training_frame() -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame(
        {
            "state_S01": [-2.0, -1.5, -0.2, 0.1, 1.4, 1.8],
            "state_S04": [-1.2, -0.9, 0.0, 0.4, 1.0, 1.5],
            "qc_signal": [0.1, 0.8, 0.2, 0.9, 0.3, 0.7],
            "task": ["picture", "picture", "picture", "fluency", "fluency", "fluency"],
        }
    )
    return frame, ["HC", "HC", "MCI", "MCI", "AD", "AD"]


def _expert(frame: pd.DataFrame, y: list[str]) -> TaskConditionedStatisticalExpert:
    return TaskConditionedStatisticalExpert(
        ["MCI", "HC", "AD"], c=0.5, task_column="task"
    ).fit(
        frame,
        y,
        feature_columns=["state_S01", "state_S04", "qc_signal"],
        feature_branches={"state_S01": "timing", "state_S04": "language", "qc_signal": "qc"},
        artifact_snapshot={"fit_manifest": "fold-1"},
    )


def test_module_a_contributions_reconstruct_softmax_logits(training_frame) -> None:
    frame, y = training_frame
    packet = _expert(frame, y).explain_case(
        frame.iloc[0], consumed_evidence_ids=["m2", "m1"], evidence_snapshot={"revision": 2},
        state_snapshot={"S01": -2.0}, fold_disagreement={"std": 0.12},
    )

    assert packet.schema_version == EXPLANATION_PACKET_VERSION
    for label in packet.class_order:
        reconstructed = packet.intercepts[label] + sum(packet.feature_contributions[label].values())
        assert reconstructed == pytest.approx(packet.logits[label])
        assert sum(packet.branch_contributions[label].values()) == pytest.approx(
            sum(packet.feature_contributions[label].values())
        )
    assert packet.consumed_evidence_ids == ("m1", "m2")
    assert packet.fold_disagreement == {"std": 0.12}


def test_module_a_class_order_and_json_are_fixed(training_frame) -> None:
    frame, y = training_frame
    expert = _expert(frame, y)
    first = expert.predict_packet(frame.iloc[2], evidence_snapshot={"e": 1}, state_snapshot={"s": 1})
    second = expert.predict_packet(frame.iloc[2], evidence_snapshot={"e": 1}, state_snapshot={"s": 1})

    assert first.class_order == ("MCI", "HC", "AD")
    assert list(first.raw_probabilities) == ["MCI", "HC", "AD"]
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json())["class_order"] == ["MCI", "HC", "AD"]
    np.testing.assert_allclose(expert.predict_proba(frame.iloc[:2]).sum(axis=1), 1.0)


def test_module_a_snapshot_hashes_change_with_evidence_and_state(training_frame) -> None:
    frame, y = training_frame
    expert = _expert(frame, y)
    baseline = expert.explain_case(frame.iloc[1], evidence_snapshot={"revision": 1}, state_snapshot={"S01": 0.2})
    changed_evidence = expert.explain_case(frame.iloc[1], evidence_snapshot={"revision": 2}, state_snapshot={"S01": 0.2})
    changed_state = expert.explain_case(frame.iloc[1], evidence_snapshot={"revision": 1}, state_snapshot={"S01": 0.3})

    assert baseline.hashes["artifact_hash"] == changed_evidence.hashes["artifact_hash"]
    assert baseline.hashes["evidence_snapshot_hash"] != changed_evidence.hashes["evidence_snapshot_hash"]
    assert baseline.hashes["state_snapshot_hash"] != changed_state.hashes["state_snapshot_hash"]


def test_module_a_excludes_qc_from_disease_contributions_and_marks_missing_calibration(training_frame) -> None:
    frame, y = training_frame
    packet = _expert(frame, y).explain_case(frame.iloc[4])

    assert packet.calibration_status == "not_calibrated"
    assert packet.calibrated_probabilities is None
    assert "qc_signal" not in {feature for values in packet.feature_contributions.values() for feature in values}
    assert "qc_signal" not in {feature for values in packet.branch_contributions.values() for feature in values}

