from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from advoice.condition_c_advisor import FrozenConditionCAdvisor
from advoice.module_a import ExplanationPacket, snapshot_hash
from advoice.utils import sha256_file


LABELS = ("HC", "MCI", "AD")


def _artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset_id": "fixture",
                "subject_id": "s1",
                "label": "HC",
                "split": "test",
                "prob_HC": 0.7,
                "prob_MCI": 0.2,
                "prob_AD": 0.1,
                "predicted_label": "HC",
                "condition": "Ours",
            },
            {
                "dataset_id": "fixture",
                "subject_id": "s2",
                "label": "AD",
                "split": "test",
                "prob_HC": 0.1,
                "prob_MCI": 0.2,
                "prob_AD": 0.7,
                "predicted_label": "AD",
                "condition": "Ours",
            },
        ]
    ).to_csv(artifact_dir / "ours_predictions.csv", index=False)
    (artifact_dir / "ours_model.json").write_text(
        json.dumps(
            {
                "labels": list(LABELS),
                "schema_version": "condition-c-evidence-agent-v1",
                "final_probability_temperature": 1.25,
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "ours_model.joblib").write_bytes(b"frozen-condition-c-model")
    return artifact_dir


def test_packet_is_label_blind_and_probability_identical(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    advisor = FrozenConditionCAdvisor.from_artifact_dir(
        artifact_dir,
        expected_dataset_id="fixture",
        expected_subject_ids={"s1", "s2"},
    )

    evidence = {"state_graph_hash": "state-v1", "evidence_ids": ["ev-2", "ev-1"]}
    packet = advisor.explain_subject("s1", evidence_snapshot=evidence)

    assert isinstance(packet, ExplanationPacket)
    assert packet.class_order == LABELS
    assert tuple(packet.raw_probabilities) == LABELS
    assert tuple(packet.calibrated_probabilities or {}) == LABELS
    assert list(packet.raw_probabilities.values()) == [0.7, 0.2, 0.1]
    assert list((packet.calibrated_probabilities or {}).values()) == [0.7, 0.2, 0.1]
    assert packet.predicted_label == "HC"
    assert packet.calibration_status == "frozen_historical_calibrated"
    assert packet.applicability_status == "static_frozen_lookup_only"

    public = packet.to_dict()
    assert "label" not in public
    assert "split" not in public
    assert "dataset_id" not in public
    assert "subject_id" not in public
    assert set(public) == {
        "schema_version",
        "module_version",
        "class_order",
        "predicted_label",
        "raw_probabilities",
        "calibrated_probabilities",
        "calibration_status",
        "logits",
        "intercepts",
        "feature_contributions",
        "branch_contributions",
        "consumed_evidence_ids",
        "uncertainty",
        "fold_disagreement",
        "ood",
        "applicability_status",
        "hashes",
    }


def test_packet_binds_model_prediction_and_evidence_hashes(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    advisor = FrozenConditionCAdvisor.from_artifact_dir(artifact_dir)
    snapshot = {"evidence": [{"id": "ev-1", "value": 0.5}]}

    packet = advisor.explain_subject("s2", evidence_snapshot=snapshot)

    assert packet.hashes["model_joblib_sha256"] == sha256_file(
        artifact_dir / "ours_model.joblib"
    )
    assert packet.hashes["model_metadata_sha256"] == sha256_file(
        artifact_dir / "ours_model.json"
    )
    assert packet.hashes["prediction_table_sha256"] == sha256_file(
        artifact_dir / "ours_predictions.csv"
    )
    assert packet.hashes["evidence_snapshot_sha256"] == snapshot_hash(snapshot)
    assert packet.hashes["prediction_row_sha256"]
    assert packet.hashes["model_snapshot_sha256"]

    changed = advisor.explain_subject("s2", evidence_snapshot={"evidence": []})
    assert changed.raw_probabilities == packet.raw_probabilities
    assert changed.hashes["evidence_snapshot_sha256"] != packet.hashes[
        "evidence_snapshot_sha256"
    ]


@pytest.mark.parametrize("missing_name", ["ours_predictions.csv", "ours_model.json", "ours_model.joblib"])
def test_missing_required_artifact_is_rejected(tmp_path: Path, missing_name: str) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    (artifact_dir / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        FrozenConditionCAdvisor.from_artifact_dir(artifact_dir)


def test_unknown_and_duplicate_subjects_are_rejected(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    advisor = FrozenConditionCAdvisor.from_artifact_dir(artifact_dir)
    with pytest.raises(KeyError, match="Unknown subject_id"):
        advisor.explain_subject("missing")

    frame = pd.read_csv(artifact_dir / "ours_predictions.csv")
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(
        artifact_dir / "ours_predictions.csv", index=False
    )
    with pytest.raises(ValueError, match="Duplicate subject_id"):
        FrozenConditionCAdvisor.from_artifact_dir(artifact_dir)


@pytest.mark.parametrize(
    "probabilities",
    [
        (0.7, 0.2, 0.2),
        (-0.1, 0.4, 0.7),
        (math.nan, 0.4, 0.6),
    ],
)
def test_invalid_probability_vector_is_rejected(
    tmp_path: Path, probabilities: tuple[float, float, float]
) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    frame = pd.read_csv(artifact_dir / "ours_predictions.csv")
    frame.loc[0, ["prob_HC", "prob_MCI", "prob_AD"]] = probabilities
    frame.to_csv(artifact_dir / "ours_predictions.csv", index=False)

    with pytest.raises(ValueError, match="probabilit"):
        FrozenConditionCAdvisor.from_artifact_dir(artifact_dir)


def test_fixed_class_order_and_predicted_label_are_validated(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    metadata = json.loads((artifact_dir / "ours_model.json").read_text(encoding="utf-8"))
    metadata["labels"] = ["AD", "MCI", "HC"]
    (artifact_dir / "ours_model.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="probability columns do not match fixed class order"):
        FrozenConditionCAdvisor.from_artifact_dir(artifact_dir)

    artifact_dir = _artifact_dir(tmp_path / "second")
    frame = pd.read_csv(artifact_dir / "ours_predictions.csv")
    frame.loc[0, "predicted_label"] = "AD"
    frame.to_csv(artifact_dir / "ours_predictions.csv", index=False)
    with pytest.raises(ValueError, match="predicted_label does not match"):
        FrozenConditionCAdvisor.from_artifact_dir(artifact_dir)


def test_dataset_and_cohort_mismatch_are_rejected(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    with pytest.raises(ValueError, match="dataset cohort mismatch"):
        FrozenConditionCAdvisor.from_artifact_dir(
            artifact_dir, expected_dataset_id="another-dataset"
        )
    with pytest.raises(ValueError, match="subject cohort mismatch"):
        FrozenConditionCAdvisor.from_artifact_dir(
            artifact_dir, expected_subject_ids={"s1", "s3"}
        )


def test_mixed_dataset_rows_are_rejected(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    frame = pd.read_csv(artifact_dir / "ours_predictions.csv")
    frame.loc[1, "dataset_id"] = "other"
    frame.to_csv(artifact_dir / "ours_predictions.csv", index=False)

    with pytest.raises(ValueError, match="exactly one dataset cohort"):
        FrozenConditionCAdvisor.from_artifact_dir(artifact_dir)
