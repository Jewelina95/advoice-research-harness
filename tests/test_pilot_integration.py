"""Independent offline integration checks for the evidence-state pilot."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from advoice.evidence import EvidenceProvenance, MetricEvidenceV2
from advoice.pilot import contracts as c
from advoice.pilot import learning as l
from advoice.pilot.labels import LabelRow, read_labels_jsonl, serialize_labels_jsonl


FIXTURE = json.loads((Path(__file__).parent / "fixtures/pilot/synthetic_cases.json").read_text())


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _subject(index: int, *, partition: str = "development", fold_id: str = "fold_0") -> c.SubjectRow:
    return c.SubjectRow(
        dataset_id="synthetic", subject_id=f"sub_{index:016x}", partition=partition,
        task="hc_ad", class_order=("HC", "AD"), source_group_id=f"grp_{index:016x}",
        channel="picture_description", language="en", task_ids=("picture_description",),
        role="participant", fold_id=fold_id,
        raw_hashes={f"asset_{index:016x}": _hash(f"asset-{index}")}, source_version="fixture_v1",
    )


def _snapshot(subject: c.SubjectRow, *, value: float = 0.0, reference_id: str = "reference_pending",
              reference_hash: str = "a" * 64) -> c.EvidenceSnapshot:
    evidence = MetricEvidenceV2(
        evidence_id=f"e{subject.subject_id[-6:]}", metric_id="pause", state_id="timing",
        subject_id=subject.subject_id, case_id=f"case_{subject.subject_id[4:]}", value=value,
        provenance=EvidenceProvenance(source_asset_id=next(iter(subject.raw_hashes)),
                                      source_segment_ids=(), method_version="fixture_v1",
                                      measurement_version="fixture_v1", generated_by="fixture"),
    )
    return c.EvidenceSnapshot(
        subject=subject, case_id=f"case_{subject.subject_id[4:]}", state_version="state_v1",
        reference_fit_id=reference_id, reference_fit_hash=reference_hash,
        evidence=(c.MetricEvidenceBody.from_evidence(evidence),),
        state_cards=(c.StateCardBody(body={"state_id": "timing", "state_z": value,
            "available": True, "supporting_evidence_ids": [evidence.evidence_id],
            "counter_evidence_ids": []}),), source_segments=(),
        observability={"timing": c.Observability(status="observed", reason=None)},
        confounds={"potential": (), "observed": (), "ruled_out": ()},
        skill_hash="b" * 64, extractor_hash="c" * 64,
    )


def _head(arm: str, coefficients: tuple[float, float, float, float]) -> l.HeadCalibrator:
    return l.HeadCalibrator(arm=arm, head="binary", class_order=("HC", "AD"), estimable=True,
        refusal_reason=None, optimizer_success=True, objective=0.0, coefficients=coefficients,
        agent_mean=0.0, agent_scale=1.0, state_mean=0.0, state_scale=1.0,
        fit_ids=("sub_0000000000000001",), fit_count=1, seed=1)


def _manual_calibrators() -> l.JointCalibrators:
    arms = {
        "B": {"binary": _head("B", (1.0, 0.0, 0.0, 0.0))},
        "J-A": {"binary": _head("J-A", (1.0, 0.25, 0.0, 0.0))},
        "J-S": {"binary": _head("J-S", (1.0, 0.0, 0.25, 0.0))},
        "J-AS": {"binary": _head("J-AS", (1.0, 0.25, 1.0, 0.0))},
    }
    matched = {arm: {"binary": _head(f"B_matched_{arm}", (1.0, 0.0, 0.0, 0.0))}
               for arm in ("J-A", "J-S", "J-AS")}
    return l.JointCalibrators(task="hc_ad", class_order=("HC", "AD"), arms=arms,
        matched_baselines=matched, fold_manifest_id="manual_fixture_manifest",
        oof_prediction_hashes=("manual_fixture_prediction",), seed=1)


def _relation(actual: float, base: float) -> str:
    return "up" if actual > base + 1e-10 else "down" if actual < base - 1e-10 else "same"


@pytest.mark.parametrize("case", FIXTURE["conflict_matrix"], ids=lambda case: "-".join(
    str(case[key]) for key in ("base", "evidence", "agent", "state")))
def test_hand_calculated_conflict_matrix(case: dict[str, str]) -> None:
    """Every matrix expectation is fixed in JSON, never copied from model output."""
    base_ad = 0.50 if case["base"] == "low" else 0.90
    agent_scores = ({"HC": 0, "AD": 4} if case["agent"] == "plus" else {"HC": 4, "AD": 0})
    if case["evidence"] == "weak":
        agent_scores = {"HC": 2, "AD": 2}
    invalid = case["evidence"] == "invalid"
    base_log_odds = math.log(base_ad / (1.0 - base_ad))
    state_shift = {"plus": 1.0, "minus": -1.0, "zero": 0.0}[case["state"]]
    state_ad = 1.0 / (1.0 + math.exp(-(base_log_odds + state_shift)))
    row = l.CalibrationRow(
        subject=_subject(99), fusion_hash=_hash(json.dumps(case, sort_keys=True)),
        base_probabilities=(1 - base_ad, base_ad), agent_v0_scores=None if invalid else agent_scores,
        agent_v1_scores=None if invalid else agent_scores, v1_required=not invalid,
        state_probabilities_before=(1 - base_ad, base_ad),
        state_probabilities_after=(1 - state_ad, state_ad),
        arm_refusal_reasons={"J-A": "v0_failed" if invalid else None, "J-S": None,
                             "J-AS": "v0_failed" if invalid else None},
    )
    predictions = {item.arm: item for item in l.predict_joint(row, _manual_calibrators())}
    for arm, expected in (("J-A", case["j_a"]), ("J-S", case["j_s"]), ("J-AS", case["j_as"])):
        prediction = predictions[arm]
        if expected == "fallback":
            assert prediction.status == "fallback"
            assert prediction.probabilities == predictions["B"].probabilities
        else:
            assert prediction.status == "ok"
            assert _relation(prediction.probabilities[1], predictions["B"].probabilities[1]) == expected


def _oof_pipeline() -> tuple[list[l.FoldPrediction], dict[str, str], l.FoldArtifactManifest, l.FoldInputs, l.FrozenFoldConfig]:
    cases: dict[str, l.FoldCaseInput] = {}
    labels: dict[str, str] = {}
    for index in range(FIXTURE["pipeline"]["development_subjects"]):
        label = "HC" if index % 2 == 0 else "AD"
        subject = _subject(index, fold_id=f"fold_{(index // 2) % 2}")
        cases[subject.subject_id] = l.FoldCaseInput(
            subject=subject, base_features={"signal": -2.0 if label == "HC" else 2.0, "nonstate": index % 2},
            state_features={"state": -1.0 if label == "HC" else 1.0}, evidence_snapshot=_snapshot(subject),
        )
        labels[subject.subject_id] = label
    inputs = l.FoldInputs(cases=cases, labels=labels)
    config = l.FrozenFoldConfig(class_order=("HC", "AD"), base_feature_names=("signal", "nonstate"),
        state_feature_names=("state",), base_model_id="synthetic_full_base_v1",
        replay_model_id="synthetic_replay_v1", analytical_run=False,
        development_manifest=l.seal_development_manifest(inputs))
    artifacts, predictions = [], []
    for fold in ("fold_0", "fold_1"):
        validation = tuple(subject_id for subject_id, case in cases.items() if case.subject.fold_id == fold)
        artifact = l.fit_fold(inputs, tuple(subject_id for subject_id in cases if subject_id not in validation), validation, config)
        artifacts.append(artifact)
        for subject_id in validation:
            case = cases[subject_id]
            predictions.append(l.predict_fold(artifact, replace(case, evidence_snapshot=_snapshot(
                case.subject, reference_id=artifact.reference_fit_id, reference_hash=artifact.reference_fit_hash))))
    return predictions, labels, l.seal_fold_artifacts(artifacts, predictions), inputs, config


def test_tiny_offline_pipeline_uses_oof_only_and_scores_all_arms() -> None:
    predictions, labels, manifest, inputs, config = _oof_pipeline()
    rows = []
    for prediction in predictions:
        signal = prediction.subject.subject_id[-1]
        agent = {"HC": 4, "AD": 0} if int(signal, 16) % 2 == 0 else {"HC": 0, "AD": 4}
        before = prediction.state_probabilities
        after = (before[0] * 0.9 + 0.05, before[1] * 0.9 + 0.05)
        rows.append(l.CalibrationRow.from_fold_prediction(prediction, fusion_hash=_hash(prediction.subject.subject_id),
            agent_v0_scores=agent, agent_v1_scores=agent, v1_required=True,
            state_probabilities_after=after,
            arm_refusal_reasons={"J-A": None, "J-S": None, "J-AS": None}))
    calibrators = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    final = l.refit_full_development(inputs, predictions, manifest, config)
    heldout = _subject(100, partition="holdout", fold_id="holdout")
    holdout_prediction = l.predict_fold(final, l.FoldCaseInput(heldout, {"signal": 2.0, "nonstate": 0},
        {"state": 1.0}, _snapshot(heldout, reference_id=final.reference_fit_id, reference_hash=final.reference_fit_hash)))
    row = l.CalibrationRow.from_fold_prediction(holdout_prediction, fusion_hash=_hash("holdout"),
        agent_v0_scores={"HC": 0, "AD": 4}, agent_v1_scores={"HC": 0, "AD": 4}, v1_required=True,
        state_probabilities_after=(0.1, 0.9),
        arm_refusal_reasons={"J-A": None, "J-S": None, "J-AS": None})
    scored = l.predict_joint(row, calibrators)
    assert len(predictions) == len(labels) == 12
    assert {item.subject.subject_id for item in predictions} == set(labels)
    assert all(item.subject.subject_id not in item.fit_ids for item in predictions)
    assert {item.arm for item in scored} == {"B_raw", "B", "J-A", "J-S", "J-AS"}
    assert all(item.status == "ok" for item in scored)


def test_labels_are_isolated_from_provider_and_holdout_reading() -> None:
    holdout = _subject(77, partition="holdout", fold_id="holdout")
    label = LabelRow(subject=holdout, source_label="ad", label="AD", label_mapping={"ad": "AD"}, source="fixture")
    serialized = serialize_labels_jsonl((label,), purpose="scoring")
    assert read_labels_jsonl(serialized.splitlines(), purpose="scoring") == (label,)
    with pytest.raises(c.PilotContractError):
        read_labels_jsonl(serialized.splitlines(), purpose="inference")
    with pytest.raises(c.PilotContractError):
        label.to_inference_dict()
    with pytest.raises(c.PilotContractError):
        c.safe_inference_payload({"holdout_label": label.to_dict()})
