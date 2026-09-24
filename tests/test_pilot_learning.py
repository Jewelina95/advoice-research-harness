"""Synthetic, offline tests for fold-safe pilot learning."""
from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from advoice.evidence import MetricEvidenceV2
from advoice.pilot import contracts as c
from advoice.pilot import learning as l


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _subject(
    index: int,
    *,
    task: str = "hc_mci_ad",
    class_order: tuple[str, ...] = ("HC", "MCI", "AD"),
    partition: str = "development",
    fold_id: str = "fold_0",
    group_index: int | None = None,
) -> c.SubjectRow:
    group = index if group_index is None else group_index
    return c.SubjectRow(
        dataset_id="fixture", subject_id=f"sub_{index:016x}", partition=partition,
        task=task, class_order=class_order, source_group_id=f"grp_{group:016x}",
        channel="picture_description", language="en",
        task_ids=("picture_description",), role="participant", fold_id=fold_id,
        raw_hashes={f"asset_{index:016x}": _hash(f"asset-{index}")}, source_version="fixture_v1",
    )


def _snapshot(
    subject: c.SubjectRow,
    *,
    reference_fit_id: str = "reference_pending",
    reference_fit_hash: str = "a" * 64,
) -> c.EvidenceSnapshot:
    evidence = MetricEvidenceV2(
        evidence_id=f"e{subject.subject_id[-6:]}", metric_id="pause", state_id="timing",
        subject_id=subject.subject_id, case_id=f"case_{subject.subject_id[4:]}", value=0.0,
    )
    return c.EvidenceSnapshot(
        subject=subject, case_id=f"case_{subject.subject_id[4:]}", state_version="state_v1",
        reference_fit_id=reference_fit_id, reference_fit_hash=reference_fit_hash,
        evidence=(c.MetricEvidenceBody.from_evidence(evidence),),
        state_cards=(c.StateCardBody(body={
            "state_id": "timing", "state_z": 0.0, "available": True,
            "supporting_evidence_ids": [evidence.evidence_id], "counter_evidence_ids": [],
        }),),
        source_segments=(), observability={
            "timing": c.Observability(status="observed", reason=None),
        }, confounds={"potential": (), "observed": (), "ruled_out": ()},
        skill_hash="b" * 64, extractor_hash="c" * 64,
    )


def _fold_fixture(*, leaky_group: bool = False) -> tuple[l.FoldInputs, tuple[str, ...], tuple[str, ...], l.FrozenFoldConfig]:
    cases: dict[str, l.FoldCaseInput] = {}
    labels: dict[str, str] = {}
    fit_ids: list[str] = []
    validation_ids: list[str] = []
    for index in range(10):
        label = "HC" if index % 2 == 0 else "AD"
        fold_id = "fold_1" if index >= 8 else "fold_0"
        group = 0 if leaky_group and index == 8 else index
        subject = _subject(
            index, task="hc_ad", class_order=("HC", "AD"), fold_id=fold_id,
            group_index=group,
        )
        base_value = -2.0 if label == "HC" else 2.0
        state_value = -1.0 if label == "HC" else 1.0
        cases[subject.subject_id] = l.FoldCaseInput(
            subject=subject, base_features={"base_signal": base_value, "constant": 1.0},
            state_features={"state_signal": state_value, "missing": None},
            evidence_snapshot=_snapshot(subject),
        )
        labels[subject.subject_id] = label
        (validation_ids if index >= 8 else fit_ids).append(subject.subject_id)
    config = l.FrozenFoldConfig(
        class_order=("HC", "AD"), base_feature_names=("base_signal", "constant"),
        state_feature_names=("state_signal", "missing"), base_model_id="condition_c_fixture_v1",
        replay_model_id="state_replay_fixture_v1",
    )
    return l.FoldInputs(cases=cases, labels=labels), tuple(fit_ids), tuple(validation_ids), config


def _provenance(subject: c.SubjectRow, index: int) -> l.OOFProvenance:
    other = f"sub_{(index + 100):016x}"
    return l.OOFProvenance(
        fold_artifact_id=f"fold_artifact_{index}", fit_ids=(other,),
        fit_group_ids=(f"grp_{(index + 100):016x}",), validation_id=subject.subject_id,
        validation_group_id=subject.source_group_id, excluded_ids=(subject.subject_id,),
    )


def _three_class_rows(
    *,
    uninformative_agent: bool = False,
    invalid_second_pass: int | None = None,
) -> tuple[list[l.CalibrationRow], dict[str, str]]:
    rows: list[l.CalibrationRow] = []
    labels: dict[str, str] = {}
    base = {
        "HC": (0.72, 0.18, 0.10),
        "MCI": (0.22, 0.58, 0.20),
        "AD": (0.12, 0.23, 0.65),
    }
    scores = {
        "HC": {"HC": 4, "MCI": 1, "AD": 0},
        "MCI": {"HC": 1, "MCI": 4, "AD": 2},
        "AD": {"HC": 0, "MCI": 2, "AD": 4},
    }
    after = {
        "HC": (0.80, 0.14, 0.06),
        "MCI": (0.15, 0.67, 0.18),
        "AD": (0.08, 0.20, 0.72),
    }
    for index in range(18):
        label = ("HC", "MCI", "AD")[index % 3]
        subject = _subject(index, fold_id=f"fold_{index % 3}")
        ordinal = {"HC": 2, "MCI": 2, "AD": 2} if uninformative_agent else scores[label]
        refusal = {"J-A": None, "J-S": None, "J-AS": None}
        v1 = ordinal
        if invalid_second_pass == index:
            refusal["J-AS"] = "v1_failed"
            v1 = None
        row = l.CalibrationRow(
            subject=subject, fusion_hash=_hash(f"fusion-{index}"),
            base_probabilities=base[label], agent_v0_scores=ordinal,
            agent_v1_scores=v1, v1_required=True, state_probabilities_before=base[label],
            state_probabilities_after=after[label], arm_refusal_reasons=refusal,
            oof_provenance=_provenance(subject, index),
        )
        rows.append(row)
        labels[subject.subject_id] = label
    return rows, labels


def _binary_rows() -> tuple[list[l.CalibrationRow], dict[str, str]]:
    rows: list[l.CalibrationRow] = []
    labels: dict[str, str] = {}
    for index in range(12):
        label = "HC" if index % 2 == 0 else "AD"
        subject = _subject(
            index, task="hc_ad", class_order=("HC", "AD"), fold_id=f"fold_{index % 2}",
        )
        probability = (0.75, 0.25) if label == "HC" else (0.20, 0.80)
        scores = {"HC": 3, "AD": 1} if label == "HC" else {"HC": 1, "AD": 3}
        rows.append(l.CalibrationRow(
            subject=subject, fusion_hash=_hash(f"binary-fusion-{index}"),
            base_probabilities=probability, agent_v0_scores=scores, agent_v1_scores=None,
            v1_required=False,
            state_probabilities_before=probability, state_probabilities_after=probability,
            arm_refusal_reasons={"J-A": None, "J-S": None, "J-AS": None},
            oof_provenance=_provenance(subject, index),
        ))
        labels[subject.subject_id] = label
    return rows, labels


def _manual_head(
    arm: str,
    head: str,
    coefficients: tuple[float, float, float, float],
    *,
    class_order: tuple[str, ...] = ("HC", "AD"),
) -> l.HeadCalibrator:
    return l.HeadCalibrator(
        arm=arm, head=head, class_order=class_order, estimable=True, refusal_reason=None,
        optimizer_success=True, objective=0.0, coefficients=coefficients,
        agent_mean=0.0, agent_scale=1.0, state_mean=0.0, state_scale=1.0,
        fit_ids=("sub_0000000000000001",), fit_count=1, seed=1,
    )


def _manual_binary_calibrators() -> l.JointCalibrators:
    arms = {
        "B": {"binary": _manual_head("B", "binary", (1.0, 0.0, 0.0, 0.0))},
        "J-A": {"binary": _manual_head("J-A", "binary", (1.0, 0.5, 0.0, 0.0))},
        "J-S": {"binary": _manual_head("J-S", "binary", (1.0, 0.0, 0.5, 0.0))},
        "J-AS": {"binary": _manual_head("J-AS", "binary", (1.0, 0.5, 0.5, 0.0))},
    }
    matched = {
        arm: {"binary": _manual_head(f"B_matched_{arm}", "binary", (1.0, 0.0, 0.0, 0.0))}
        for arm in ("J-A", "J-S", "J-AS")
    }
    return l.JointCalibrators(
        task="hc_ad", class_order=("HC", "AD"), arms=arms,
        matched_baselines=matched, seed=1,
    )


def test_real_model_and_fusion_symbols_are_distinct_and_current():
    assert l.FULL_BASE_PREDICTOR_SYMBOL == "advoice.condition_c.train_condition_c"
    assert l.STATE_REPLAY_PREDICTOR_SYMBOL == "advoice.module_a.TaskConditionedStatisticalExpert"
    assert l.EXISTING_FUSION_SYMBOL == "advoice.authority_joint_fusion.fuse_authority_joint"
    assert len({l.FULL_BASE_PREDICTOR_SYMBOL, l.STATE_REPLAY_PREDICTOR_SYMBOL,
                l.EXISTING_FUSION_SYMBOL}) == 3
    manifest = l.model_identity_manifest()
    assert manifest["base"]["symbol"] == l.FULL_BASE_PREDICTOR_SYMBOL
    assert "subject_transcripts" in manifest["base"]["feature_signature"]
    assert manifest["replay"]["feature_signature"][0] == "explicit_state_feature_whitelist"


def test_state_only_signature_cannot_masquerade_as_full_base():
    with pytest.raises(l.LearningError, match="state-only baseline"):
        l.FrozenFoldConfig(
            class_order=("HC", "AD"), base_feature_names=("state",),
            state_feature_names=("state",), base_model_id="base_v1",
            replay_model_id="replay_v1",
        )


def test_fit_fold_rejects_identity_group_leakage():
    inputs, fit_ids, validation_ids, config = _fold_fixture(leaky_group=True)
    with pytest.raises(l.FoldLeakageError, match="identity group"):
        l.fit_fold(inputs, fit_ids, validation_ids, config)


def test_holdout_label_mutation_does_not_change_fold_artifact_or_predictions():
    inputs, fit_ids, validation_ids, config = _fold_fixture()
    first = l.fit_fold(inputs, fit_ids, validation_ids, config)
    mutated_labels = dict(inputs.labels)
    for subject_id in validation_ids:
        mutated_labels[subject_id] = "HC" if mutated_labels[subject_id] == "AD" else "AD"
    second = l.fit_fold(replace(inputs, labels=mutated_labels), fit_ids, validation_ids, config)
    assert first.artifact_id == second.artifact_id
    assert first.base_model.to_dict() == second.base_model.to_dict()
    assert first.replay_model.to_dict() == second.replay_model.to_dict()

    subject_id = validation_ids[0]
    case = inputs.cases[subject_id]
    bound = replace(case, evidence_snapshot=_snapshot(
        case.subject, reference_fit_id=first.reference_fit_id,
        reference_fit_hash=first.reference_fit_hash,
    ))
    prediction_one = l.predict_fold(first, bound)
    prediction_two = l.predict_fold(second, bound)
    assert prediction_one.base_probabilities == prediction_two.base_probabilities
    assert prediction_one.state_probabilities == prediction_two.state_probabilities
    assert prediction_one.provenance is not None
    assert subject_id not in prediction_one.provenance.fit_ids
    assert case.subject.source_group_id not in prediction_one.provenance.fit_group_ids


def test_fold_preprocessing_is_fit_only_and_constant_or_missing_features_are_finite():
    inputs, fit_ids, validation_ids, config = _fold_fixture()
    artifact = l.fit_fold(inputs, fit_ids, validation_ids, config)
    assert artifact.base_model.scales[1] == 1.0
    assert artifact.replay_model.impute_values[1] == 0.0
    assert all(np.isfinite(artifact.base_model.predict_proba((
        {"base_signal": np.inf, "constant": None},
    ))[0]))


def test_predict_fold_rejects_wrong_reference_context():
    inputs, fit_ids, validation_ids, config = _fold_fixture()
    artifact = l.fit_fold(inputs, fit_ids, validation_ids, config)
    with pytest.raises(l.LearningError, match="paired fold reference"):
        l.predict_fold(artifact, inputs.cases[validation_ids[0]])


def test_full_refit_is_blocked_until_oof_is_complete():
    inputs, fit_ids, validation_ids, config = _fold_fixture()
    artifact = l.fit_fold(inputs, fit_ids, validation_ids, config)
    predictions = []
    for subject_id in validation_ids:
        case = inputs.cases[subject_id]
        bound = replace(case, evidence_snapshot=_snapshot(
            case.subject, reference_fit_id=artifact.reference_fit_id,
            reference_fit_hash=artifact.reference_fit_hash,
        ))
        predictions.append(l.predict_fold(artifact, bound))
    with pytest.raises(l.LearningError, match="exactly one OOF"):
        l.refit_full_development(inputs, fit_ids + validation_ids, predictions, config)


def test_oof_duplicate_and_final_fit_provenance_are_rejected():
    rows, labels = _three_class_rows()
    with pytest.raises(l.FoldLeakageError, match="exactly one"):
        l.fit_joint_calibrators(rows + [rows[0]], labels, l.CalibrationConfig())
    with pytest.raises(l.FoldLeakageError, match="Final-refit"):
        replace(rows[0].oof_provenance, final_fit=True)


def test_prior_coefficients_reconstruct_raw_base_log_odds():
    head = _manual_head("B", "binary", l.PRIOR_COEFFICIENTS)
    for probability in (0.01, 0.25, 0.5, 0.91, 0.9999999):
        scored = l.apply_head(head, base_log_odds=l._logit(probability))
        assert scored == pytest.approx(l._clip_probability(probability), abs=1e-12)


def test_uninformative_agent_has_no_forced_positive_coefficient():
    rows, labels = _three_class_rows(uninformative_agent=True)
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig())
    for head in fitted.arms["J-A"].values():
        assert head.estimable
        assert head.coefficients is not None
        assert head.coefficients[1] == pytest.approx(0.0, abs=1e-10)


def test_fixed_contrasts_and_conflicting_agent_state_directions():
    assert l.agent_contrast({"HC": 1, "MCI": 3, "AD": 4}, "impairment") == 2.5
    assert l.agent_contrast({"HC": 1, "MCI": 3, "AD": 4}, "stage") == 1.0
    assert l.agent_contrast({"HC": 4, "AD": 1}, "binary") == -3.0

    subject = _subject(90, task="hc_ad", class_order=("HC", "AD"))
    row = l.CalibrationRow(
        subject=subject, fusion_hash=_hash("conflict"), base_probabilities=(0.5, 0.5),
        agent_v0_scores={"HC": 1, "AD": 4}, agent_v1_scores={"HC": 1, "AD": 4},
        v1_required=True,
        state_probabilities_before=(0.5, 0.5), state_probabilities_after=(0.8, 0.2),
        arm_refusal_reasons={"J-A": None, "J-S": None, "J-AS": None},
    )
    predictions = {item.arm: item for item in l.predict_joint(row, _manual_binary_calibrators())}
    assert predictions["J-A"].probabilities[1] > 0.5
    assert predictions["J-S"].probabilities[1] < 0.5
    assert (
        predictions["J-S"].probabilities[1]
        < predictions["J-AS"].probabilities[1]
        < predictions["J-A"].probabilities[1]
    )


def test_uncertain_base_and_weak_agent_are_finite_and_deterministic():
    head = _manual_head("J-A", "binary", (1.0, 0.25, 0.0, 0.0))
    first = l.apply_head(head, base_log_odds=l._logit(0.5), agent_value=0.01)
    second = l.apply_head(head, base_log_odds=l._logit(0.5), agent_value=0.01)
    assert math_is_finite(first)
    assert first == second


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


def test_binary_task_has_one_head_and_never_fabricates_mci():
    rows, labels = _binary_rows()
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig())
    assert set(fitted.arms["B"]) == {"binary"}
    predictions = l.predict_joint(rows[0], fitted)
    assert {item.arm for item in predictions} == set(l.ARMS)
    for prediction in predictions:
        assert prediction.class_order == ("HC", "AD")
        assert prediction.probabilities is None or len(prediction.probabilities) == 2


def test_invalid_second_pass_falls_back_directly_to_frozen_b():
    rows, labels = _three_class_rows(invalid_second_pass=0)
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig())
    predictions = {item.arm: item for item in l.predict_joint(rows[0], fitted)}
    assert predictions["J-AS"].status == "fallback"
    assert predictions["J-AS"].fallback_reason == "v1_failed"
    assert predictions["J-AS"].fallback_arm == "B"
    assert predictions["J-AS"].probabilities == predictions["B"].probabilities


def test_three_class_head_combination_and_probability_sum():
    result = l.combine_head_probabilities(("HC", "MCI", "AD"), impairment=0.7, stage=0.25)
    assert result == pytest.approx((0.3, 0.525, 0.175))
    assert sum(result) == pytest.approx(1.0)
    rows, labels = _three_class_rows()
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig())
    predictions = l.predict_joint(rows[1], fitted)
    assert tuple(item.arm for item in predictions) == l.ARMS
    for prediction in predictions:
        assert prediction.probabilities is not None
        assert sum(prediction.probabilities) == pytest.approx(1.0)
        assert np.isfinite(prediction.probabilities).all()


def test_model_class_permutation_round_trip():
    artifact = l.LinearModelArtifact(
        model_id="permuted_fixture", source_symbol="fixture.full_base",
        class_order=("HC", "MCI", "AD"), feature_names=("x",), impute_values=(0.0,),
        means=(0.0,), scales=(1.0,), learned_classes=("AD", "HC", "MCI"),
        coefficients=((2.0,), (-2.0,), (0.0,)), intercepts=(0.0, 0.0, 0.0),
        fit_ids=("sub_0000000000000001",), excluded_ids=(), seed=1,
    )
    probabilities = artifact.predict_proba(({"x": 1.0},))[0]
    assert probabilities[0] < probabilities[1] < probabilities[2]
    restored = l.LinearModelArtifact.from_dict(artifact.to_dict())
    assert restored.predict_proba(({"x": 1.0},))[0] == pytest.approx(probabilities)


def test_clipping_is_finite_at_probability_extremes():
    assert np.isfinite(l._logit(0.0))
    assert np.isfinite(l._logit(1.0))
    result = l.combine_head_probabilities(("HC", "AD"), binary=0.0)
    assert all(np.isfinite(result))
    assert sum(result) == pytest.approx(1.0)


def test_optimizer_failure_blocks_heads_without_invented_coefficients(monkeypatch):
    rows, labels = _binary_rows()

    class Failed:
        success = False
        fun = 1.0
        x = np.asarray(l.PRIOR_COEFFICIENTS)

    monkeypatch.setattr(l, "minimize", lambda *args, **kwargs: Failed())
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig())
    for heads in fitted.arms.values():
        assert not heads["binary"].estimable
        assert heads["binary"].refusal_reason == "optimizer_failure"
        assert heads["binary"].coefficients is None


def test_zero_state_replay_delta_is_neutral():
    subject = _subject(91, task="hc_ad", class_order=("HC", "AD"))
    row = l.CalibrationRow(
        subject=subject, fusion_hash=_hash("zero-delta"), base_probabilities=(0.4, 0.6),
        agent_v0_scores={"HC": 2, "AD": 2}, agent_v1_scores=None, v1_required=False,
        state_probabilities_before=(0.4, 0.6), state_probabilities_after=(0.4, 0.6),
        arm_refusal_reasons={"J-A": None, "J-S": None, "J-AS": None},
    )
    predictions = {item.arm: item for item in l.predict_joint(row, _manual_binary_calibrators())}
    assert l.replay_log_odds_delta(row, "binary") == pytest.approx(0.0)
    assert predictions["J-S"].probabilities == pytest.approx(predictions["B"].probabilities)


def test_labels_are_not_mutated_or_reassigned_to_improve_fit():
    rows, labels = _three_class_rows()
    original = dict(labels)
    l.fit_joint_calibrators(rows, labels, l.CalibrationConfig())
    assert labels == original
    assert [labels[row.subject.subject_id] for row in rows].count("MCI") == 6


def test_matched_fit_baselines_use_exact_joint_fit_ids():
    rows, labels = _three_class_rows(invalid_second_pass=0)
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig())
    for head in ("impairment", "stage"):
        assert (
            fitted.matched_baselines["J-AS"][head].fit_ids
            == fitted.arms["J-AS"][head].fit_ids
        )
    assert rows[0].subject.subject_id not in fitted.arms["J-AS"]["impairment"].fit_ids


def test_calibrator_serialization_preserves_predictions_within_tolerance():
    rows, labels = _three_class_rows()
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig())
    restored = l.JointCalibrators.from_json(fitted.to_json())
    before = l.predict_joint(rows[5], fitted)
    after = l.predict_joint(rows[5], restored)
    assert fitted.artifact_id == restored.artifact_id
    for left, right in zip(before, after, strict=True):
        assert left.status == right.status
        assert left.probabilities == pytest.approx(right.probabilities, abs=1e-10)


def test_missing_agent_or_replay_inputs_are_not_treated_as_healthy_evidence():
    subject = _subject(92, task="hc_ad", class_order=("HC", "AD"))
    row = l.CalibrationRow(
        subject=subject, fusion_hash=_hash("missing-inputs"), base_probabilities=(0.45, 0.55),
        agent_v0_scores=None, agent_v1_scores=None, v1_required=False,
        state_probabilities_before=None,
        state_probabilities_after=None,
        arm_refusal_reasons={"J-A": "v0_missing", "J-S": "replay_missing", "J-AS": "v0_missing"},
    )
    predictions = {item.arm: item for item in l.predict_joint(row, _manual_binary_calibrators())}
    for arm in ("J-A", "J-S", "J-AS"):
        assert predictions[arm].status == "fallback"
        assert predictions[arm].probabilities == predictions["B"].probabilities


def test_absent_stage_training_class_is_explicitly_unestimable():
    rows, labels = _three_class_rows()
    no_ad_labels = {
        subject_id: ("MCI" if label == "AD" else label) for subject_id, label in labels.items()
    }
    fitted = l.fit_joint_calibrators(rows, no_ad_labels, l.CalibrationConfig())
    assert not fitted.arms["B"]["stage"].estimable
    assert fitted.arms["B"]["stage"].refusal_reason == "absent_training_class"
    predictions = {item.arm: item for item in l.predict_joint(rows[0], fitted)}
    assert predictions["B"].status == "unavailable"
    assert predictions["J-AS"].status == "unavailable"
