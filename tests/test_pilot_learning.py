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
    dataset_id: str = "fixture",
    source_version: str = "fixture_v1",
    channel: str = "picture_description",
) -> c.SubjectRow:
    group = index if group_index is None else group_index
    return c.SubjectRow(
        dataset_id=dataset_id, subject_id=f"sub_{index:016x}", partition=partition,
        task=task, class_order=class_order, source_group_id=f"grp_{group:016x}",
        channel=channel, language="en",
        task_ids=("picture_description",), role="participant", fold_id=fold_id,
        raw_hashes={f"asset_{index:016x}": _hash(f"asset-{index}")},
        source_version=source_version,
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
    inputs = l.FoldInputs(cases=cases, labels=labels)
    config = l.FrozenFoldConfig(
        class_order=("HC", "AD"), base_feature_names=("base_signal", "constant"),
        state_feature_names=("state_signal", "missing"), base_model_id="condition_c_fixture_v1",
        replay_model_id="state_replay_fixture_v1", analytical_run=False,
        development_manifest=l.seal_development_manifest(inputs),
    )
    return inputs, tuple(fit_ids), tuple(validation_ids), config


def _actual_oof_predictions(
    *, task: str, class_order: tuple[str, ...], ordered_labels: tuple[str, ...], count: int,
) -> tuple[list[l.FoldPrediction], dict[str, str], l.FoldArtifactManifest, l.FoldInputs, l.FrozenFoldConfig]:
    cases: dict[str, l.FoldCaseInput] = {}
    labels: dict[str, str] = {}
    signal = {label: float(index) for index, label in enumerate(class_order)}
    fold_count = 3 if len(class_order) == 3 else 2
    for index in range(count):
        label = ordered_labels[index % len(ordered_labels)]
        subject = _subject(
            index, task=task, class_order=class_order,
            fold_id=f"fold_{(index // len(ordered_labels)) % fold_count}",
        )
        cases[subject.subject_id] = l.FoldCaseInput(
            subject=subject,
            base_features={"base_signal": signal[label], "base_nonstate": float(index % 2)},
            state_features={"state_signal": signal[label]},
            evidence_snapshot=_snapshot(subject),
        )
        labels[subject.subject_id] = label
    inputs = l.FoldInputs(cases=cases, labels=labels)
    config = l.FrozenFoldConfig(
        class_order=class_order, base_feature_names=("base_signal", "base_nonstate"),
        state_feature_names=("state_signal",), base_model_id="synthetic_full_base_v1",
        replay_model_id="synthetic_state_replay_v1", analytical_run=False,
        development_manifest=l.seal_development_manifest(inputs),
    )
    artifacts: list[l.FoldArtifact] = []
    predictions: list[l.FoldPrediction] = []
    for fold_index in range(fold_count):
        validation = tuple(
            subject_id for subject_id, case in cases.items()
            if case.subject.fold_id == f"fold_{fold_index}"
        )
        fit = tuple(subject_id for subject_id in cases if subject_id not in validation)
        artifact = l.fit_fold(inputs, fit, validation, config)
        artifacts.append(artifact)
        for subject_id in validation:
            case = cases[subject_id]
            bound = replace(case, evidence_snapshot=_snapshot(
                case.subject, reference_fit_id=artifact.reference_fit_id,
                reference_fit_hash=artifact.reference_fit_hash,
            ))
            predictions.append(l.predict_fold(artifact, bound))
    return predictions, labels, l.seal_fold_artifacts(artifacts, predictions), inputs, config


def _three_class_rows(
    *,
    uninformative_agent: bool = False,
    invalid_second_pass: int | None = None,
) -> tuple[list[l.CalibrationRow], dict[str, str], l.FoldArtifactManifest]:
    rows: list[l.CalibrationRow] = []
    predictions, labels, manifest, _, _ = _actual_oof_predictions(
        task="hc_mci_ad", class_order=("HC", "MCI", "AD"),
        ordered_labels=("HC", "MCI", "AD"), count=18,
    )
    scores = {
        "HC": {"HC": 4, "MCI": 1, "AD": 0},
        "MCI": {"HC": 1, "MCI": 4, "AD": 2},
        "AD": {"HC": 0, "MCI": 2, "AD": 4},
    }
    for index, prediction in enumerate(predictions):
        label = labels[prediction.subject.subject_id]
        ordinal = {"HC": 2, "MCI": 2, "AD": 2} if uninformative_agent else scores[label]
        refusal = {"J-A": None, "J-S": None, "J-AS": None}
        v1 = ordinal
        if invalid_second_pass == index:
            refusal["J-AS"] = "v1_failed"
            v1 = None
        before = np.asarray(prediction.state_probabilities)
        target = np.zeros(3)
        target[("HC", "MCI", "AD").index(label)] = 1.0
        after = tuple((0.8 * before + 0.2 * target).tolist())
        row = l.CalibrationRow.from_fold_prediction(
            prediction, fusion_hash=_hash(f"fusion-{index}"),
            agent_v0_scores=ordinal, agent_v1_scores=v1, v1_required=True,
            state_probabilities_after=after, arm_refusal_reasons=refusal,
        )
        rows.append(row)
    return rows, labels, manifest


def _binary_rows() -> tuple[list[l.CalibrationRow], dict[str, str], l.FoldArtifactManifest]:
    predictions, labels, manifest, _, _ = _actual_oof_predictions(
        task="hc_ad", class_order=("HC", "AD"), ordered_labels=("HC", "AD"), count=12,
    )
    return _binary_rows_for_predictions(predictions, labels), labels, manifest


def _binary_rows_for_predictions(
    predictions: list[l.FoldPrediction], labels: dict[str, str],
) -> list[l.CalibrationRow]:
    rows: list[l.CalibrationRow] = []
    for index, prediction in enumerate(predictions):
        label = labels[prediction.subject.subject_id]
        scores = {"HC": 3, "AD": 1} if label == "HC" else {"HC": 1, "AD": 3}
        rows.append(l.CalibrationRow.from_fold_prediction(
            prediction, fusion_hash=_hash(f"binary-fusion-{index}"),
            agent_v0_scores=scores, agent_v1_scores=None, v1_required=False,
            state_probabilities_after=prediction.state_probabilities,
            arm_refusal_reasons={"J-A": None, "J-S": None, "J-AS": None},
        ))
    return rows


def _analytical_fixture(
    *, origin: str,
) -> tuple[l.FoldInputs, tuple[str, ...], tuple[str, ...], l.FrozenFoldConfig]:
    inputs, fit_ids, validation_ids, synthetic_config = _fold_fixture()
    cases: dict[str, l.FoldCaseInput] = {}
    for subject_id, case in inputs.cases.items():
        subject = replace(
            case.subject,
            dataset_id="dementiabank_prepared",
            source_version="dbank_2026_09",
        )
        provenance = l.PreparedFeatureProvenance.from_features(
            subject=subject,
            base_features=case.base_features,
            state_features=case.state_features,
            origin=origin,
            source_manifest_hash=_hash("prepared-source-manifest"),
        )
        cases[subject_id] = l.FoldCaseInput(
            subject=subject,
            base_features=case.base_features,
            state_features=case.state_features,
            evidence_snapshot=_snapshot(subject),
            feature_provenance=provenance,
        )
    analytical_inputs = l.FoldInputs(cases=cases, labels=inputs.labels)
    base = l.FixedPilotBasePredictorAdapter()
    replay = l.FixedPilotReplayPredictorAdapter()
    registry = l.create_trusted_adapter_registry(
        (base, replay),
        class_order=synthetic_config.class_order,
        feature_names_by_role={
            "base": synthetic_config.base_feature_names,
            "replay": synthetic_config.state_feature_names,
        },
    )
    config = replace(
        synthetic_config,
        base_model_id="fixed_pilot_base_v1",
        replay_model_id="fixed_pilot_replay_v1",
        analytical_run=True,
        base_adapter=base,
        replay_adapter=replay,
        trusted_adapter_registry=registry,
        development_manifest=l.seal_development_manifest(analytical_inputs),
    )
    return analytical_inputs, fit_ids, validation_ids, config


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
        matched_baselines=matched, fold_manifest_id="manual_fixture_manifest",
        oof_prediction_hashes=("manual_fixture_prediction",), seed=1,
    )


def test_real_model_and_fusion_symbols_are_distinct_and_current():
    assert l.FULL_BASE_PREDICTOR_SYMBOL == "advoice.condition_c.train_condition_c"
    assert l.STATE_REPLAY_PREDICTOR_SYMBOL == "advoice.module_a.TaskConditionedStatisticalExpert"
    assert l.EXISTING_FUSION_SYMBOL == "advoice.authority_joint_fusion.fuse_authority_joint"
    assert len({l.FULL_BASE_PREDICTOR_SYMBOL, l.STATE_REPLAY_PREDICTOR_SYMBOL,
                l.EXISTING_FUSION_SYMBOL}) == 3
    manifest = l.model_identity_manifest()
    assert manifest["base"]["symbol"] == "advoice.pilot.learning.FixedPilotBasePredictorAdapter"
    assert manifest["base"]["reference_symbol"] == l.FULL_BASE_PREDICTOR_SYMBOL
    assert manifest["base"]["scope"] == "fixed_pilot_predictor_over_real_prepared_features"
    assert manifest["base"]["validated_condition_c_equivalence"] is False
    assert manifest["base"]["validated_full_extraction"] is False
    assert manifest["replay"]["feature_signature"][0] == "explicit_state_feature_whitelist"


def test_analytical_adapter_rejects_fixture_provenance_and_binds_reference_ranges():
    inputs, fit_ids, validation_ids, config = _analytical_fixture(
        origin="synthetic_fixture",
    )
    with pytest.raises(l.LearningError, match="real prepared feature provenance"):
        l.fit_fold(inputs, fit_ids, validation_ids, config)

    inputs, fit_ids, validation_ids, config = _analytical_fixture(
        origin="real_prepared",
    )
    artifact = l.fit_fold(inputs, fit_ids, validation_ids, config)
    assert artifact.purpose == "analytical"
    assert artifact.predictor_scope == "fixed_pilot_predictor_over_real_prepared_features"
    assert artifact.base_model.adapter_implementation.endswith("fixed_prepared_base_logistic")
    assert l.FULL_BASE_PREDICTOR_SYMBOL not in artifact.base_model.adapter_implementation
    assert artifact.reference_ranges
    assert len(artifact.reference_range_hash) == 64
    assert artifact.reference_range_hash in artifact.reference_fit_hash_inputs


def test_synthetic_adapter_is_typed_persisted_and_blocked_for_analytical_runs():
    inputs, fit_ids, validation_ids, config = _fold_fixture()
    assert isinstance(l.SyntheticLinearPredictorAdapter("base"), l.PredictorAdapter)
    artifact = l.fit_fold(inputs, fit_ids, validation_ids, config)
    assert artifact.purpose == "synthetic_test"
    assert artifact.base_model.adapter_implementation.endswith("synthetic_base_linear")
    assert artifact.replay_model.adapter_implementation.endswith("synthetic_replay_linear")
    assert artifact.base_model.feature_pipeline_hash != artifact.replay_model.feature_pipeline_hash
    assert len(artifact.base_model.feature_pipeline_hash) == 64
    assert l.FULL_BASE_PREDICTOR_SYMBOL not in artifact.base_model.adapter_implementation
    analytical_inputs, analytical_fit, analytical_validation, analytical_config = (
        _analytical_fixture(origin="real_prepared")
    )
    with pytest.raises(l.LearningError, match="production predictor adapter"):
        l.fit_fold(
            analytical_inputs, analytical_fit, analytical_validation,
            replace(
                analytical_config, base_adapter=None, replay_adapter=None,
                trusted_adapter_registry=None,
            ),
        )


def test_synthetic_adapter_cannot_self_attest_for_analytical_use(monkeypatch):
    inputs, fit_ids, validation_ids, config = _analytical_fixture(
        origin="real_prepared",
    )
    base = l.SyntheticLinearPredictorAdapter("base")
    replay = l.SyntheticLinearPredictorAdapter("replay")
    monkeypatch.setattr(l.SyntheticLinearPredictorAdapter, "analytical_capable", True, raising=False)
    with pytest.raises(l.LearningError, match="trusted analytical adapter"):
        l.create_trusted_adapter_registry((base, replay))
    with pytest.raises(l.LearningError, match="trusted adapter registry"):
        l.fit_fold(
            inputs, fit_ids, validation_ids,
            replace(
                config, base_adapter=base, replay_adapter=replay,
                trusted_adapter_registry=None,
            ),
        )


def test_trusted_adapter_registry_cannot_be_caller_constructed():
    with pytest.raises(l.LearningError, match="created by create_trusted_adapter_registry"):
        l.TrustedAdapterRegistry(())


@pytest.mark.parametrize("partition", ["holdout", "stress"])
@pytest.mark.parametrize("target", ["fit", "validation"])
def test_fit_fold_rejects_non_development_fit_and_validation_subjects(partition, target):
    inputs, fit_ids, validation_ids, config = _fold_fixture()
    subject_id = fit_ids[0] if target == "fit" else validation_ids[0]
    original = inputs.cases[subject_id]
    changed_subject = replace(original.subject, partition=partition)
    changed_case = replace(
        original, subject=changed_subject, evidence_snapshot=_snapshot(changed_subject),
    )
    changed_cases = dict(inputs.cases)
    changed_cases[subject_id] = changed_case
    with pytest.raises(l.FoldLeakageError, match="development subjects"):
        l.fit_fold(
            replace(inputs, cases=changed_cases), fit_ids, validation_ids, config,
        )


def test_public_fit_fold_cannot_create_final_refit_artifact():
    inputs, fit_ids, _, config = _fold_fixture()
    with pytest.raises(l.LearningError, match="validation_ids"):
        l.fit_fold(inputs, fit_ids, (), config)
    with pytest.raises(TypeError):
        l.FrozenFoldConfig(
            class_order=("HC", "AD"), base_feature_names=("base",),
            state_feature_names=("state",), base_model_id="base_v1",
            replay_model_id="replay_v1", final_refit=True,
        )


def test_private_full_refit_requires_verified_complete_development_oof():
    predictions, labels, manifest, inputs, config = _actual_oof_predictions(
        task="hc_ad", class_order=("HC", "AD"), ordered_labels=("HC", "AD"), count=12,
    )
    final = l.refit_full_development(
        inputs, predictions, manifest, config,
    )
    development_manifest = config.development_manifest
    assert set(development_manifest.subject_ids) == set(labels)
    for proof in development_manifest.subjects:
        subject = inputs.cases[proof.subject_id].subject
        assert proof.source_group_id == subject.source_group_id
        assert proof.fold_id == subject.fold_id
        assert proof.subject_hash == subject.content_hash
    assert final.final_refit
    assert not final.validation_ids

    subject_id = next(iter(labels))
    original = inputs.cases[subject_id]
    holdout_subject = replace(original.subject, partition="holdout")
    changed_cases = dict(inputs.cases)
    changed_cases[subject_id] = replace(
        original, subject=holdout_subject, evidence_snapshot=_snapshot(holdout_subject),
    )
    with pytest.raises(l.FoldLeakageError, match="canonical development manifest"):
        l.refit_full_development(
            replace(inputs, cases=changed_cases), predictions, manifest, config,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_model_id", "changed_base"),
        ("replay_model_id", "changed_replay"),
        ("seed", 7),
        ("c", 0.25),
        ("max_iter", 17),
        ("excluded_ids", ("sub_external_exclusion",)),
    ],
)
def test_full_refit_rejects_recipe_changes(field, value):
    predictions, _, manifest, inputs, config = _actual_oof_predictions(
        task="hc_ad", class_order=("HC", "AD"), ordered_labels=("HC", "AD"), count=12,
    )
    with pytest.raises(l.FoldLeakageError, match="model recipe"):
        l.refit_full_development(
            inputs, predictions, manifest, replace(config, **{field: value}),
        )


def test_full_refit_rejects_subset_against_canonical_development_manifest():
    predictions, _, manifest, inputs, config = _actual_oof_predictions(
        task="hc_ad", class_order=("HC", "AD"), ordered_labels=("HC", "AD"), count=12,
    )
    omitted = predictions[0].subject.subject_id
    with pytest.raises(l.LearningError, match="exactly one OOF"):
        l.refit_full_development(inputs, predictions[1:], manifest, config)

    subset_cases = {key: value for key, value in inputs.cases.items() if key != omitted}
    subset_labels = {key: value for key, value in inputs.labels.items() if key != omitted}
    with pytest.raises(l.FoldLeakageError, match="canonical development manifest"):
        l.refit_full_development(
            l.FoldInputs(cases=subset_cases, labels=subset_labels),
            predictions[1:], manifest, config,
        )


def test_fold_manifest_rejects_incomplete_sealed_development_membership():
    inputs, fit_ids, validation_ids, config = _fold_fixture()
    artifact = l.fit_fold(inputs, fit_ids, validation_ids, config)
    predictions = []
    for subject_id in validation_ids:
        case = inputs.cases[subject_id]
        bound = replace(case, evidence_snapshot=_snapshot(
            case.subject,
            reference_fit_id=artifact.reference_fit_id,
            reference_fit_hash=artifact.reference_fit_hash,
        ))
        predictions.append(l.predict_fold(artifact, bound))
    with pytest.raises(l.FoldLeakageError, match="complete development OOF membership"):
        l.seal_fold_artifacts((artifact,), predictions)


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
    holdout = _subject(
        99, task="hc_ad", class_order=("HC", "AD"), partition="holdout",
        fold_id="holdout",
    )
    cases = dict(inputs.cases)
    cases[holdout.subject_id] = l.FoldCaseInput(
        subject=holdout, base_features={"base_signal": 50.0, "constant": 1.0},
        state_features={"state_signal": 50.0, "missing": None},
        evidence_snapshot=_snapshot(holdout),
    )
    labels = {**inputs.labels, holdout.subject_id: "AD"}
    with_holdout = l.FoldInputs(cases=cases, labels=labels)
    first = l.fit_fold(with_holdout, fit_ids, validation_ids, config)
    mutated_labels = {**labels, holdout.subject_id: "HC"}
    second = l.fit_fold(
        replace(with_holdout, labels=mutated_labels), fit_ids, validation_ids, config,
    )
    assert first.artifact_id == second.artifact_id
    assert first.base_model.to_dict() == second.base_model.to_dict()
    assert first.replay_model.to_dict() == second.replay_model.to_dict()

    subject_id = validation_ids[0]
    case = with_holdout.cases[subject_id]
    bound = replace(case, evidence_snapshot=_snapshot(
        case.subject, reference_fit_id=first.reference_fit_id,
        reference_fit_hash=first.reference_fit_hash,
    ))
    prediction_one = l.predict_fold(first, bound)
    prediction_two = l.predict_fold(second, bound)
    assert prediction_one.base_probabilities == prediction_two.base_probabilities
    assert prediction_one.state_probabilities == prediction_two.state_probabilities
    assert not prediction_one.final_refit
    assert subject_id not in prediction_one.fit_ids
    assert case.subject.source_group_id not in prediction_one.fit_group_ids


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
    with pytest.raises(l.FoldLeakageError, match="complete development OOF membership"):
        l.seal_fold_artifacts((artifact,), predictions)


def test_oof_duplicate_and_fabricated_fold_binding_are_rejected():
    rows, labels, manifest = _three_class_rows()
    with pytest.raises(l.LearningError, match="seal_fold_artifacts"):
        l.FoldArtifactManifest(manifest.proofs)
    with pytest.raises(l.FoldLeakageError, match="exactly one"):
        l.fit_joint_calibrators(
            rows + [rows[0]], labels, l.CalibrationConfig(), manifest,
        )
    fabricated_prediction = replace(
        rows[0].fold_prediction, fold_artifact_id="fold_fabricated", prediction_hash="",
    )
    with pytest.raises(l.FoldLeakageError, match="sealed OOF calibration receipt"):
        replace(rows[0], fold_prediction=fabricated_prediction)


def test_oof_binding_retains_and_verifies_resolved_fold_artifact_fields():
    rows, labels, manifest = _three_class_rows()
    prediction = rows[0].fold_prediction
    proof = manifest.resolve(prediction.fold_artifact_id)
    assert prediction.base_fit_hash == proof.base_model_artifact_hash
    assert prediction.reference_fit_id == proof.reference_fit_id
    assert prediction.reference_fit_hash == proof.reference_fit_hash
    assert prediction.fold_id == proof.fold_id
    assert prediction.excluded_ids == proof.excluded_ids
    assert prediction.validation_id in prediction.excluded_ids
    assert prediction.validation_group_id not in prediction.fit_group_ids

    tampered = replace(prediction, base_fit_hash="model_fabricated", prediction_hash="")
    with pytest.raises(l.FoldLeakageError, match="sealed OOF calibration receipt"):
        replace(rows[0], fold_prediction=tampered)


def test_oof_probability_tamper_with_recomputed_prediction_hash_is_rejected():
    rows, labels, manifest = _three_class_rows()
    prediction = rows[0].fold_prediction
    forged_probabilities = tuple(reversed(prediction.base_probabilities))
    forged_prediction = replace(
        prediction, base_probabilities=forged_probabilities, prediction_hash="",
    )
    assert forged_prediction.prediction_hash != prediction.prediction_hash
    with pytest.raises(l.FoldLeakageError, match="sealed OOF calibration receipt"):
        replace(
            rows[0], base_probabilities=forged_probabilities,
            fold_prediction=forged_prediction,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"agent_v0_scores": {"HC": 0, "MCI": 4, "AD": 4}},
        {"assessment_hashes": (("v0", "d" * 64),)},
        {"source_trace_hashes": ("e" * 64,)},
        {"replay_hash": "f" * 64},
        {"fusion_hash": "0" * 64},
        {"state_probabilities_after": (0.2, 0.3, 0.5)},
    ],
)
def test_oof_calibration_receipt_rejects_agent_trace_replay_and_fusion_fabrication(change):
    rows, labels, manifest = _three_class_rows()
    assert rows[0].oof_receipt is not None
    with pytest.raises(l.FoldLeakageError, match="sealed OOF calibration receipt"):
        forged = replace(rows[0], **change)
        l.fit_joint_calibrators(
            [forged, *rows[1:]], labels, l.CalibrationConfig(), manifest,
        )


def test_calibration_requires_manifest_complete_rows_and_labels_not_matching_subsets():
    rows, labels, manifest = _three_class_rows()
    omitted = rows[0].subject.subject_id
    subset_labels = {key: value for key, value in labels.items() if key != omitted}
    with pytest.raises(l.FoldLeakageError, match="complete sealed development OOF membership"):
        l.fit_joint_calibrators(
            rows[1:], subset_labels, l.CalibrationConfig(), manifest,
        )


def test_final_refit_binds_frozen_calibrators_and_rejects_relabelled_training_identity():
    predictions, labels, manifest, inputs, config = _actual_oof_predictions(
        task="hc_ad", class_order=("HC", "AD"), ordered_labels=("HC", "AD"), count=12,
    )
    rows = _binary_rows_for_predictions(predictions, labels)
    calibrators = l.fit_joint_calibrators(
        rows, labels, l.CalibrationConfig(), manifest,
    )
    final = l.refit_full_development(
        inputs, predictions, manifest, config, calibrators=calibrators,
    )
    assert final.calibrator_artifact_id == calibrators.artifact_id
    assert final.calibration_recipe_hash == calibrators.calibration_recipe_hash

    training_case = inputs.cases[predictions[0].subject.subject_id]
    relabelled = replace(training_case.subject, partition="holdout", fold_id="holdout")
    holdout_case = l.FoldCaseInput(
        subject=relabelled,
        base_features=training_case.base_features,
        state_features=training_case.state_features,
        evidence_snapshot=_snapshot(
            relabelled,
            reference_fit_id=final.reference_fit_id,
            reference_fit_hash=final.reference_fit_hash,
        ),
    )
    with pytest.raises(l.FoldLeakageError, match="training identity"):
        l.predict_fold(final, holdout_case)

    disguised = replace(
        training_case.subject,
        subject_id="sub_fffffffffffffff0",
        source_group_id="grp_fffffffffffffff0",
        partition="holdout",
        fold_id="holdout",
    )
    disguised_case = l.FoldCaseInput(
        subject=disguised,
        base_features=training_case.base_features,
        state_features=training_case.state_features,
        evidence_snapshot=_snapshot(
            disguised,
            reference_fit_id=final.reference_fit_id,
            reference_fit_hash=final.reference_fit_hash,
        ),
    )
    with pytest.raises(l.FoldLeakageError, match="training identity"):
        l.predict_fold(final, disguised_case)


def test_analytical_final_refit_requires_matching_frozen_calibrators_and_recipe():
    inputs, _, _, config = _analytical_fixture(origin="real_prepared")
    artifacts: list[l.FoldArtifact] = []
    predictions: list[l.FoldPrediction] = []
    for fold_id in ("fold_0", "fold_1"):
        validation = tuple(
            subject_id for subject_id, case in inputs.cases.items()
            if case.subject.fold_id == fold_id
        )
        fit = tuple(subject_id for subject_id in inputs.cases if subject_id not in validation)
        artifact = l.fit_fold(inputs, fit, validation, config)
        artifacts.append(artifact)
        for subject_id in validation:
            case = inputs.cases[subject_id]
            bound = replace(case, evidence_snapshot=_snapshot(
                case.subject,
                reference_fit_id=artifact.reference_fit_id,
                reference_fit_hash=artifact.reference_fit_hash,
            ))
            predictions.append(l.predict_fold(artifact, bound))
    manifest = l.seal_fold_artifacts(artifacts, predictions)
    labels = dict(inputs.labels)
    rows = _binary_rows_for_predictions(predictions, labels)
    with pytest.raises(l.LearningError, match="frozen calibrators"):
        l.refit_full_development(inputs, predictions, manifest, config)

    with pytest.raises(l.FoldLeakageError, match="synthetic fixture OOF receipt"):
        l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)

    fixture_rows, fixture_labels, fixture_manifest = _binary_rows()
    fixture_calibrators = l.fit_joint_calibrators(
        fixture_rows, fixture_labels, l.CalibrationConfig(), fixture_manifest,
    )
    proof = manifest.proofs[0]
    changed = replace(
        fixture_calibrators,
        fold_manifest_id=manifest.manifest_id,
        dataset_id=proof.dataset_id,
        source_version=proof.source_version,
        channel=proof.channel,
        development_manifest_id=proof.development_manifest_id,
        model_recipe_hash="0" * 64,
        purpose="analytical",
        artifact_id="",
    )
    with pytest.raises(l.FoldLeakageError, match="calibrator.*recipe"):
        l.refit_full_development(
            inputs, predictions, manifest, config, calibrators=changed,
        )


def test_prior_coefficients_reconstruct_raw_base_log_odds():
    head = _manual_head("B", "binary", l.PRIOR_COEFFICIENTS)
    for probability in (0.01, 0.25, 0.5, 0.91, 0.9999999):
        scored = l.apply_head(head, base_log_odds=l._logit(probability))
        assert scored == pytest.approx(l._clip_probability(probability), abs=1e-12)


def test_uninformative_agent_has_no_forced_positive_coefficient():
    rows, labels, manifest = _three_class_rows(uninformative_agent=True)
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
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
    rows, labels, manifest = _binary_rows()
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    assert set(fitted.arms["B"]) == {"binary"}
    predictions = l.predict_joint(rows[0], fitted)
    assert {item.arm for item in predictions} == set(l.ARMS)
    for prediction in predictions:
        assert prediction.class_order == ("HC", "AD")
        assert prediction.probabilities is None or len(prediction.probabilities) == 2


def test_invalid_second_pass_falls_back_directly_to_frozen_b():
    rows, labels, manifest = _three_class_rows(invalid_second_pass=0)
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    predictions = {item.arm: item for item in l.predict_joint(rows[0], fitted)}
    assert predictions["J-AS"].status == "fallback"
    assert predictions["J-AS"].fallback_reason == "v1_failed"
    assert predictions["J-AS"].fallback_arm == "B"
    assert predictions["J-AS"].probabilities == predictions["B"].probabilities


def test_three_class_head_combination_and_probability_sum():
    result = l.combine_head_probabilities(("HC", "MCI", "AD"), impairment=0.7, stage=0.25)
    assert result == pytest.approx((0.3, 0.525, 0.175))
    assert sum(result) == pytest.approx(1.0)
    rows, labels, manifest = _three_class_rows()
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    predictions = l.predict_joint(rows[1], fitted)
    assert tuple(item.arm for item in predictions) == l.ARMS
    for prediction in predictions:
        assert prediction.probabilities is not None
        assert sum(prediction.probabilities) == pytest.approx(1.0)
        assert np.isfinite(prediction.probabilities).all()


def test_model_class_permutation_round_trip():
    artifact = l.LinearModelArtifact(
        model_id="permuted_fixture", adapter_implementation="fixture.full_base",
        adapter_version="v1", feature_pipeline_hash="a" * 64,
        adapter_attestation_id=None,
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
    rows, labels, manifest = _binary_rows()

    class Failed:
        success = False
        fun = 1.0
        x = np.asarray(l.PRIOR_COEFFICIENTS)

    monkeypatch.setattr(l, "minimize", lambda *args, **kwargs: Failed())
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
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
    rows, labels, manifest = _three_class_rows()
    original = dict(labels)
    l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    assert labels == original
    assert [labels[row.subject.subject_id] for row in rows].count("MCI") == 6


def test_matched_fit_baselines_use_exact_joint_fit_ids():
    rows, labels, manifest = _three_class_rows(invalid_second_pass=0)
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    for head in ("impairment", "stage"):
        assert (
            fitted.matched_baselines["J-AS"][head].fit_ids
            == fitted.arms["J-AS"][head].fit_ids
        )
    assert rows[0].subject.subject_id not in fitted.arms["J-AS"]["impairment"].fit_ids


def test_matched_fit_baseline_predictions_are_exposed_for_locked_scoring():
    rows, labels, manifest = _three_class_rows(invalid_second_pass=0)
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    predictions = {
        item.arm: item for item in l.predict_matched_baselines(rows[1], fitted)
    }
    assert set(predictions) == {
        "B_matched_J-A", "B_matched_J-S", "B_matched_J-AS",
    }
    for prediction in predictions.values():
        assert prediction.status == "ok"
        assert prediction.probabilities is not None
        assert sum(prediction.probabilities) == pytest.approx(1.0)
        assert fitted.artifact_id in prediction.trace_ids


def test_calibrator_binding_rejects_cross_dataset_source_version_and_channel_reuse():
    rows, labels, manifest = _binary_rows()
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    original = rows[0]
    changed_subject = replace(
        original.subject,
        dataset_id="different_dataset",
        source_version="different_version",
        channel="different_channel",
        partition="holdout",
        fold_id="holdout",
    )
    holdout_row = l.CalibrationRow(
        subject=changed_subject,
        fusion_hash=_hash("cross-dataset"),
        base_probabilities=original.base_probabilities,
        agent_v0_scores=original.agent_v0_scores,
        agent_v1_scores=original.agent_v1_scores,
        v1_required=original.v1_required,
        state_probabilities_before=original.state_probabilities_before,
        state_probabilities_after=original.state_probabilities_after,
        arm_refusal_reasons=original.arm_refusal_reasons,
    )
    with pytest.raises(l.FoldLeakageError, match="dataset/source/channel/task/class binding"):
        l.predict_joint(holdout_row, fitted)


def test_calibrator_serialization_preserves_predictions_within_tolerance():
    rows, labels, manifest = _three_class_rows()
    fitted = l.fit_joint_calibrators(rows, labels, l.CalibrationConfig(), manifest)
    restored = l.JointCalibrators.from_json(fitted.to_json())
    before = l.predict_joint(rows[5], fitted)
    after = l.predict_joint(rows[5], restored)
    assert fitted.artifact_id == restored.artifact_id
    assert restored.dataset_id == fitted.dataset_id
    assert restored.source_version == fitted.source_version
    assert restored.channel == fitted.channel
    assert restored.development_manifest_id == fitted.development_manifest_id
    assert restored.model_recipe_hash == fitted.model_recipe_hash
    assert restored.calibration_recipe_hash == fitted.calibration_recipe_hash
    assert restored.oof_calibration_receipt_ids == fitted.oof_calibration_receipt_ids
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
    rows, labels, manifest = _three_class_rows()
    no_ad_labels = {
        subject_id: ("MCI" if label == "AD" else label) for subject_id, label in labels.items()
    }
    fitted = l.fit_joint_calibrators(rows, no_ad_labels, l.CalibrationConfig(), manifest)
    assert not fitted.arms["B"]["stage"].estimable
    assert fitted.arms["B"]["stage"].refusal_reason == "absent_training_class"
    predictions = {item.arm: item for item in l.predict_joint(rows[0], fitted)}
    assert predictions["B"].status == "unavailable"
    assert predictions["J-AS"].status == "unavailable"
