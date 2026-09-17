from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from advoice.agent_led_evaluation import (
    apply_agent_temperature,
    evaluate_agent_decisions,
    fit_agent_temperature,
)


LABELS = ["HC", "AD", "MCI"]


def decision(case_id="one", predicted="HC", **overrides):
    result = {
        "case_id": case_id,
        "status": "decided",
        "predicted_label": predicted,
        "scores": {"HC": 4, "AD": 0, "MCI": 1},
        "probabilities": None,
        "schema_fingerprint": "schema-v1",
    }
    result.update(overrides)
    return result


def test_final_decision_is_authoritative_and_abstentions_are_noncorrect():
    rows = [decision(), decision("two", "AD"),
            decision("three", None, status="abstained", scores=None),
            decision("four", None, status="provider_disabled", scores=None)]
    original = copy.deepcopy(rows)
    report = evaluate_agent_decisions(
        rows, {"one": "HC", "two": "AD", "three": "AD", "four": "MCI"}, LABELS
    )
    assert report["accuracy"] == 0.5
    assert report["coverage"] == 0.5
    assert report["conditional_accuracy"] == 1.0
    assert report["denominators"] == {
        "accuracy": 4, "coverage": 4, "conditional_accuracy": 2
    }
    assert report["per_class"]["AD"]["support"] == 2
    assert report["per_class"]["AD"]["tp"] == 1
    assert report["per_class"]["AD"]["fn"] == 1
    assert report["per_class"]["AD"]["abstained"] == 1
    assert report["per_class"]["MCI"]["predicted"] == 0
    assert report["status_counts"]["provider_disabled"] == 1
    assert report["probability_metrics"]["log_loss"] is None
    assert rows == original
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("rows,truth", [
    ([decision(), decision()], {"one": "HC"}),
    ([decision()], {"other": "HC"}),
    ([], {"one": "HC"}),
    ([decision()], {}),
    ([decision()], {"one": "unknown"}),
])
def test_invalid_cohorts_fail_closed(rows, truth):
    with pytest.raises(ValueError):
        evaluate_agent_decisions(rows, truth, LABELS)


@pytest.mark.parametrize("update", [
    {"predicted_label": None}, {"predicted_label": "unknown"},
    {"status": "abstained"}, {"status": ""}, {"case_id": 1},
    {"scores": {"HC": 1}},
    {"scores": {"HC": 1, "AD": 1, "MCI": float("nan")}},
    {"scores": {"HC": 1, "AD": 1, "MCI": float("inf")}},
    {"scores": {"HC": 1, "AD": 1, "MCI": 5}},
    {"scores": {"HC": 1, "AD": 1, "MCI": 1.5}},
    {"scores": {"HC": 1, "AD": 1, "MCI": True}},
    {"scores": {"HC": 1, "AD": 1, "MCI": "2"}},
    {"probabilities": {"HC": 0.9, "AD": 0.2, "MCI": 0.1}},
    {"probabilities": {"HC": 1.1, "AD": -0.1, "MCI": 0}},
    {"probabilities": {"HC": 1, "AD": float("nan"), "MCI": 0}},
    {"probabilities": {"HC": 1}}, {"schema_fingerprint": None},
])
def test_malformed_outputs_are_rejected(update):
    with pytest.raises(ValueError):
        evaluate_agent_decisions([decision(**update)], {"one": "HC"}, LABELS)


def test_mixed_schema_fingerprints_are_rejected():
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_agent_decisions(
            [decision(), decision("two", schema_fingerprint="other")],
            {"one": "HC", "two": "AD"}, LABELS,
        )


@pytest.mark.parametrize("labels", [[], ["HC", "HC"], ["HC", 2]])
def test_invalid_class_order_is_rejected(labels):
    with pytest.raises(ValueError):
        evaluate_agent_decisions([], {}, labels)


def test_ranking_auc_uses_raw_complete_scores_and_explicit_subset():
    rows = [decision(),
            decision("two", "AD", scores={"HC": 0, "AD": 4, "MCI": 1}),
            decision("three", "MCI", scores=None)]
    report = evaluate_agent_decisions(
        rows, {"one": "HC", "two": "AD", "three": "MCI"}, LABELS
    )
    ranking = report["ranking_metrics"]
    assert ranking["source"] == "ranking_scores"
    assert ranking["denominator"] == 2
    assert ranking["case_ids"] == ["one", "two"]
    assert ranking["per_class_auroc"] == {"HC": 1.0, "AD": 1.0, "MCI": None}
    assert ranking["macro_auroc_ovr"] is None
    assert report["probability_metrics"]["denominator"] == 0
    json.dumps(report, allow_nan=False)


def test_probabilities_are_supplied_not_derived_and_keep_class_order():
    rows = [decision(probabilities={"MCI": 0.1, "AD": 0.1, "HC": 0.8}),
            decision("two", "AD", scores=None,
                     probabilities={"MCI": 0.1, "AD": 0.7, "HC": 0.2})]
    report = evaluate_agent_decisions(rows, {"one": "HC", "two": "AD"}, LABELS)
    metrics = report["probability_metrics"]
    assert metrics["denominator"] == 2
    assert metrics["log_loss"] == pytest.approx(-np.log([0.8, 0.7]).mean())
    assert metrics["multiclass_brier"] == pytest.approx(0.1)
    assert metrics["source"] == "supplied_probabilities"


def test_zero_true_probability_has_explicit_undefined_log_loss():
    report = evaluate_agent_decisions(
        [decision(probabilities={"HC": 0, "AD": 1, "MCI": 0})], {"one": "HC"}, LABELS
    )
    assert report["probability_metrics"]["log_loss"] is None
    assert report["probability_metrics"]["log_loss_status"] == "infinite_zero_true_probability"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("rows,truth", [([], {}), (
    [decision(predicted=None, status="provider_disabled", scores=None)], {"one": "HC"}
)])
def test_empty_or_no_decisions_is_json_safe(rows, truth):
    report = evaluate_agent_decisions(rows, truth, LABELS)
    assert report["conditional_accuracy"] is None
    assert report["accuracy"] == (0.0 if rows else None)
    assert report["per_class"]["MCI"]["support"] == 0
    assert report["ranking_metrics"]["macro_auroc_ovr"] is None
    json.dumps(report, allow_nan=False)


def test_layer_b_is_structural_and_missing_fields_are_not_zero():
    rows = [decision(trace={"steps": [{}, {}]}, revisions=[{}],
                     clinical_claims=[{"claim": "synthetic"}]), decision("two")]
    report = evaluate_agent_decisions(rows, {"one": "HC", "two": "AD"}, LABELS)
    layer = report["layer_b"]
    assert layer["interpretation"] == "structural_only_not_clinical_efficacy"
    assert layer["trace_length"] == {"total": 2, "denominator": 1, "mean": 2.0}
    assert layer["revisions"] == {"total": 1, "denominator": 1, "mean": 1.0}
    assert layer["clinical_claims"] == {"total": 1, "denominator": 1, "mean": 1.0}


def test_runtime_model_fingerprint_and_state_revisions_contract():
    row = decision(schema_version="agent-led-v1", model_fingerprint="model-v1",
                   state_revisions=2, trace=[{}, {}, {}])
    del row["schema_fingerprint"]
    report = evaluate_agent_decisions([row], {"one": "HC"}, LABELS)
    assert report["schema_fingerprint"] is None
    assert report["schema_version"] == "agent-led-v1"
    assert report["model_fingerprint"] == "model-v1"
    assert report["layer_b"]["revisions"]["total"] == 2


def test_differing_models_are_not_pooled_even_with_matching_schemas():
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_agent_decisions(
            [decision(model_fingerprint="model-one"),
             decision("two", model_fingerprint="model-two")],
            {"one": "HC", "two": "HC"}, LABELS,
        )


def test_runtime_result_evaluates_without_an_adapter():
    from advoice.agent_led import EvidenceSession

    session = EvidenceSession({"case_id": "synthetic"}, LABELS,
                              model_id="synthetic-model", skill_hash="synthetic-skill")
    report = evaluate_agent_decisions(
        [session.finish("provider_disabled")], {"synthetic": "HC"}, LABELS
    )
    assert report["accuracy"] == 0
    assert report["coverage"] == 0
    assert report["layer_b"]["trace_length"]["mean"] == 0
    assert report["layer_b"]["revisions"]["mean"] == 0
    assert report["layer_b"]["clinical_claims"]["mean"] is None


def test_full_per_class_counts_include_false_positives_and_nondecisions():
    rows = [decision(), decision("two", "AD"), decision("three", "HC"),
            decision("four", None, status="abstained", scores=None)]
    report = evaluate_agent_decisions(
        rows, {"one": "HC", "two": "HC", "three": "AD", "four": "MCI"}, LABELS
    )
    assert report["accuracy"] == 0.25
    assert report["conditional_accuracy"] == pytest.approx(1 / 3)
    assert {key: report["per_class"]["HC"][key] for key in ("tp", "fp", "fn", "tn")} == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 1,
    }
    assert report["confusion_matrix"]["counts"] == [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]


def test_tied_complete_ranking_vectors_and_probability_only_outputs():
    rows = [decision(str(i), label, scores=dict.fromkeys(LABELS, 2))
            for i, label in enumerate(LABELS)]
    truth = dict(zip(["0", "1", "2"], LABELS))
    ranking = evaluate_agent_decisions(rows, truth, LABELS)["ranking_metrics"]
    assert ranking["macro_auroc_ovr"] == 0.5
    assert ranking["macro_class_denominator"] == 3
    for row in rows:
        row["scores"] = None
        row["probabilities"] = dict.fromkeys(LABELS, 1 / 3)
    report = evaluate_agent_decisions(rows, truth, LABELS)
    assert report["ranking_metrics"]["denominator"] == 0
    assert report["ranking_metrics"]["macro_auroc_ovr"] is None
    assert report["probability_metrics"]["denominator"] == 3


@pytest.mark.parametrize("field,value", [
    ("trace", {}), ("trace", "invalid"), ("revisions", -1),
    ("revisions", True), ("clinical_claims", float("nan")),
])
def test_malformed_structural_metadata_is_not_silently_counted(field, value):
    with pytest.raises(ValueError):
        evaluate_agent_decisions([decision(**{field: value})], {"one": "HC"}, LABELS)


def fit_synthetic(**overrides):
    kwargs = dict(
        scores=np.array([[4, 0, 1], [4, 0, 1], [0, 4, 1], [0, 1, 4]]),
        y=["HC", "AD", "AD", "MCI"], case_ids=["a", "b", "c", "d"],
        labels=LABELS, model_fingerprint="frozen-model-v1",
        training_case_ids=["train-only"],
    )
    kwargs.update(overrides)
    return fit_agent_temperature(**kwargs)


def test_temperature_artifact_and_argmax_are_preserved_on_disjoint_cases():
    artifact = fit_synthetic()
    assert artifact["temperature"] > 0
    assert artifact["fit_case_ids"] == ["a", "b", "c", "d"]
    assert artifact["class_order"] == LABELS
    assert artifact["model_fingerprint"] == "frozen-model-v1"
    assert artifact["fit_log_loss"] <= artifact["uncalibrated_log_loss"]
    scores = np.array([[4, 0, 1], [0, 4, 1], [0, 1, 4], [1, 1, 1]])
    p = apply_agent_temperature(scores, artifact, ["e", "f", "g", "h"],
                                "frozen-model-v1", labels=LABELS)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_array_equal(p.argmax(axis=1), scores.argmax(axis=1))
    json.dumps(artifact, allow_nan=False)


@pytest.mark.parametrize("overrides", [
    {"partition": "test"}, {"partition": "training"},
    {"case_ids": ["a", "a", "c", "d"]}, {"training_case_ids": ["a"]},
    {"y": ["HC", "HC", "AD", "bad"]}, {"y": ["HC"] * 4},
    {"model_fingerprint": ""}, {"scores": np.ones((3, 3))},
    {"scores": np.full((4, 3), np.inf)},
])
def test_invalid_calibration_contract_is_rejected(overrides):
    with pytest.raises(ValueError):
        fit_synthetic(**overrides)


@pytest.mark.parametrize("ids,fingerprint,labels", [
    (["a"], "frozen-model-v1", LABELS),
    (["e"], "other", LABELS),
    (["e"], "frozen-model-v1", list(reversed(LABELS))),
    (["e", "e"], "frozen-model-v1", LABELS),
])
def test_application_rejects_overlap_fingerprint_and_class_order(ids, fingerprint, labels):
    with pytest.raises(ValueError):
        apply_agent_temperature(np.ones((len(ids), 3)), fit_synthetic(), ids,
                                fingerprint, labels=labels)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True, "1"])
def test_application_rejects_invalid_temperature(temperature):
    artifact = fit_synthetic()
    artifact["temperature"] = temperature
    with pytest.raises(ValueError):
        apply_agent_temperature(np.ones((1, 3)), artifact, ["e"],
                                "frozen-model-v1", labels=LABELS)
