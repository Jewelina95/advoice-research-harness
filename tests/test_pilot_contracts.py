"""Offline tests for the versioned pilot boundaries."""
from dataclasses import FrozenInstanceError, replace
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from advoice.evidence import EvidenceProvenance, MetricEvidenceV2
from advoice.state_graph import build_state_graph_frame


@pytest.fixture
def c():
    return importlib.import_module("advoice.pilot.contracts")


def subject(c, **changes):
    return c.SubjectRow(**{
        "dataset_id": "fixture", "subject_id": "sub_0123456789abcdef",
        "partition": "development", "task": "hc_mci_ad",
        "class_order": ("HC", "MCI", "AD"),
        "source_group_id": "grp_0123456789abcdef", "channel": "picture_description",
        "language": "en", "task_ids": ("picture_description",), "role": "participant",
        "fold_id": "fold_0", "raw_hashes": {"asset_0123456789abcdef": "d" * 64},
        "source_version": "fixture_v1",
        **changes,
    })


def snapshot(c, value=0.0, **changes):
    evidence = MetricEvidenceV2(
        evidence_id="e001", metric_id="pause", state_id="timing",
        subject_id="sub_0123456789abcdef", value=value,
    )
    card = c.StateCardBody(body={
        "state_id": "timing", "state_z": value, "available": value is not None,
        "supporting_evidence_ids": ["e001"], "counter_evidence_ids": [],
        "provenance_trace": [{"source_segment_ids": ["seg_0123456789abcdef"],
                               "method_version": "fixture_v1"}],
        "confidence": 0.7, "raw_state_z": value, "task_scope": "overall",
    })
    return c.EvidenceSnapshot(**{
        "subject": subject(c), "case_id": "case_0123456789abcdef", "state_version": "state_v1",
        "reference_fit_id": "ref_0", "reference_fit_hash": "a" * 64,
        "evidence": (c.MetricEvidenceBody.from_evidence(evidence),),
        "state_cards": (card,), "source_segments": (c.SourceSegment(
            segment_id="seg_0123456789abcdef", source_asset_id="asset_0123456789abcdef",
            start_seconds=0.0, end_seconds=10.0, role="participant"),),
        "observability": {"timing": c.Observability(
            status="observed" if value is not None else "missing",
            reason=None if value is not None else "measurement_missing")},
        "confounds": {"potential": (), "observed": (), "ruled_out": ()},
        "skill_hash": "a" * 64, "extractor_hash": "b" * 64, **changes,
    })


def usage(c, **changes):
    return c.UsageRow(**{
        "request_id": "req_001", "cache_key": "d" * 64, "cache_status": "miss", "attempt": 1,
        "response_status": "ok",
        "model_id": "fixture-model-2026-01-01", "input_tokens": None,
        "output_tokens": None, "reasoning_tokens": None, "cost_usd": None,
        "wall_seconds": None, "response_id": None,
        "unavailable_reasons": {key: "provider_not_reported" for key in (
            "input_tokens", "output_tokens", "reasoning_tokens", "cost_usd", "wall_seconds", "response_id")},
        **changes,
    })


def assessment(c, snap, **changes):
    usage_row = changes.get("usage", usage(c))
    observed = snap.observability["timing"].status == "observed"
    payload = {
        "subject_id": snap.subject.subject_id, "case_id": snap.case_id, "task": snap.subject.task,
        "class_order": snap.subject.class_order, "snapshot_hash": snap.snapshot_hash,
        "revision": "v0", "ordinal_scores": {key: 2 for key in snap.subject.class_order},
        "citations": ("e001",), "state_judgments": (
            c.StateJudgment(state_id="timing", status=snap.observability["timing"].status,
                            score=2 if observed else None,
                            citations=("e001",) if observed else (),
                            reason=None if observed else snap.observability["timing"].reason),
        ), "operations": (), "model_id": "fixture-model-2026-01-01",
        "usage": usage_row, "usage_id": usage_row.content_hash, "status": "ok",
    }
    payload.update(changes)
    return c.AgentAssessment(**payload)


def operation(c, **changes):
    return c.ReviewOperation(**{
        "operation_id": "op_0", "state_id": "timing", "action": "deterministic_remeasurement",
        "citations": ("e001",), "source_segment_ids": ("seg_0123456789abcdef",),
        "reason": "reviewed", "method_hash": "c" * 64, **changes,
    })


def replay_result(c, *, changed=True, accepted=True, **changes):
    before, after = snapshot(c), snapshot(c, 1.0 if changed else 0.0)
    accepted_ops = (operation(c),) if accepted else ()
    payload = {
        "before": before, "after": after, "parent_hash": before.snapshot_hash,
        "child_hash": after.snapshot_hash, "accepted_ops": accepted_ops,
        "rejected_ops": (), "invalidated_ids": (), "recomputed_ids": ("e001",) if accepted else (),
        "before_state_scores": {"timing": 0.0}, "after_state_scores": {"timing": 1.0 if changed else 0.0},
        "state_delta": {"timing": 1.0 if changed else 0.0}, "replay_model_id": "deterministic_replay_v1",
        "repair_batch_id": "repair_0" if accepted else None,
        "executor_registry_hash": "e" * 64,
        "authorized_operations": {item.operation_id: item.content_hash for item in accepted_ops},
        "valid": True,
    }
    payload.update(changes)
    return c.ReplayResult(**payload)


def arm(c, reason=None, fallback_arm=None):
    return c.ArmAvailability(available=reason is None, valid=reason is None,
                             fallback_reason=reason, fallback_arm=fallback_arm)


def fusion(c, *, changed=True, accepted=True, **changes):
    replay = changes.pop("replay", replay_result(c, changed=changed, accepted=accepted))
    meaningful = any(getattr(replay.before, key) != getattr(replay.after, key) for key in (
        "evidence", "state_cards", "observability", "confounds", "source_segments"))
    replay_reason = (
        "replay_failed" if not replay.valid else
        "replay_inputs_missing" if all(
            value is None for values in (
                replay.before_state_scores, replay.after_state_scores, replay.state_delta,
            ) for value in values.values()
        ) else None
    )
    proposed = replay.accepted_ops + tuple(item.operation for item in replay.rejected_ops)
    payload = {
        "subject": replay.before.subject, "dataset_id": "fixture", "subject_id": replay.before.subject.subject_id,
        "partition": "development", "fold_id": "fold_0", "base_fit_id": "base_0",
        "base_fit_hash": "b" * 64, "reference_fit_id": "ref_0", "reference_fit_hash": "a" * 64,
        "base_probabilities": (0.2, 0.3, 0.5),
        "assessment_v0": assessment(c, replay.before, operations=proposed),
        "assessment_v1": assessment(c, replay.after, revision="v1") if meaningful else None,
        "replay": replay, "parent_snapshot_hash": replay.parent_hash, "child_snapshot_hash": replay.child_hash,
        "replay_delta": replay.state_delta, "v1_required": meaningful,
        "j_a": arm(c), "j_s": arm(c, replay_reason, "B" if replay_reason else None),
        "j_as": arm(c, replay_reason, "B" if replay_reason else None),
        "validity": "partial" if replay_reason else "valid",
    }
    payload.update(changes)
    return c.FusionRow(**payload)


def prediction(c, **changes):
    return c.PredictionRow(**{
        "subject": subject(c), "fusion_hash": "a" * 64, "arm": "J-AS",
        "class_order": ("HC", "MCI", "AD"), "probabilities": (0.2, 0.3, 0.5),
        "predicted": "AD", "status": "ok", "calibrator_id": "calibrator_0",
        "trace_ids": ("trace_0",), **changes,
    })


def test_contract_module_is_available():
    assert importlib.util.find_spec("advoice.pilot") is not None


@pytest.mark.parametrize("partition", ["engineering_canary", "development", "holdout", "stress"])
def test_partition_contract(c, partition):
    assert subject(c, partition=partition).partition == partition


@pytest.mark.parametrize("partition", ["train", "test", "val", "HC", "", None])
def test_other_partitions_rejected(c, partition):
    with pytest.raises(c.PilotContractError):
        subject(c, partition=partition)


@pytest.mark.parametrize("key,value", [
    ("label", "AD"), ("diagnosis_truth", "AD"), ("groundTruth", "AD"),
    ("split_truth", "AD"), ("base_probabilities", [0.1, 0.2, 0.7]),
    ("baseProbs", {"AD": 0.9}), ("patient_name", "Jane Doe"),
    ("raw_path", "/private/HC/sample.wav"), ("transcript", "private speech"),
])
def test_recursive_inference_boundary_rejects_private_fields(c, key, value):
    with pytest.raises(c.PilotContractError):
        c.safe_inference_payload({"state_judgments": [{key: value}]})
    payload = assessment(c, snapshot(c)).to_dict()
    payload["state_judgments"][0][key] = value
    with pytest.raises(c.PilotContractError):
        c.AgentAssessment.from_provider_payload(payload, snapshot(c))


@pytest.mark.parametrize("value", ["/data/AD/a.wav", "HC/a.wav", "C:\\data\\MCI\\a.wav",
                                   "../raw/audio.wav", "Jane Doe", "person@example.org"])
def test_unsafe_values_rejected_even_under_safe_keys(c, value):
    with pytest.raises(c.PilotContractError):
        c.safe_inference_payload({"reason": value})


def test_provider_unknown_fields_and_non_pseudonymous_identity_rejected(c):
    payload = assessment(c, snapshot(c)).to_dict()
    payload["extra"] = {"secret": "value"}
    with pytest.raises(c.PilotContractError):
        c.AgentAssessment.from_provider_payload(payload, snapshot(c))
    with pytest.raises(c.PilotContractError):
        subject(c, subject_id="patient_John")


@pytest.mark.parametrize("task,order", [
    ("hc_mci_ad", ("AD", "MCI", "HC")), ("hc_ad", ("HC", "MCI", "AD")),
    ("hc_mci_ad", ("HC", "AD")), ("hc_impairment", ("HC", "AD")),
    ("ad_stage", ("HC", "MCI", "AD")),
])
def test_task_class_order_is_exact(c, task, order):
    with pytest.raises(c.PilotContractError):
        subject(c, task=task, class_order=order)


@pytest.mark.parametrize("task,order", [("hc_ad", ("HC", "AD")),
                                        ("hc_impairment", ("HC", "IMPAIRED"))])
def test_binary_tasks_do_not_invent_mci_or_ad(c, task, order):
    snap = snapshot(c, subject=subject(c, task=task, class_order=order))
    row = assessment(c, snap)
    row.validate_snapshot(snap)
    assert tuple(row.ordinal_scores) == order
    with pytest.raises(c.PilotContractError):
        replace(row, ordinal_scores={"HC": 1, "MCI": 2, "AD": 3})


@pytest.mark.parametrize("status", ["not_observable", "missing", "conflicted"])
def test_unavailable_state_is_not_neutral(c, status):
    missing = c.StateJudgment(state_id="timing", status=status, score=None, reason="insufficient_evidence")
    neutral = c.StateJudgment(state_id="timing", status="observed", score=2, citations=("e001",))
    assert missing.to_dict()["score"] is None
    assert neutral.to_dict()["score"] == 2
    with pytest.raises(c.PilotContractError):
        replace(missing, score=2)
    with pytest.raises(c.PilotContractError):
        replace(missing, reason=None)
    with pytest.raises(c.PilotContractError):
        replace(neutral, score=None)


@pytest.mark.parametrize("score", [-1, 5, 0.5, True, None])
def test_assessment_scores_are_ordinals(c, score):
    with pytest.raises(c.PilotContractError):
        assessment(c, snapshot(c), ordinal_scores={"HC": score, "MCI": 2, "AD": 2})


def test_assessment_stale_hash_citations_and_state_ids(c):
    snap = snapshot(c)
    row = assessment(c, snap)
    row.validate_snapshot(snap)
    with pytest.raises(c.PilotContractError, match="snapshot"):
        row.validate_snapshot(snapshot(c, 1.0))
    for changes in ({"citations": ("unknown",)}, {"state_judgments": ()},
                    {"model_id": "different-model-2026-01-01"}):
        with pytest.raises(c.PilotContractError):
            replace(row, **changes).validate_snapshot(snap)
    with pytest.raises(c.PilotContractError):
        replace(snap, snapshot_hash="f" * 64)
    with pytest.raises(c.PilotContractError):
        replace(row, ad_stage="mild")


@pytest.mark.parametrize("revision", ["v0", "v1"])
def test_state_judgment_status_must_match_snapshot_observability(c, revision):
    snap = snapshot(c, None)
    row = assessment(
        c, snap, revision=revision,
        state_judgments=(c.StateJudgment(
            state_id="timing", status="not_observable", score=None,
            reason="task_not_applicable"),),
    )
    with pytest.raises(c.PilotContractError):
        row.validate_snapshot(snap)


def test_max_three_operations(c):
    operations = tuple(operation(c, operation_id=f"op_{i}") for i in range(4))
    with pytest.raises(c.PilotContractError):
        assessment(c, snapshot(c), operations=operations)


def test_metric_domain_rules_reused_and_raw_provenance_rejected(c):
    body = snapshot(c).evidence[0]
    payload = body.to_dict()
    payload["body"]["consumed_by_supervised"] = True
    payload["body"]["incremental_for_agent"] = True
    with pytest.raises(c.PilotContractError):
        c.MetricEvidenceBody.from_mapping(payload)
    private = MetricEvidenceV2(evidence_id="e001", metric_id="pause",
                               provenance=EvidenceProvenance(source_asset_id="/data/AD/raw.wav"))
    with pytest.raises(c.PilotContractError):
        c.MetricEvidenceBody.from_evidence(private)
    serialized = snapshot(c).to_inference_dict()
    assert "reference_label" not in json.dumps(serialized)
    assert "transcript" not in json.dumps(serialized)


@pytest.mark.parametrize("field,value", [("direction", True), ("observable", "false"),
                                        ("predicted_class", "AD")])
def test_metric_serialized_body_cannot_coerce_or_discard_fields(c, field, value):
    payload = snapshot(c).evidence[0].to_dict()
    payload["body"][field] = value
    with pytest.raises(c.PilotContractError):
        c.MetricEvidenceBody.from_mapping(payload)


@pytest.mark.parametrize("field,value", [
    ("subject_id", 123), ("case_id", "Jane_Doe"), ("session_id", "John_visit_1"),
    ("source_asset_id", "patient_123"), ("source_segment_ids", ["John_segment_1"]),
])
def test_auxiliary_identity_fields_require_pseudonyms(c, field, value):
    with pytest.raises(c.PilotContractError):
        c.safe_inference_payload({field: value})


def test_pseudonymous_provenance_is_allowed(c):
    evidence = MetricEvidenceV2(
        evidence_id="e001", metric_id="pause", subject_id="sub_0123456789abcdef",
        case_id="case_0123456789abcdef", session_id="ses_0123456789abcdef",
        provenance=EvidenceProvenance(source_asset_id="asset_0123456789abcdef",
                                      source_segment_ids=("seg_0123456789abcdef",)),
    )
    assert c.MetricEvidenceBody.from_evidence(evidence).to_inference_dict()


def test_deep_immutability_and_deterministic_roundtrip(c):
    rows = [subject(c), snapshot(c), usage(c), assessment(c, snapshot(c)), fusion(c)]
    rows.append(rows[-1].replay)
    rows.append(prediction(c, fusion_hash=rows[-2].content_hash))
    for row in rows:
        restored = type(row).from_json(row.to_json())
        assert restored == row
        assert restored.to_json() == row.to_json()
        assert restored.content_hash == row.content_hash
        with pytest.raises(FrozenInstanceError):
            row.schema_version = "changed"
    with pytest.raises(TypeError):
        rows[1].evidence[0].body["value"] = 123
    with pytest.raises(TypeError):
        rows[3].ordinal_scores["HC"] = 0
    assert snapshot(c).snapshot_hash != snapshot(c, 2.0).snapshot_hash
    with pytest.raises(c.PilotContractError):
        c.SubjectRow.from_mapping({**subject(c).to_dict(), "schema_version": "v99"})


def test_records_are_hashable_by_canonical_content(c):
    rows = (subject(c), snapshot(c), usage(c), assessment(c, snapshot(c)), fusion(c), prediction(c))
    for row in rows:
        assert hash(row) == hash(type(row).from_json(row.to_json()))


def test_record_equality_and_hash_use_canonical_json_types(c):
    base = snapshot(c).state_cards[0].to_state_card()
    integer = c.StateCardBody(body={**base, "extension_count": 1})
    floating = c.StateCardBody(body={**base, "extension_count": 1.0})
    assert integer != floating
    assert hash(integer) != hash(floating)
    restored = c.StateCardBody.from_json(integer.to_json())
    assert restored == integer and hash(restored) == hash(integer)


def test_from_mapping_requires_schema_version_and_json_rejects_duplicate_keys(c):
    rows = (subject(c), snapshot(c), usage(c), assessment(c, snapshot(c)),
            replay_result(c), fusion(c), prediction(c), resolved_config(c))
    for row in rows:
        payload = row.to_dict()
        del payload["schema_version"]
        with pytest.raises(c.PilotContractError):
            type(row).from_mapping(payload)
    duplicate = '{"schema_version":"advoice.pilot.v1","schema_version":"advoice.pilot.v1"}'
    with pytest.raises(c.PilotContractError, match="Duplicate"):
        c.SubjectRow.from_json(duplicate)


def test_usage_null_requires_reason_and_reported_zero_is_distinct(c):
    row = usage(c)
    assert row.input_tokens is row.output_tokens is row.cost_usd is None
    for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cost_usd", "wall_seconds"):
        reasons = {key: value for key, value in row.unavailable_reasons.items() if key != field}
        with pytest.raises(c.PilotContractError):
            replace(row, unavailable_reasons=reasons)
        with pytest.raises(c.PilotContractError):
            replace(row, **{field: 0})
        assert getattr(replace(row, **{field: 0, "unavailable_reasons": reasons}), field) == 0
        with pytest.raises(c.PilotContractError):
            replace(row, **{field: -1, "unavailable_reasons": reasons})
    with pytest.raises(c.PilotContractError):
        usage(c, input_tokens=True)


def test_fusion_binds_replay_fits_revisions_and_validity(c):
    row = fusion(c)
    assert "label" not in row.to_fusion_dict()
    with pytest.raises(c.PilotContractError):
        row.to_inference_dict()
    with pytest.raises(c.PilotContractError):
        row.to_publishable_dict()
    changes = [
        {"fold_id": "fold_1"}, {"reference_fit_hash": "f" * 64},
        {"subject": subject(c, partition="holdout")},
        {"assessment_v1": row.assessment_v0},
        {"assessment_v0": replace(row.assessment_v0, snapshot_hash="c" * 64)},
    ]
    for change in changes:
        with pytest.raises(c.PilotContractError):
            replace(row, **change)
    with pytest.raises(c.PilotContractError):
        replace(row.replay, state_delta={"timing": 0.0})
    missing = snapshot(c, None)
    replay = replay_result(c, changed=False, accepted=False, before=missing, after=missing,
                           parent_hash=missing.snapshot_hash, child_hash=missing.snapshot_hash,
                           before_state_scores={"timing": None}, after_state_scores={"timing": None},
                           state_delta={"timing": None})
    assert replay.state_delta["timing"] is None


@pytest.mark.parametrize("changes", [
    {"probabilities": (0.2, 0.8)}, {"probabilities": (0.2, 0.2, 0.2)},
    {"probabilities": (float("nan"), 0.0, 1.0)}, {"predicted": "MCI"},
    {"ad_stage": "moderate"},
])
def test_prediction_protocol(c, changes):
    with pytest.raises(c.PilotContractError):
        prediction(c, **changes)


def config_payload():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs/pilot/evidence_state_v1.yaml").read_text())


def resolved_config(c):
    payload = config_payload()
    payload.update(skill_hash="a" * 64, prompt_hash="b" * 64,
                   provider="fixture", model_id="fixture-model-2026-01-01",
                   pricing={"schema_version": "advoice.pilot.v1", "source": "fixture_pricing", "date": "2026-09-24",
                            "input_usd_per_million": 1.0, "output_usd_per_million": 2.0,
                            "reasoning_usd_per_million": 2.0},
                   dataset_manifests=[{"schema_version": "advoice.pilot.v1", "dataset_id": "fixture", "manifest_id": "manifest_0",
                                       "manifest_hash": "c" * 64, "task": "hc_ad",
                                       "class_order": ["HC", "AD"]}],
                   routes=[{"schema_version": "advoice.pilot.v1", "dataset_id": "fixture", "observation_route": "picture_description",
                            "target_route": "diagnosis"}])
    return c.ResolvedConfig.from_mapping(payload)


def test_config_disabled_by_default_and_requires_resolution(c):
    template = config_payload()
    assert template["paid_execution"] is False
    with pytest.raises(c.PilotContractError):
        c.ResolvedConfig.from_mapping(template)
    row = resolved_config(c)
    assert row.paid_execution is False
    payload = row.to_dict()
    del payload["paid_execution"]
    assert c.ResolvedConfig.from_mapping(payload).paid_execution is False
    assert c.ResolvedConfig.from_json(row.to_json()) == row
    for changes in ({"model_id": "gpt-latest"}, {"routes": ()},
                    {"skill_hash": ""}, {"paid_execution": "false"}):
        with pytest.raises(c.PilotContractError):
            replace(row, **changes)
    with pytest.raises(c.PilotContractError):
        replace(row.limits, max_usd=-1)
    with pytest.raises(c.PilotContractError):
        replace(row.limits, concurrency=0)


def test_v1_omitted_for_noop_and_not_forced_by_accepted_noop(c):
    for accepted in (False, True):
        row = fusion(c, changed=False, accepted=accepted)
        assert row.assessment_v1 is None and not row.v1_required
        assert row.j_s.valid and row.j_as.valid
        with pytest.raises(c.PilotContractError):
            replace(row, v1_required=True)
        with pytest.raises(c.PilotContractError):
            replace(row, assessment_v1=assessment(c, row.replay.after, revision="v1"))


@pytest.mark.parametrize("failure", ["absent", "failed"])
def test_required_v1_fallback_does_not_disable_j_s(c, failure):
    row = fusion(c)
    v1 = None if failure == "absent" else assessment(
        c, row.replay.after, revision="v1", status="failed", failure_reason="transport_failure",
        ordinal_scores=None, citations=(), state_judgments=(), operations=())
    reason = "v1_missing" if failure == "absent" else "v1_failed"
    fallback = replace(row, assessment_v1=v1, j_as=arm(c, reason, "B"), validity="partial")
    assert fallback.v1_required and fallback.j_s.available and fallback.j_s.valid
    assert not fallback.j_as.valid and fallback.j_as.fallback_reason == reason
    assert c.FusionRow.from_json(fallback.to_json()) == fallback
    with pytest.raises(c.PilotContractError):
        replace(fallback, j_as=arm(c))
    with pytest.raises(c.PilotContractError):
        replace(fallback, j_s=arm(c, reason, "B"))
    with pytest.raises(c.PilotContractError):
        replace(fallback, validity="valid")


def test_optional_v0_and_failed_replay_have_explicit_independent_arms(c):
    row = fusion(c, changed=False, accepted=False)
    missing = replace(row, assessment_v0=None, j_a=arm(c, "v0_missing", "B"),
                      j_as=arm(c, "v0_missing", "B"), validity="partial")
    assert missing.j_s.valid and not missing.j_a.valid and not missing.j_as.valid
    failed = replace(row.replay, valid=False, reason="transport_failure")
    result = replace(row, replay=failed, validity="partial", j_s=arm(c, "replay_failed", "B"),
                     j_as=arm(c, "replay_failed", "B"))
    assert result.j_a.valid and not result.j_s.valid and result.j_s.fallback_arm == "B"
    with pytest.raises(c.PilotContractError):
        replace(row, assessment_v0=None)


@pytest.mark.parametrize("action", ["retain", "downweight", "invalidate", "mark_unavailable"])
def test_old_authority_operations_are_not_pilot_operations(c, action):
    with pytest.raises(c.PilotContractError):
        operation(c, action=action)


def test_v1_has_zero_operations_even_if_failed(c):
    for status in ("ok", "failed"):
        with pytest.raises(c.PilotContractError):
            assessment(c, snapshot(c), revision="v1", operations=(operation(c),), status=status)


@pytest.mark.parametrize("change", [
    {"method_hash": None}, {"source_segment_ids": ()},
    {"source_segment_ids": ("seg_ffffffffffffffff",)},
    {"action": "source_role_span_correction", "method_hash": None},
    {"action": "source_role_span_correction", "method_hash": None, "corrected_role": "patient"},
    {"action": "source_role_span_correction", "method_hash": None, "span_start": 0.0, "span_end": 20.0},
    {"action": "unsupported_interpretation_flag", "method_hash": None},
    {"action": "confound_flag", "method_hash": None},
])
def test_operation_requires_determinism_or_source_backing(c, change):
    with pytest.raises(c.PilotContractError):
        assessment(c, snapshot(c), operations=(operation(c, **change),)).validate_snapshot(snapshot(c))


@pytest.mark.parametrize("change", [
    {"action": "source_role_span_correction", "method_hash": None, "corrected_role": "examiner"},
    {"action": "source_role_span_correction", "method_hash": None, "span_start": 2.0, "span_end": 8.0},
    {"action": "unsupported_interpretation_flag", "method_hash": None, "flag_code": "unsupported_claim"},
    {"action": "confound_flag", "method_hash": None, "flag_code": "overlapping_speech"},
])
def test_supported_operations_roundtrip(c, change):
    row = operation(c, **change)
    assessment(c, snapshot(c), operations=(row,)).validate_snapshot(snapshot(c))
    assert c.ReviewOperation.from_json(row.to_json()) == row


@pytest.mark.parametrize("record,required", [
    ("subject", ("dataset_id", "subject_id", "source_group_id", "channel", "language", "task_ids", "role",
                 "partition", "fold_id", "raw_hashes", "source_version")),
    ("snapshot", ("case_id", "state_version", "reference_fit_id", "source_segments", "evidence",
                  "state_cards", "observability", "confounds", "skill_hash", "extractor_hash")),
    ("assessment", ("subject_id", "case_id", "snapshot_hash", "revision", "ordinal_scores",
                    "operations", "model_id", "usage", "usage_id", "status")),
    ("replay", ("parent_hash", "child_hash", "accepted_ops", "rejected_ops", "invalidated_ids",
                "recomputed_ids", "before_state_scores", "after_state_scores", "replay_model_id",
                "repair_batch_id", "executor_registry_hash", "authorized_operations")),
    ("fusion", ("subject_id", "dataset_id", "partition", "fold_id", "base_fit_id", "reference_fit_id",
                "base_probabilities", "replay_delta", "validity", "parent_snapshot_hash", "child_snapshot_hash",
                "j_a", "j_s", "j_as", "v1_required")),
    ("prediction", ("arm", "class_order", "probabilities", "predicted", "status", "calibrator_id", "trace_ids")),
    ("usage", ("request_id", "cache_key", "cache_status", "response_status", "model_id", "attempt", "input_tokens",
               "output_tokens", "reasoning_tokens", "cost_usd", "wall_seconds", "response_id", "unavailable_reasons")),
])
def test_required_record_fields_cannot_be_omitted(c, record, required):
    row = {"subject": subject, "snapshot": snapshot,
           "assessment": lambda module: assessment(module, snapshot(module)), "replay": replay_result,
           "fusion": fusion, "prediction": prediction, "usage": usage}[record](c)
    for field in required:
        payload = row.to_dict()
        del payload[field]
        with pytest.raises(c.PilotContractError):
            type(row).from_mapping(payload)


def test_statecard_adapter_preserves_full_body_and_nested_provenance(c):
    payload = {
        **snapshot(c).state_cards[0].to_state_card(), "family_support_count": 3,
        "correlation_families": {"timing": {"reliability": 0.6}},
        "future_state_metadata": {"some_new_field": [1, 2, 3]},
    }
    card = c.StateCardBody.from_state_card(payload)
    assert card.to_state_card() == payload
    assert c.StateCardBody.from_json(card.to_json()).to_state_card() == payload
    payload["future_state_metadata"]["some_new_field"].append(4)
    assert card.to_state_card()["future_state_metadata"]["some_new_field"] == [1, 2, 3]
    with pytest.raises(TypeError):
        card.body["provenance_trace"][0]["method_version"] = "changed"
    class ExistingCard:
        def to_dict(self):
            return card.to_state_card()
    assert c.StateCardBody.from_state_card(ExistingCard()).to_state_card() == card.to_state_card()
    with pytest.raises(c.PilotContractError):
        card.to_inference_dict()  # Unknown extension is preserved locally, never silently published.


@pytest.mark.parametrize("field,value", [("label", "AD"), ("raw_path", "/private/AD/a.wav"),
                                        ("transcript", "raw speech"), ("base_probabilities", [0.1, 0.9])])
def test_lossless_cards_do_not_bypass_privacy(c, field, value):
    payload = snapshot(c).state_cards[0].to_state_card()
    payload["provenance_trace"][0][field] = value
    card = c.StateCardBody.from_state_card(payload)
    assert card.to_state_card() == payload
    for row in (card, snapshot(c, state_cards=(card,))):
        for serializer in (row.to_inference_dict, row.to_publishable_dict):
            with pytest.raises(c.PilotContractError):
                serializer()


@pytest.mark.parametrize("field", [
    "predicted", "predicted_class", "calibrator", "calibrator_id", "fusion",
    "fusion_hash", "arm", "base_fit_id", "base_prediction", "base_predictions",
])
def test_state_card_extensions_cannot_smuggle_prediction_or_fusion_fields(c, field):
    payload = snapshot(c).state_cards[0].to_state_card()
    payload["provenance_trace"][0][field] = "opaque_value"
    card = c.StateCardBody.from_state_card(payload)
    assert card.to_state_card()["provenance_trace"][0][field] == "opaque_value"
    with pytest.raises(c.PilotContractError):
        card.to_inference_dict()


@pytest.mark.parametrize("change", [
    {"parent_hash": "f" * 64}, {"child_hash": "f" * 64},
    {"accepted_ops": ()}, {"invalidated_ids": ("unknown",)}, {"recomputed_ids": ("unknown",)},
    {"before_state_scores": {"timing": 5.0}}, {"after_state_scores": {"timing": None}},
    {"replay_model_id": ""},
])
def test_replay_cannot_claim_unbound_changes(c, change):
    with pytest.raises(c.PilotContractError):
        replay_result(c, **change)


def test_rejected_operations_are_disjoint_and_do_not_require_v1(c):
    rejected = c.RejectedOperation(operation=operation(c), reason="source_not_sufficient")
    replay = replay_result(c, changed=False, accepted=False, rejected_ops=(rejected,))
    row = fusion(c, changed=False, accepted=False, replay=replay,
                 assessment_v0=assessment(c, replay.before, operations=(operation(c),)))
    assert not row.v1_required
    with pytest.raises(c.PilotContractError):
        replace(replay, accepted_ops=(operation(c),))
    with pytest.raises(c.PilotContractError):
        replace(rejected, reason="")


def _replay_between(c, before, after, op, **changes):
    before_scores = {card.state_id: card.state_z for card in before.state_cards}
    after_scores = {card.state_id: card.state_z for card in after.state_cards}
    delta = {key: (None if before_scores[key] is None or after_scores[key] is None
                   else after_scores[key] - before_scores[key]) for key in before_scores}
    payload = dict(
        before=before, after=after, parent_hash=before.snapshot_hash,
        child_hash=after.snapshot_hash, accepted_ops=(op,), rejected_ops=(),
        recomputed_ids=(("e001",) if op.action in {
            "deterministic_remeasurement", "source_role_span_correction",
        } else ()), before_state_scores=before_scores,
        after_state_scores=after_scores, state_delta=delta, repair_batch_id="repair_0",
        authorized_operations={op.operation_id: op.content_hash},
    )
    payload.update(changes)
    return replay_result(c, **payload)


def test_replay_meaningful_change_covers_every_mutable_snapshot_component(c):
    base = snapshot(c)
    evidence_body = base.evidence[0].to_dict()
    evidence_body["body"]["value"] = 1.0
    evidence_after = replace(base, evidence=(c.MetricEvidenceBody.from_mapping(evidence_body),),
                             snapshot_hash="")

    card_body = base.state_cards[0].to_state_card()
    card_body["confidence"] = 0.8
    card_after = replace(base, state_cards=(c.StateCardBody(body=card_body),), snapshot_hash="")

    unavailable = snapshot(c, None)
    observation_after = replace(
        unavailable,
        observability={"timing": c.Observability(status="conflicted", reason="source_conflict")},
        snapshot_hash="",
    )
    confound_after = replace(base, confounds={"potential": ("overlapping_speech",),
                                              "observed": (), "ruled_out": ()}, snapshot_hash="")
    segment_after = replace(base, source_segments=(replace(base.source_segments[0], role="examiner"),),
                            snapshot_hash="")

    cases = (
        (base, evidence_after, operation(c)),
        (base, card_after, operation(c)),
        (unavailable, observation_after, operation(c)),
        (base, confound_after, operation(c, action="confound_flag", method_hash=None,
                                         flag_code="overlapping_speech")),
        (base, segment_after, operation(c, action="source_role_span_correction",
                                        method_hash=None, corrected_role="examiner")),
    )
    for before, after, op in cases:
        replay = _replay_between(c, before, after, op)
        assert replay.meaningful_change
        with pytest.raises(c.PilotContractError):
            fusion(c, replay=replay, assessment_v1=None)
        assert fusion(c, replay=replay).v1_required


def test_replay_operation_type_authorizes_the_changed_domain(c):
    before = snapshot(c)
    changed_segment = replace(
        before, source_segments=(replace(before.source_segments[0], role="examiner"),), snapshot_hash="")
    changed_confound = replace(
        before, confounds={"potential": (), "observed": ("overlapping_speech",), "ruled_out": ()},
        snapshot_hash="")
    with pytest.raises(c.PilotContractError):
        _replay_between(c, before, changed_segment, operation(c))
    with pytest.raises(c.PilotContractError):
        _replay_between(c, before, changed_confound, operation(c))


@pytest.mark.parametrize("action,flag", [
    ("unsupported_interpretation_flag", "unsupported_claim"),
    ("confound_flag", "overlapping_speech"),
])
def test_flag_operations_cannot_mutate_metric_evidence(c, action, flag):
    before = snapshot(c)
    payload = before.evidence[0].to_dict()
    payload["body"]["value"] = 1.0
    after = replace(before, evidence=(c.MetricEvidenceBody.from_mapping(payload),), snapshot_hash="")
    op = operation(c, action=action, method_hash=None, flag_code=flag)
    with pytest.raises(c.PilotContractError):
        _replay_between(c, before, after, op)


def test_source_correction_cannot_mutate_confounds_without_confound_flag(c):
    before = snapshot(c)
    after = replace(before, confounds={"potential": (), "observed": ("overlapping_speech",),
                                       "ruled_out": ()}, snapshot_hash="")
    op = operation(c, action="source_role_span_correction", method_hash=None,
                   corrected_role="examiner")
    with pytest.raises(c.PilotContractError):
        _replay_between(c, before, after, op)


def test_recomputed_same_state_descendant_need_not_be_directly_cited(c):
    before = snapshot(c)
    descendant = c.MetricEvidenceBody.from_evidence(MetricEvidenceV2(
        evidence_id="e002", metric_id="pause_variant", state_id="timing",
        subject_id=before.subject.subject_id, value=0.0,
    ))
    card_payload = before.state_cards[0].to_state_card()
    card_payload["supporting_evidence_ids"] = ["e001", "e002"]
    before = replace(before, evidence=before.evidence + (descendant,),
                     state_cards=(c.StateCardBody(body=card_payload),), snapshot_hash="")
    changed = descendant.to_dict()
    changed["body"]["value"] = 1.0
    after = replace(before, evidence=(before.evidence[0], c.MetricEvidenceBody.from_mapping(changed)),
                    snapshot_hash="")
    replay = _replay_between(c, before, after, operation(c), recomputed_ids=("e002",))
    assert replay.recomputed_ids == ("e002",)


def test_changed_evidence_and_state_domains_require_matching_audit_coverage(c):
    before = snapshot(c)
    evidence_payload = before.evidence[0].to_dict()
    evidence_payload["body"]["value"] = 1.0
    evidence_after = replace(before, evidence=(c.MetricEvidenceBody.from_mapping(evidence_payload),),
                             snapshot_hash="")
    audited = _replay_between(c, before, evidence_after, operation(c))
    with pytest.raises(c.PilotContractError):
        replace(audited, recomputed_ids=())

    second_evidence = c.MetricEvidenceBody.from_evidence(MetricEvidenceV2(
        evidence_id="e002", metric_id="lexical", state_id="language",
        subject_id=before.subject.subject_id, value=0.0,
    ))
    second_card = c.StateCardBody(body={
        "state_id": "language", "state_z": 0.0, "available": True,
        "supporting_evidence_ids": ["e002"], "counter_evidence_ids": [],
        "provenance_trace": [{"source_segment_ids": ["seg_0123456789abcdef"],
                               "method_version": "fixture_v1"}],
    })
    two_states = replace(
        before, evidence=before.evidence + (second_evidence,),
        state_cards=before.state_cards + (second_card,),
        observability={**before.observability, "language": c.Observability(status="observed", reason=None)},
        snapshot_hash="",
    )
    changed_card_body = second_card.to_state_card()
    changed_card_body["confidence"] = 0.8
    changed_state = replace(
        two_states,
        state_cards=(two_states.state_cards[0], c.StateCardBody(body=changed_card_body)),
        snapshot_hash="",
    )
    before_scores = {"timing": 0.0, "language": 0.0}
    with pytest.raises(c.PilotContractError):
        replay_result(
            c, before=two_states, after=changed_state, parent_hash=two_states.snapshot_hash,
            child_hash=changed_state.snapshot_hash, before_state_scores=before_scores,
            after_state_scores=before_scores, state_delta={"timing": 0.0, "language": 0.0},
        )


def test_replay_executor_authorization_is_exact_and_required(c):
    row = replay_result(c)
    for changes in (
        {"repair_batch_id": None}, {"repair_batch_id": ""},
        {"executor_registry_hash": "bad"}, {"authorized_operations": {}},
        {"authorized_operations": {"op_0": "f" * 64}},
        {"authorized_operations": {"op_other": row.accepted_ops[0].content_hash}},
    ):
        with pytest.raises(c.PilotContractError):
            replace(row, **changes)
    noop = replay_result(c, changed=False, accepted=False)
    for changes in ({"repair_batch_id": "repair_0"},
                    {"authorized_operations": {"op_0": operation(c).content_hash}}):
        with pytest.raises(c.PilotContractError):
            replace(noop, **changes)


def test_state_version_changes_only_with_meaningful_accepted_replay(c):
    before = snapshot(c)
    version_only = replace(before, state_version="state_v2", snapshot_hash="")
    with pytest.raises(c.PilotContractError):
        replay_result(c, changed=False, accepted=False, before=before, after=version_only,
                      parent_hash=before.snapshot_hash, child_hash=version_only.snapshot_hash)
    card_body = before.state_cards[0].to_state_card()
    card_body["confidence"] = 0.8
    changed = replace(before, state_version="state_v2",
                      state_cards=(c.StateCardBody(body=card_body),), snapshot_hash="")
    assert _replay_between(c, before, changed, operation(c)).meaningful_change


def test_snapshot_binds_observability_evidence_states_and_confounds(c):
    observed = snapshot(c)
    with pytest.raises(c.PilotContractError):
        replace(observed, observability={"timing": c.Observability(
            status="missing", reason="measurement_missing")}, snapshot_hash="")
    unavailable = snapshot(c, None)
    with pytest.raises(c.PilotContractError):
        replace(unavailable, observability={"timing": c.Observability(
            status="observed", reason=None)}, snapshot_hash="")
    evidence_payload = observed.evidence[0].to_dict()
    evidence_payload["body"]["state_id"] = "language"
    with pytest.raises(c.PilotContractError):
        replace(observed, evidence=(c.MetricEvidenceBody.from_mapping(evidence_payload),), snapshot_hash="")
    with pytest.raises(c.PilotContractError):
        replace(observed, confounds={"potential": ("noise",), "observed": ("noise",),
                                     "ruled_out": ()}, snapshot_hash="")


def test_state_card_normalizes_integer_state_z_before_snapshot_hashing(c):
    card = c.StateCardBody(body={
        **snapshot(c).state_cards[0].to_state_card(), "state_z": 1, "raw_state_z": 1,
    })
    assert card.state_z == 1.0 and type(card.body["state_z"]) is float
    assert c.StateCardBody.from_json(card.to_json()) == card


def test_state_card_accepts_real_state_graph_payload_and_normalizes_numeric_values(c):
    evidence = pd.DataFrame([{
        "dataset_id": "fixture", "subject_id": "sub_0123456789abcdef",
        "metric_id": "pause", "metric_instance_id": "pause", "task_scope": "overall",
        "directional_z": np.float64(1.25), "reliability": np.float64(0.8),
        "missing": np.bool_(False), "evidence_status": "available",
        "report_permission": np.bool_(True), "confound_tags": "[]", "evidence_segments": "[]",
    }])
    states = {"metric_contribution_clip_z": 5.0, "states": [{
        "id": "timing", "name_zh": "timing", "branch": "speech_behavior",
        "clinical_question": "pause", "metrics": ["pause"], "weights": [1.0],
    }]}
    cards, _ = build_state_graph_frame(evidence, states)
    payload = cards.query("graph_level == 'shared'").iloc[0].to_dict()
    payload["nested_numpy"] = {"value": np.int64(2), "missing": np.float64(np.nan)}
    card = c.StateCardBody.from_state_card(payload)
    for field in ("state_z", "raw_state_z", "task_state_z", "report_state_z",
                  "residual_shrinkage_factor"):
        assert card.body[field] is None or type(card.body[field]) is float
    assert card.body["nested_numpy"] == {"value": 2, "missing": None}
    assert c.StateCardBody.from_json(card.to_json()) == card


def test_assessment_binds_case_and_usage_content(c):
    snap = snapshot(c)
    row = assessment(c, snap)
    for changes in ({"case_id": "case_ffffffffffffffff"}, {"usage_id": "f" * 64}):
        with pytest.raises(c.PilotContractError):
            replace(row, **changes).validate_snapshot(snap)


@pytest.mark.parametrize("status", ["success", "failed", "HTTP 500", ""])
def test_usage_response_status_is_strictly_coded(c, status):
    with pytest.raises(c.PilotContractError):
        usage(c, response_status=status)


@pytest.mark.parametrize("response_status", [
    "transport_failure", "provider_failure", "timeout", "malformed_response",
])
def test_successful_assessment_requires_successful_usage(c, response_status):
    snap = snapshot(c)
    bad_usage = usage(c, response_status=response_status)
    with pytest.raises(c.PilotContractError):
        assessment(c, snap, usage=bad_usage, usage_id=bad_usage.content_hash)


def test_all_prediction_arms_and_direct_baseline_fallback(c):
    row = fusion(c)
    for prediction_arm in ("B_raw", "B", "J-A", "J-S", "J-AS"):
        prediction(c, fusion_hash=row.content_hash, arm=prediction_arm).validate_fusion(row)
    fallback = replace(row, assessment_v1=None, j_as=arm(c, "v1_missing", "B"), validity="partial")
    with pytest.raises(c.PilotContractError):
        replace(fallback, j_as=arm(c, "v1_missing", "J-S"))


def test_missing_replay_disables_only_replay_dependent_arms(c):
    row = fusion(c, changed=False, accepted=False)
    missing = replace(row, replay=None, assessment_v1=None, v1_required=False,
                      j_s=arm(c, "replay_missing", "B"),
                      j_as=arm(c, "replay_missing", "B"), validity="partial")
    assert missing.j_a.valid and not missing.j_s.valid and not missing.j_as.valid
    proposed = replace(
        row, replay=None, assessment_v0=assessment(c, row.replay.before, operations=(operation(c),)),
        assessment_v1=None, v1_required=False, j_s=arm(c, "replay_missing", "B"),
        j_as=arm(c, "replay_missing", "B"), validity="partial",
    )
    assert proposed.j_a.valid and proposed.assessment_v0.operations
    with pytest.raises(c.PilotContractError):
        replace(missing, parent_snapshot_hash="f" * 64)
    other = subject(c, subject_id="sub_ffffffffffffffff")
    with pytest.raises(c.PilotContractError):
        replace(missing, subject=other, subject_id=other.subject_id)


def test_all_unavailable_replay_inputs_fall_back_to_baseline(c):
    missing = snapshot(c, None)
    replay = replay_result(
        c, changed=False, accepted=False, before=missing, after=missing,
        parent_hash=missing.snapshot_hash, child_hash=missing.snapshot_hash,
        before_state_scores={"timing": None}, after_state_scores={"timing": None},
        state_delta={"timing": None},
    )
    row = fusion(
        c, replay=replay, j_s=arm(c, "replay_inputs_missing", "B"),
        j_as=arm(c, "replay_inputs_missing", "B"), validity="partial",
    )
    assert row.j_a.valid and not row.j_s.valid and not row.j_as.valid


@pytest.mark.parametrize("change", [
    {"arm": "unknown"}, {"class_order": ("AD", "MCI", "HC")}, {"trace_ids": ()},
    {"calibrator_id": None}, {"status": "fallback"}, {"status": "unavailable"},
    {"fallback_reason": "v1_missing"},
])
def test_prediction_status_and_trace_are_explicit(c, change):
    with pytest.raises(c.PilotContractError):
        prediction(c, **change)


def test_prediction_fallback_binds_arm_and_fusion(c):
    row = fusion(c, assessment_v1=None, j_as=arm(c, "v1_missing", "B"), validity="partial")
    pred = prediction(c, fusion_hash=row.content_hash, status="fallback", fallback_reason="v1_missing",
                      fallback_arm="B")
    pred.validate_fusion(row)
    with pytest.raises(c.PilotContractError):
        replace(pred, fusion_hash="f" * 64).validate_fusion(row)
    with pytest.raises(c.PilotContractError):
        replace(pred, arm="J-S").validate_fusion(row)
    with pytest.raises(c.PilotContractError):
        replace(pred, fallback_reason="other").validate_fusion(row)


@pytest.mark.parametrize("change", [{"attempt": 0}, {"request_id": ""}, {"cache_key": "bad"},
                                     {"cache_status": "unknown"}, {"wall_seconds": -1.0},
                                     {"response_id": "raw response text"}])
def test_usage_metadata_rejects_incomplete_or_unsafe_values(c, change):
    with pytest.raises(c.PilotContractError):
        usage(c, **change)


@pytest.mark.parametrize("field,value", [
    ("family", "logistic_regression"), ("l2_lambda", 0.5), ("loss", "sum_cross_entropy"),
    ("prior", [0, 0, 0, 0]),
    ("non_intercept_nonnegative", False), ("base_log_odds_scaling", "standardized"),
    ("other_features_scaling", "all_data"), ("hyperparameter_search", True),
    ("optimizer", "lbfgs"), ("probability_clip", [1e-5, 1 - 1e-5]),
    ("contrast_definitions", {"impaired": "MCI-HC", "stage": "AD-MCI", "binary": "AD-HC"}),
    ("feature_order", ["intercept", "base_log_odds", "agent_contrast", "replay_log_odds_delta"]),
])
def test_fixed_constrained_fusion_policy_cannot_be_relaxed(c, field, value):
    payload = config_payload()["fusion"]
    payload[field] = value
    with pytest.raises(c.PilotContractError):
        c.FusionConfig.from_mapping(payload)


def test_yaml_execution_policy_and_unresolved_provider_pricing(c):
    payload = config_payload()
    assert payload["provider"] is payload["model_id"] is None
    assert payload["pricing"]["source"] is payload["pricing"]["date"] is None
    assert payload["paid_execution"] is False
    limits = payload["limits"]
    assert limits["concurrency"] == 2
    assert limits["request_timeout_seconds"] == 180
    assert limits["semantic_retry_max"] == 1
    assert limits["pause_after_consecutive_transport_failures"] == 2
    assert {key: limits[key] for key in (
        "core_call_cap", "comparison_call_cap", "canary_call_cap", "total_call_cap",
        "comparison_structured_cell_max_calls", "comparison_flat_cell_max_calls",
        "comparison_subject_max_calls",
    )} == {
        "core_call_cap": 742, "comparison_call_cap": 144, "canary_call_cap": 24,
        "total_call_cap": 910, "comparison_structured_cell_max_calls": 2,
        "comparison_flat_cell_max_calls": 1, "comparison_subject_max_calls": 6,
    }
    for field in ("max_input_tokens", "max_output_tokens", "max_reasoning_tokens", "max_total_tokens", "max_calls", "max_usd"):
        assert limits[field] == 0
        with pytest.raises(c.PilotContractError):
            c.ExecutionCaps.from_mapping({**limits, field: -1})
        without = dict(limits)
        del without[field]
        with pytest.raises(c.PilotContractError):
            c.ExecutionCaps.from_mapping(without)
    row = resolved_config(c)
    with pytest.raises(c.PilotContractError):
        replace(row, paid_execution=True)
    for change in ({"provider": None}, {"model_id": None}, {"pricing": payload["pricing"]}):
        with pytest.raises(c.PilotContractError):
            replace(row, **change)


@pytest.mark.parametrize("change", [
    {"concurrency": 0}, {"concurrency": 3},
    {"request_timeout_seconds": 0}, {"request_timeout_seconds": 181},
    {"semantic_retry_max": -1}, {"semantic_retry_max": 2},
    {"pause_after_consecutive_transport_failures": 0},
    {"pause_after_consecutive_transport_failures": 3},
    {"core_call_cap": 0}, {"core_call_cap": 743},
    {"comparison_call_cap": 0}, {"comparison_call_cap": 145},
    {"canary_call_cap": 0}, {"canary_call_cap": 25},
    {"total_call_cap": 0}, {"total_call_cap": 911},
    {"comparison_structured_cell_max_calls": 0},
    {"comparison_structured_cell_max_calls": 3},
    {"comparison_flat_cell_max_calls": 0}, {"comparison_flat_cell_max_calls": 2},
    {"comparison_subject_max_calls": 0}, {"comparison_subject_max_calls": 7},
])
def test_execution_ceilings_reject_nonpositive_or_relaxed_values(c, change):
    with pytest.raises(c.PilotContractError):
        c.ExecutionCaps.from_mapping({**config_payload()["limits"], **change})


def test_execution_ceilings_accept_stricter_values(c):
    limits = c.ExecutionCaps.from_mapping({
        **config_payload()["limits"],
        "concurrency": 1, "request_timeout_seconds": 1, "semantic_retry_max": 0,
        "pause_after_consecutive_transport_failures": 1,
        "core_call_cap": 1, "comparison_call_cap": 1, "canary_call_cap": 1,
        "total_call_cap": 3, "comparison_structured_cell_max_calls": 1,
        "comparison_flat_cell_max_calls": 1, "comparison_subject_max_calls": 1,
    })
    assert limits.concurrency == 1 and limits.total_call_cap == 3
    assert replace(limits, max_calls=1).max_calls == 1
    with pytest.raises(c.PilotContractError):
        replace(limits, max_calls=4)


def test_component_call_caps_must_fit_total(c):
    payload = config_payload()["limits"]
    payload.update(core_call_cap=2, comparison_call_cap=2, canary_call_cap=1,
                   total_call_cap=4)
    with pytest.raises(c.PilotContractError):
        c.ExecutionCaps.from_mapping(payload)


def test_config_records_reject_unknown_or_missing_fields(c):
    records = (
        (c.FusionConfig, config_payload()["fusion"]),
        (c.ExecutionCaps, config_payload()["limits"]),
        (c.Pricing, resolved_config(c).pricing.to_dict()),
    )
    for cls, payload in records:
        with pytest.raises(c.PilotContractError):
            cls.from_mapping({**payload, "unexpected": 1})
        for field in tuple(payload):
            if field == "schema_version":
                continue
            incomplete = dict(payload)
            del incomplete[field]
            with pytest.raises(c.PilotContractError):
                cls.from_mapping(incomplete)


@pytest.mark.parametrize("field,value", [
    ("concurrency", True),
    ("request_timeout_seconds", True), ("semantic_retry_max", True),
    ("pause_after_consecutive_transport_failures", True),
    ("core_call_cap", True), ("comparison_call_cap", True), ("canary_call_cap", True),
    ("total_call_cap", True), ("comparison_structured_cell_max_calls", True),
    ("comparison_flat_cell_max_calls", True), ("comparison_subject_max_calls", True),
    ("max_calls", True), ("max_input_tokens", True), ("max_output_tokens", True),
    ("max_reasoning_tokens", True), ("max_total_tokens", True), ("max_usd", True),
])
def test_execution_caps_are_never_boolean(c, field, value):
    with pytest.raises(c.PilotContractError):
        c.ExecutionCaps.from_mapping({**config_payload()["limits"], field: value})


@pytest.mark.parametrize("change", [
    {"source": ""}, {"date": ""}, {"input_usd_per_million": -0.01},
    {"output_usd_per_million": -0.01}, {"reasoning_usd_per_million": -0.01},
])
def test_resolved_pricing_requires_provenance_and_nonnegative_rates(c, change):
    with pytest.raises(c.PilotContractError):
        replace(resolved_config(c).pricing, **change)


@pytest.mark.parametrize("model_id", ["gpt-latest", "gpt-latest-2026-01-01", "gpt-alias-2026-01-01"])
def test_resolved_config_requires_provider_and_exact_model_id(c, model_id):
    row = resolved_config(c)
    for change in ({"provider": ""}, {"model_id": model_id}):
        with pytest.raises(c.PilotContractError):
            replace(row, **change)


def test_paid_execution_requires_every_positive_budget_cap(c):
    row = resolved_config(c)
    for field in ("max_calls", "max_input_tokens", "max_output_tokens", "max_reasoning_tokens",
                  "max_total_tokens", "max_usd"):
        caps = replace(row.limits, **{field: 1 if getattr(row.limits, field) == 0 else 0})
        with pytest.raises(c.PilotContractError):
            replace(row, limits=caps, paid_execution=True)
    paid_caps = replace(row.limits, max_calls=1, max_input_tokens=1, max_output_tokens=1,
                        max_reasoning_tokens=1, max_total_tokens=1, max_usd=0.01)
    assert replace(row, limits=paid_caps, paid_execution=True).paid_execution is True


def test_runtime_call_cap_is_zero_when_disabled_and_never_exceeds_total(c):
    row = resolved_config(c)
    with pytest.raises(c.PilotContractError):
        replace(row, limits=replace(row.limits, max_calls=1))
    with pytest.raises(c.PilotContractError):
        replace(row.limits, max_calls=911)


def test_fusion_probability_clip_contrasts_and_feature_order_are_exact(c):
    fusion_config = resolved_config(c).fusion
    assert fusion_config.probability_clip == (1e-6, 1 - 1e-6)
    assert dict(fusion_config.contrast_definitions) == {
        "impaired": "(MCI+AD)/2-HC", "stage": "AD-MCI", "binary": "AD-HC",
    }
    assert fusion_config.feature_order == (
        "base_log_odds", "agent_contrast", "replay_log_odds_delta", "intercept",
    )
