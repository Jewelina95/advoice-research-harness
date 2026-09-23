from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from advoice.authority_review_runtime import (
    AuthorityReviewRuntime,
    AuthorityReviewValidationError,
    LikelihoodEvidenceCitation,
    REVIEW_AVAILABLE,
    REVIEW_MODE_LEGACY_TWO_PASS,
    REVIEW_MODE_SINGLE_BLIND,
    REVIEW_PROVIDER_ERROR,
    agent_evidence_strength,
    agent_staging_evidence_strength,
    build_blind_payload,
    _parse_blind,
    _parse_reconciliation,
)
from advoice.conditional_authority import PreparedAuthorityCase
from advoice.decision_lock import canonical_json, hash_artifact
from advoice.evidence import EvidencePermissions, EvidenceProvenance, MetricEvidenceV2
from advoice.routing import ObservationRoute, RouteDecision, TargetRoute


LABELS = ("HC", "AD")


def _prepared() -> PreparedAuthorityCase:
    evidence = MetricEvidenceV2(
        evidence_id="metric:pause",
        metric_id="pause_rate",
        subject_id="raw-subject-9",
        session_id="raw-session-2",
        case_id="raw-subject-9",
        state_id="S01",
        task_id="cookie",
        value=2.0,
        direction=1,
        provenance=EvidenceProvenance(source_asset_id="/private/raw/audio.wav"),
        permissions=EvidencePermissions(inference=True, report=True),
        consumed_by_supervised=True,
    )
    route = RouteDecision(
        ObservationRoute("picture_description", "picture_description", task_id="cookie", language="en"),
        TargetRoute("diagnosis", "cross_sectional_diagnosis", LABELS),
    )
    packet = SimpleNamespace(
        class_order=LABELS,
        raw_probabilities={"HC": 0.7, "AD": 0.3},
        calibrated_probabilities={"HC": 0.6, "AD": 0.4},
        feature_contributions={"HC": {"state_S01": -1.0}, "AD": {"state_S01": 1.0}},
        branch_contributions={"HC": {"clinical": -1.0}, "AD": {"clinical": 1.0}},
        uncertainty={"entropy": 0.4},
        hashes={"packet": "a" * 64},
    )
    evidence_artifact = {"evidence": ["frozen"]}
    state_artifact = {"evidence_hash": "e" * 64, "state_cards": ["frozen"]}
    state_hash = hash_artifact(state_artifact)
    packet_artifact = {
        "state_graph_hash": state_hash,
        "evidence_snapshot_hash": hash_artifact(evidence_artifact),
        "packet": {"frozen": True},
    }
    packet_hash = hash_artifact(packet_artifact)
    return PreparedAuthorityCase(
        case_id="raw-subject-9",
        route=route,
        case_metadata={"truth": "AD", "split": "test", "subject_id": "raw-subject-9"},
        case_context={"source_path": "/private/raw/case.json"},
        evidence=(evidence,),
        module_a_evidence=(evidence,),
        incremental_evidence=(),
        pre_replay=SimpleNamespace(packet=packet),
        pre_state_cards=({
            "case_id": "raw-subject-9",
            "state_card_id": "S01:cookie:0",
            "state_id": "S01",
            "task_id": "cookie",
            "task_ids": ["cookie"],
            "supporting_evidence_ids": ["metric:pause"],
            "counter_evidence_ids": [],
            "segment_ids": ["raw-segment"],
            "available": True,
            "report_permission": True,
        },),
        pre_evidence_artifact=evidence_artifact,
        pre_state_artifact=state_artifact,
        pre_packet_artifact=packet_artifact,
        reviewed_evidence_hash="e" * 64,
        reviewed_state_graph_hash=state_hash,
        reviewed_packet_hash=packet_hash,
        advisor_packet_hash=packet_hash,
    )


def _skill(tmp_path: Path) -> Path:
    path = tmp_path / "skills" / "ad_evidence_skill"
    path.mkdir(parents=True)
    skill = path / "skill.md"
    skill.write_text("Use typed evidence only.", encoding="utf-8")
    (path / "POLICY.md").write_text("Never invent measurements.", encoding="utf-8")
    return skill


def _blind(prepared: PreparedAuthorityCase) -> dict[str, object]:
    return {
        "case_id": "case_" + __import__("hashlib").sha256(
            b"advoice-8.27::raw-subject-9"
        ).hexdigest()[:12],
        "reviewed_packet_hash": prepared.reviewed_packet_hash,
        "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
        "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
        "state_actions": [{
            "state_id": "S01", "action": "downweight", "cited_metric_evidence_ids": ["metric:pause"],
            "reliability_multiplier": 0.75, "rationale": "Measurement reliability is reduced.",
        }],
        "ordinal_scores": {"HC": 2, "AD": 2},
        "likelihood_evidence": [{
            "class_label": "AD",
            "state_id": "S01",
            "relation": "support",
            "cited_metric_evidence_ids": ["metric:pause"],
        }, {
            "class_label": "HC",
            "state_id": "S01",
            "relation": "counter",
            "cited_metric_evidence_ids": ["metric:pause"],
        }],
        "report_trace": ["S01 reviewed against metric:pause."],
    }


def _advisor(prepared: PreparedAuthorityCase) -> dict[str, object]:
    return {
        "case_id": _blind(prepared)["case_id"],
        "reviewed_packet_hash": prepared.reviewed_packet_hash,
        "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
        "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
        "advisor_packet_hash": prepared.advisor_packet_hash,
        "disposition": "amend",
        "amendments": [{
            "state_id": "S01", "action": "downweight", "cited_metric_evidence_ids": ["metric:pause"],
            "reliability_multiplier": 0.5, "rationale": "Advisor uncertainty supports a bounded downweight.",
        }],
        "cited_metric_evidence_ids": ["metric:pause"],
        "rationale": "Amendment changes evidence handling only.",
    }


def _payload_from_prompt(prompt: str) -> dict[str, object]:
    return json.loads(prompt.split("\n", 1)[1])


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value)) if value else set()
    return set()


def test_two_pass_payloads_are_blind_then_advisor_bound(monkeypatch, tmp_path: Path) -> None:
    prepared = _prepared()
    calls: list[dict[str, object]] = []

    def provider(root, prompt, schema_path, output_path, model, provider):
        calls.append(_payload_from_prompt(prompt))
        return _blind(prepared) if len(calls) == 1 else _advisor(prepared)

    monkeypatch.setattr("advoice.authority_review_runtime.run_structured_batch", provider)
    runtime = AuthorityReviewRuntime(
        root=tmp_path, provider="openai_api", model="test",
        skill_path=_skill(tmp_path), review_mode=REVIEW_MODE_LEGACY_TWO_PASS,
    )
    result = runtime.review(prepared)

    assert result.status == REVIEW_AVAILABLE
    assert len(calls) == 2
    assert "probabilities" not in _keys(calls[0])
    assert "advisor_packet" not in calls[0]
    assert calls[1]["advisor_packet"]["probabilities"]["raw"] == {"HC": 0.7, "AD": 0.3}
    assert result.effective_state_actions["S01"].action == "downweight"


def test_default_runtime_uses_one_blind_evidence_call(monkeypatch, tmp_path: Path) -> None:
    prepared = _prepared()
    calls: list[dict[str, object]] = []

    def provider(root, prompt, schema_path, output_path, model, provider):
        calls.append(_payload_from_prompt(prompt))
        return _blind(prepared)

    monkeypatch.setattr("advoice.authority_review_runtime.run_structured_batch", provider)
    result = AuthorityReviewRuntime(
        root=tmp_path, provider="openai_api", model="test", skill_path=_skill(tmp_path),
    ).review(prepared)

    assert result.status == REVIEW_AVAILABLE
    assert result.review_mode == REVIEW_MODE_SINGLE_BLIND
    assert len(calls) == 1
    assert "advisor_packet" not in calls[0]
    assert result.reconciliation is None
    assert result.reconciliation_request_hash is None
    assert result.effective_state_actions == result.blind_assessment.state_actions


def test_blind_review_accepts_sparse_actions_and_omits_retained_states() -> None:
    prepared = _prepared()
    response = _blind(prepared)
    response["state_actions"] = []

    assessment = _parse_blind(response, prepared)

    assert assessment.state_actions == {}


def test_blind_review_rejects_retain_unknown_state_and_cross_state_citations() -> None:
    prepared = _prepared()
    retain = _blind(prepared)
    retain["state_actions"][0]["action"] = "retain"
    retain["state_actions"][0]["reliability_multiplier"] = 1.0
    with pytest.raises(AuthorityReviewValidationError, match="omit retained"):
        _parse_blind(retain, prepared)

    unknown = _blind(prepared)
    unknown["state_actions"][0]["state_id"] = "S99"
    with pytest.raises(AuthorityReviewValidationError, match="outside the frozen"):
        _parse_blind(unknown, prepared)

    second_evidence = replace(
        prepared.evidence[0], evidence_id="metric:other", state_id="S02"
    )
    cross_state = replace(prepared, evidence=prepared.evidence + (second_evidence,))
    response = _blind(cross_state)
    response["state_actions"][0]["cited_metric_evidence_ids"] = ["metric:other"]
    with pytest.raises(AuthorityReviewValidationError, match="outside the reviewed state"):
        _parse_blind(response, cross_state)

    unsupported_strength = _blind(prepared)
    unsupported_strength["state_actions"][0]["reliability_multiplier"] = 0.1
    with pytest.raises(AuthorityReviewValidationError, match="pre-registered multiplier"):
        _parse_blind(unsupported_strength, prepared)


@pytest.mark.parametrize("action", ["invalidate", "mark_unavailable"])
def test_provider_removal_action_canonicalizes_stale_multiplier(action: str) -> None:
    prepared = _prepared()
    response = _blind(prepared)
    response["state_actions"][0]["action"] = action
    response["state_actions"][0]["reliability_multiplier"] = 0.25

    assessment = _parse_blind(response, prepared)

    assert assessment.state_actions["S01"].reliability_multiplier == 0.0


def test_review_rejects_citations_without_inference_permission() -> None:
    prepared = _prepared()
    blocked = replace(
        prepared.evidence[0],
        permissions=EvidencePermissions(inference=False, report=False),
    )
    blocked_case = replace(prepared, evidence=(blocked,))

    with pytest.raises(AuthorityReviewValidationError, match="unknown MetricEvidence"):
        _parse_blind(_blind(blocked_case), blocked_case)

    blind_response = _blind(blocked_case)
    blind_response["state_actions"] = []
    with pytest.raises(AuthorityReviewValidationError, match="unknown MetricEvidence"):
        _parse_blind(blind_response, blocked_case)
    blind = _parse_blind(_blind(prepared), prepared)
    with pytest.raises(AuthorityReviewValidationError, match="unknown MetricEvidence"):
        _parse_reconciliation(_advisor(blocked_case), blocked_case, blind)


def test_agent_likelihood_requires_directional_class_citations() -> None:
    prepared = _prepared()
    response = _blind(prepared)
    response["ordinal_scores"] = {"HC": 0, "AD": 4}
    response["likelihood_evidence"] = [response["likelihood_evidence"][0]]

    with pytest.raises(AuthorityReviewValidationError, match="lowest Agent score"):
        _parse_blind(response, prepared)


def test_agent_evidence_strength_is_citation_bound_not_uncertainty_bound() -> None:
    prepared = _prepared()
    response = _blind(prepared)
    response["ordinal_scores"] = {"HC": 0, "AD": 4}
    assessment = _parse_blind(response, prepared)

    assert agent_evidence_strength(prepared, assessment) == pytest.approx(0.5)


def test_agent_evidence_strength_does_not_double_count_one_metric_across_states() -> None:
    prepared = _prepared()
    duplicate_family = replace(
        prepared.evidence[0], evidence_id="metric:pause:s02", state_id="S02"
    )
    prepared = replace(prepared, evidence=prepared.evidence + (duplicate_family,))
    assessment = _parse_blind(_blind(_prepared()), _prepared())
    assessment = replace(
        assessment,
        likelihood_evidence=assessment.likelihood_evidence + (
            LikelihoodEvidenceCitation(
                class_label="AD",
                state_id="S02",
                relation="support",
                cited_metric_evidence_ids=("metric:pause:s02",),
            ),
        ),
    )

    assert agent_evidence_strength(prepared, assessment) == pytest.approx(0.5)


def test_mci_ad_stage_preference_requires_direct_two_sided_citations() -> None:
    prepared = _prepared()
    target = replace(prepared.route.target_route, labels=("HC", "MCI", "AD"))
    prepared = replace(prepared, route=replace(prepared.route, target_route=target))
    response = _blind(prepared)
    response["ordinal_scores"] = {"HC": 0, "MCI": 1, "AD": 4}

    with pytest.raises(AuthorityReviewValidationError, match="MCI/AD stage preference"):
        _parse_blind(response, prepared)

    response["likelihood_evidence"].append({
        "class_label": "MCI",
        "state_id": "S01",
        "relation": "counter",
        "cited_metric_evidence_ids": ["metric:pause"],
    })
    assessment = _parse_blind(response, prepared)

    assert agent_staging_evidence_strength(prepared, assessment) == pytest.approx(0.5)


def test_reconciliation_rejects_cross_state_amendment_citations() -> None:
    prepared = _prepared()
    second_evidence = replace(
        prepared.evidence[0], evidence_id="metric:other", state_id="S02"
    )
    prepared = replace(prepared, evidence=prepared.evidence + (second_evidence,))
    blind = _parse_blind(_blind(prepared), prepared)
    response = _advisor(prepared)
    response["amendments"][0]["cited_metric_evidence_ids"] = ["metric:other"]

    with pytest.raises(AuthorityReviewValidationError, match="outside its state"):
        _parse_reconciliation(response, prepared, blind)


def test_runtime_schema_enumerates_true_state_ids_and_forbids_blind_retain(
    monkeypatch, tmp_path: Path,
) -> None:
    prepared = _prepared()
    schemas: list[dict[str, object]] = []

    def provider(root, prompt, schema_path, output_path, model, provider):
        schemas.append(json.loads(Path(schema_path).read_text(encoding="utf-8")))
        return _blind(prepared) if len(schemas) == 1 else _advisor(prepared)

    monkeypatch.setattr("advoice.authority_review_runtime.run_structured_batch", provider)
    result = AuthorityReviewRuntime(
        root=tmp_path, provider="openai_api", model="test", skill_path=_skill(tmp_path),
        review_mode=REVIEW_MODE_LEGACY_TWO_PASS,
    ).review(prepared)

    assert result.status == REVIEW_AVAILABLE
    variants = schemas[0]["properties"]["state_actions"]["items"]["anyOf"]
    assert len(variants) == 3
    by_action = {
        item["properties"]["action"]["enum"][0]: item["properties"]
        for item in variants
    }
    assert set(by_action) == {"downweight", "invalidate", "mark_unavailable"}
    assert all(item["state_id"]["enum"] == ["S01"] for item in by_action.values())
    assert by_action["downweight"]["reliability_multiplier"]["enum"] == [0.25, 0.5, 0.75]
    assert by_action["invalidate"]["reliability_multiplier"]["enum"] == [0.0]
    assert by_action["mark_unavailable"]["reliability_multiplier"]["enum"] == [0.0]
    assert all(
        item["cited_metric_evidence_ids"]["items"]["enum"] == ["E001"]
        for item in by_action.values()
    )


def test_runtime_schema_binds_each_action_to_same_state_evidence(
    monkeypatch, tmp_path: Path,
) -> None:
    prepared = _prepared()
    second_evidence = replace(
        prepared.evidence[0], evidence_id="metric:other", state_id="S02"
    )
    prepared = replace(
        prepared,
        evidence=prepared.evidence + (second_evidence,),
        pre_state_cards=prepared.pre_state_cards + ({
            **prepared.pre_state_cards[0],
            "state_card_id": "S02:cookie:0",
            "state_id": "S02",
            "supporting_evidence_ids": ["metric:other"],
        },),
    )
    schemas: list[dict[str, object]] = []

    def provider(root, prompt, schema_path, output_path, model, provider):
        schemas.append(json.loads(Path(schema_path).read_text(encoding="utf-8")))
        return _blind(prepared) if len(schemas) == 1 else _advisor(prepared)

    monkeypatch.setattr("advoice.authority_review_runtime.run_structured_batch", provider)
    result = AuthorityReviewRuntime(
        root=tmp_path, provider="openai_api", model="test", skill_path=_skill(tmp_path),
        review_mode=REVIEW_MODE_LEGACY_TWO_PASS,
    ).review(prepared)

    assert result.status == REVIEW_AVAILABLE
    for schema in schemas:
        key = "state_actions" if "state_actions" in schema["properties"] else "amendments"
        variants = schema["properties"][key]["items"]["anyOf"]
        for variant in variants:
            state_id = variant["properties"]["state_id"]["enum"][0]
            allowed = variant["properties"]["cited_metric_evidence_ids"]["items"]["enum"]
            assert allowed == (["E002"] if state_id == "S01" else ["E001"])


def test_runtime_uses_short_transport_ids_and_restores_full_audit_ids(
    monkeypatch, tmp_path: Path,
) -> None:
    prepared = _prepared()
    prompts: list[dict[str, object]] = []

    def provider(root, prompt, schema_path, output_path, model, provider):
        prompts.append(_payload_from_prompt(prompt))
        response = _blind(prepared) if len(prompts) == 1 else _advisor(prepared)
        text = canonical_json(response).replace("metric:pause", "E001")
        return json.loads(text)

    monkeypatch.setattr("advoice.authority_review_runtime.run_structured_batch", provider)
    result = AuthorityReviewRuntime(
        root=tmp_path, provider="openai_api", model="test", skill_path=_skill(tmp_path),
        review_mode=REVIEW_MODE_LEGACY_TWO_PASS,
    ).review(prepared)

    assert result.status == REVIEW_AVAILABLE
    assert prompts[0]["metric_evidence"][0]["evidence_id"] == "E001"
    assert prompts[0]["state_cards"][0]["supporting_evidence_ids"] == ["E001"]
    assert result.blind_assessment.state_actions["S01"].cited_metric_evidence_ids == (
        "metric:pause",
    )
    assert result.reconciliation.amendments["S01"].cited_metric_evidence_ids == (
        "metric:pause",
    )


def test_payload_strips_leakage_and_chat_residue_without_mutating_input(tmp_path: Path) -> None:
    prepared = _prepared()
    transcript = {
        "text": "The patient says pro:sub|I described the cookie 9|5|CJCT clearly.",
        "source_path": "/private/raw/chat.cha",
        "subject_id": "raw-subject-9",
        "truth": "AD",
        "raw_test_answer": "AD",
        "diagnosis": "dementia",
    }
    original = deepcopy(transcript)
    payload = build_blind_payload(
        prepared, policy_documents={"skill.md": "typed evidence"}, transcript=transcript,
    )
    rendered = canonical_json(payload)
    transcript_payload = payload["sanitized_transcript"]

    assert transcript == original
    assert "pro:sub|" not in transcript_payload["text"]
    assert "9|5|CJCT" not in transcript_payload["text"]
    assert "described the cookie" in transcript_payload["text"]
    assert "/private/raw" not in rendered
    assert not {"truth", "split", "subject_id", "source_path", "raw_test_answer", "diagnosis"} & _keys(payload)


def test_provider_leakage_and_stale_hashes_are_rejected(monkeypatch, tmp_path: Path) -> None:
    prepared = _prepared()
    response = _blind(prepared)
    response["truth"] = "AD"
    monkeypatch.setattr("advoice.authority_review_runtime.run_structured_batch", lambda *args: response)
    runtime = AuthorityReviewRuntime(root=tmp_path, provider="openai_api", skill_path=_skill(tmp_path))
    with pytest.raises(AuthorityReviewValidationError, match="prohibited leakage"):
        runtime.review(prepared)

    stale = replace(prepared, reviewed_packet_hash="0" * 64)
    with pytest.raises(AuthorityReviewValidationError, match="stale"):
        runtime.review(stale)
    for stale_hash in ("reviewed_evidence_hash", "reviewed_state_graph_hash"):
        with pytest.raises(AuthorityReviewValidationError, match="stale"):
            runtime.review(replace(prepared, **{stale_hash: "0" * 64}))


def test_provider_failure_is_closed_without_fabricated_review(monkeypatch, tmp_path: Path) -> None:
    prepared = _prepared()
    monkeypatch.setattr(
        "advoice.authority_review_runtime.run_structured_batch",
        lambda *args: (_ for _ in ()).throw(RuntimeError("provider down")),
    )
    result = AuthorityReviewRuntime(
        root=tmp_path, provider="openai_api", skill_path=_skill(tmp_path)
    ).review(prepared)
    assert result.status == REVIEW_PROVIDER_ERROR
    assert result.blind_assessment is None
    assert result.reconciliation is None
    assert result.effective_state_actions is None


def test_request_hashes_are_deterministic_and_disabled_never_calls_provider(monkeypatch, tmp_path: Path) -> None:
    prepared = _prepared()
    calls = 0

    def provider(*args):
        nonlocal calls
        calls += 1
        return _blind(prepared) if calls % 2 else _advisor(prepared)

    monkeypatch.setattr("advoice.authority_review_runtime.run_structured_batch", provider)
    skill = _skill(tmp_path)
    first = AuthorityReviewRuntime(
        root=tmp_path, provider="openai_api", skill_path=skill,
        review_mode=REVIEW_MODE_LEGACY_TWO_PASS,
    ).review(prepared)
    second = AuthorityReviewRuntime(
        root=tmp_path, provider="openai_api", skill_path=skill,
        review_mode=REVIEW_MODE_LEGACY_TWO_PASS,
    ).review(prepared)
    disabled = AuthorityReviewRuntime(root=tmp_path, provider="disabled", skill_path=skill).review(prepared)

    assert calls == 4
    assert first.blind_request_hash == second.blind_request_hash
    assert first.reconciliation_request_hash == second.reconciliation_request_hash
    assert disabled.status == "provider_unavailable"
    assert disabled.blind_assessment is None


def test_request_hashes_change_when_provider_contract_or_policy_changes(monkeypatch, tmp_path: Path) -> None:
    prepared = _prepared()
    output_paths: list[Path] = []
    calls = 0

    def provider(root, prompt, schema_path, output_path, model, provider_name):
        nonlocal calls
        calls += 1
        output_paths.append(Path(output_path))
        return _blind(prepared) if calls % 2 else _advisor(prepared)

    monkeypatch.setattr("advoice.authority_review_runtime.run_structured_batch", provider)
    skill = _skill(tmp_path)

    def review(**kwargs):
        return AuthorityReviewRuntime(
            root=tmp_path, skill_path=skill,
            review_mode=REVIEW_MODE_LEGACY_TWO_PASS, **kwargs,
        ).review(prepared)

    baseline = review(provider="openai_api", model="model-a")
    changed_provider = review(provider="other_provider", model="model-a")
    changed_model = review(provider="openai_api", model="model-b")
    skill.write_text("Use revised typed evidence only.", encoding="utf-8")
    changed_skill = review(provider="openai_api", model="model-a")
    (skill.parent / "POLICY.md").write_text("Require verified measurements.", encoding="utf-8")
    changed_policy = review(provider="openai_api", model="model-a")
    monkeypatch.setattr("advoice.authority_review_runtime.SCHEMA_VERSION", "test.runtime.v2")
    changed_schema = review(provider="openai_api", model="model-a")

    results = (baseline, changed_provider, changed_model, changed_skill, changed_policy, changed_schema)
    assert len({result.blind_request_hash for result in results}) == len(results)
    assert len({result.reconciliation_request_hash for result in results}) == len(results)
    assert len(set(output_paths)) == len(output_paths)
