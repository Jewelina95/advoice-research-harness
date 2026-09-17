from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from advoice.evidence import (
    ConfoundSets,
    EvidencePermissions,
    EvidenceProvenance,
    MetricEvidenceV2,
    ReferenceMetadata,
    ReliabilityComponents,
)
from advoice.routing import RouteValidationError, route_case


def test_observation_and_target_routes_are_independent() -> None:
    metadata = {
        "channel": "picture_description",
        "language": "en",
        "task_id": "cookie_theft",
        "target": "diagnosis",
    }
    diagnosis = route_case(metadata)
    progression = route_case({
        **metadata,
        "target": "progression",
        "visit_ids": ["baseline", "followup"],
        "visit_interval_days": 365,
    })
    assert diagnosis.observation_route == progression.observation_route
    assert diagnosis.target_route.endpoint != progression.target_route.endpoint


def test_structured_task_audio_routes_picture_description_by_task_metadata() -> None:
    decision = route_case({
        "channel": "structured_task_audio",
        "task_type": "long_picture_description",
        "language": "zh",
        "target": "diagnosis",
    })

    assert decision.observation_route.id == "structured_task_audio"
    assert decision.observation_route.family == "picture_description"
    assert decision.observation_route.task_id == "long_picture_description"


def test_structured_task_audio_preserves_non_picture_multitask_family() -> None:
    decision = route_case({
        "channel": "structured_task_audio",
        "task_type": "semantic_fluency",
        "language": "en",
        "target": "diagnosis",
    })

    assert decision.observation_route.id == "structured_task_audio"
    assert decision.observation_route.family == "structured_cognitive_multitask"


def test_progression_requires_paired_visits_and_interval() -> None:
    with pytest.raises(RouteValidationError, match="paired visits"):
        route_case({"channel": "picture_description", "target": "progression", "visit_ids": ["v1"]})
    with pytest.raises(RouteValidationError, match="interval"):
        route_case({
            "channel": "picture_description", "target": "progression",
            "visit_ids": ["v1", "v2"],
        })
    decision = route_case({
        "channel": "picture_description", "target": "progression",
        "visit_ids": ["v1", "v2"], "visit_interval_days": 180,
    })
    assert decision.target_route.requires_paired_visits is True
    assert decision.interval_days == 180.0


def test_permissions_are_separate_and_immutable() -> None:
    evidence = MetricEvidenceV2(
        evidence_id="metric:x", metric_id="x", subject_id="s1", session_id="v1",
        permissions=EvidencePermissions(inference=True, report=False),
    )
    assert evidence.inference_permission is True
    assert evidence.report_permission is False
    with pytest.raises(FrozenInstanceError):
        evidence.permissions.report = True  # type: ignore[misc]
    assert evidence.to_dict()["permissions"] == {"inference": True, "report": False}


def test_reference_records_sample_count_and_content_hash() -> None:
    first = ReferenceMetadata.from_values([1.0, 2.0, 3.0], artifact_id="fold-1")
    second = ReferenceMetadata.from_values([1.0, 2.0, 4.0], artifact_id="fold-1")
    assert first.sample_size == 3
    assert first.artifact_hash
    assert first.artifact_hash != second.artifact_hash


def test_reliability_components_and_confound_statuses_remain_separate() -> None:
    evidence = MetricEvidenceV2(
        evidence_id="metric:x", metric_id="x", subject_id="s1", session_id="v1",
        reliability_components=ReliabilityComponents(source=.9, role=.4, alignment=.8, asr=.7),
        confounds=ConfoundSets(
            potential=("device",), observed=("editing",), ruled_out=("role_overlap",)
        ),
    )
    payload = evidence.to_dict()
    assert payload["reliability_components"]["role"] == .4
    assert payload["reliability_components"]["asr"] == .7
    assert payload["confounds"] == {
        "potential": ["device"], "observed": ["editing"], "ruled_out": ["role_overlap"]
    }


def test_consumption_flags_are_explicit_not_derived() -> None:
    evidence = MetricEvidenceV2(
        evidence_id="metric:x", metric_id="x", subject_id="s1", session_id="v1",
        consumed_by_supervised=True,
    )
    payload = evidence.to_dict()
    assert payload["consumed_by_supervised"] is True
    assert payload["incremental_for_agent"] is False
    incremental = MetricEvidenceV2(
        evidence_id="metric:y", metric_id="y", subject_id="s1", session_id="v1",
        incremental_for_agent=True,
    )
    assert incremental.consumed_by_supervised is False
    with pytest.raises(ValueError, match="both consumed_by_supervised"):
        MetricEvidenceV2(
            evidence_id="metric:z", metric_id="z", subject_id="s1", session_id="v1",
            consumed_by_supervised=True, incremental_for_agent=True,
        )


@pytest.mark.parametrize("direction", [-2, 2, True, False])
def test_direction_is_closed_to_signed_unit_values(direction: int) -> None:
    with pytest.raises(ValueError, match="exactly -1, 0, or 1"):
        MetricEvidenceV2(
            evidence_id="metric:x", metric_id="x", subject_id="s1", session_id="v1",
            direction=direction,
        )


def test_contract_serialization_is_deterministic() -> None:
    evidence = MetricEvidenceV2(
        evidence_id="metric:x", metric_id="x", subject_id="s1", session_id="v1",
        provenance=EvidenceProvenance(source_segment_ids=("seg-2", "seg-1")),
    )
    first = evidence.to_json()
    second = MetricEvidenceV2.from_mapping(json.loads(first)).to_json()
    assert first == second


def test_mapping_parses_string_booleans_without_truthiness() -> None:
    evidence = MetricEvidenceV2.from_mapping({
        "evidence_id": "metric:x",
        "metric_id": "x",
        "subject_id": "s1",
        "observable": "False",
        "inference_permission": "False",
        "report_permission": "False",
        "consumed_by_supervised": "0",
        "incremental_for_agent": "true",
        "reliability": "0.5",
    })
    assert evidence.observable is False
    assert evidence.inference_permission is False
    assert evidence.report_permission is False
    assert evidence.consumed_by_supervised is False
    assert evidence.incremental_for_agent is True
    assert evidence.reliability_migration == "legacy_scalar_total"
    assert evidence.reliability_components.source == 0.5
    assert evidence.reliability_components.role == 1.0

    with pytest.raises(ValueError, match="report_permission"):
        MetricEvidenceV2.from_mapping({
            "evidence_id": "metric:x", "metric_id": "x", "report_permission": "no",
            "reliability": 1.0,
        })


def test_missing_or_empty_reliability_fails_closed() -> None:
    missing = MetricEvidenceV2.from_mapping({
        "evidence_id": "metric:x", "metric_id": "x",
    })
    assert missing.reliability_components.source == 0.0
    assert missing.reliability_migration == "missing_fail_closed"
    assert sum(missing.reliability_components.to_dict().values()) == 5.0
    with pytest.raises(ValueError, match="cannot be empty"):
        MetricEvidenceV2.from_mapping({
            "evidence_id": "metric:x", "metric_id": "x",
            "reliability_components": {},
        })


def test_downweight_applies_target_multiplier_once() -> None:
    from advoice.evidence_replay import _downweighted_reliability, _scalar_reliability

    updated = _downweighted_reliability(ReliabilityComponents(), 0.5)
    evidence = MetricEvidenceV2(
        evidence_id="metric:x", metric_id="x", reliability_components=updated,
    )
    assert _scalar_reliability(evidence) == pytest.approx(0.5)
