from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from advoice.evidence import MetricEvidenceV2, ReferenceMetadata
from advoice.evidence_replay import (
    EvidenceRevision,
    EvidenceRevisionError,
    apply_evidence_revision,
    evidence_snapshot_hash,
    replay_evidence,
)
from advoice.module_a import TaskConditionedStatisticalExpert


def _snapshot() -> tuple[MetricEvidenceV2, ...]:
    reference = ReferenceMetadata(median=0.0, scale=1.0, sample_size=12)
    return (
        MetricEvidenceV2(
            evidence_id="metric:a", metric_id="a", metric_instance_id="a", subject_id="case-1",
            state_id="S01", value=1.0, direction=1, reference=reference,
        ),
        MetricEvidenceV2(
            evidence_id="metric:b", metric_id="b", metric_instance_id="b", subject_id="case-1",
            state_id="S01", value=5.0, direction=1, reference=reference,
        ),
    )


def _expert() -> TaskConditionedStatisticalExpert:
    frame = pd.DataFrame({"state_S01": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]})
    return TaskConditionedStatisticalExpert(["HC", "AD"], c=1.0).fit(
        frame, ["HC", "HC", "HC", "AD", "AD", "AD"], feature_columns=["state_S01"],
        artifact_snapshot={"fold": "frozen-1"},
    )


def _states() -> dict:
    return {"states": [{"id": "S01", "metrics": ["a", "b"], "weights": [1.0, 1.0]}]}


def test_accepted_revision_rebuilds_numeric_state_and_packet() -> None:
    snapshot = _snapshot()
    expert = _expert()
    baseline = replay_evidence(snapshot, None, states_config=_states(), module_a=expert)
    revision = EvidenceRevision(
        evidence_id="metric:a", action="downweight",
        expected_evidence_hash=evidence_snapshot_hash(snapshot), reliability_multiplier=0.25,
        rationale="Review found an alignment defect.", cited_evidence_ids=("qc:alignment",),
    )
    replayed = replay_evidence(snapshot, revision, states_config=_states(), module_a=expert)

    assert baseline.state_graph.wide.loc[0, "state_S01"] != replayed.state_graph.wide.loc[0, "state_S01"]
    assert baseline.packet.raw_probabilities != replayed.packet.raw_probabilities
    assert replayed.audit.parent_evidence_hash == evidence_snapshot_hash(snapshot)
    assert replayed.audit.evidence_hash != baseline.audit.evidence_hash
    assert replayed.audit.state_hash == replayed.state_graph.state_hash
    assert replayed.audit.model_hash == expert.artifact_hash_


def test_replay_audit_is_idempotent() -> None:
    snapshot = _snapshot()
    revision = EvidenceRevision(
        evidence_id="metric:a", action="downweight", expected_evidence_hash=evidence_snapshot_hash(snapshot),
        reliability_multiplier=0.5,
    )
    first = replay_evidence(snapshot, revision, states_config=_states(), module_a=_expert())
    second = replay_evidence(snapshot, revision, states_config=_states(), module_a=_expert())

    assert first.audit == second.audit
    assert first.packet.to_json() == second.packet.to_json()


def test_stale_hash_and_forbidden_edits_are_rejected() -> None:
    snapshot = _snapshot()
    stale = EvidenceRevision(evidence_id="metric:a", action="invalidate", expected_evidence_hash="stale")
    with pytest.raises(EvidenceRevisionError, match="stale"):
        apply_evidence_revision(snapshot, stale)
    with pytest.raises(TypeError):
        EvidenceRevision(  # type: ignore[call-arg]
            evidence_id="metric:a", action="invalidate", expected_evidence_hash=evidence_snapshot_hash(snapshot), value=2.0
        )
    with pytest.raises(TypeError):
        EvidenceRevision(  # type: ignore[call-arg]
            evidence_id="metric:a", action="invalidate", expected_evidence_hash=evidence_snapshot_hash(snapshot),
            direction=-1, observable=False, permissions={"report": True},
        )
    with pytest.raises(FrozenInstanceError):
        stale.action = "downweight"  # type: ignore[misc]


def test_no_revision_preserves_evidence_state_packet_and_audit() -> None:
    snapshot = _snapshot()
    first = replay_evidence(snapshot, None, states_config=_states(), module_a=_expert())
    second = replay_evidence(snapshot, None, states_config=_states(), module_a=_expert())

    assert first.revised_evidence == snapshot
    assert first.packet.to_json() == second.packet.to_json()
    assert first.audit == second.audit


def test_unavailability_never_rewrites_observability() -> None:
    snapshot = _snapshot()
    revised, _ = apply_evidence_revision(
        snapshot,
        EvidenceRevision(
            evidence_id="metric:a", action="request_remeasurement",
            expected_evidence_hash=evidence_snapshot_hash(snapshot),
        ),
    )
    assert revised[0].observable is snapshot[0].observable
    assert revised[0].unavailable_reason == "remeasurement_requested_by_evidence_revision"
