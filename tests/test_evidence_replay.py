from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from advoice.evidence import MetricEvidenceV2, ReferenceMetadata
from advoice.evidence_replay import (
    EvidenceRevision,
    EvidenceRevisionError,
    atomic_revision_batch_hash,
    apply_evidence_revision_batch,
    apply_evidence_revision,
    build_state_graph_v2,
    evidence_snapshot_hash,
    replay_revision_hash,
    replay_evidence,
)
from advoice.evidence_revision_batch import EvidenceRevisionBatch
from advoice.module_a import TaskConditionedStatisticalExpert, snapshot_hash


def _snapshot() -> tuple[MetricEvidenceV2, ...]:
    reference = ReferenceMetadata(median=0.0, scale=1.0, sample_size=12)
    return (
        MetricEvidenceV2(
            evidence_id="metric:a", metric_id="a", metric_instance_id="a", subject_id="case-1",
            state_id="S01", value=1.0, direction=1, reference=reference,
            consumed_by_supervised=True,
        ),
        MetricEvidenceV2(
            evidence_id="metric:b", metric_id="b", metric_instance_id="b", subject_id="case-1",
            state_id="S01", value=5.0, direction=1, reference=reference,
            consumed_by_supervised=True,
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


def _batch(
    snapshot: tuple[MetricEvidenceV2, ...],
    *,
    expected_hash: str | None = None,
) -> EvidenceRevisionBatch:
    snapshot_hash = expected_hash or evidence_snapshot_hash(snapshot)
    revisions = tuple(
        EvidenceRevision(
            evidence_id=evidence_id,
            action="downweight",
            expected_evidence_hash=snapshot_hash,
            reliability_multiplier=0.25,
        )
        for evidence_id in ("metric:a", "metric:b")
    )
    return EvidenceRevisionBatch(
        case_id="case-1",
        state_id="S01",
        action="downweight",
        expected_evidence_hash=snapshot_hash,
        revisions=revisions,
    )


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


def test_batch_replay_applies_every_metric_atomically_and_binds_combined_hash() -> None:
    snapshot = _snapshot()
    batch = _batch(snapshot)

    result = replay_evidence(snapshot, batch, states_config=_states(), module_a=_expert())

    assert [
        item.reliability_components.measurement_stability
        for item in result.revised_evidence
    ] == [0.25, 0.25]
    assert result.audit.revision_hash == replay_revision_hash(batch)
    assert result.audit.revision_hash not in {item.revision_hash for item in batch.revisions}
    assert result.audit.revision_hash == batch.batch_hash
    assert set(result.state_graph.cards["revision_hash"]) == {result.audit.revision_hash}
    assert set(result.state_graph.cards["state_revision_hash"]) == {result.audit.revision_hash}
    assert result.packet.hashes["evidence_snapshot_hash"] == snapshot_hash({
        "evidence_hash": result.audit.evidence_hash,
        "revision_hash": result.audit.revision_hash,
    })
    assert result.packet.hashes["state_snapshot_hash"] == snapshot_hash({
        "state_hash": result.state_graph.state_hash,
        "state_wide": result.state_graph.wide.to_dict("records"),
    })


def test_batch_replay_hash_is_deterministic_for_ordered_atomic_revisions() -> None:
    snapshot = _snapshot()
    first = replay_evidence(snapshot, _batch(snapshot), states_config=_states(), module_a=_expert())
    second = replay_evidence(snapshot, _batch(snapshot), states_config=_states(), module_a=_expert())

    assert first.audit.revision_hash == second.audit.revision_hash
    assert first.audit == second.audit
    assert first.packet.to_json() == second.packet.to_json()
    assert atomic_revision_batch_hash(_batch(snapshot).revisions) != atomic_revision_batch_hash(
        tuple(reversed(_batch(snapshot).revisions))
    )


def test_batch_prevalidation_rejects_stale_member_without_partial_application() -> None:
    snapshot = _snapshot()
    stale_batch = _batch(snapshot, expected_hash="f" * 64)
    mixed = (
        _batch(snapshot).revisions[0],
        stale_batch.revisions[1],
    )
    before = tuple(item.to_json() for item in snapshot)

    with pytest.raises(EvidenceRevisionError, match="stale"):
        apply_evidence_revision_batch(snapshot, stale_batch)
    with pytest.raises(EvidenceRevisionError, match="mixed"):
        apply_evidence_revision_batch(snapshot, mixed)

    assert tuple(item.to_json() for item in snapshot) == before


def test_batch_prevalidation_rejects_duplicate_and_unknown_evidence_ids() -> None:
    snapshot = _snapshot()
    snapshot_hash = evidence_snapshot_hash(snapshot)
    duplicate = EvidenceRevision(
        evidence_id="metric:a", action="invalidate", expected_evidence_hash=snapshot_hash,
    )
    unknown = EvidenceRevision(
        evidence_id="metric:unknown", action="invalidate", expected_evidence_hash=snapshot_hash,
    )

    with pytest.raises(EvidenceRevisionError, match="duplicate"):
        apply_evidence_revision_batch(snapshot, (duplicate, duplicate))
    with pytest.raises(EvidenceRevisionError, match="absent"):
        apply_evidence_revision_batch(snapshot, (duplicate, unknown))


def test_batch_rejects_mixed_actions_and_mismatched_state_provenance() -> None:
    snapshot = _snapshot()
    snapshot_hash = evidence_snapshot_hash(snapshot)
    mixed = (
        EvidenceRevision(
            evidence_id="metric:a", action="invalidate",
            expected_evidence_hash=snapshot_hash,
        ),
        EvidenceRevision(
            evidence_id="metric:b", action="downweight",
            expected_evidence_hash=snapshot_hash, reliability_multiplier=0.5,
        ),
    )
    with pytest.raises(EvidenceRevisionError, match="mix revision actions"):
        apply_evidence_revision_batch(snapshot, mixed)

    wrong_state = EvidenceRevisionBatch(
        case_id="case-1", state_id="S99", action="downweight",
        expected_evidence_hash=snapshot_hash,
        revisions=_batch(snapshot).revisions,
    )
    with pytest.raises(EvidenceRevisionError, match="state_id"):
        apply_evidence_revision_batch(snapshot, wrong_state)


def test_retain_batch_still_rejects_duplicate_snapshot_ids() -> None:
    snapshot = (_snapshot()[0], _snapshot()[0])
    batch = EvidenceRevisionBatch(
        case_id="case-1",
        state_id="S01",
        action="retain",
        expected_evidence_hash=evidence_snapshot_hash(snapshot),
    )

    with pytest.raises(EvidenceRevisionError, match="duplicate"):
        apply_evidence_revision_batch(snapshot, batch)


def test_replay_binds_revision_hash_to_every_state_card_and_state_hash() -> None:
    snapshot = _snapshot()
    baseline = replay_evidence(snapshot, None, states_config=_states(), module_a=_expert())
    expected_no_revision = replay_revision_hash(None)
    assert set(baseline.state_graph.cards["revision_hash"]) == {expected_no_revision}
    assert set(baseline.state_graph.cards["state_revision_hash"]) == {expected_no_revision}
    assert baseline.audit.revision_hash == expected_no_revision
    unbound = build_state_graph_v2(snapshot, _states())
    assert baseline.state_graph.state_hash != unbound.state_hash

    revision = EvidenceRevision(
        evidence_id="metric:a", action="downweight",
        expected_evidence_hash=evidence_snapshot_hash(snapshot),
        reliability_multiplier=0.5,
    )
    accepted = replay_evidence(snapshot, revision, states_config=_states(), module_a=_expert())
    assert set(accepted.state_graph.cards["revision_hash"]) == {revision.revision_hash}
    assert set(accepted.state_graph.cards["state_revision_hash"]) == {revision.revision_hash}
    assert accepted.audit.revision_hash == revision.revision_hash


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


@pytest.mark.parametrize("action", ["invalidate", "request_remeasurement"])
def test_withdrawn_evidence_is_absent_from_module_a_consumed_ids(action: str) -> None:
    snapshot = _snapshot()
    result = replay_evidence(
        snapshot,
        EvidenceRevision(
            evidence_id="metric:a", action=action, expected_evidence_hash=evidence_snapshot_hash(snapshot),
        ),
        states_config=_states(), module_a=_expert(),
    )

    assert "metric:a" not in result.packet.consumed_evidence_ids
    assert result.packet.consumed_evidence_ids == ("metric:b",)


def test_replay_consumed_ids_require_supervised_flag_and_reached_state_feature() -> None:
    reference = ReferenceMetadata(median=0.0, scale=1.0, sample_size=12)
    snapshot = (
        MetricEvidenceV2(
            evidence_id="metric:used", metric_id="used", metric_instance_id="used",
            subject_id="case-1", state_id="S01", value=1.0, direction=1,
            reference=reference, consumed_by_supervised=True,
        ),
        MetricEvidenceV2(
            evidence_id="metric:unflagged", metric_id="unflagged", metric_instance_id="unflagged",
            subject_id="case-1", state_id="S01", value=2.0, direction=1,
            reference=reference,
        ),
        MetricEvidenceV2(
            evidence_id="metric:not-in-graph", metric_id="not-in-graph", metric_instance_id="not-in-graph",
            subject_id="case-1", state_id="S99", value=3.0, direction=1,
            reference=reference, consumed_by_supervised=True,
        ),
    )
    result = replay_evidence(
        snapshot, None,
        states_config={"states": [{"id": "S01", "metrics": ["used", "unflagged"], "weights": [1.0, 1.0]}]},
        module_a=_expert(),
    )
    assert result.packet.consumed_evidence_ids == ("metric:used",)


def test_replay_overwrites_stale_graph_context_feature() -> None:
    snapshot = _snapshot()
    baseline = replay_evidence(snapshot, None, states_config=_states(), module_a=_expert())
    replayed = replay_evidence(
        snapshot, None, states_config=_states(), module_a=_expert(),
        case_context={"state_S01": -999.0},
    )

    assert replayed.packet.to_json() == baseline.packet.to_json()


@pytest.mark.parametrize("feature", ["state_S02", "rel_S02", "available_S02"])
def test_replay_rejects_stale_context_when_trained_graph_feature_is_missing(feature: str) -> None:
    training = pd.DataFrame({feature: [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]})
    expert = TaskConditionedStatisticalExpert(["HC", "AD"], c=1.0).fit(
        training, ["HC", "HC", "HC", "AD", "AD", "AD"], feature_columns=[feature],
    )

    with pytest.raises(EvidenceRevisionError, match="missing trained feature"):
        replay_evidence(
            _snapshot(), None, states_config=_states(), module_a=expert,
            case_context={feature: 999.0},
        )
