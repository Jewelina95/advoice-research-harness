from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from advoice.evidence import EvidencePermissions, MetricEvidenceV2
from advoice.evidence_replay import evidence_snapshot_hash
from advoice.evidence_revision_batch import (
    EvidenceRevisionBatchError,
    compile_evidence_revision_batch,
)
from advoice.state_graph import StateGraphV2


def _evidence(*, evidence_id: str, state_id: str = "S01", case_id: str = "case-1", **kwargs) -> MetricEvidenceV2:
    return MetricEvidenceV2(
        evidence_id=evidence_id,
        metric_id=evidence_id,
        metric_instance_id=evidence_id,
        subject_id=case_id,
        case_id=case_id,
        state_id=state_id,
        task_id="picture",
        value=kwargs.pop("value", 1.0),
        direction=1,
        consumed_by_supervised=kwargs.pop("consumed_by_supervised", True),
        incremental_for_agent=kwargs.pop("incremental_for_agent", False),
        permissions=kwargs.pop("permissions", EvidencePermissions(inference=True, report=True)),
        **kwargs,
    )


def _graph(evidence: list[MetricEvidenceV2]) -> StateGraphV2:
    frame = pd.DataFrame([
        {
            "dataset_id": "fixture",
            "subject_id": item.subject_id,
            "case_id": item.case_id,
            "label": "unknown",
            "split": "test",
            "evidence_id": item.evidence_id,
            "metric_id": item.metric_id,
            "metric_instance_id": item.metric_instance_id,
            "state_id": item.state_id,
            "task_scope": item.task_id or "overall",
            "directional_z": 1.0,
            "reliability": 1.0,
            "missing": False,
            "evidence_status": "available",
            "report_permission": True,
        }
        for item in evidence
    ])
    return StateGraphV2.from_evidence_frame(
        frame,
        {"states": [{"id": "S01", "metrics": ["m1", "m2"], "weights": [1.0, 1.0]}]},
    )


def _compile(evidence: list[MetricEvidenceV2], action: str = "invalidate", **kwargs):
    return compile_evidence_revision_batch(
        "case-1",
        "S01",
        action,
        _graph(evidence),
        evidence,
        evidence_snapshot_hash(evidence),
        **kwargs,
    )


def test_compiles_every_consumed_inferable_metric_atomically_and_sorts() -> None:
    evidence = [_evidence(evidence_id="m2"), _evidence(evidence_id="m1")]
    batch = _compile(evidence, "downweight")

    assert batch.action == "downweight"
    assert batch.evidence_ids == ("m1", "m2")
    assert all(item.reliability_multiplier == 0.5 for item in batch.revisions)
    assert all(item.expected_evidence_hash == evidence_snapshot_hash(evidence) for item in batch.revisions)
    assert all("value" not in item.to_dict() for item in batch.revisions)


def test_retain_is_an_explicit_immutable_noop() -> None:
    evidence = [_evidence(evidence_id="m1")]
    batch = _compile(evidence, "retain")

    assert batch.revisions == ()
    assert batch.evidence_ids == ()
    assert batch.to_json() == batch.to_json()
    with pytest.raises(FrozenInstanceError):
        batch.action = "invalidate"


def test_batch_serialization_is_order_invariant() -> None:
    first = [_evidence(evidence_id="m1"), _evidence(evidence_id="m2")]
    second = list(reversed(first))

    left = _compile(first, "mark_unavailable")
    right = _compile(second, "mark_unavailable")

    assert left.to_json() == right.to_json()
    assert left.batch_hash == right.batch_hash


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda rows: rows + [_evidence(evidence_id="foreign", case_id="case-2")], "Cross-case"),
        (lambda rows: [_evidence(evidence_id="m1", permissions=EvidencePermissions(inference=False, report=True))], "inferable"),
        (lambda rows: [_evidence(evidence_id="m1", consumed_by_supervised=False, incremental_for_agent=True)], "Incremental Agent"),
    ],
)
def test_invalid_authority_inputs_fail_closed(mutate, match) -> None:
    evidence = mutate([_evidence(evidence_id="m1")])
    with pytest.raises(EvidenceRevisionBatchError, match=match):
        _compile(evidence)


def test_stale_or_mixed_hash_fails_closed() -> None:
    evidence = [_evidence(evidence_id="m1"), _evidence(evidence_id="m2")]
    with pytest.raises(EvidenceRevisionBatchError, match="hash"):
        compile_evidence_revision_batch(
            "case-1", "S01", "invalidate", _graph(evidence), evidence, "stale-hash"
        )


def test_non_supervised_state_evidence_is_not_promoted_into_the_batch() -> None:
    evidence = [
        _evidence(evidence_id="m1"),
        _evidence(evidence_id="agent-only", consumed_by_supervised=False),
    ]
    batch = _compile(evidence, "invalidate")
    assert batch.evidence_ids == ("m1",)


def test_unknown_state_and_empty_snapshot_fail_closed() -> None:
    evidence = [_evidence(evidence_id="m1")]
    with pytest.raises(EvidenceRevisionBatchError, match="Unknown state"):
        compile_evidence_revision_batch(
            "case-1", "S99", "invalidate", _graph(evidence), evidence, evidence_snapshot_hash(evidence)
        )
    with pytest.raises(EvidenceRevisionBatchError, match="empty"):
        compile_evidence_revision_batch(
            "case-1", "S01", "retain", _graph(evidence), (), evidence_snapshot_hash(())
        )


def test_compilation_does_not_mutate_raw_evidence() -> None:
    evidence = [_evidence(evidence_id="m1"), _evidence(evidence_id="m2")]
    before = tuple(item.to_json() for item in evidence)
    _compile(evidence, "invalidate")
    assert before == tuple(item.to_json() for item in evidence)
