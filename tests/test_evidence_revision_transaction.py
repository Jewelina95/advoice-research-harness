from __future__ import annotations

import pandas as pd
import pytest

from advoice.evidence import MetricEvidenceV2, ReferenceMetadata
from advoice.evidence_replay import (
    EvidenceRevision,
    EvidenceRevisionError,
    apply_evidence_revision_transaction,
    evidence_snapshot_hash,
    replay_evidence,
    replay_revision_hash,
)
from advoice.evidence_revision_batch import EvidenceRevisionBatch
from advoice.evidence_revision_transaction import (
    EvidenceRevisionTransaction,
    EvidenceRevisionTransactionError,
)
from advoice.module_a import TaskConditionedStatisticalExpert


def _snapshot(*, case_id: str = "case-1") -> tuple[MetricEvidenceV2, ...]:
    reference = ReferenceMetadata(median=0.0, scale=1.0, sample_size=12)
    return (
        MetricEvidenceV2(
            evidence_id="metric:s01", metric_id="s01", metric_instance_id="s01",
            subject_id=case_id, case_id=case_id, state_id="S01", value=1.0,
            direction=1, reference=reference, consumed_by_supervised=True,
        ),
        MetricEvidenceV2(
            evidence_id="metric:s02", metric_id="s02", metric_instance_id="s02",
            subject_id=case_id, case_id=case_id, state_id="S02", value=3.0,
            direction=1, reference=reference, consumed_by_supervised=True,
        ),
    )


def _states() -> dict:
    return {
        "states": [
            {"id": "S01", "metrics": ["s01"], "weights": [1.0]},
            {"id": "S02", "metrics": ["s02"], "weights": [1.0]},
        ]
    }


def _expert() -> TaskConditionedStatisticalExpert:
    frame = pd.DataFrame({
        "state_S01": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        "state_S02": [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0],
    })
    return TaskConditionedStatisticalExpert(["HC", "AD"], c=1.0).fit(
        frame,
        ["HC", "HC", "HC", "AD", "AD", "AD"],
        feature_columns=["state_S01", "state_S02"],
        artifact_snapshot={"fold": "transaction-fixture"},
    )


def _batch(
    snapshot: tuple[MetricEvidenceV2, ...],
    *,
    state_id: str,
    evidence_id: str,
    case_id: str = "case-1",
    expected_hash: str | None = None,
) -> EvidenceRevisionBatch:
    evidence_hash = expected_hash or evidence_snapshot_hash(snapshot)
    return EvidenceRevisionBatch(
        case_id=case_id,
        state_id=state_id,
        action="downweight",
        expected_evidence_hash=evidence_hash,
        revisions=(
            EvidenceRevision(
                evidence_id=evidence_id,
                action="downweight",
                expected_evidence_hash=evidence_hash,
                reliability_multiplier=0.5,
            ),
        ),
    )


def _transaction(snapshot: tuple[MetricEvidenceV2, ...]) -> EvidenceRevisionTransaction:
    return EvidenceRevisionTransaction(
        case_id="case-1",
        expected_evidence_hash=evidence_snapshot_hash(snapshot),
        batches=(
            _batch(snapshot, state_id="S02", evidence_id="metric:s02"),
            _batch(snapshot, state_id="S01", evidence_id="metric:s01"),
        ),
    )


def test_transaction_canonicalizes_state_order_and_hashes_full_serialized_batches() -> None:
    snapshot = _snapshot()
    first = _transaction(snapshot)
    second = EvidenceRevisionTransaction(
        case_id="case-1",
        expected_evidence_hash=evidence_snapshot_hash(snapshot),
        batches=tuple(reversed(first.batches)),
    )

    assert tuple(batch.state_id for batch in first.batches) == ("S01", "S02")
    assert first.to_json() == second.to_json()
    assert first.transaction_hash == second.transaction_hash
    assert first.transaction_hash != first.batches[0].batch_hash


def test_transaction_replay_changes_multiple_states_and_binds_one_transaction_hash() -> None:
    snapshot = _snapshot()
    transaction = _transaction(snapshot)

    replayed = replay_evidence(
        snapshot, transaction, states_config=_states(), module_a=_expert(),
    )

    assert [
        item.reliability_components.measurement_stability
        for item in replayed.revised_evidence
    ] == [0.5, 0.5]
    assert replayed.audit.revision_hash == transaction.transaction_hash
    assert replayed.audit.revision_hash == replay_revision_hash(transaction)
    assert set(replayed.state_graph.cards["revision_hash"]) == {transaction.transaction_hash}


@pytest.mark.parametrize(
    "factory, match",
    [
        (
            lambda snapshot: EvidenceRevisionTransaction(
                case_id="case-1",
                expected_evidence_hash="f" * 64,
                batches=(
                    _batch(snapshot, state_id="S01", evidence_id="metric:s01", expected_hash="f" * 64),
                ),
            ),
            "stale",
        ),
        (
            lambda snapshot: EvidenceRevisionTransaction(
                case_id="case-1",
                expected_evidence_hash=evidence_snapshot_hash(snapshot),
                batches=(
                    _batch(snapshot, state_id="S99", evidence_id="metric:s01"),
                ),
            ),
            "state_id",
        ),
        (
            lambda snapshot: EvidenceRevisionTransaction(
                case_id="case-1",
                expected_evidence_hash=evidence_snapshot_hash(snapshot),
                batches=(
                    _batch(snapshot, state_id="S01", evidence_id="metric:missing"),
                ),
            ),
            "absent",
        ),
    ],
)
def test_transaction_rejects_stale_wrong_state_and_unknown_evidence(factory, match) -> None:
    snapshot = _snapshot()
    with pytest.raises(EvidenceRevisionError, match=match):
        apply_evidence_revision_transaction(snapshot, factory(snapshot))


def test_transaction_rejects_duplicate_ids_wrong_batch_case_and_cross_case_snapshot() -> None:
    snapshot = _snapshot()
    expected_hash = evidence_snapshot_hash(snapshot)
    first = _batch(snapshot, state_id="S01", evidence_id="metric:s01")
    duplicate = _batch(snapshot, state_id="S02", evidence_id="metric:s01")
    with pytest.raises(EvidenceRevisionTransactionError, match="multiple state batches"):
        EvidenceRevisionTransaction(
            case_id="case-1", expected_evidence_hash=expected_hash, batches=(first, duplicate)
        )

    foreign_batch = _batch(
        snapshot, state_id="S02", evidence_id="metric:s02", case_id="case-2"
    )
    with pytest.raises(EvidenceRevisionTransactionError, match="another case"):
        EvidenceRevisionTransaction(
            case_id="case-1", expected_evidence_hash=expected_hash, batches=(first, foreign_batch)
        )

    foreign_snapshot = _snapshot(case_id="case-2")
    foreign_hash = evidence_snapshot_hash(foreign_snapshot)
    transaction = EvidenceRevisionTransaction(
        case_id="case-1",
        expected_evidence_hash=foreign_hash,
        batches=(
            _batch(
                foreign_snapshot,
                state_id="S01",
                evidence_id="metric:s01",
                expected_hash=foreign_hash,
            ),
        ),
    )
    with pytest.raises(EvidenceRevisionError, match="case_id"):
        apply_evidence_revision_transaction(foreign_snapshot, transaction)


def test_transaction_rejects_mixed_snapshot_batches_and_never_partially_replays() -> None:
    snapshot = _snapshot()
    before = tuple(item.to_json() for item in snapshot)
    with pytest.raises(EvidenceRevisionTransactionError, match="mixed evidence"):
        EvidenceRevisionTransaction(
            case_id="case-1",
            expected_evidence_hash=evidence_snapshot_hash(snapshot),
            batches=(
                _batch(snapshot, state_id="S01", evidence_id="metric:s01"),
                _batch(
                    snapshot,
                    state_id="S02",
                    evidence_id="metric:s02",
                    expected_hash="e" * 64,
                ),
            ),
        )

    wrong_state = EvidenceRevisionTransaction(
        case_id="case-1",
        expected_evidence_hash=evidence_snapshot_hash(snapshot),
        batches=(
            _batch(snapshot, state_id="S01", evidence_id="metric:s01"),
            _batch(snapshot, state_id="S99", evidence_id="metric:s02"),
        ),
    )
    with pytest.raises(EvidenceRevisionError, match="state_id"):
        apply_evidence_revision_transaction(snapshot, wrong_state)
    assert tuple(item.to_json() for item in snapshot) == before


def test_transaction_mapping_round_trip_and_tamper_rejection() -> None:
    transaction = _transaction(_snapshot())
    assert EvidenceRevisionTransaction.from_mapping(transaction.to_dict()) == transaction

    tampered = transaction.to_dict()
    tampered["transaction_hash"] = "0" * 64
    with pytest.raises(EvidenceRevisionTransactionError, match="transaction_hash"):
        EvidenceRevisionTransaction.from_mapping(tampered)
