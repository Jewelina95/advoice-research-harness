from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from advoice.post_agent_evaluation import (
    _prediction_metrics,
    audit_correction_propagation,
    audit_permission_compliance,
    audit_trace_integrity,
    evaluate_agent_gain,
    evaluate_perturbation_stability,
    portable_source_descriptor,
)


LABELS = ("HC", "MCI", "AD")


def test_portable_source_descriptor_omits_local_path(tmp_path: Path) -> None:
    source = tmp_path / "case_audit.json"
    source.write_text('{"case": 1}', encoding="utf-8")

    descriptor = portable_source_descriptor(source)

    assert descriptor["filename"] == "case_audit.json"
    assert len(descriptor["sha256"]) == 64
    assert str(tmp_path) not in json.dumps(descriptor)


def test_probability_metrics_respect_declared_nonlexicographic_class_order() -> None:
    probability = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.20, 0.70],
        ]
    )

    result = _prediction_metrics(
        ["HC", "MCI", "AD"], probability, class_order=LABELS
    )

    expected = -sum(math.log(value) for value in (0.70, 0.80, 0.70)) / 3.0
    assert result["accuracy"] == 1.0
    assert result["log_loss"] == pytest.approx(expected)


def _case(
    case_id: str,
    *,
    frozen: str,
    fused: str,
    transaction: bool = True,
    state_changed: bool = True,
) -> dict:
    evidence_id = (
        f"metric:dataset=fixture:session={case_id}:case={case_id}:subject={case_id}:"
        "task=picture_description:metric=pause_rate:state=S01"
    )
    transaction_hash = "a" * 64
    pre_hash = "b" * 64
    post_hash = "c" * 64 if state_changed else pre_hash
    frozen_probabilities = {
        "HC": 0.8 if frozen == "HC" else 0.1,
        "MCI": 0.8 if frozen == "MCI" else 0.1,
        "AD": 0.8 if frozen == "AD" else 0.1,
    }
    fused_probabilities = {
        "HC": 0.8 if fused == "HC" else 0.1,
        "MCI": 0.8 if fused == "MCI" else 0.1,
        "AD": 0.8 if fused == "AD" else 0.1,
    }
    return {
        "case_id": case_id,
        "status": "completed",
        "prepared": {"class_order": list(LABELS)},
        "frozen": {
            "predicted_label": frozen,
            "probabilities": frozen_probabilities,
            "packet_hash": "d" * 64,
        },
        "pre_state": {
            "predicted_label": frozen,
            "probabilities": frozen_probabilities,
            "packet_hash": pre_hash,
            "hashes": {"state_hash": "e" * 64, "evidence_hash": "f" * 64},
        },
        "post_state": {
            "predicted_label": fused,
            "probabilities": fused_probabilities,
            "packet_hash": post_hash,
            "hashes": {
                "state_hash": "1" * 64 if state_changed else "e" * 64,
                "evidence_hash": "2" * 64 if state_changed else "f" * 64,
            },
        },
        "fusion": {
            "predicted_label": fused,
            "probabilities": fused_probabilities,
            "correction_applied": frozen != fused,
            "state_authority_gate": 1.0 if state_changed else 0.0,
            "agent_authority_gate": 0.5 if frozen != fused else 0.0,
            "staging_authority_gate": 0.0,
            "provenance": {
                "frozen_packet_hash": "d" * 64,
                "pre_packet_hash": pre_hash,
                "post_packet_hash": post_hash,
                "revision_hash": transaction_hash if transaction else None,
            },
            "audit_hash": "3" * 64,
            "input_hash": "4" * 64,
        },
        "provenance": {
            "frozen_packet_hash": "d" * 64,
            "pre_packet_hash": pre_hash,
            "post_packet_hash": post_hash,
            "transaction_hash": transaction_hash if transaction else None,
            "revision_hash": transaction_hash if transaction else None,
            "pre_replay_audit_hash": "5" * 64,
            "post_replay_audit_hash": "6" * 64,
        },
        "transaction": (
            {
                "transaction_hash": transaction_hash,
                "batches": [
                    {
                        "state_id": "S01",
                        "revisions": [
                            {
                                "action": "downweight",
                                "evidence_id": evidence_id,
                                "cited_evidence_ids": [evidence_id],
                            }
                        ],
                    }
                ],
            }
            if transaction
            else None
        ),
    }


def test_agent_gain_reports_paired_help_and_harm() -> None:
    cases = [
        _case("a", frozen="HC", fused="AD"),
        _case("b", frozen="MCI", fused="HC"),
        _case("c", frozen="HC", fused="HC", transaction=False, state_changed=False),
    ]
    truth = {"a": "AD", "b": "MCI", "c": "HC"}

    result = evaluate_agent_gain(cases, truth, class_order=LABELS)

    assert result["paired_counts"] == {"helped": 1, "harmed": 1, "unchanged": 1}
    assert result["frozen"]["accuracy"] == result["fused"]["accuracy"]
    assert result["claim_status"] == "pilot_only"


def test_permission_audit_rejects_cross_state_and_cross_case_citations() -> None:
    case = _case("a", frozen="HC", fused="AD")
    revision = case["transaction"]["batches"][0]["revisions"][0]
    revision["cited_evidence_ids"].append(
        "metric:dataset=fixture:session=b:case=b:subject=b:task=x:metric=y:state=S02"
    )

    result = audit_permission_compliance([case])

    assert result["citation_count"] == 2
    assert result["violation_count"] == 1
    assert result["violations"][0]["reasons"] == ["case_mismatch", "state_mismatch"]


def test_trace_audit_requires_matching_hash_chain() -> None:
    valid = _case("a", frozen="HC", fused="AD")
    invalid = json.loads(json.dumps(valid))
    invalid["case_id"] = "b"
    invalid["fusion"]["provenance"]["post_packet_hash"] = "9" * 64

    result = audit_trace_integrity([valid, invalid])

    assert result["complete_case_count"] == 1
    assert result["case_count"] == 2
    assert result["trace_complete_rate"] == 0.5


def test_propagation_separates_state_change_from_final_decision_change() -> None:
    changed = _case("a", frozen="HC", fused="AD")
    blocked = _case("b", frozen="HC", fused="HC", state_changed=True)

    result = audit_correction_propagation([changed, blocked])

    assert result["transaction_to_state_change"] == 2
    assert result["state_change_to_prediction_change"] == 1
    assert result["state_change_blocked_or_subthreshold"] == 1
    assert result["prediction_consistency_rate"] == 1.0
    assert result["report_consistency_status"] == "not_testable_report_generation_deferred"


def test_stability_is_not_claimed_without_paired_repeats() -> None:
    result = evaluate_perturbation_stability([])
    assert result["status"] == "not_run"
    assert result["reason"] == "paired perturbation outputs were not supplied"
