from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from math import fsum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def portable_source_descriptor(path: Path) -> dict[str, str]:
    """Describe an input reproducibly without publishing a local filesystem path."""

    return {
        "filename": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _probability_matrix(
    cases: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    class_order: Sequence[str],
) -> np.ndarray:
    rows: list[list[float]] = []
    for case in cases:
        probabilities = case[arm]["probabilities"]
        row = [max(0.0, float(probabilities[label])) for label in class_order]
        total = fsum(row)
        if total <= 0.0:
            raise ValueError(f"{arm} probabilities must have positive mass")
        rows.append([value / total for value in row])
    return np.asarray(rows, dtype=float)


def _prediction_metrics(
    truth: Sequence[str],
    probability: np.ndarray,
    *,
    class_order: Sequence[str],
) -> dict[str, float | None]:
    labels = tuple(str(label) for label in class_order)
    predicted = [labels[index] for index in probability.argmax(axis=1)]
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "observed_class_macro_f1": float(
            f1_score(truth, predicted, average="macro", zero_division=0)
        ),
        # sklearn orders string labels lexicographically inside LabelBinarizer.
        # Encode explicitly so probability columns remain bound to class_order.
        "log_loss": float(
            log_loss(
                [labels.index(str(item)) for item in truth],
                probability,
                labels=list(range(len(labels))),
            )
        ),
        "macro_auroc_ovr": None,
    }
    present = set(truth)
    if len(present) == len(labels):
        binary = np.asarray([[int(item == label) for label in labels] for item in truth])
        result["macro_auroc_ovr"] = float(
            roc_auc_score(binary, probability, average="macro", multi_class="ovr")
        )
    return result


def evaluate_agent_gain(
    cases: Sequence[Mapping[str, Any]],
    truth_by_case: Mapping[str, str],
    *,
    class_order: Sequence[str],
) -> dict[str, Any]:
    """Evaluate frozen versus final decisions on exactly paired cases.

    The result deliberately uses a pilot label for small cohorts. A positive
    point estimate is not promoted to a general Agent-effect claim.
    """

    completed = [case for case in cases if case.get("status") == "completed"]
    if not completed:
        raise ValueError("at least one completed case is required")
    missing_truth = [case["case_id"] for case in completed if case["case_id"] not in truth_by_case]
    if missing_truth:
        raise KeyError(f"truth missing for cases: {missing_truth}")
    truth = [str(truth_by_case[case["case_id"]]) for case in completed]
    frozen_probability = _probability_matrix(completed, arm="frozen", class_order=class_order)
    fused_probability = _probability_matrix(completed, arm="fusion", class_order=class_order)
    labels = tuple(str(label) for label in class_order)
    frozen_predicted = [labels[index] for index in frozen_probability.argmax(axis=1)]
    fused_predicted = [labels[index] for index in fused_probability.argmax(axis=1)]

    helped = harmed = unchanged = 0
    changed_cases: list[dict[str, str]] = []
    case_effects: list[dict[str, Any]] = []
    for row_index, (case, expected, before, after) in enumerate(zip(
        completed, truth, frozen_predicted, fused_predicted, strict=True
    )):
        truth_index = labels.index(expected)
        before_true_probability = float(frozen_probability[row_index, truth_index])
        after_true_probability = float(fused_probability[row_index, truth_index])
        case_effects.append(
            {
                "case_id": str(case["case_id"]),
                "truth": expected,
                "before": before,
                "after": after,
                "before_true_probability": before_true_probability,
                "after_true_probability": after_true_probability,
                "true_probability_delta": after_true_probability - before_true_probability,
            }
        )
        if before == after:
            unchanged += 1
        elif before != expected and after == expected:
            helped += 1
        elif before == expected and after != expected:
            harmed += 1
        else:
            unchanged += 1
        if before != after:
            changed_cases.append(
                {"case_id": str(case["case_id"]), "truth": expected, "before": before, "after": after}
            )

    discordant = helped + harmed
    paired_p = float(binomtest(helped, discordant, 0.5).pvalue) if discordant else 1.0
    frozen_metrics = _prediction_metrics(truth, frozen_probability, class_order=labels)
    fused_metrics = _prediction_metrics(truth, fused_probability, class_order=labels)
    deltas = {
        key: (
            None
            if frozen_metrics[key] is None or fused_metrics[key] is None
            else float(fused_metrics[key] - frozen_metrics[key])
        )
        for key in frozen_metrics
    }
    return {
        "case_count": len(completed),
        "class_order": list(labels),
        "frozen": frozen_metrics,
        "fused": fused_metrics,
        "delta": deltas,
        "paired_counts": {"helped": helped, "harmed": harmed, "unchanged": unchanged},
        "paired_exact_p_value": paired_p,
        "changed_cases": changed_cases,
        "case_effects": case_effects,
        "claim_status": "pilot_only" if len(completed) < 30 else "formal_cohort",
    }


def _evidence_scope(evidence_id: str) -> dict[str, str]:
    scope: dict[str, str] = {}
    for token in str(evidence_id).split(":"):
        if "=" in token:
            key, value = token.split("=", 1)
            scope[key] = value
    return scope


def audit_permission_compliance(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit citation scope; inference permission itself is runtime-enforced.

    Serialized case audits do not contain the complete hidden evidence pool, so
    this function does not pretend to reconstruct report or inference access.
    It verifies that every emitted citation stays inside its case and state.
    """

    violations: list[dict[str, Any]] = []
    citation_count = 0
    for case in cases:
        case_id = str(case.get("case_id", ""))
        transaction = case.get("transaction") or {}
        for batch in transaction.get("batches", []):
            state_id = str(batch.get("state_id", ""))
            for revision in batch.get("revisions", []):
                citations = revision.get("cited_evidence_ids") or []
                for evidence_id in citations:
                    citation_count += 1
                    scope = _evidence_scope(str(evidence_id))
                    reasons: list[str] = []
                    if scope.get("case") != case_id or scope.get("subject") != case_id:
                        reasons.append("case_mismatch")
                    if scope.get("state") != state_id:
                        reasons.append("state_mismatch")
                    if reasons:
                        violations.append(
                            {
                                "case_id": case_id,
                                "state_id": state_id,
                                "evidence_id": str(evidence_id),
                                "reasons": reasons,
                            }
                        )
    return {
        "case_count": len(cases),
        "citation_count": citation_count,
        "violation_count": len(violations),
        "violations": violations,
        "citation_scope_compliance": (
            1.0 if citation_count == 0 else (citation_count - len(violations)) / citation_count
        ),
        "inference_permission_status": "interface_verified_by_runtime_schema_and_validator",
        "report_permission_status": "not_testable_report_generation_deferred",
    }


def audit_trace_integrity(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check the serialized linkage chain without claiming causal faithfulness."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        provenance = case.get("provenance") or {}
        fusion = case.get("fusion") or {}
        fusion_provenance = fusion.get("provenance") or {}
        transaction = case.get("transaction")
        expected_transaction = None if not transaction else transaction.get("transaction_hash")
        checks = {
            "frozen_packet_link": provenance.get("frozen_packet_hash")
            == (case.get("frozen") or {}).get("packet_hash"),
            "pre_packet_link": provenance.get("pre_packet_hash")
            == (case.get("pre_state") or {}).get("packet_hash"),
            "post_packet_link": provenance.get("post_packet_hash")
            == (case.get("post_state") or {}).get("packet_hash"),
            "transaction_link": provenance.get("transaction_hash") == expected_transaction,
            "revision_link": provenance.get("revision_hash") == expected_transaction,
            "fusion_frozen_link": fusion_provenance.get("frozen_packet_hash")
            == (case.get("frozen") or {}).get("packet_hash"),
            "fusion_pre_link": fusion_provenance.get("pre_packet_hash")
            == (case.get("pre_state") or {}).get("packet_hash"),
            "fusion_post_link": fusion_provenance.get("post_packet_hash")
            == (case.get("post_state") or {}).get("packet_hash"),
            "fusion_revision_link": fusion_provenance.get("revision_hash") == expected_transaction,
            "fusion_audit_hash_present": bool(fusion.get("audit_hash")),
            "fusion_input_hash_present": bool(fusion.get("input_hash")),
            "replay_audit_hashes_present": bool(provenance.get("pre_replay_audit_hash"))
            and bool(provenance.get("post_replay_audit_hash")),
        }
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "complete": all(checks.values()),
                "checks": checks,
            }
        )
    complete = sum(int(row["complete"]) for row in rows)
    return {
        "case_count": len(rows),
        "complete_case_count": complete,
        "trace_complete_rate": 0.0 if not rows else complete / len(rows),
        "cases": rows,
        "interpretation": "structural_linkage_only_not_causal_faithfulness",
    }


def _probability_l1(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    labels = set(before) | set(after)
    return float(fsum(abs(float(before.get(label, 0.0)) - float(after.get(label, 0.0))) for label in labels))


def audit_correction_propagation(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        transaction = case.get("transaction")
        pre = case.get("pre_state") or {}
        post = case.get("post_state") or {}
        frozen = case.get("frozen") or {}
        fusion = case.get("fusion") or {}
        state_l1 = _probability_l1(pre.get("probabilities", {}), post.get("probabilities", {}))
        state_changed = (
            pre.get("packet_hash") != post.get("packet_hash")
            or pre.get("hashes", {}).get("state_hash") != post.get("hashes", {}).get("state_hash")
            or state_l1 > 1e-12
        )
        prediction_probability_l1 = _probability_l1(
            frozen.get("probabilities", {}), fusion.get("probabilities", {})
        )
        prediction_changed = frozen.get("predicted_label") != fusion.get("predicted_label")
        prediction_probability_changed = prediction_probability_l1 > 1e-12
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "has_transaction": bool(transaction),
                "state_changed": state_changed,
                "state_probability_l1": state_l1,
                "prediction_changed": prediction_changed,
                "prediction_probability_changed": prediction_probability_changed,
                "prediction_probability_l1": prediction_probability_l1,
                "correction_applied": bool(fusion.get("correction_applied")),
                "state_gate": float(fusion.get("state_authority_gate", 0.0)),
                "agent_gate": float(fusion.get("agent_authority_gate", 0.0)),
                "staging_gate": float(fusion.get("staging_authority_gate", 0.0)),
            }
        )
    transaction_to_state = sum(row["has_transaction"] and row["state_changed"] for row in rows)
    state_to_prediction = sum(row["state_changed"] and row["prediction_changed"] for row in rows)
    blocked = sum(row["state_changed"] and not row["prediction_changed"] for row in rows)
    failed_state = sum(row["has_transaction"] and not row["state_changed"] for row in rows)
    return {
        "case_count": len(rows),
        "transaction_case_count": sum(row["has_transaction"] for row in rows),
        "transaction_to_state_change": transaction_to_state,
        "transaction_without_state_change": failed_state,
        "state_change_to_prediction_change": state_to_prediction,
        "state_change_blocked_or_subthreshold": blocked,
        "mean_state_probability_l1": (
            0.0 if not rows else float(np.mean([row["state_probability_l1"] for row in rows]))
        ),
        "prediction_consistency_rate": (
            0.0
            if not rows
            else sum(
                row["prediction_probability_changed"] == row["correction_applied"] for row in rows
            )
            / len(rows)
        ),
        "state_gate_active_count": sum(row["state_gate"] > 0.0 for row in rows),
        "agent_gate_active_count": sum(row["agent_gate"] > 0.0 for row in rows),
        "staging_gate_active_count": sum(row["staging_gate"] > 0.0 for row in rows),
        "report_consistency_status": "not_testable_report_generation_deferred",
        "cases": rows,
    }


def _jensen_shannon(first: Sequence[float], second: Sequence[float]) -> float:
    p = np.asarray(first, dtype=float)
    q = np.asarray(second, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    middle = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        first_term = np.where(p > 0.0, p * np.log2(p / middle), 0.0)
        second_term = np.where(q > 0.0, q * np.log2(q / middle), 0.0)
    return float(0.5 * first_term.sum() + 0.5 * second_term.sum())


def evaluate_perturbation_stability(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare baseline and perturbation Agent outputs for each case.

    Required row fields are ``case_id``, ``variant``, ``predicted_label`` and
    ``probabilities``. A baseline row must use ``variant='baseline'``.
    """

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["case_id"])][str(row["variant"])] = row
    paired = {case_id: variants for case_id, variants in grouped.items() if "baseline" in variants and len(variants) > 1}
    if not paired:
        return {"status": "not_run", "reason": "paired perturbation outputs were not supplied"}

    comparisons: list[dict[str, Any]] = []
    for case_id, variants in paired.items():
        baseline = variants["baseline"]
        base_probabilities = baseline["probabilities"]
        labels = tuple(base_probabilities)
        base_vector = [float(base_probabilities[label]) for label in labels]
        base_citations = set(baseline.get("cited_evidence_ids", []))
        for variant_name, variant in variants.items():
            if variant_name == "baseline":
                continue
            vector = [float(variant["probabilities"][label]) for label in labels]
            citations = set(variant.get("cited_evidence_ids", []))
            union = base_citations | citations
            comparisons.append(
                {
                    "case_id": case_id,
                    "variant": variant_name,
                    "label_agreement": baseline["predicted_label"] == variant["predicted_label"],
                    "probability_js_divergence": _jensen_shannon(base_vector, vector),
                    "citation_jaccard": 1.0 if not union else len(base_citations & citations) / len(union),
                }
            )
    return {
        "status": "completed",
        "paired_case_count": len(paired),
        "comparison_count": len(comparisons),
        "label_agreement": float(np.mean([row["label_agreement"] for row in comparisons])),
        "mean_probability_js_divergence": float(
            np.mean([row["probability_js_divergence"] for row in comparisons])
        ),
        "mean_citation_jaccard": float(np.mean([row["citation_jaccard"] for row in comparisons])),
        "comparisons": comparisons,
    }


def summarize_representation_stage(model_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the pretrained-representation and downstream-supervision split."""

    text = model_metadata.get("deep_text_encoder") or {}
    audio = model_metadata.get("deep_audio_encoder") or {}
    base_f1 = model_metadata.get("base_oof_macro_f1")
    final_f1 = model_metadata.get("final_oof_macro_f1")
    base_auc = model_metadata.get("base_oof_macro_auroc")
    final_auc = model_metadata.get("final_oof_macro_auroc")
    return {
        "text_backbone": {"model": text.get("model"), "revision": text.get("revision")},
        "audio_backbone": {"model": audio.get("model"), "revision": audio.get("revision")},
        "audio_window_count": audio.get("window_count"),
        "audio_subject_coverage": audio.get("subject_coverage"),
        "downstream_training": model_metadata.get("base_architecture"),
        "base_oof_macro_f1": base_f1,
        "final_oof_macro_f1": final_f1,
        "base_oof_macro_auroc": base_auc,
        "final_oof_macro_auroc": final_auc,
        "within_training_delta_macro_f1": (
            None if base_f1 is None or final_f1 is None else float(final_f1 - base_f1)
        ),
        "within_training_delta_macro_auroc": (
            None if base_auc is None or final_auc is None else float(final_auc - base_auc)
        ),
        "interpretation": (
            "pretrained self-supervised/representation backbones followed by supervised downstream fitting; "
            "these deltas are not the post-Agent causal effect"
        ),
        "representation_ablation_status": "not_identified_without_a_matched_no_backbone_arm",
    }
