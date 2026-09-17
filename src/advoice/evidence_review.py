"""Immutable evidence snapshots after deterministic validation of Agent edits."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .utils import hash_values


def apply_reviewed_snapshot(workspace: dict[str, Any], candidate: dict[str, Any],
                            audit: dict[str, Any]) -> dict[str, Any]:
    reviewed = deepcopy(workspace)
    if not audit.get("valid"):
        return reviewed
    updates = [u for u in candidate.get("state_updates", []) if u.get("action") != "keep"]
    if not updates:
        return reviewed
    excluded = set(audit.get("invalidated_evidence_ids", []))
    state_keys = {"state_observations", "reportable_state_observations",
                  "inference_only_state_observations", "model_only_state_observations"}
    for key in state_keys:
        retained = []
        for state in reviewed.get(key, []):
            if state.get("evidence_id") in excluded:
                continue
            for update in updates:
                state_id = str(update.get("state_id", ""))
                scope = str(update.get("task_scope", "overall"))
                if (state.get("state_id") in {state_id, f"{state_id}__task_{scope}"}
                        and str(state.get("task_scope", "overall")) == scope
                        and update["action"] == "downweight"):
                    state["confidence"] = float(state.get("confidence", 0.0)) * .5
                    state["review_status"] = "downweighted"
            state["metric_evidence_ids"] = [i for i in state.get("metric_evidence_ids", []) if i not in excluded]
            for field in ("supporting_metrics", "counter_evidence"):
                state[field] = [m for m in state.get(field, []) if
                    f"metric:{m.get('metric_instance_id', m.get('metric_id', ''))}" not in excluded]
            state["evidence_segments"] = [s for s in state.get("evidence_segments", []) if s.get("segment_id") not in excluded]
            retained.append(state)
        if key in reviewed:
            reviewed[key] = retained
    for key in ("selected_supporting_evidence", "selected_counterevidence",
                "inference_only_metric_observations", "evidence_registry"):
        if key in reviewed:
            reviewed[key] = [item for item in reviewed[key] if item.get("evidence_id") not in excluded]
    # These derived summaries are stale after a state edit, not new evidence.
    reviewed.pop("cognitive_state_reference", None)
    reviewed.pop("class_support", None)
    reviewed["evidence_revision"] = {
        "schema_version": "reviewed-evidence-v1", "parent_hash": hash_values([workspace]),
        "actions": deepcopy(updates), "excluded_evidence_ids": sorted(excluded),
        "state_replay_required": True,
    }
    reviewed["evidence_revision"]["snapshot_hash"] = hash_values([{
        key: reviewed.get(key) for key in sorted(state_keys | {
            "selected_supporting_evidence", "selected_counterevidence", "evidence_registry"})
    }])
    return reviewed
