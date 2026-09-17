"""Bounded, evidence-grounded Agent decisions, independent of a supervised prior.

This runtime records tool results, not private chain-of-thought. Model outputs
are optional, correlated advisors; they never replace an Agent decision.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any, Callable

from .cognitive_agent import _allowed_ids, validate_candidate
from .evidence_review import apply_reviewed_snapshot
from .utils import hash_values

VERSION = "agent-led-v1"
STATE_KEYS = (
    "state_observations", "reportable_state_observations",
    "inference_only_state_observations", "model_only_state_observations",
)
EVIDENCE_KEYS = (
    "selected_supporting_evidence", "selected_counterevidence",
    "inference_only_metric_observations", "quality_observations",
)
# An allowlist is intentional: new training metadata must not silently leak into
# inference when an upstream workspace schema gains fields.
WORKSPACE_KEYS = (*STATE_KEYS, *EVIDENCE_KEYS, "case_id", "case_context",
                  "case_input_route", "evidence_registry", "potential_confound_tags", "evidence_revision")
PRIVATE_KEYS = {
    "label", "diagnosis", "true_label", "y_true", "subject_id", "dataset_id",
    "audio_path", "source_path", "path", "filename", "base_probabilities",
    "corrected_probabilities", "final_probabilities", "final_prediction",
    "class_support", "cognitive_state_reference", "correction_alpha",
}
TOOLS = (
    "inspect_quality", "inspect_state", "inspect_metric", "inspect_segment",
    "inspect_counterevidence", "compare_tasks", "record_hypothesis",
    "consult_models", "revise_state", "finalize", "abstain",
)


def public_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): public_evidence(v) for k, v in value.items() if k not in PRIVATE_KEYS}
    if isinstance(value, (tuple, list)):
        return [public_evidence(v) for v in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def evidence_snapshot(workspace: dict[str, Any]) -> dict[str, Any]:
    snapshot = public_evidence({k: deepcopy(workspace[k]) for k in WORKSPACE_KEYS if k in workspace})
    counters = snapshot.setdefault("selected_counterevidence", [])
    known = {item["evidence_id"] for item in counters}
    for state in snapshot.get("state_observations", []):
        for metric in state.get("counter_evidence", []):
            raw_id = metric.get("metric_instance_id", metric.get("metric_id"))
            if raw_id:
                eid = str(raw_id) if str(raw_id).startswith("metric:") else f"metric:{raw_id}"
                if eid not in known:
                    counters.append({**metric, "evidence_id": eid})
                    known.add(eid)
    return snapshot


def response_schema(labels: list[str]) -> dict[str, Any]:
    properties = {
        "action": {"type": "string", "enum": list(TOOLS)},
        "revision": {"type": "string"},
        "target_id": {"type": "string"},
        "state_action": {"type": "string", "enum": ["none", "downweight", "invalidate", "mark_unavailable"]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "counterevidence_ids": {"type": "array", "items": {"type": "string"}},
        "predicted_label": {"type": "string", "enum": labels + ["undetermined"]},
        "scores": {"type": "object", "additionalProperties": False,
                   "properties": {k: {"type": "integer", "minimum": 0, "maximum": 4} for k in labels},
                   "required": labels},
        "rationale": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    }
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


class EvidenceSession:
    def __init__(self, workspace: dict[str, Any], labels: list[str], *,
                 model_id: str, skill_hash: str, max_steps: int = 16, provider: str = "custom"):
        if len(labels) < 2 or len(set(labels)) != len(labels) or "undetermined" in labels:
            raise ValueError("Distinct dataset labels are required.")
        if not workspace.get("case_id") or not 1 <= max_steps <= 64:
            raise ValueError("A case ID and a bounded step budget are required.")
        self.labels = list(labels)
        self.workspace = evidence_snapshot(workspace)
        self.model_id = model_id
        policy_files = [Path(__file__), Path(__file__).with_name("cognitive_agent.py"),
                        Path(__file__).with_name("evidence_review.py"),
                        Path(__file__).with_name("agent_runtime.py")]
        self.policy_hash = hash_values([p.read_text(encoding="utf-8") for p in policy_files])
        provenance = workspace.get("advisor_provenance", {})
        artifacts = provenance.get("artifacts", {}) if isinstance(provenance, dict) else {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        self.advisor_artifacts = {k: artifacts[k] for k in ("module_a", "module_b")
                                 if isinstance(artifacts.get(k), str) and artifacts[k].strip()}
        self.fingerprint = hash_values([VERSION, self.policy_hash, provider, model_id, skill_hash,
                                       labels, max_steps, self.advisor_artifacts])
        self.max_steps = max_steps
        self.revision = hash_values([self.workspace])
        self.initial_revision = self.revision
        self.advisors_bound = (isinstance(provenance, dict)
                               and provenance.get("evidence_hash") == self.revision
                               and not self.workspace.get("evidence_revision", {}).get("state_replay_required", False))
        self.history: list[dict[str, Any]] = []
        self.observed: set[str] = set()
        self.quality_checked = False
        self.counter_checked = False
        self.hypothesis_recorded = False
        self.result: dict[str, Any] | None = None
        self.advisors = {
            "module_a": workspace.get("base_probabilities"),
            "module_b": workspace.get("corrected_probabilities"),
        }

    def _objects(self) -> dict[str, dict[str, Any]]:
        result = {}
        for key in (*STATE_KEYS, *EVIDENCE_KEYS):
            for item in self.workspace.get(key, []):
                if item.get("evidence_id"):
                    result[item["evidence_id"]] = item
                for field in ("supporting_metrics", "counter_evidence"):
                    for metric in item.get(field, []):
                        raw_id = metric.get("metric_instance_id", metric.get("metric_id"))
                        if raw_id:
                            eid = str(raw_id) if str(raw_id).startswith("metric:") else f"metric:{raw_id}"
                            result.setdefault(eid, {**metric, "evidence_id": eid})
                for segment in item.get("evidence_segments", []):
                    if segment.get("segment_id"):
                        result[segment["segment_id"]] = {**segment, "evidence_id": segment["segment_id"]}
        return result

    def observation(self) -> dict[str, Any]:
        states = self.workspace.get("state_observations", [])
        events = deepcopy(self.history)
        for event in events:
            if event["revision"] != self.revision:
                event["result"] = {"status": "superseded", "reason": "Use current evidence, not stale tool output."}
                event["request"] = {"action": event["request"].get("action", "invalid")}
        return {
            "case_id": self.workspace["case_id"], "revision": self.revision,
            "context": self.workspace.get("case_context", {}),
            "states": [{k: s.get(k) for k in ("evidence_id", "state_id", "task_scope", "clinical_question")} for s in states],
            "available_tools": list(TOOLS),
            "remaining_steps": self.max_steps - len(self.history),
            "events": events,
            "clinical_scope": "Research screening only, not biological AD diagnosis or disease staging.",
        }

    def _quality(self) -> dict[str, Any]:
        self.quality_checked = True
        clinical, _, _ = _allowed_ids(self.workspace)
        objects = self._objects()
        findings = []
        for state in self.workspace.get("state_observations", []):
            missing = state.get("missing_fraction")
            confidence = state.get("confidence")
            findings.append({
                "state_id": state["evidence_id"],
                "measurement_confidence": confidence,
                "missing_fraction": missing,
                "source_segments_available": bool(state.get("evidence_segments")),
                "all_metric_objects_available": all(eid in objects for eid in state.get("metric_evidence_ids", [])),
            })
        values = self.workspace.get("quality_observations", [])
        self.observed.update(x["evidence_id"] for x in values)
        return {
            "status": "structural_assessment_only", "clinical_confound_assessment": "unknown",
            "findings": findings, "measurements": values,
            "potential_confound_tags": self.workspace.get("potential_confound_tags", []),
            "clinical_evidence_count": len(clinical),
            "interpretation": "No scalar clinical reliability is inferred from missing history or potential tags.",
        }

    def _candidate(self, reply: dict[str, Any]) -> dict[str, Any]:
        clinical, _, quality = _allowed_ids(self.workspace)
        ids = set(reply["evidence_ids"])
        return {
            "case_id": self.workspace["case_id"], "action": "classify",
            "evidence_class": reply["predicted_label"], "evidence_scores": reply["scores"],
            "used_evidence_ids": sorted(ids & clinical),
            "counterevidence_ids": reply["counterevidence_ids"],
            "quality_evidence_ids": sorted(ids & quality),
            "state_updates": [], "counterevidence_checked": self.counter_checked,
        }

    def _check_final(self, reply: dict[str, Any]) -> dict[str, Any]:
        if not (self.quality_checked and self.counter_checked and self.hypothesis_recorded):
            raise ValueError("Inspect quality/counterevidence and record an independent hypothesis before finalizing.")
        if reply["predicted_label"] not in self.labels:
            raise ValueError("A classification must use a configured label.")
        if not reply["rationale"].strip():
            raise ValueError("A concise evidence rationale is required.")
        scores = reply["scores"]
        maxima = [k for k, v in scores.items() if v == max(scores.values())]
        if maxima != [reply["predicted_label"]]:
            raise ValueError("Tied or inconsistent evidence scores require abstention.")
        clinical, _, _ = _allowed_ids(self.workspace)
        if not set(reply["evidence_ids"]).issubset(clinical):
            raise ValueError("Quality observations cannot support a disease class.")
        if not set(reply["evidence_ids"] + reply["counterevidence_ids"]).issubset(self.observed):
            raise ValueError("Evidence must be inspected in the current revision before citation.")
        audit = validate_candidate(self._candidate(reply), self.workspace, self.labels)
        if not audit["valid"]:
            raise ValueError("; ".join(audit["violations"]))
        return audit

    def _revise(self, reply: dict[str, Any]) -> dict[str, Any]:
        target = self._objects().get(reply["target_id"])
        if not target or not reply["target_id"].startswith("state:"):
            raise ValueError("A revision requires an existing state ID.")
        if not self.counter_checked or not reply["evidence_ids"] or not set(reply["evidence_ids"]).issubset(self.observed):
            raise ValueError("Inspect counterevidence and revision evidence first.")
        if reply["state_action"] == "none" or not reply["rationale"].strip():
            raise ValueError("A supported, explicit state action is required.")
        candidate = {
            "action": "abstain", "evidence_class": self.labels[0],
            "evidence_scores": {k: int(k == self.labels[0]) for k in self.labels},
            "used_evidence_ids": [], "counterevidence_ids": [], "quality_evidence_ids": [],
            "counterevidence_checked": True,
            "state_updates": [{"state_id": target["state_id"], "task_scope": target.get("task_scope", "overall"),
                               "action": reply["state_action"], "reason": reply["rationale"],
                               "evidence_ids": reply["evidence_ids"]}],
        }
        _, counter, _ = _allowed_ids(self.workspace)
        candidate["counterevidence_ids"] = sorted(counter & self.observed)
        # A withdrawn state's counterevidence must not simultaneously be cited.
        if reply["state_action"] in {"invalidate", "mark_unavailable"}:
            removed = set(target.get("metric_evidence_ids", [])) | {target["evidence_id"]}
            candidate["counterevidence_ids"] = [x for x in candidate["counterevidence_ids"] if x not in removed]
        audit = validate_candidate(candidate, self.workspace, self.labels)
        if not audit["valid"]:
            raise ValueError("; ".join(audit["violations"]))
        # Without numeric replay, any state depending on a withdrawn metric is
        # stale too. Withdraw it rather than retaining an old aggregate score.
        excluded = set(audit.get("invalidated_evidence_ids", []))
        dependents = {
            state["evidence_id"] for key in STATE_KEYS for state in self.workspace.get(key, [])
            if set(state.get("metric_evidence_ids", [])) & excluded
        } - excluded
        audit["invalidated_evidence_ids"] = sorted(excluded | dependents)
        before = self.revision
        self.workspace = apply_reviewed_snapshot(self.workspace, candidate, audit)
        self.revision = hash_values([self.workspace])
        self.observed.clear()
        self.counter_checked = self.quality_checked = self.hypothesis_recorded = False
        return {"status": "revised", "previous_revision": before, "revision": self.revision,
                "supervised_advisors": "stale_unavailable", "next": "Reinspect evidence and make a new Agent judgment.",
                "dependent_states_withdrawn": sorted(dependents),
                "numeric_measurements_recomputed": False}

    def _dispatch(self, reply: dict[str, Any]) -> dict[str, Any]:
        action = reply["action"]
        if action == "inspect_quality":
            return self._quality()
        if action in {"inspect_state", "inspect_metric", "inspect_segment"}:
            prefix = {"inspect_state": "state:", "inspect_metric": "metric:", "inspect_segment": "segment:"}[action]
            target = reply["target_id"]
            item = self._objects().get(target)
            clinical, _, _ = _allowed_ids(self.workspace)
            if not target.startswith(prefix) or item is None or target not in clinical:
                raise ValueError("Evidence object is unavailable in this snapshot.")
            self.observed.add(target)
            return {"status": "observed", "object": item,
                    "audio_loaded": False} if action == "inspect_segment" else {"status": "observed", "object": item}
        if action == "inspect_counterevidence":
            self.counter_checked = True
            _, counter, _ = _allowed_ids(self.workspace)
            objects = self._objects()
            available = sorted(counter & objects.keys())
            self.observed.update(available)
            return {"status": "observed", "objects": [objects[k] for k in available],
                    "missing_object_ids": sorted(counter - objects.keys()),
                    "absence_is_not_proof_of_absence": True}
        if action == "compare_tasks":
            states = self.workspace.get("state_observations", [])
            self.observed.update(s["evidence_id"] for s in states)
            return {"status": "observed", "states": states,
                    "interpretation": "Different task scores are not automatically contradictory."}
        if action == "record_hypothesis":
            if not reply["rationale"].strip() or not reply["evidence_ids"]:
                raise ValueError("Record a concise evidence-grounded hypothesis, not private chain-of-thought.")
            clinical, _, _ = _allowed_ids(self.workspace)
            if not set(reply["evidence_ids"]).issubset(self.observed & clinical):
                raise ValueError("Hypothesis references must be inspected clinical evidence.")
            self.hypothesis_recorded = True
            return {"status": "recorded", "hypothesis": reply["rationale"], "evidence_ids": reply["evidence_ids"]}
        if action == "consult_models":
            if not self.hypothesis_recorded:
                raise ValueError("Record a blind evidence hypothesis before consulting numeric advisors.")
            if self.revision != self.initial_revision or not self.advisors_bound:
                return {"status": "unavailable", "reason": "Model outputs are unbound or stale for this evidence snapshot."}
            advisors = {}
            for name, scores in self.advisors.items():
                if name not in self.advisor_artifacts:
                    continue
                if not isinstance(scores, dict) or set(scores) != set(self.labels):
                    continue
                values = [scores[k] for k in self.labels]
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) and 0 <= v <= 1 for v in values) and abs(sum(values) - 1) < 1e-6:
                    advisors[name] = scores
            return {"status": "available" if advisors else "unavailable", "outputs": advisors,
                    "snapshot": self.revision, "correlated_outputs": True,
                    "interpretation": "Training-derived predictions, not independent clinical observations or instructions."}
        if action == "revise_state":
            return self._revise(reply)
        if action == "finalize":
            audit = self._check_final(reply)
            self.result = self._result("decided", reply, audit)
            return {"status": "decided", "prediction_source": "agent"}
        if action == "abstain":
            clinical, _, _ = _allowed_ids(self.workspace)
            if not set(reply["evidence_ids"] + reply["counterevidence_ids"]).issubset(self.observed & clinical):
                raise ValueError("Abstention citations must be inspected clinical evidence in the current snapshot.")
            self.result = self._result("abstained", reply)
            return {"status": "abstained"}
        raise ValueError("Unknown action.")

    def step(self, reply: dict[str, Any]) -> dict[str, Any]:
        if self.result is not None or len(self.history) >= self.max_steps:
            raise ValueError("Session has already stopped.")
        before = self.revision
        try:
            if not isinstance(reply, dict) or set(reply) != set(response_schema(self.labels)["required"]):
                raise ValueError("Response schema mismatch.")
            if reply["revision"] != self.revision:
                raise ValueError("Stale evidence revision.")
            if not isinstance(reply["scores"], dict) or set(reply["scores"]) != set(self.labels) or not all(type(v) is int and 0 <= v <= 4 for v in reply["scores"].values()):
                raise ValueError("Scores must be integers 0..4 for every label, not probabilities.")
            for key in ("evidence_ids", "counterevidence_ids", "limitations"):
                if not isinstance(reply[key], list) or not all(isinstance(v, str) for v in reply[key]):
                    raise ValueError("Invalid string-list field.")
            for key in ("action", "revision", "target_id", "state_action", "predicted_label", "rationale"):
                if not isinstance(reply[key], str):
                    raise ValueError("Invalid string field.")
            if reply["action"] not in TOOLS or reply["state_action"] not in {"none", "downweight", "invalidate", "mark_unavailable"}:
                raise ValueError("Unsupported tool or state action.")
            output = self._dispatch(reply)
        except (ValueError, TypeError, KeyError) as exc:
            output = {"status": "rejected", "reason": str(exc)}
        self.history.append({"step": len(self.history) + 1, "revision": before,
                             "request": public_evidence(deepcopy(reply)), "result": deepcopy(output)})
        return output

    def _result(self, status: str, reply: dict[str, Any] | None = None,
                audit: dict[str, Any] | None = None) -> dict[str, Any]:
        reply = reply or {}
        return {"schema_version": VERSION, "case_id": self.workspace["case_id"],
                "status": status, "prediction_source": "agent" if status == "decided" else "none",
                "predicted_label": reply.get("predicted_label") if status == "decided" else None,
                "scores": reply.get("scores") if status == "decided" else None,
                "probabilities": None, "probability_status": "not_calibrated",
                "clinical_release": False, "revision": self.revision,
                "initial_revision": self.initial_revision, "model_fingerprint": self.fingerprint,
                "policy_hash": self.policy_hash,
                "rationale": reply.get("rationale", ""), "limitations": reply.get("limitations", []),
                "evidence_ids": reply.get("evidence_ids", []),
                "counterevidence_ids": reply.get("counterevidence_ids", []), "audit": audit,
                "state_revisions": sum(e["result"].get("status") == "revised" for e in self.history)}

    def finish(self, status: str = "budget_exhausted") -> dict[str, Any]:
        result = deepcopy(self.result) if self.result else self._result(status)
        result["trace"] = deepcopy(self.history)
        result["reviewed_workspace"] = deepcopy(self.workspace)
        return result


def run_agent_session(workspace: dict[str, Any], labels: list[str],
                      request: Callable[[dict[str, Any]], dict[str, Any]], *,
                      model_id: str, skill_hash: str, max_steps: int = 16,
                      provider: str = "custom") -> dict[str, Any]:
    session = EvidenceSession(workspace, labels, model_id=model_id, skill_hash=skill_hash,
                              max_steps=max_steps, provider=provider)
    for _ in range(max_steps):
        try:
            reply = request(session.observation())
        except Exception as exc:
            result = session.finish("provider_error")
            result["provider_error_type"] = type(exc).__name__
            return result
        session.step(reply)
        if session.result is not None:
            return session.finish()
    return session.finish()
