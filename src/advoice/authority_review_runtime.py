"""Bounded Agent review of frozen authority evidence.

This module is deliberately only an Agent boundary.  It never chooses a
class, changes a probability, replays evidence, or imports the compiler that
turns reviewed states into evidence revisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from .agent_runtime import case_pseudonym, run_structured_batch
from .conditional_authority import PreparedAuthorityCase
from .condition_c import clean_model_transcript
from .decision_lock import canonical_json, hash_artifact
from .transcript_sanitization import sanitize_transcript_payload


SCHEMA_VERSION = "advoice.authority_review_runtime.v3-single-blind-default"
STATE_ACTIONS = frozenset({"retain", "downweight", "invalidate", "mark_unavailable"})
DOWNWEIGHT_MULTIPLIERS = (0.25, 0.5, 0.75)
REVIEW_AVAILABLE = "available"
REVIEW_UNAVAILABLE = "provider_unavailable"
REVIEW_PROVIDER_ERROR = "failed_closed_provider_error"
REVIEW_MODE_SINGLE_BLIND = "single_blind"
REVIEW_MODE_LEGACY_TWO_PASS = "legacy_two_pass"
REVIEW_MODES = frozenset({REVIEW_MODE_SINGLE_BLIND, REVIEW_MODE_LEGACY_TWO_PASS})
_DECISION_POLICY_EXCLUSIONS = frozenset({"REFERENCES.md", "REPORT_CONTRACT.md"})


class AuthorityReviewError(ValueError):
    """Base error for a rejected authority-review boundary."""


class AuthorityReviewValidationError(AuthorityReviewError):
    """Raised when frozen inputs or structured provider output are invalid."""


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): value[key] for key in sorted(value, key=str)})


def _hash(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise AuthorityReviewValidationError(f"{field} must be a SHA-256 hash.")
    return text


def _unit(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise AuthorityReviewValidationError(f"{field} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AuthorityReviewValidationError(f"{field} must be numeric.") from exc
    if not 0.0 <= parsed <= 1.0:
        raise AuthorityReviewValidationError(f"{field} must be in [0, 1].")
    return parsed


def _strings(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not str(item).strip() for item in value):
        raise AuthorityReviewValidationError(f"{field} must be a non-empty string sequence.")
    values = tuple(str(item) for item in value)
    if len(values) != len(set(values)):
        raise AuthorityReviewValidationError(f"{field} cannot contain duplicates.")
    return values


@dataclass(frozen=True, slots=True)
class StateReviewAction:
    """One bounded review action for one frozen StateCard state."""

    state_id: str
    action: Literal["retain", "downweight", "invalidate", "mark_unavailable"]
    cited_metric_evidence_ids: tuple[str, ...]
    reliability_multiplier: float
    rationale: str

    def __post_init__(self) -> None:
        if not self.state_id.strip() or self.action not in STATE_ACTIONS:
            raise AuthorityReviewValidationError("State review action is unsupported or missing.")
        object.__setattr__(self, "cited_metric_evidence_ids", _strings(
            self.cited_metric_evidence_ids, field="cited_metric_evidence_ids"
        ))
        multiplier = _unit(self.reliability_multiplier, field="reliability_multiplier")
        if self.action == "retain" and multiplier != 1.0:
            raise AuthorityReviewValidationError("retain requires reliability_multiplier=1.0.")
        if self.action == "downweight" and multiplier not in DOWNWEIGHT_MULTIPLIERS:
            raise AuthorityReviewValidationError(
                "downweight requires a pre-registered multiplier in "
                f"{list(DOWNWEIGHT_MULTIPLIERS)}."
            )
        if self.action in {"invalidate", "mark_unavailable"} and multiplier != 0.0:
            raise AuthorityReviewValidationError(
                f"{self.action} requires reliability_multiplier=0.0."
            )
        if not self.rationale.strip():
            raise AuthorityReviewValidationError("State review rationale is required.")
        object.__setattr__(self, "reliability_multiplier", multiplier)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "action": self.action,
            "cited_metric_evidence_ids": list(self.cited_metric_evidence_ids),
            "reliability_multiplier": self.reliability_multiplier,
            "rationale": self.rationale,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StateReviewAction":
        action = str(value.get("action", ""))
        multiplier = value.get("reliability_multiplier")
        # These actions have protocol-defined coefficients. Canonicalizing
        # stale provider numbers is fail-safe because it cannot increase the
        # reviewer's authority; downweight values remain strictly validated.
        if action == "retain":
            multiplier = 1.0
        elif action in {"invalidate", "mark_unavailable"}:
            multiplier = 0.0
        return cls(
            state_id=str(value.get("state_id", "")),
            action=action,  # type: ignore[arg-type]
            cited_metric_evidence_ids=tuple(value.get("cited_metric_evidence_ids", ())),
            reliability_multiplier=multiplier,
            rationale=str(value.get("rationale", "")),
        )


def _actions_by_state(actions: Sequence[StateReviewAction]) -> Mapping[str, StateReviewAction]:
    values = {action.state_id: action for action in actions}
    if len(values) != len(actions):
        raise AuthorityReviewValidationError("Each state may have only one review action.")
    return MappingProxyType({key: values[key] for key in sorted(values)})


def _scores(value: Mapping[str, Any], class_order: Sequence[str]) -> Mapping[str, int]:
    expected = tuple(str(label) for label in class_order)
    if not isinstance(value, Mapping) or set(str(key) for key in value) != set(expected):
        raise AuthorityReviewValidationError("Ordinal score labels must exactly match the provided class order.")
    parsed: dict[str, int] = {}
    for label in expected:
        score = value[label]
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
            raise AuthorityReviewValidationError("Ordinal evidence scores must be integers from 0 through 4.")
        parsed[label] = score
    return MappingProxyType(parsed)


@dataclass(frozen=True, slots=True)
class BlindEvidenceAssessment:
    """Validated first-pass assessment with no advisor prediction information."""

    case_id: str
    reviewed_packet_hash: str
    reviewed_evidence_hash: str
    reviewed_state_graph_hash: str
    state_actions: Mapping[str, StateReviewAction]
    ordinal_scores: Mapping[str, int]
    report_trace: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise AuthorityReviewValidationError("Blind assessment requires a pseudonymous case_id.")
        for field in ("reviewed_packet_hash", "reviewed_evidence_hash", "reviewed_state_graph_hash"):
            object.__setattr__(self, field, _hash(getattr(self, field), field=field))
        object.__setattr__(self, "state_actions", _actions_by_state(tuple(self.state_actions.values())))
        object.__setattr__(self, "ordinal_scores", MappingProxyType(dict(self.ordinal_scores)))
        if not self.ordinal_scores:
            raise AuthorityReviewValidationError("Blind assessment requires ordinal evidence scores.")
        if not isinstance(self.report_trace, tuple) or any(not str(item).strip() for item in self.report_trace):
            raise AuthorityReviewValidationError("report_trace must be a tuple of non-empty strings.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "reviewed_packet_hash": self.reviewed_packet_hash,
            "reviewed_evidence_hash": self.reviewed_evidence_hash,
            "reviewed_state_graph_hash": self.reviewed_state_graph_hash,
            "state_actions": {key: action.to_dict() for key, action in self.state_actions.items()},
            "ordinal_scores": dict(self.ordinal_scores),
            "report_trace": list(self.report_trace),
        }


@dataclass(frozen=True, slots=True)
class AdvisorReconciliation:
    """Second-pass decision that can only retain or amend state actions."""

    case_id: str
    reviewed_packet_hash: str
    reviewed_evidence_hash: str
    reviewed_state_graph_hash: str
    advisor_packet_hash: str
    disposition: Literal["retain", "amend"]
    amendments: Mapping[str, StateReviewAction]
    cited_metric_evidence_ids: tuple[str, ...]
    rationale: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.case_id.strip() or self.disposition not in {"retain", "amend"}:
            raise AuthorityReviewValidationError("Advisor reconciliation disposition is invalid.")
        for field in (
            "reviewed_packet_hash", "reviewed_evidence_hash", "reviewed_state_graph_hash", "advisor_packet_hash",
        ):
            object.__setattr__(self, field, _hash(getattr(self, field), field=field))
        object.__setattr__(self, "amendments", _actions_by_state(tuple(self.amendments.values())))
        object.__setattr__(self, "cited_metric_evidence_ids", _strings(
            self.cited_metric_evidence_ids, field="cited_metric_evidence_ids"
        ))
        if not self.rationale.strip():
            raise AuthorityReviewValidationError("Advisor reconciliation rationale is required.")
        if self.disposition == "retain" and self.amendments:
            raise AuthorityReviewValidationError("retain cannot contain state amendments.")
        if self.disposition == "amend" and not self.amendments:
            raise AuthorityReviewValidationError("amend requires at least one state amendment.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "reviewed_packet_hash": self.reviewed_packet_hash,
            "reviewed_evidence_hash": self.reviewed_evidence_hash,
            "reviewed_state_graph_hash": self.reviewed_state_graph_hash,
            "advisor_packet_hash": self.advisor_packet_hash,
            "disposition": self.disposition,
            "amendments": {key: action.to_dict() for key, action in self.amendments.items()},
            "cited_metric_evidence_ids": list(self.cited_metric_evidence_ids),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class AuthorityReviewResult:
    """Result of one bounded review attempt under an explicit review mode."""

    status: str
    case_id: str
    blind_request_hash: str
    reconciliation_request_hash: str | None
    blind_assessment: BlindEvidenceAssessment | None = None
    reconciliation: AdvisorReconciliation | None = None
    effective_state_actions: Mapping[str, StateReviewAction] | None = None
    error: str | None = None
    # Legacy is the deserialization default for archived two-pass artifacts.
    # New runtimes explicitly stamp their selected mode.
    review_mode: str = REVIEW_MODE_LEGACY_TWO_PASS

    def __post_init__(self) -> None:
        if self.review_mode not in REVIEW_MODES:
            raise AuthorityReviewValidationError("Authority review mode is unsupported.")
        if self.status == REVIEW_AVAILABLE:
            if self.blind_assessment is None or self.effective_state_actions is None:
                raise AuthorityReviewValidationError(
                    "Available review requires a validated blind assessment and effective actions."
                )
            if self.review_mode == REVIEW_MODE_SINGLE_BLIND:
                if self.reconciliation is not None or self.reconciliation_request_hash is not None:
                    raise AuthorityReviewValidationError(
                        "Single-blind review cannot contain advisor reconciliation."
                    )
            elif self.reconciliation is None or self.reconciliation_request_hash is None:
                raise AuthorityReviewValidationError(
                    "Legacy two-pass review requires advisor reconciliation."
                )
            object.__setattr__(self, "effective_state_actions", _actions_by_state(
                tuple(self.effective_state_actions.values())
            ))
        elif any(value is not None for value in (
            self.blind_assessment, self.reconciliation, self.effective_state_actions,
        )):
            raise AuthorityReviewValidationError("Unavailable reviews must not fabricate a partial assessment.")

    @property
    def cache_key(self) -> str:
        return hash_artifact({
            "blind_request_hash": self.blind_request_hash,
            "reconciliation_request_hash": self.reconciliation_request_hash,
        })


_FORBIDDEN_KEYS = frozenset({
    "label", "labels", "truth", "truelabel", "trueclass", "groundtruth", "split", "fold",
    "subjectid", "participantid", "patientid", "sourcepath", "filepath", "rawtestanswer",
    "testanswer", "predictedlabel", "prediction", "diagnosis",
})


def _normalized_key(key: Any) -> str:
    return "".join(character for character in str(key).lower() if character.isalnum())


def sanitize_provider_payload(value: Any) -> Any:
    """Strip leakage keys recursively before a payload crosses the provider boundary."""

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_provider_payload(item)
            for key, item in value.items()
            if _normalized_key(key) not in _FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_provider_payload(item) for item in value]
    return value


def _reject_leakage(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = [str(key) for key in value if _normalized_key(key) in _FORBIDDEN_KEYS]
        if forbidden:
            raise AuthorityReviewValidationError("Provider response contains prohibited leakage fields: " + ", ".join(forbidden))
        for item in value.values():
            _reject_leakage(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_leakage(item)


def _evidence_aliases(prepared: PreparedAuthorityCase) -> Mapping[str, str]:
    """Return stable, case-local transport IDs while retaining full IDs server-side."""

    evidence_ids = sorted({item.evidence_id for item in prepared.evidence})
    return MappingProxyType({
        evidence_id: f"E{index:03d}"
        for index, evidence_id in enumerate(evidence_ids, start=1)
    })


def _safe_evidence(
    prepared: PreparedAuthorityCase,
    evidence_aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in sorted(prepared.evidence, key=lambda current: current.evidence_id):
        evidence.append({
            "evidence_id": (evidence_aliases or {}).get(item.evidence_id, item.evidence_id),
            "metric_id": item.metric_id,
            "state_id": item.state_id,
            "task_id": item.task_id,
            "value": item.value,
            "unit": item.unit,
            "source_modality": item.source_modality,
            "direction": item.direction,
            "direction_provenance": item.direction_provenance,
            "observable": item.observable,
            "unavailable_reason": item.unavailable_reason,
            "reference": {
                "median": item.reference.median,
                "scale": item.reference.scale,
                "sample_size": item.reference.sample_size,
                "scope": item.reference.scope,
            },
            "quality": {
                "reliability_components": item.reliability_components.to_dict(),
                "inference_permission": item.inference_permission,
                "report_permission": item.report_permission,
                "confounds": item.confounds.to_dict(),
            },
            "consumed_by_supervised": item.consumed_by_supervised,
            "incremental_for_agent": item.incremental_for_agent,
        })
    return evidence


def _safe_state_cards(
    prepared: PreparedAuthorityCase,
    evidence_aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    aliases = evidence_aliases or {}
    return [{
        "state_card_id": str(card["state_card_id"]),
        "state_id": str(card["state_id"]),
        "task_id": str(card["task_id"]),
        "task_ids": list(card["task_ids"]),
        "supporting_evidence_ids": [aliases.get(value, value) for value in card["supporting_evidence_ids"]],
        "counter_evidence_ids": [aliases.get(value, value) for value in card["counter_evidence_ids"]],
        "available": bool(card["available"]),
        "report_permission": bool(card["report_permission"]),
    } for card in prepared.pre_state_cards]


def _safe_transcript(transcript: Mapping[str, Any] | str | None, language: str | None) -> Mapping[str, Any] | None:
    if transcript is None:
        return None
    source: Mapping[str, Any] = {"text": transcript, "language": language or "unknown"} if isinstance(transcript, str) else transcript
    # ``condition_c`` normalizes CHAT/parser residue before the disclosure
    # screen sees the text.  Copy only text-bearing fields so caller input is
    # never modified at this boundary.
    normalized = dict(source)
    for key in ("text", "transcript", "utterance"):
        if key in normalized:
            normalized[key] = clean_model_transcript(normalized[key])
    if isinstance(source.get("segments"), list):
        segments: list[dict[str, Any]] = []
        for segment in source["segments"]:
            if not isinstance(segment, Mapping):
                continue
            clean_segment = dict(segment)
            for key in ("text", "transcript", "utterance"):
                if key in clean_segment:
                    clean_segment[key] = clean_model_transcript(clean_segment[key])
            segments.append(clean_segment)
        normalized["segments"] = segments
    cleaned = sanitize_transcript_payload(normalized)
    # Keep only the content-bearing, already screened transcript view.
    result: dict[str, Any] = {
        "text": str(cleaned.get("text", cleaned.get("transcript", cleaned.get("utterance", "")))),
        "prediction_eligible": bool(cleaned.get("prediction_eligible", False)),
    }
    if isinstance(cleaned.get("segments"), list):
        result["segments"] = [{
            "text": str(segment.get("text", segment.get("transcript", segment.get("utterance", "")))),
            "speaker_role": str(segment.get("speaker_role", segment.get("role", ""))),
            "prediction_eligible": bool(segment.get("prediction_eligible", False)),
        } for segment in cleaned["segments"] if isinstance(segment, Mapping)]
    return _freeze_mapping(sanitize_provider_payload(result))


def _safe_advisor_packet(prepared: PreparedAuthorityCase) -> dict[str, Any]:
    packet = prepared.pre_replay.packet
    return {
        "class_order": list(packet.class_order),
        "probabilities": {
            "raw": dict(packet.raw_probabilities),
            "calibrated": None if packet.calibrated_probabilities is None else dict(packet.calibrated_probabilities),
        },
        "feature_contributions": {label: dict(packet.feature_contributions[label]) for label in packet.class_order},
        "branch_contributions": {label: dict(packet.branch_contributions[label]) for label in packet.class_order},
        "uncertainty": dict(packet.uncertainty),
        "hashes": dict(packet.hashes),
    }


def _policy_documents(root: Path, skill_path: Path | None) -> Mapping[str, str]:
    skill = skill_path or root / "skills" / "ad_evidence_skill" / "skill.md"
    if not skill.exists():
        # The current repository predates the renamed skill.  This fallback is
        # only a read-only compatibility path, not a different policy source.
        skill = root / "skills" / "ad_evidence_diagnostic" / "SKILL.md"
    if not skill.is_file():
        raise AuthorityReviewError("The AD evidence skill is required for an enabled provider.")
    documents = {"skill.md": skill.read_text(encoding="utf-8")}
    for policy in sorted(skill.parent.glob("*.md")):
        if policy != skill and policy.name not in _DECISION_POLICY_EXCLUSIONS:
            documents[policy.name] = policy.read_text(encoding="utf-8")
    return _freeze_mapping(documents)


def _request_runtime_fingerprint(
    *,
    provider: str,
    model: str,
    policy_documents: Mapping[str, str],
    review_mode: str,
) -> Mapping[str, str]:
    """Return the configuration identity that makes a provider response reusable.

    ``run_structured_batch`` treats an existing output path as a cache hit.  The
    path therefore has to change whenever a request could be answered under a
    different provider contract, model, runtime schema, or clinical policy.
    """

    return _freeze_mapping({
        "provider": str(provider),
        "model": str(model),
        "runtime_schema_version": SCHEMA_VERSION,
        "review_mode": review_mode,
        "policy_content_hash": hash_artifact(dict(policy_documents)),
    })


def _verify_prepared(prepared: PreparedAuthorityCase) -> tuple[str, tuple[str, ...], set[str]]:
    if not isinstance(prepared, PreparedAuthorityCase):
        raise TypeError("AuthorityReviewRuntime requires PreparedAuthorityCase.")
    pseudo = case_pseudonym(prepared.case_id)
    if hash_artifact(prepared.pre_state_artifact) != prepared.reviewed_state_graph_hash:
        raise AuthorityReviewValidationError("Prepared state-card artifact hash is stale.")
    if hash_artifact(prepared.pre_packet_artifact) != prepared.reviewed_packet_hash:
        raise AuthorityReviewValidationError("Prepared advisor packet hash is stale.")
    if prepared.advisor_packet_hash != prepared.reviewed_packet_hash:
        raise AuthorityReviewValidationError("Prepared advisor packet hash is stale.")
    if str(prepared.pre_state_artifact.get("evidence_hash", "")) != prepared.reviewed_evidence_hash:
        raise AuthorityReviewValidationError("Prepared evidence hash is stale.")
    if str(prepared.pre_packet_artifact.get("state_graph_hash", "")) != prepared.reviewed_state_graph_hash:
        raise AuthorityReviewValidationError("Prepared packet state hash is stale.")
    if str(prepared.pre_packet_artifact.get("evidence_snapshot_hash", "")) != hash_artifact(prepared.pre_evidence_artifact):
        raise AuthorityReviewValidationError("Prepared packet evidence artifact hash is stale.")
    audit = getattr(prepared.pre_replay, "audit", None)
    if audit is not None:
        if str(getattr(audit, "evidence_hash", "")) != prepared.reviewed_evidence_hash:
            raise AuthorityReviewValidationError("Prepared replay evidence hash is stale.")
        revision_hash = str(getattr(audit, "revision_hash", ""))
        if not revision_hash or any(
            str(card.get("revision_hash", "")) != revision_hash for card in prepared.pre_state_cards
        ):
            raise AuthorityReviewValidationError("Prepared StateCard revision hash is stale.")
    states = tuple(sorted({str(card["state_id"]) for card in prepared.pre_state_cards}))
    evidence_ids = {item.evidence_id for item in prepared.evidence}
    if not states or not evidence_ids:
        raise AuthorityReviewValidationError("Prepared authority case has no reviewable states or evidence.")
    return pseudo, states, evidence_ids


def build_blind_payload(
    prepared: PreparedAuthorityCase,
    *,
    policy_documents: Mapping[str, str],
    transcript: Mapping[str, Any] | str | None = None,
    evidence_aliases: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Build the first-pass, label-blind provider payload."""

    pseudo, states, _ = _verify_prepared(prepared)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "review_pass": "blind_evidence_assessment",
        "case_id": pseudo,
        "route": {
            "observation_route": prepared.route.observation_route.id,
            "task_id": prepared.route.observation_route.task_id,
            "language": prepared.route.observation_route.language,
        },
        "class_order": list(prepared.route.target_route.labels),
        "review_contract": {
            "reviewable_state_ids": list(states),
            "state_id_rule": "Use state_id exactly; never use state_card_id.",
            "output_rule": "Return only states requiring a non-retain evidence action; omitted states are retained.",
            "citation_rule": "Each action may cite only MetricEvidence IDs whose state_id matches the action state_id.",
            "action_multiplier_rule": {
                "downweight": list(DOWNWEIGHT_MULTIPLIERS),
                "invalidate": [0.0],
                "mark_unavailable": [0.0],
            },
        },
        "reviewed_packet_hash": prepared.reviewed_packet_hash,
        "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
        "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
        "metric_evidence": _safe_evidence(prepared, evidence_aliases),
        "state_cards": _safe_state_cards(prepared, evidence_aliases),
        "sanitized_transcript": _safe_transcript(transcript, prepared.route.observation_route.language),
        "ad_evidence_policy": dict(policy_documents),
    }
    return _freeze_mapping(sanitize_provider_payload(payload))


def build_advisor_payload(
    prepared: PreparedAuthorityCase,
    blind: BlindEvidenceAssessment,
    *,
    policy_documents: Mapping[str, str],
    evidence_aliases: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Build the second pass; it exposes advisor values but never outcomes."""

    pseudo, _, _ = _verify_prepared(prepared)
    if blind.case_id != pseudo:
        raise AuthorityReviewValidationError("Blind assessment case pseudonym does not match prepared case.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "review_pass": "advisor_reconciliation",
        "case_id": pseudo,
        "reviewed_packet_hash": prepared.reviewed_packet_hash,
        "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
        "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
        "advisor_packet_hash": prepared.advisor_packet_hash,
        "blind_assessment": _alias_response_citations(blind.to_dict(), evidence_aliases or {}),
        "advisor_packet": _safe_advisor_packet(prepared),
        "reconciliation_contract": {
            "citation_rule": "Cite only same-state MetricEvidence transport IDs.",
            "action_multiplier_rule": {
                "retain": [1.0],
                "downweight": list(DOWNWEIGHT_MULTIPLIERS),
                "invalidate": [0.0],
                "mark_unavailable": [0.0],
            },
        },
        "ad_evidence_policy": dict(policy_documents),
    }
    return _freeze_mapping(sanitize_provider_payload(payload))


def _alias_response_citations(value: Any, aliases: Mapping[str, str]) -> Any:
    """Translate only typed citation fields; free text remains untouched."""

    citation_fields = {
        "evidence_id", "cited_metric_evidence_ids",
        "supporting_evidence_ids", "counter_evidence_ids",
    }
    if isinstance(value, Mapping):
        translated: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in citation_fields:
                if isinstance(item, (list, tuple)):
                    translated[str(key)] = [aliases.get(str(entry), str(entry)) for entry in item]
                else:
                    translated[str(key)] = aliases.get(str(item), str(item))
            else:
                translated[str(key)] = _alias_response_citations(item, aliases)
        return translated
    if isinstance(value, (list, tuple)):
        return [_alias_response_citations(item, aliases) for item in value]
    return value


def _action_schema(
    evidence_ids_by_state: Mapping[str, Sequence[str]], *, allow_retain: bool = True,
) -> dict[str, Any]:
    actions = sorted(STATE_ACTIONS if allow_retain else STATE_ACTIONS - {"retain"})
    variants = []
    for state_id in sorted(evidence_ids_by_state):
        evidence_ids = sorted(set(str(value) for value in evidence_ids_by_state[state_id]))
        if not evidence_ids:
            continue
        for action in actions:
            multiplier_values = (
                [1.0] if action == "retain"
                else list(DOWNWEIGHT_MULTIPLIERS) if action == "downweight"
                else [0.0]
            )
            variants.append({
                "type": "object", "additionalProperties": False,
                "properties": {
                    "state_id": {"type": "string", "enum": [state_id]},
                    "action": {"type": "string", "enum": [action]},
                    "cited_metric_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string", "enum": evidence_ids},
                        "minItems": 1,
                    },
                    "reliability_multiplier": {
                        "type": "number", "enum": multiplier_values,
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "state_id", "action", "cited_metric_evidence_ids",
                    "reliability_multiplier", "rationale",
                ],
            })
    if not variants:
        raise AuthorityReviewValidationError(
            "Structured review schema has no state-bound MetricEvidence IDs."
        )
    return {"anyOf": variants}


def _blind_schema(
    class_order: Sequence[str], evidence_ids_by_state: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    score_properties = {label: {"type": "integer", "minimum": 0, "maximum": 4} for label in class_order}
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "case_id": {"type": "string"},
            "reviewed_packet_hash": {"type": "string"},
            "reviewed_evidence_hash": {"type": "string"},
            "reviewed_state_graph_hash": {"type": "string"},
            "state_actions": {
                "type": "array", "items": _action_schema(evidence_ids_by_state, allow_retain=False),
                "minItems": 0, "maxItems": len(evidence_ids_by_state),
            },
            "ordinal_scores": {"type": "object", "properties": score_properties, "required": list(class_order), "additionalProperties": False},
            "report_trace": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["case_id", "reviewed_packet_hash", "reviewed_evidence_hash", "reviewed_state_graph_hash", "state_actions", "ordinal_scores", "report_trace"],
    }


def _reconciliation_schema(
    evidence_ids_by_state: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "case_id": {"type": "string"},
            "reviewed_packet_hash": {"type": "string"},
            "reviewed_evidence_hash": {"type": "string"},
            "reviewed_state_graph_hash": {"type": "string"},
            "advisor_packet_hash": {"type": "string"},
            "disposition": {"type": "string", "enum": ["retain", "amend"]},
            "amendments": {
                "type": "array",
                "items": _action_schema(evidence_ids_by_state),
                "maxItems": len(evidence_ids_by_state),
            },
            "cited_metric_evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "rationale": {"type": "string"},
        },
        "required": ["case_id", "reviewed_packet_hash", "reviewed_evidence_hash", "reviewed_state_graph_hash", "advisor_packet_hash", "disposition", "amendments", "cited_metric_evidence_ids", "rationale"],
    }


def _validate_actions(
    actions: Sequence[StateReviewAction],
    states: Sequence[str],
    evidence_states: Mapping[str, str],
) -> Mapping[str, StateReviewAction]:
    mapped = _actions_by_state(actions)
    if not set(mapped).issubset(states):
        raise AuthorityReviewValidationError("Provider reviewed a state outside the frozen state graph.")
    for action in mapped.values():
        if action.action == "retain":
            raise AuthorityReviewValidationError(
                "Blind review must omit retained states and return only non-retain actions."
            )
        unknown = set(action.cited_metric_evidence_ids) - set(evidence_states)
        if unknown:
            raise AuthorityReviewValidationError("Provider cited unknown MetricEvidence IDs: " + ", ".join(sorted(unknown)))
        foreign = sorted(
            evidence_id for evidence_id in action.cited_metric_evidence_ids
            if evidence_states[evidence_id] != action.state_id
        )
        if foreign:
            raise AuthorityReviewValidationError(
                "Provider cited MetricEvidence outside the reviewed state: " + ", ".join(foreign)
            )
    return mapped


def _parse_blind(response: Any, prepared: PreparedAuthorityCase) -> BlindEvidenceAssessment:
    _reject_leakage(response)
    if not isinstance(response, Mapping):
        raise AuthorityReviewValidationError("Blind provider response must be an object.")
    pseudo, states, _ = _verify_prepared(prepared)
    evidence_states = {
        item.evidence_id: item.state_id
        for item in prepared.evidence
        if item.inference_permission
    }
    actions_value = response.get("state_actions")
    if not isinstance(actions_value, list):
        raise AuthorityReviewValidationError("Blind provider response requires state_actions.")
    actions = tuple(StateReviewAction.from_mapping(item) for item in actions_value if isinstance(item, Mapping))
    if len(actions) != len(actions_value):
        raise AuthorityReviewValidationError("Every state action must be an object.")
    assessment = BlindEvidenceAssessment(
        case_id=str(response.get("case_id", "")),
        reviewed_packet_hash=str(response.get("reviewed_packet_hash", "")),
        reviewed_evidence_hash=str(response.get("reviewed_evidence_hash", "")),
        reviewed_state_graph_hash=str(response.get("reviewed_state_graph_hash", "")),
        state_actions=_validate_actions(actions, states, evidence_states),
        ordinal_scores=_scores(response.get("ordinal_scores", {}), prepared.route.target_route.labels),
        report_trace=tuple(response.get("report_trace", ())),
    )
    if (assessment.case_id, assessment.reviewed_packet_hash, assessment.reviewed_evidence_hash, assessment.reviewed_state_graph_hash) != (
        pseudo, prepared.reviewed_packet_hash, prepared.reviewed_evidence_hash, prepared.reviewed_state_graph_hash,
    ):
        raise AuthorityReviewValidationError("Blind provider response is bound to stale or foreign artifacts.")
    return assessment


def _parse_reconciliation(response: Any, prepared: PreparedAuthorityCase, blind: BlindEvidenceAssessment) -> AdvisorReconciliation:
    _reject_leakage(response)
    if not isinstance(response, Mapping):
        raise AuthorityReviewValidationError("Advisor provider response must be an object.")
    pseudo, states, evidence_ids = _verify_prepared(prepared)
    evidence_states = {
        item.evidence_id: item.state_id
        for item in prepared.evidence
        if item.inference_permission
    }
    evidence_ids = frozenset(evidence_states)
    amendments_value = response.get("amendments")
    if not isinstance(amendments_value, list):
        raise AuthorityReviewValidationError("Advisor provider response requires amendments.")
    amendments = tuple(StateReviewAction.from_mapping(item) for item in amendments_value if isinstance(item, Mapping))
    if len(amendments) != len(amendments_value):
        raise AuthorityReviewValidationError("Every amendment must be an object.")
    mapped_amendments = _actions_by_state(amendments)
    if not set(mapped_amendments).issubset(states):
        raise AuthorityReviewValidationError("Advisor response amended an unknown state.")
    cited = _strings(response.get("cited_metric_evidence_ids", ()), field="cited_metric_evidence_ids")
    if set(cited) - evidence_ids:
        raise AuthorityReviewValidationError("Advisor response cited unknown MetricEvidence IDs.")
    for action in mapped_amendments.values():
        if set(action.cited_metric_evidence_ids) - evidence_ids:
            raise AuthorityReviewValidationError("Advisor amendment cited unknown MetricEvidence IDs.")
        if any(evidence_states[item] != action.state_id for item in action.cited_metric_evidence_ids):
            raise AuthorityReviewValidationError("Advisor amendment cited MetricEvidence outside its state.")
    reconciliation = AdvisorReconciliation(
        case_id=str(response.get("case_id", "")),
        reviewed_packet_hash=str(response.get("reviewed_packet_hash", "")),
        reviewed_evidence_hash=str(response.get("reviewed_evidence_hash", "")),
        reviewed_state_graph_hash=str(response.get("reviewed_state_graph_hash", "")),
        advisor_packet_hash=str(response.get("advisor_packet_hash", "")),
        disposition=str(response.get("disposition", "")),  # type: ignore[arg-type]
        amendments=mapped_amendments,
        cited_metric_evidence_ids=cited,
        rationale=str(response.get("rationale", "")),
    )
    if (reconciliation.case_id, reconciliation.reviewed_packet_hash, reconciliation.reviewed_evidence_hash,
        reconciliation.reviewed_state_graph_hash, reconciliation.advisor_packet_hash) != (
        pseudo, prepared.reviewed_packet_hash, prepared.reviewed_evidence_hash,
        prepared.reviewed_state_graph_hash, prepared.advisor_packet_hash,
    ):
        raise AuthorityReviewValidationError("Advisor response is bound to stale or foreign artifacts.")
    return reconciliation


class AuthorityReviewRuntime:
    """Run a label-blind review, optionally followed by legacy reconciliation."""

    def __init__(
        self,
        *,
        root: Path,
        provider: str | None,
        model: str = "",
        skill_path: Path | None = None,
        cache_dir: Path | None = None,
        review_mode: str = REVIEW_MODE_SINGLE_BLIND,
    ) -> None:
        if review_mode not in REVIEW_MODES:
            raise ValueError(f"Unsupported authority review mode: {review_mode!r}")
        self.root = Path(root)
        self.provider = provider
        self.model = model
        self.skill_path = None if skill_path is None else Path(skill_path)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.root / ".authority_review_cache"
        self.review_mode = review_mode

    def study_identity(self) -> Mapping[str, str]:
        """Bind resumable cohort outputs to the exact decision runtime."""

        return _request_runtime_fingerprint(
            provider=str(self.provider),
            model=self.model,
            policy_documents=_policy_documents(self.root, self.skill_path),
            review_mode=self.review_mode,
        )

    def review(
        self,
        prepared: PreparedAuthorityCase,
        *,
        transcript: Mapping[str, Any] | str | None = None,
    ) -> AuthorityReviewResult:
        pseudo, _, _ = _verify_prepared(prepared)
        if self.provider in {None, "", "disabled"}:
            unavailable_hash = hash_artifact({"case_id": pseudo, "provider": "disabled", "pass": 1})
            return AuthorityReviewResult(
                REVIEW_UNAVAILABLE, pseudo, unavailable_hash, None,
                error="provider_disabled", review_mode=self.review_mode,
            )

        policy = _policy_documents(self.root, self.skill_path)
        runtime_fingerprint = _request_runtime_fingerprint(
            provider=str(self.provider),
            model=self.model,
            policy_documents=policy,
            review_mode=self.review_mode,
        )
        evidence_aliases = _evidence_aliases(prepared)
        full_evidence_ids = {alias: evidence_id for evidence_id, alias in evidence_aliases.items()}
        blind_payload = build_blind_payload(
            prepared,
            policy_documents=policy,
            transcript=transcript,
            evidence_aliases=evidence_aliases,
        )
        _, state_ids, _ = _verify_prepared(prepared)
        evidence_ids_by_state = {
            state_id: tuple(sorted(
                evidence_aliases[item.evidence_id]
                for item in prepared.evidence
                if item.state_id == state_id and item.inference_permission
            ))
            for state_id in state_ids
        }
        blind_schema = _blind_schema(prepared.route.target_route.labels, evidence_ids_by_state)
        blind_hash = hash_artifact({
            "pass": 1,
            "runtime": runtime_fingerprint,
            "payload": blind_payload,
            "schema": blind_schema,
        })
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            blind_response = run_structured_batch(
                self.root,
                "Return only the requested blind evidence assessment JSON.\n" + canonical_json(blind_payload),
                self._write_schema("blind", blind_hash, blind_schema),
                self.cache_dir / f"{blind_hash}.blind.output.json",
                self.model,
                self.provider,
            )
            blind = _parse_blind(
                _alias_response_citations(blind_response, full_evidence_ids), prepared
            )
            if self.review_mode == REVIEW_MODE_SINGLE_BLIND:
                return AuthorityReviewResult(
                    status=REVIEW_AVAILABLE,
                    case_id=pseudo,
                    blind_request_hash=blind_hash,
                    reconciliation_request_hash=None,
                    blind_assessment=blind,
                    reconciliation=None,
                    effective_state_actions=blind.state_actions,
                    review_mode=self.review_mode,
                )
            advisor_payload = build_advisor_payload(
                prepared, blind,
                policy_documents=policy,
                evidence_aliases=evidence_aliases,
            )
            reconciliation_schema = _reconciliation_schema(evidence_ids_by_state)
            advisor_hash = hash_artifact({
                "pass": 2,
                "runtime": runtime_fingerprint,
                "payload": advisor_payload,
                "schema": reconciliation_schema,
            })
            advisor_response = run_structured_batch(
                self.root,
                "Return only the requested advisor reconciliation JSON.\n" + canonical_json(advisor_payload),
                self._write_schema("advisor", advisor_hash, reconciliation_schema),
                self.cache_dir / f"{advisor_hash}.advisor.output.json",
                self.model,
                self.provider,
            )
            reconciliation = _parse_reconciliation(
                _alias_response_citations(advisor_response, full_evidence_ids),
                prepared,
                blind,
            )
        except AuthorityReviewValidationError:
            raise
        except Exception as error:
            return AuthorityReviewResult(
                REVIEW_PROVIDER_ERROR, pseudo, blind_hash,
                locals().get("advisor_hash"), error=f"{type(error).__name__}: {error}",
                review_mode=self.review_mode,
            )
        effective = dict(blind.state_actions)
        effective.update(reconciliation.amendments)
        return AuthorityReviewResult(
            REVIEW_AVAILABLE, pseudo, blind_hash, advisor_hash, blind, reconciliation, effective,
            review_mode=self.review_mode,
        )

    def _write_schema(self, name: str, request_hash: str, schema: Mapping[str, Any]) -> Path:
        path = self.cache_dir / f"{request_hash}.{name}.schema.json"
        path.write_text(canonical_json(schema), encoding="utf-8")
        return path


__all__ = [
    "AdvisorReconciliation", "AuthorityReviewError", "AuthorityReviewResult", "AuthorityReviewRuntime",
    "AuthorityReviewValidationError", "BlindEvidenceAssessment", "REVIEW_AVAILABLE",
    "REVIEW_MODE_LEGACY_TWO_PASS", "REVIEW_MODE_SINGLE_BLIND", "REVIEW_MODES",
    "REVIEW_PROVIDER_ERROR", "REVIEW_UNAVAILABLE", "SCHEMA_VERSION", "STATE_ACTIONS", "StateReviewAction", "build_advisor_payload",
    "build_blind_payload", "sanitize_provider_payload",
]
