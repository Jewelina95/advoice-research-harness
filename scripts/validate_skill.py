#!/usr/bin/env python3
"""Static contract and safety checks for the 8.27 initial Skill.

This script deliberately performs no training and imports no project runtime.
"""

from __future__ import annotations

import json
import math
import re
import csv
import copy
import hashlib
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "ad_evidence_diagnostic"
SCHEMA_DIR = ROOT / "schemas"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
PROVENANCE_KEYS = ("run_id", "config_hash", "data_snapshot_hash", "skill_hash")

FORBIDDEN_CASE_KEYS = {
    "true_label",
    "heldout_label",
    "ground_truth",
    "source_diagnosis",
    "diagnosis_filename",
}

FORBIDDEN_IDENTIFIER_TOKENS = re.compile(
    r"(?:^|[_/.-])(ad|dementia|control|healthy|hc|mci|patient)(?:$|[_/.-])",
    re.IGNORECASE,
)
DIAGNOSTIC_DISCLOSURE_TOKENS = re.compile(
    r"alzheimer|dementia|demencia|阿尔茨海默|痴呆", re.IGNORECASE
)
FORBIDDEN_REFERENCE_TOKENS = re.compile(
    r"test|held.?out|full.?dataset|including.?labels|ground.?truth",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return value


def walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(walk_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(walk_keys(child))
        return keys
    return set()


def walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for child in value.values() for text in walk_strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in walk_strings(child)]
    return []


def assert_required(mapping: dict[str, Any], required: list[str], context: str) -> None:
    missing = [key for key in required if key not in mapping]
    if missing:
        raise AssertionError(f"{context} missing required fields: {missing}")


def assert_rejected(callback: Any, expected_message: str) -> None:
    try:
        callback()
    except AssertionError:
        return
    raise AssertionError(expected_message)


def validate_manifest() -> None:
    manifest = load_json(SKILL_DIR / "manifest.json")
    assert manifest["entrypoint"] == "SKILL.md"
    for name in [manifest["entrypoint"], *manifest["resources"]]:
        path = SKILL_DIR / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty Skill resource: {path}")
    for key in ("output_schema", "case_schema", "trace_schema"):
        path = (SKILL_DIR / manifest[key]).resolve()
        if not path.is_file():
            raise AssertionError(f"missing schema referenced by {key}: {path}")


def validate_skill_language() -> None:
    skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    required_phrases = [
        "do not independently diagnose biological Alzheimer disease",
        "Never use filenames",
        "QC evidence",
        "counterevidence",
        "COGNITIVE_ROLLBACK_PROTOCOL.md",
    ]
    for phrase in required_phrases:
        if phrase not in skill_text:
            raise AssertionError(f"Skill is missing required rule: {phrase}")


def validate_schema_documents() -> None:
    expected = {
        "case_evidence_package.schema.json": ["endpoint", "sessions", "tasks", "supervised_prior", "state_cards", "metric_evidence", "metric_validation_records", "decision_bounds"],
        "agent_decision.schema.json": ["endpoint_type", "proposed_probabilities", "proposed_value", "used_evidence_ids", "counterevidence_ids", "quality_evidence_ids"],
        "cognitive_trace.schema.json": ["trace_id", "steps", "final_status"],
        "module_a_output.schema.json": ["output_id", "case_id", "fold_provenance"],
        "validator_result.schema.json": ["validation_id", "accepted", "violation_codes"],
        "module_b_output.schema.json": ["output_id", "eligible", "applied_correction"],
        "locked_decision.schema.json": ["decision_id", "module_a_output_id", "module_b_output_id"],
        "clinician_report_data.schema.json": ["report_id", "decision_id", "trace_map"],
        "agent_tools.schema.json": ["tools", "default_timeout_ms", "max_retries"],
    }
    for name, fields in expected.items():
        schema = load_json(SCHEMA_DIR / name)
        assert schema.get("$schema", "").endswith("2020-12/schema")
        required = schema.get("required", [])
        for field in fields:
            if field not in required:
                raise AssertionError(f"{name} does not require {field}")


def validate_json_schema_instances() -> None:
    fixtures_by_schema = {
        "case_evidence_package.schema.json": ["valid_case.json", "valid_longitudinal_case.json", "valid_regression_case.json", "valid_ordinal_case.json"],
        "agent_decision.schema.json": ["valid_decision.json", "valid_longitudinal_decision.json", "valid_regression_decision.json", "valid_ordinal_decision.json", "invalid_qc_as_disease_decision.json"],
        "cognitive_trace.schema.json": ["valid_trace.json"],
        "module_a_output.schema.json": ["valid_module_a_output.json"],
        "validator_result.schema.json": ["valid_validator_result.json", "valid_validator_rejection.json"],
        "module_b_output.schema.json": ["valid_module_b_output.json"],
        "locked_decision.schema.json": ["valid_locked_decision.json"],
        "clinician_report_data.schema.json": ["valid_clinician_report_data.json"],
        "trusted_artifact_registry.schema.json": ["trusted_artifact_registry.json"],
    }
    for schema_name, fixture_names in fixtures_by_schema.items():
        schema = load_json(SCHEMA_DIR / schema_name)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for fixture_name in fixture_names:
            errors = sorted(
                validator.iter_errors(load_json(FIXTURE_DIR / fixture_name)),
                key=lambda error: list(error.absolute_path),
            )
            if errors:
                first = errors[0]
                location = ".".join(str(part) for part in first.absolute_path) or "<root>"
                raise AssertionError(
                    f"{fixture_name} violates {schema_name} at {location}: {first.message}"
                )


def validate_evidence_registry() -> None:
    path = SKILL_DIR / "EVIDENCE_REGISTRY.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "state_id", "metric_id", "task_scope", "language_scope",
        "abnormal_direction", "evidence_status", "confounds",
        "report_permission", "implementation_status", "reference_key",
    }
    if not rows or not required.issubset(rows[0]):
        raise AssertionError("evidence registry is empty or incomplete")
    states = {row["state_id"] for row in rows}
    expected_states = {f"S{index:02d}" for index in range(1, 15)} | {"QC"}
    if states != expected_states:
        raise AssertionError(f"evidence registry state coverage mismatch: {states ^ expected_states}")
    if not any(row["reference_key"] == "R_PROJECT" for row in rows):
        raise AssertionError("project hypotheses must be explicitly marked")
    allowed_permissions = {"yes", "no", "conditional"}
    invalid_permissions = {
        row["report_permission"] for row in rows
        if row["report_permission"] not in allowed_permissions
    }
    if invalid_permissions:
        raise AssertionError(f"invalid registry report permissions: {invalid_permissions}")


def load_evidence_registry() -> dict[str, dict[str, str]]:
    with (SKILL_DIR / "EVIDENCE_REGISTRY.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_metric: dict[str, dict[str, str]] = {}
    for row in rows:
        metric_id = row["metric_id"]
        if metric_id in by_metric:
            raise AssertionError(f"duplicate metric registry row: {metric_id}")
        by_metric[metric_id] = row
    return by_metric


def validate_tools() -> None:
    registry = load_json(SKILL_DIR / "TOOLS.json")
    names = [tool["name"] for tool in registry["tools"]]
    if len(names) != len(set(names)):
        raise AssertionError("Agent tool names must be unique")
    for tool in registry["tools"]:
        if tool["name"] != "submit_candidate_decision" and not tool["read_only"]:
            raise AssertionError(f"tool {tool['name']} must be read-only")


def trusted_run_for_case(
    trusted_registry: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    provenance = tuple(case[key] for key in PROVENANCE_KEYS)
    matches = [
        run for run in trusted_registry["runs"]
        if tuple(run[key] for key in PROVENANCE_KEYS) == provenance
    ]
    if len(matches) != 1:
        raise AssertionError("case provenance does not resolve to one trusted artifact run")
    return matches[0]


def validate_trusted_artifact(record: dict[str, Any]) -> None:
    path = (ROOT / record["artifact_path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise AssertionError("trusted artifact path is missing or outside the project")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["artifact_hash"]:
        artifact_key = record.get("artifact_id", record.get("metric_validation_id", "unknown"))
        raise AssertionError(f"trusted artifact hash mismatch: {artifact_key}")


def validate_case(case: dict[str, Any], trusted_registry: dict[str, Any]) -> None:
    assert_required(
        case,
        ["case_id", "subject_id", "endpoint", "sessions", "tasks", "quality_summary", "supervised_prior", "state_cards", "metric_evidence", "segments", "decision_bounds"],
        "case",
    )
    forbidden = walk_keys(case) & FORBIDDEN_CASE_KEYS
    if forbidden:
        raise AssertionError(f"case package leaks forbidden fields: {sorted(forbidden)}")
    identifiers = [case["case_id"], case["subject_id"]]
    identifiers.extend(session["session_id"] for session in case["sessions"])
    identifiers.extend(task["task_id"] for task in case["tasks"])
    identifiers.extend(
        segment["audio_asset_id"] for segment in case["segments"]
        if segment["audio_asset_id"] is not None
    )
    leaked = [value for value in identifiers if FORBIDDEN_IDENTIFIER_TOKENS.search(value)]
    if leaked:
        raise AssertionError(f"de-identified identifiers contain diagnostic tokens: {leaked}")
    endpoint_type = case["endpoint"]["endpoint_type"]
    if case["supervised_prior"]["endpoint_type"] != endpoint_type:
        raise AssertionError("prior endpoint type must match case endpoint")
    allowed = case["endpoint"]["allowed_classes"]
    probabilities = case["supervised_prior"]["probabilities"]
    is_classification = "classification" in endpoint_type
    if is_classification:
        if len(allowed) < 2:
            raise AssertionError("classification endpoint requires at least two classes")
        if not isinstance(probabilities, dict) or set(probabilities) != set(allowed):
            raise AssertionError("classification prior classes must match allowed_classes")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise AssertionError("prior probabilities must sum to one")
        if case["supervised_prior"]["point_estimate"] is not None:
            raise AssertionError("classification prior must not contain a point estimate")
    else:
        if allowed or case["endpoint"]["value_range"] is None:
            raise AssertionError("regression endpoint requires an empty class list and value range")
        if case["endpoint"]["value_range"][1] <= case["endpoint"]["value_range"][0]:
            raise AssertionError("regression endpoint value range must have positive width")
        if probabilities is not None or case["supervised_prior"]["point_estimate"] is None:
            raise AssertionError("regression prior requires a point estimate and no probabilities")
    session_ids = {item["session_id"] for item in case["sessions"]}
    task_ids = {item["task_id"] for item in case["tasks"]}
    if any(task["session_id"] not in session_ids for task in case["tasks"]):
        raise AssertionError("task references an unknown session")
    for session in case["sessions"]:
        if not set(session["task_ids"]).issubset(task_ids):
            raise AssertionError("session references an unknown task")
    if endpoint_type.startswith("paired_change") and len(session_ids) < 2:
        raise AssertionError("paired-change endpoint requires at least two sessions")
    evidence_ids = [item["evidence_id"] for item in case["metric_evidence"]]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise AssertionError("evidence IDs must be unique")
    segment_ids = [item["segment_id"] for item in case["segments"]]
    if len(segment_ids) != len(set(segment_ids)):
        raise AssertionError("segment IDs must be unique")
    segments = {item["segment_id"]: item for item in case["segments"]}
    known_segments = set(segments)
    registry = load_evidence_registry()
    trusted_run = trusted_run_for_case(trusted_registry, case)
    trusted_references = {
        item["artifact_id"]: item for item in trusted_run["reference_artifacts"]
    }
    trusted_metric_validations = {
        item["metric_validation_id"]: item
        for item in trusted_run["metric_validation_artifacts"]
    }
    validation_records = {
        item["metric_validation_id"]: item for item in case["metric_validation_records"]
    }
    if len(validation_records) != len(case["metric_validation_records"]):
        raise AssertionError("metric validation IDs must be unique")
    for item in case["metric_evidence"]:
        if item["session_id"] not in session_ids or item["task_id"] not in task_ids:
            raise AssertionError(f"{item['evidence_id']} has invalid session/task references")
        unknown = set(item["segment_ids"]) - known_segments
        if unknown:
            raise AssertionError(f"{item['evidence_id']} cites unknown segments: {unknown}")
        for segment_id in item["segment_ids"]:
            segment = segments[segment_id]
            if (segment["session_id"], segment["task_id"]) != (
                item["session_id"], item["task_id"]
            ):
                raise AssertionError(
                    f"{item['evidence_id']} cites a segment from another session/task"
                )
            if item["speaker_role"] == "participant" and segment["speaker_role"] != "participant":
                raise AssertionError(
                    f"{item['evidence_id']} participant evidence cites a non-participant segment"
                )
            if item["evidence_role"] != "quality_control" and not segment["prediction_eligible"]:
                raise AssertionError(
                    f"{item['evidence_id']} predictive evidence cites a prediction-ineligible segment"
                )
        components = item["reliability_components"]
        required_components = {
            "coverage",
            "role_confidence",
            "alignment_confidence",
            "measurement_stability",
            "reference_support",
            "qc_penalty",
        }
        if set(components) != required_components:
            raise AssertionError(f"{item['evidence_id']} has incomplete reliability decomposition")
        expected_reliability = math.prod(float(value) for value in components.values())
        if not math.isclose(item["reliability"], expected_reliability, abs_tol=0.002):
            raise AssertionError(
                f"{item['evidence_id']} reliability does not match its declared components"
            )
        registry_row = registry.get(item["metric_id"])
        if registry_row is None:
            raise AssertionError(f"{item['evidence_id']} metric is absent from the evidence registry")
        if registry_row["state_id"] != item["state_id"]:
            raise AssertionError(
                f"{item['evidence_id']} maps {item['metric_id']} to {item['state_id']} "
                f"instead of registry state {registry_row['state_id']}"
            )
        if FORBIDDEN_REFERENCE_TOKENS.search(item["reference_scope"]):
            raise AssertionError(f"{item['evidence_id']} reference scope may include held-out data")
        provenance = item["reference_provenance"]
        if not provenance["training_only"]:
            raise AssertionError(f"{item['evidence_id']} reference is not training-only")
        trusted_reference = trusted_references.get(provenance["artifact_id"])
        if trusted_reference is None:
            raise AssertionError(f"{item['evidence_id']} reference is absent from trusted registry")
        if (
            trusted_reference["fold_id"] != provenance["fold_id"]
            or trusted_reference["population"] != provenance["population"]
            or trusted_reference["status"] != "frozen_training_only"
        ):
            raise AssertionError(f"{item['evidence_id']} reference provenance is not trusted")
        validate_trusted_artifact(trusted_reference)
        registry_permission = registry_row["report_permission"]
        basis = item["report_permission_basis"]
        minimum_reliability = case["decision_bounds"]["minimum_reliability"]
        hard_eligible = (
            item["evidence_role"] == "clinical_support"
            and item["observability"] == "observable"
            and not item["missing"]
            and item["reliability"] >= minimum_reliability
            and bool(item["reference_scope"])
        )
        if registry_permission == "no":
            if item["report_permission"] or basis != "blocked_registry_no":
                raise AssertionError(f"{item['evidence_id']} violates registry-level report ban")
        elif item["report_permission"]:
            expected_basis = "registry_yes" if registry_permission == "yes" else "conditional_validated"
            if not hard_eligible or basis != expected_basis:
                raise AssertionError(f"{item['evidence_id']} is not eligible for clinical reporting")
            if registry_permission == "conditional" and item["confound_tags"]:
                raise AssertionError(
                    f"{item['evidence_id']} conditional evidence has unresolved confounds"
                )
            if registry_permission == "conditional":
                validation_id = item["metric_validation_id"]
                record = validation_records.get(validation_id)
                task = next(task for task in case["tasks"] if task["task_id"] == item["task_id"])
                if record is None or record["status"] != "validated":
                    raise AssertionError(
                        f"{item['evidence_id']} conditional permission lacks validated credential"
                    )
                trusted_validation = trusted_metric_validations.get(validation_id)
                if trusted_validation is None or any(
                    trusted_validation[key] != record[key]
                    for key in [
                        "artifact_hash", "metric_id", "task_family", "language",
                        "method_version", "status",
                    ]
                ):
                    raise AssertionError(
                        f"{item['evidence_id']} validation credential is absent from trusted registry"
                    )
                validate_trusted_artifact(trusted_validation)
                if (
                    record["metric_id"] != item["metric_id"]
                    or record["task_family"] != task["task_family"]
                    or record["language"] != task["language"]
                ):
                    raise AssertionError(
                        f"{item['evidence_id']} metric validation credential has wrong scope"
                    )
        elif not basis.startswith("blocked_"):
            raise AssertionError(f"{item['evidence_id']} blocked evidence lacks a blocking basis")
        if not item["report_permission"] and item["metric_validation_id"] is not None:
            if item["metric_validation_id"] not in validation_records:
                raise AssertionError(f"{item['evidence_id']} cites unknown metric validation")
    evidence = {item["evidence_id"]: item for item in case["metric_evidence"]}
    known_states = {item["state_id"] for item in case["state_cards"]}
    for card in case["state_cards"]:
        if card["session_id"] not in session_ids or card["task_id"] not in task_ids:
            raise AssertionError(f"state {card['state_id']} has invalid session/task references")
        if card["observability"] == "unavailable" and (
            card["report_permission"] or card["supporting_evidence_ids"] or card["counterevidence_ids"]
        ):
            raise AssertionError("unavailable state cannot be reportable or cite clinical evidence")
        for evidence_id in [*card["supporting_evidence_ids"], *card["counterevidence_ids"]]:
            if evidence_id not in evidence:
                raise AssertionError(f"state {card['state_id']} cites unknown evidence {evidence_id}")
            if evidence[evidence_id]["state_id"] != card["state_id"]:
                raise AssertionError(
                    f"state {card['state_id']} cites evidence assigned to {evidence[evidence_id]['state_id']}"
                )
            if (evidence[evidence_id]["session_id"], evidence[evidence_id]["task_id"]) != (
                card["session_id"], card["task_id"]
            ):
                raise AssertionError(
                    f"state {card['state_id']} cites evidence from another session/task"
                )
        for segment_id in card["segment_ids"]:
            if segment_id not in segments:
                raise AssertionError(f"state {card['state_id']} cites unknown segment {segment_id}")
            segment = segments[segment_id]
            if (segment["session_id"], segment["task_id"]) != (
                card["session_id"], card["task_id"]
            ):
                raise AssertionError(
                    f"state {card['state_id']} cites a segment from another session/task"
                )
            if not segment["prediction_eligible"]:
                raise AssertionError(
                    f"state {card['state_id']} cites a prediction-ineligible segment"
                )
        cited = [
            evidence[evidence_id]
            for evidence_id in [*card["supporting_evidence_ids"], *card["counterevidence_ids"]]
        ]
        if card["report_permission"] and not any(item["report_permission"] for item in cited):
            raise AssertionError("reportable StateCard requires reportable cited evidence")
    for item in case["metric_evidence"]:
        if item["state_id"] != "QC" and item["state_id"] not in known_states:
            raise AssertionError(f"evidence {item['evidence_id']} has no StateCard")
        if item["evidence_role"] in {"quality_control", "model_auxiliary", "planned_unavailable"} and item["report_permission"]:
            raise AssertionError(f"non-clinical evidence {item['evidence_id']} cannot be reportable")
    for segment in case["segments"]:
        if segment["session_id"] not in session_ids or segment["task_id"] not in task_ids:
            raise AssertionError("segment has invalid session/task references")
        if segment["transcript"] is None and segment["audio_asset_id"] is None:
            raise AssertionError("segment requires transcript or opaque audio asset")
        if (segment["start_sec"] is None) != (segment["end_sec"] is None):
            raise AssertionError("segment start/end must both be present or absent")
        if segment["start_sec"] is not None and segment["end_sec"] <= segment["start_sec"]:
            raise AssertionError("segment end must be greater than start")
        disclosed = segment["diagnostic_disclosure"] != "none"
        transcript_has_diagnosis = bool(
            segment["transcript"]
            and DIAGNOSTIC_DISCLOSURE_TOKENS.search(segment["transcript"])
        )
        if transcript_has_diagnosis and not disclosed:
            raise AssertionError("diagnostic disclosure in transcript is not annotated")
        if disclosed and segment["prediction_eligible"]:
            raise AssertionError("diagnostic disclosure segment cannot be prediction-eligible")


def validate_decision(case: dict[str, Any], decision: dict[str, Any]) -> None:
    assert decision["case_id"] == case["case_id"]
    if decision["action"] not in case["decision_bounds"]["allowed_actions"]:
        raise AssertionError("decision action is not allowed for this case")
    endpoint_type = case["endpoint"]["endpoint_type"]
    if decision["endpoint_type"] != endpoint_type:
        raise AssertionError("decision endpoint type does not match case")
    probabilities = decision["proposed_probabilities"]
    is_classification = "classification" in endpoint_type
    if decision["action"] in {"abstain", "retest"} and decision["proposed_class"] is not None:
        raise AssertionError("abstain/retest cannot assert a class")
    if is_classification:
        allowed_classes = case["endpoint"]["allowed_classes"]
        if not isinstance(probabilities, dict) or set(probabilities) != set(allowed_classes):
            raise AssertionError("decision classes do not match case classes")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise AssertionError("decision probabilities must sum to one")
        if decision["proposed_value"] is not None:
            raise AssertionError("classification decision cannot contain a proposed value")
    else:
        if probabilities is not None or decision["proposed_class"] is not None:
            raise AssertionError("regression decision requires no probabilities or class")
        if decision["action"] in {"estimate", "hold_prior"} and decision["proposed_value"] is None:
            raise AssertionError("regression estimate/hold_prior requires a proposed value")
        if decision["action"] in {"abstain", "retest"} and decision["proposed_value"] is not None:
            raise AssertionError("regression abstain/retest cannot assert a value")
    if is_classification and decision["action"] in {"classify", "hold_prior"}:
        predicted = max(probabilities, key=probabilities.get)
        if decision["proposed_class"] != predicted:
            raise AssertionError("proposed_class must match the largest proposed probability")
    if is_classification and decision["action"] == "estimate":
        raise AssertionError("classification endpoint cannot use estimate action")
    if not is_classification and decision["action"] == "classify":
        raise AssertionError("regression endpoint cannot use classify action")
    if is_classification and len(allowed_classes) == 2:
        positive = allowed_classes[1]
        prior_positive = case["supervised_prior"]["probabilities"][positive]
        proposed_positive = probabilities[positive]
        epsilon = 1e-6
        prior_logit = math.log(max(prior_positive, epsilon) / max(1.0 - prior_positive, epsilon))
        proposed_logit = math.log(max(proposed_positive, epsilon) / max(1.0 - proposed_positive, epsilon))
        max_delta = case["decision_bounds"]["max_logit_delta"]
        if max_delta is None or abs(proposed_logit - prior_logit) > max_delta + 1e-6:
            raise AssertionError("proposed probability exceeds the permitted logit correction")
    if is_classification and len(allowed_classes) > 2:
        max_delta = case["decision_bounds"]["max_logit_delta"]
        if max_delta is None:
            raise AssertionError("multiclass correction requires a logit bound")
        epsilon = 1e-6
        prior_logs = [
            math.log(max(case["supervised_prior"]["probabilities"][label], epsilon))
            for label in allowed_classes
        ]
        proposed_logs = [
            math.log(max(probabilities[label], epsilon)) for label in allowed_classes
        ]
        prior_mean = sum(prior_logs) / len(prior_logs)
        proposed_mean = sum(proposed_logs) / len(proposed_logs)
        centered_change = max(
            abs((proposed - proposed_mean) - (prior - prior_mean))
            for proposed, prior in zip(proposed_logs, prior_logs)
        )
        if centered_change > max_delta + 1e-6:
            raise AssertionError("multiclass decision exceeds the permitted centered-logit correction")
    if not is_classification and decision["action"] in {"estimate", "hold_prior"}:
        prior_value = case["supervised_prior"]["point_estimate"]
        proposed_value = decision["proposed_value"]
        if decision["action"] == "hold_prior" and not math.isclose(
            proposed_value, prior_value, abs_tol=1e-9
        ):
            raise AssertionError("regression hold_prior must preserve the prior value")
        if decision["action"] == "estimate":
            value_range = case["endpoint"]["value_range"]
            max_delta = case["decision_bounds"]["max_standardized_value_delta"]
            if value_range is None or max_delta is None:
                raise AssertionError("regression estimate requires a range and correction bound")
            width = value_range[1] - value_range[0]
            if width <= 0:
                raise AssertionError("regression endpoint value range must have positive width")
            standardized_delta = abs(proposed_value - prior_value) / width
            if standardized_delta > max_delta + 1e-9:
                raise AssertionError("regression estimate exceeds the permitted standardized correction")
    evidence = {item["evidence_id"]: item for item in case["metric_evidence"]}
    all_refs = set(decision["used_evidence_ids"]) | set(decision["counterevidence_ids"]) | set(decision["quality_evidence_ids"])
    unknown = all_refs - set(evidence)
    if unknown:
        raise AssertionError(f"decision cites unknown evidence: {sorted(unknown)}")
    for evidence_id in [*decision["used_evidence_ids"], *decision["counterevidence_ids"]]:
        item = evidence[evidence_id]
        if item["evidence_role"] != "clinical_support" or not item["report_permission"]:
            raise AssertionError(f"clinical citation {evidence_id} is not reportable clinical evidence")
        if item["observability"] == "unavailable" or item["missing"]:
            raise AssertionError(f"clinical citation {evidence_id} is unavailable or missing")
    for evidence_id in decision["quality_evidence_ids"]:
        if evidence[evidence_id]["evidence_role"] != "quality_control":
            raise AssertionError(f"quality citation {evidence_id} is not QC evidence")
    if case["decision_bounds"]["require_counterevidence_check"] and not decision["counterevidence_checked"]:
        raise AssertionError("counterevidence check is required")
    known_state_keys = {
        (item["state_id"], item["session_id"], item["task_id"])
        for item in case["state_cards"]
    }
    for update in decision["state_updates"]:
        if (update["state_id"], update["session_id"], update["task_id"]) not in known_state_keys:
            raise AssertionError("state update does not resolve to a StateCard")
        if not set(update["evidence_ids"]).issubset(evidence):
            raise AssertionError("state update cites unknown evidence")
        for evidence_id in update["evidence_ids"]:
            item = evidence[evidence_id]
            if (item["session_id"], item["task_id"]) != (
                update["session_id"], update["task_id"]
            ):
                raise AssertionError("state update cites evidence from another session/task")
            if item["state_id"] not in {update["state_id"], "QC"}:
                raise AssertionError("state update cites evidence from another clinical construct")


def validate_trace(
    case: dict[str, Any],
    trace: dict[str, Any],
    validator_results: dict[str, dict[str, Any]],
    require_rollback: bool = False,
) -> None:
    assert trace["case_id"] == case["case_id"]
    step_ids = [step["step_id"] for step in trace["steps"]]
    if step_ids != list(range(1, len(step_ids) + 1)):
        raise AssertionError("trace step IDs must be sequential")
    rollbacks = [step for step in trace["steps"] if step["action"] == "rollback"]
    if require_rollback and not rollbacks:
        raise AssertionError("rollback fixture must include a rollback")
    evidence_ids = {item["evidence_id"] for item in case["metric_evidence"]}
    state_ids = {item["state_id"] for item in case["state_cards"]}
    violation_rollbacks: dict[str, int] = {}
    for step in trace["steps"]:
        if not set(step["evidence_ids"]).issubset(evidence_ids):
            raise AssertionError("trace cites unknown evidence")
        if not set(step["state_ids"]).issubset(state_ids):
            raise AssertionError("trace cites unknown state")
        if not set(step["invalidated_evidence_ids"]).issubset(evidence_ids):
            raise AssertionError("trace invalidates unknown evidence")
        if not set(step["invalidated_state_ids"]).issubset(state_ids):
            raise AssertionError("trace invalidates unknown state")
        if step["rollback_to_step"] is not None and step["rollback_to_step"] >= step["step_id"]:
            raise AssertionError("rollback target must be an earlier trace step")
        validation_id = step["validator_result_id"]
        if validation_id is not None:
            result = validator_results.get(validation_id)
            if result is None or result["case_id"] != case["case_id"]:
                raise AssertionError("trace validator_result_id does not resolve")
            if tuple(result[key] for key in PROVENANCE_KEYS) != tuple(
                case[key] for key in PROVENANCE_KEYS
            ):
                raise AssertionError("trace validator result has different provenance")
            if step["violation_code"] is not None and step["violation_code"] not in result["violation_codes"]:
                raise AssertionError("trace violation does not match validator result")
            if step["action"] == "rollback" and not result["rollback_required"]:
                raise AssertionError("rollback step resolves to a validator result without rollback")
        if step["action"] == "rollback":
            code = step["violation_code"]
            violation_rollbacks[code] = violation_rollbacks.get(code, 0) + 1
    if any(count > 1 for count in violation_rollbacks.values()):
        raise AssertionError("the same violation may trigger at most one rollback")
    for step in rollbacks:
        if not step.get("violation_code") or not step.get("rollback_to_step"):
            raise AssertionError("rollback steps require violation_code and rollback_to_step")
        if not step["invalidated_evidence_ids"] and not step["invalidated_state_ids"]:
            raise AssertionError("rollback must invalidate evidence or state")
        if step["output_before"] == step["output_after"]:
            raise AssertionError("rollback must change output or explicitly fallback")
    if trace["steps"][-1]["action"] not in {"submit", "stop"}:
        raise AssertionError("trace must end with submit or stop")


def validate_decision_chain(
    case: dict[str, Any],
    decision: dict[str, Any],
    trace: dict[str, Any],
    module_a: dict[str, Any],
    validator: dict[str, Any],
    module_b: dict[str, Any],
    locked: dict[str, Any],
    report: dict[str, Any],
) -> None:
    artifacts = {
        "case": case, "Agent decision": decision, "trace": trace,
        "module A": module_a, "validator": validator, "module B": module_b,
        "locked decision": locked, "report": report,
    }
    expected_provenance = tuple(case[key] for key in PROVENANCE_KEYS)
    for name, item in artifacts.items():
        if item["case_id"] != case["case_id"]:
            raise AssertionError(f"{name} case ID mismatch")
        if tuple(item[key] for key in PROVENANCE_KEYS) != expected_provenance:
            raise AssertionError(f"{name} provenance does not match the case package")
    if not math.isclose(sum(module_a["branch_contributions"].values()), 1.0, abs_tol=1e-6):
        raise AssertionError("module A branch contributions must sum to one")
    if module_a["output_id"] != case["supervised_prior"]["module_a_output_id"]:
        raise AssertionError("case prior does not resolve to module A output")
    if validator["validated_agent_decision_id"] != decision["decision_id"]:
        raise AssertionError("validator does not resolve to the Agent decision")
    if decision["trace_id"] != trace["trace_id"] or locked["trace_id"] != trace["trace_id"]:
        raise AssertionError("decision/trace/locked trace IDs do not resolve to one chain")
    if module_b["module_a_output_id"] != module_a["output_id"]:
        raise AssertionError("module B does not resolve to module A")
    if module_b["agent_decision_id"] != decision["decision_id"]:
        raise AssertionError("module B does not resolve to the Agent decision")
    if module_b["validator_result_id"] != validator["validation_id"]:
        raise AssertionError("module B does not resolve to the validator result")
    if module_b["endpoint_type"] != case["endpoint"]["endpoint_type"]:
        raise AssertionError("module B endpoint does not match case endpoint")
    correction = module_b["applied_correction"]
    is_classification = "classification" in module_b["endpoint_type"]
    if correction["correction_type"] == "class_logit_delta" and not is_classification:
        raise AssertionError("classification correction used for non-classification endpoint")
    if correction["correction_type"] == "class_logit_delta" and (
        not isinstance(correction["class_logit_deltas"], dict)
        or set(correction["class_logit_deltas"]) != set(case["endpoint"]["allowed_classes"])
        or correction["value_delta"] is not None
    ):
        raise AssertionError("classification correction must provide one delta per allowed class")
    if correction["correction_type"] == "value_delta" and is_classification:
        raise AssertionError("value correction used for classification endpoint")
    if correction["correction_type"] == "value_delta" and (
        correction["class_logit_deltas"] is not None
        or not isinstance(correction["value_delta"], (int, float))
    ):
        raise AssertionError("regression correction must provide only a value delta")
    if correction["correction_type"] == "none" and (
        correction["class_logit_deltas"] is not None or correction["value_delta"] is not None
    ):
        raise AssertionError("none correction must not carry a delta")
    if not module_b["eligible"] and correction["correction_type"] != "none":
        raise AssertionError("ineligible module B output cannot apply a correction")
    if locked["module_a_output_id"] != module_a["output_id"]:
        raise AssertionError("locked decision does not resolve to module A")
    if locked["module_b_output_id"] != module_b["output_id"]:
        raise AssertionError("locked decision does not resolve to module B")
    if locked["validator_result_id"] != validator["validation_id"]:
        raise AssertionError("locked decision does not resolve to validator result")
    if locked["agent_decision_id"] != decision["decision_id"]:
        raise AssertionError("locked decision does not resolve to Agent decision")
    if report["decision_id"] != locked["decision_id"]:
        raise AssertionError("report does not resolve to the locked decision")
    if report["decision_action"] != locked["action"]:
        raise AssertionError("report action does not match locked decision")
    if report["reported_class"] != locked["final_class"]:
        raise AssertionError("report class does not match locked decision")
    if report["reported_value"] != locked["final_value"]:
        raise AssertionError("report value does not match locked decision")
    if locked["final_class"] is not None:
        expected_probability = locked["final_probabilities"][locked["final_class"]]
        if not math.isclose(report["reported_probability"], expected_probability, abs_tol=1e-9):
            raise AssertionError("report probability does not match locked decision")
    prohibited_report_claim = re.compile(
        r"biological\s+alzheimer|confirmed\s+(?:alzheimer|dementia)|"
        r"(?:alzheimer|dementia)\s+is\s+confirmed|"
        r"diagnos(?:e|ed|is)\w*\s+(?:with\s+)?(?:alzheimer|dementia)|"
        r"(?:已)?确诊.{0,8}(?:阿尔茨海默|痴呆)|诊断为.{0,8}(?:阿尔茨海默|痴呆)",
        re.IGNORECASE,
    )
    if any(prohibited_report_claim.search(text) for text in walk_strings(report)):
        raise AssertionError("report makes a prohibited diagnostic claim")
    if module_b["calibrated_probabilities"] != locked["final_probabilities"]:
        raise AssertionError("locked probabilities must equal module B output")
    evidence = {item["evidence_id"]: item for item in case["metric_evidence"]}
    segments = {item["segment_id"] for item in case["segments"]}
    for group in [report["main_findings"], report["counterevidence"]]:
        for finding in group:
            for evidence_id in finding["evidence_ids"]:
                if evidence_id not in evidence or not evidence[evidence_id]["report_permission"]:
                    raise AssertionError("clinician finding cites non-reportable evidence")
            if not set(finding["segment_ids"]).issubset(segments):
                raise AssertionError("clinician finding cites unknown segment")
    for item in report["trace_map"]:
        if not set(item["evidence_ids"]).issubset(evidence):
            raise AssertionError("trace map cites unknown evidence")
        if not set(item["segment_ids"]).issubset(segments):
            raise AssertionError("trace map cites unknown segment")
        if any(not evidence[evidence_id]["report_permission"] for evidence_id in item["evidence_ids"]):
            raise AssertionError("trace map cites blocked evidence")
        for evidence_id in item["evidence_ids"]:
            if evidence[evidence_id]["state_id"] != item["state_id"]:
                raise AssertionError("trace map evidence does not match its state")
            if evidence[evidence_id]["task_id"] not in item["task_ids"]:
                raise AssertionError("trace map evidence does not match its task")


def main() -> None:
    validate_manifest()
    validate_skill_language()
    validate_schema_documents()
    validate_json_schema_instances()
    validate_evidence_registry()
    validate_tools()
    trusted_registry = load_json(FIXTURE_DIR / "trusted_artifact_registry.json")
    case = load_json(FIXTURE_DIR / "valid_case.json")
    longitudinal_case = load_json(FIXTURE_DIR / "valid_longitudinal_case.json")
    regression_case = load_json(FIXTURE_DIR / "valid_regression_case.json")
    ordinal_case = load_json(FIXTURE_DIR / "valid_ordinal_case.json")
    decision = load_json(FIXTURE_DIR / "valid_decision.json")
    longitudinal_decision = load_json(FIXTURE_DIR / "valid_longitudinal_decision.json")
    regression_decision = load_json(FIXTURE_DIR / "valid_regression_decision.json")
    ordinal_decision = load_json(FIXTURE_DIR / "valid_ordinal_decision.json")
    trace = load_json(FIXTURE_DIR / "valid_trace.json")
    validator_rejection = load_json(FIXTURE_DIR / "valid_validator_rejection.json")
    validator_acceptance = load_json(FIXTURE_DIR / "valid_validator_result.json")
    invalid = load_json(FIXTURE_DIR / "invalid_qc_as_disease_decision.json")
    validate_case(case, trusted_registry)
    validate_case(longitudinal_case, trusted_registry)
    validate_case(regression_case, trusted_registry)
    validate_case(ordinal_case, trusted_registry)
    validate_decision(case, decision)
    validate_decision(longitudinal_case, longitudinal_decision)
    validate_decision(regression_case, regression_decision)
    validate_decision(ordinal_case, ordinal_decision)
    validator_results = {
        item["validation_id"]: item
        for item in [validator_rejection, validator_acceptance]
    }
    validate_trace(case, trace, validator_results, require_rollback=True)
    validate_decision_chain(
        case,
        decision,
        trace,
        load_json(FIXTURE_DIR / "valid_module_a_output.json"),
        validator_acceptance,
        load_json(FIXTURE_DIR / "valid_module_b_output.json"),
        load_json(FIXTURE_DIR / "valid_locked_decision.json"),
        load_json(FIXTURE_DIR / "valid_clinician_report_data.json"),
    )
    assert_rejected(
        lambda: validate_decision(case, invalid),
        "invalid QC-as-disease decision was not rejected",
    )
    wrong_state_case = copy.deepcopy(case)
    wrong_state_case["metric_evidence"][1]["state_id"] = "S12"
    assert_rejected(
        lambda: validate_case(wrong_state_case, trusted_registry),
        "metric-to-state registry mismatch was not rejected",
    )
    leaked_transcript_case = copy.deepcopy(case)
    leaked_transcript_case["segments"][0]["transcript"] = "The source diagnosis is Alzheimer disease."
    assert_rejected(
        lambda: validate_case(leaked_transcript_case, trusted_registry),
        "unannotated diagnostic disclosure was not rejected",
    )
    leaked_reference_case = copy.deepcopy(case)
    leaked_reference_case["metric_evidence"][1]["reference_scope"] = "full_dataset_including_test_labels"
    assert_rejected(
        lambda: validate_case(leaked_reference_case, trusted_registry),
        "held-out reference leakage was not rejected",
    )
    uncredentialed_case = copy.deepcopy(longitudinal_case)
    uncredentialed_case["metric_evidence"][0]["report_permission"] = True
    uncredentialed_case["metric_evidence"][0]["report_permission_basis"] = "conditional_validated"
    uncredentialed_case["state_cards"][0]["report_permission"] = True
    assert_rejected(
        lambda: validate_case(uncredentialed_case, trusted_registry),
        "uncredentialed conditional report permission was not rejected",
    )
    invalid_trace = copy.deepcopy(trace)
    invalid_trace["steps"][0]["evidence_ids"] = ["E999"]
    assert_rejected(
        lambda: validate_trace(case, invalid_trace, validator_results),
        "unknown trace evidence was not rejected",
    )
    invalid_rollback = copy.deepcopy(trace)
    invalid_rollback["steps"][3]["rollback_to_step"] = 99
    assert_rejected(
        lambda: validate_trace(case, invalid_rollback, validator_results),
        "forward rollback target was not rejected",
    )
    forged_reference_case = copy.deepcopy(case)
    forged_reference_case["metric_evidence"][1]["reference_provenance"]["fold_id"] = "outer_test_fold"
    assert_rejected(
        lambda: validate_case(forged_reference_case, trusted_registry),
        "untrusted reference provenance was not rejected",
    )
    auxiliary_leak_case = copy.deepcopy(case)
    auxiliary_leak_case["segments"][0]["transcript"] = "I was diagnosed with Alzheimer disease."
    auxiliary_leak_case["segments"][0]["diagnostic_disclosure"] = "participant_self_report"
    auxiliary_leak_case["segments"][0]["prediction_eligible"] = False
    auxiliary_leak_case["metric_evidence"][1]["evidence_role"] = "model_auxiliary"
    auxiliary_leak_case["metric_evidence"][1]["report_permission"] = False
    auxiliary_leak_case["metric_evidence"][1]["report_permission_basis"] = "blocked_role"
    auxiliary_leak_case["state_cards"][1]["report_permission"] = False
    assert_rejected(
        lambda: validate_case(auxiliary_leak_case, trusted_registry),
        "diagnostic disclosure entered through model auxiliary evidence",
    )
    statecard_leak_case = copy.deepcopy(case)
    statecard_leak_case["segments"][0]["transcript"] = "I was diagnosed with Alzheimer disease."
    statecard_leak_case["segments"][0]["diagnostic_disclosure"] = "participant_self_report"
    statecard_leak_case["segments"][0]["prediction_eligible"] = False
    for evidence_item in statecard_leak_case["metric_evidence"]:
        if evidence_item["evidence_role"] != "quality_control":
            evidence_item["segment_ids"] = []
    assert_rejected(
        lambda: validate_case(statecard_leak_case, trusted_registry),
        "diagnostic disclosure entered through a StateCard segment",
    )
    wrong_validator_provenance = copy.deepcopy(validator_results)
    wrong_validator_provenance["validation_demo_001"]["run_id"] = "run_999999999999"
    assert_rejected(
        lambda: validate_trace(case, trace, wrong_validator_provenance),
        "trace accepted a validator result from another run",
    )
    chain_args = [
        case, decision, trace,
        load_json(FIXTURE_DIR / "valid_module_a_output.json"),
        validator_acceptance,
        load_json(FIXTURE_DIR / "valid_module_b_output.json"),
        load_json(FIXTURE_DIR / "valid_locked_decision.json"),
        load_json(FIXTURE_DIR / "valid_clinician_report_data.json"),
    ]
    wrong_trace_decision = copy.deepcopy(decision)
    wrong_trace_decision["trace_id"] = "trace_other"
    assert_rejected(
        lambda: validate_decision_chain(case, wrong_trace_decision, *chain_args[2:]),
        "decision/trace ID mismatch was not rejected",
    )
    bad_module_b = copy.deepcopy(chain_args[5])
    bad_module_b["eligible"] = False
    bad_module_b["applied_correction"] = {
        "correction_type": "class_logit_delta",
        "class_logit_deltas": None,
        "value_delta": 100.0,
    }
    assert_rejected(
        lambda: validate_decision_chain(case, decision, trace, chain_args[3], validator_acceptance, bad_module_b, chain_args[6], chain_args[7]),
        "impossible module B correction was not rejected",
    )
    unsafe_report = copy.deepcopy(chain_args[7])
    unsafe_report["screening_impression"] = "Biological Alzheimer disease is confirmed."
    assert_rejected(
        lambda: validate_decision_chain(case, decision, trace, chain_args[3], validator_acceptance, chain_args[5], chain_args[6], unsafe_report),
        "unsupported diagnostic report claim was not rejected",
    )
    unsafe_recommendation = copy.deepcopy(chain_args[7])
    unsafe_recommendation["recommendations"].append("Biological Alzheimer disease is confirmed.")
    assert_rejected(
        lambda: validate_decision_chain(case, decision, trace, chain_args[3], validator_acceptance, chain_args[5], chain_args[6], unsafe_recommendation),
        "unsupported recommendation claim was not rejected",
    )
    unsafe_chinese_finding = copy.deepcopy(chain_args[7])
    unsafe_chinese_finding["counterevidence"][0]["interpretation"] = "患者已确诊阿尔茨海默病。"
    assert_rejected(
        lambda: validate_decision_chain(case, decision, trace, chain_args[3], validator_acceptance, chain_args[5], chain_args[6], unsafe_chinese_finding),
        "unsupported Chinese diagnostic claim was not rejected",
    )
    print("PASS: initial Skill, schemas, fixtures and safety checks are internally consistent")
    print("NOTE: no training or clinical-performance evaluation was run")


if __name__ == "__main__":
    main()
