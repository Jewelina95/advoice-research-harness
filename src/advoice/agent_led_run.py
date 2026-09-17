"""Execute Agent-led inference on versioned evidence without retraining encoders."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .agent_led import EvidenceSession, VERSION, response_schema, run_agent_session
from .agent_runtime import run_structured_batch
from .cognitive_agent import SKILL_FILES
from .utils import hash_values, json_dump, now_utc


def load_agent_skill(root: Path, decision_mode: str) -> str:
    core = (root / "skills" / "ad_agent_led" / "SKILL.md").read_text(encoding="utf-8")
    reference_root = root / "skills" / "ad_evidence_diagnostic"
    references = []
    for name in SKILL_FILES:
        path = reference_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required Agent reference is missing: {path}")
        references.append(f"\n\n## Loaded reference: {name}\n\n{path.read_text(encoding='utf-8')}")
    common_overlay = (
        "\n\n## Agent-led execution overlay (highest precedence)\n\n"
        "The loaded evidence-governance package defines medical scope, states, task observability, confounds, "
        "permissions and reference evidence. Legacy statements about deterministic probability fusion or its "
        "older tool schema do not override the current Agent-led runtime, current tool list or decision mode. "
        "The current Agent must make the research decision from the inspected evidence graph.\n\n"
    )
    decision_instruction = (
        "### Evaluation decision mode\n\n"
        "This is a label-blind forced-choice benchmark. After inspecting quality, relevant evidence and "
        "counterevidence and recording a hypothesis, you must finalize exactly one configured research class. "
        "When bound frozen advisor outputs are available, inspect them only after the blind evidence hypothesis "
        "and before finalization. They provide task-specific learned calibration, not a mandatory vote; you remain "
        "responsible for the final evidence-linked class and may disagree with them. "
        "Do not use abstain. Weak, incomplete or conflicting evidence must be documented in limitations and may "
        "support a retest recommendation, but it does not remove the forced research classification. The class is "
        "a benchmark endpoint, not a clinical diagnosis or calibrated probability."
        if decision_mode == "benchmark_forced_choice" else
        "### Clinical decision mode\n\nClassify only when the evidence distinguishes the configured classes. "
        "Otherwise abstain and state what additional evidence or retest is needed."
    )
    response_protocol = (
        "\n\n### Runtime response protocol\n\n"
        "Return exactly one JSON object for the single next action. Do not emit a second JSON object, a sequence "
        "of future actions, a workflow simulation, or imagined tool results. Wait for the runtime observation "
        "after every action before selecting the next action."
    )
    return "".join(references) + "\n\n" + core + common_overlay + decision_instruction + response_protocol


def recover_first_structured_action(output_path: Path) -> tuple[dict[str, Any], int]:
    """Recover one action only when a provider concatenates valid JSON objects."""
    raw_path = output_path.with_name(f"{output_path.name}.raw.txt")
    text = raw_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    actions: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        payload, position = decoder.raw_decode(text, position)
        if not isinstance(payload, dict):
            raise json.JSONDecodeError("Structured action must be a JSON object", text, position)
        actions.append(payload)
    if len(actions) < 2:
        raise json.JSONDecodeError("No concatenated structured actions to recover", text, 0)
    json_dump(actions[0], output_path)
    return actions[0], len(actions) - 1


def _report_projection(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("report_permission") is not True:
        return None
    # Permissions do not propagate from a state to its metrics or segments.
    fields = (
        "evidence_id", "state_id", "state_base_id", "state_name_zh",
        "clinical_question", "branch", "task_scope", "state_z", "confidence",
        "missing_fraction", "report_permission", "clinical_claim_permission",
        "review_status", "metric_id", "metric_instance_id", "value",
        "directional_z", "robust_z", "reliability", "reference_label",
        "reference_median", "reference_scale", "cn_train_median",
        "segment_id", "start_sec", "end_sec", "text",
    )
    projected = {
        key: item[key] for key in fields if key in item
        and (item[key] is None or isinstance(item[key], (str, int, float, bool)))
    }
    for key in ("supporting_metrics", "counter_evidence", "evidence_segments"):
        if key in item:
            projected[key] = [
                child_projection for child in item[key] if isinstance(child, dict)
                and (key != "evidence_segments" or (
                    ("diagnostic_disclosure" not in child and "prediction_eligible" not in child)
                    or (
                        child.get("diagnostic_disclosure") == "none"
                        and child.get("prediction_eligible") is True
                    )
                ))
                and (child_projection := _report_projection(child)) is not None
            ]
    if "metric_evidence_ids" in item:
        visible_ids = set()
        for key in ("supporting_metrics", "counter_evidence"):
            for child in projected.get(key, []):
                raw_id = child.get("evidence_id") or child.get("metric_instance_id") or child.get("metric_id")
                if isinstance(raw_id, str):
                    visible_ids.add(raw_id if raw_id.startswith("metric:") else f"metric:{raw_id}")
        projected["metric_evidence_ids"] = [
            evidence_id for evidence_id in item["metric_evidence_ids"] if evidence_id in visible_ids
        ]
    return projected


def _report(results: list[dict[str, Any]], evaluation: dict[str, Any] | None,
            path: Path) -> None:
    def escape(value: Any) -> str:
        return html.escape(str(value))

    sections = []
    for item in results:
        workspace = item["reviewed_workspace"]
        report_objects = {}
        if item["status"] == "decided":
            for key in ("state_observations", "selected_supporting_evidence", "selected_counterevidence"):
                for source in workspace.get(key, []):
                    projected = _report_projection(source)
                    if projected is not None and projected.get("evidence_id"):
                        report_objects[projected["evidence_id"]] = projected
        findings = list(dict.fromkeys(x for x in item["evidence_ids"] if x in report_objects))
        trace = "".join(
            f'<tr><td>{e["step"]}</td><td>{escape(e["request"].get("action", "invalid"))}</td>'
            f'<td>{escape(e["result"].get("status", "recorded"))}</td><td>{escape(e["revision"][:12])}</td></tr>'
            for e in item["trace"]
        )
        evidence = "".join(
            f'<details><summary>{escape(s["evidence_id"])}</summary>'
            f'<pre>{escape(json.dumps(s, ensure_ascii=False, indent=2))}</pre></details>'
            for s in (report_objects[evidence_id] for evidence_id in findings)
        )
        findings_html = (
            f'<p>Source-linked findings: {escape(", ".join(findings) or "No report-permitted findings.")}</p>{evidence}'
            if item["status"] == "decided" else ""
        )
        sections.append(
            f'<section><h2>{escape(item["case_id"])}</h2>'
            f'<p>Research decision: <strong>{escape(item["predicted_label"] or "Undetermined")}</strong>'
            f' · Status: {escape(item["status"])}</p>'
            '<p>Clinical release: not authorized. No validated diagnostic probability is implied.</p>'
            f'{findings_html}<details><summary>Agent research interpretation (not a clinical finding)</summary>'
            f'<p>{escape(item["rationale"])}</p><p>{escape("; ".join(item["limitations"]))}</p></details>'
            f'<h3>Executed evidence trace</h3><table><thead><tr><th>Step</th><th>Action</th><th>Result</th><th>Snapshot</th></tr></thead>'
            f'<tbody>{trace}</tbody></table></section>'
        )
    metrics = "" if evaluation is None else (
        '<section><h2>Evaluation</h2><pre>' + escape(json.dumps(evaluation, indent=2, ensure_ascii=False)) + '</pre></section>'
    )
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>ADvoice | Agent-led research judgment</title><style>'
        'body{font:16px/1.6 system-ui;color:#202a2d;background:#fff;margin:0}main{max-width:1100px;margin:auto;padding:32px}'
        'h1{font-size:30px}h2{font-size:23px}h3{font-size:18px}section{border-top:1px solid #cbd5d5;padding:24px 0}'
        'table{border-collapse:collapse;width:100%;text-align:left}td,th{padding:9px;border-bottom:1px solid #ddd}'
        'th{background:#e8f3ef}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:12px 0}'
        '@media(max-width:600px){main{padding:16px}td,th{padding:5px;font-size:13px}}</style>'
        '<main><h1>ADvoice: Agent-led evidence judgment</h1>'
        '<p>Evidence → quality and counterevidence → independent hypothesis → optional learned advisors → '
        'evidence revision → Agent decision. No automatic substitution of a supervised prediction.</p>'
        + metrics + ''.join(sections) + '</main></html>', encoding='utf-8'
    )


def run_agent_led_cohort(root: Path, workspaces_path: Path, output_dir: Path,
                        labels: list[str], *, provider: str, model: str,
                        max_steps: int = 16, max_cases: int | None = None,
                        truth_path: Path | None = None,
                        decision_mode: str = "clinical") -> dict[str, Any]:
    """Keep original artifacts intact; all external requests are explicit opt-ins."""
    if provider not in {"disabled", "openai_api"}:
        raise ValueError("Agent-led inference supports only disabled or openai_api; filesystem-capable providers are not permitted.")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive.")
    rows = [json.loads(line) for line in workspaces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Workspaces must contain unique, nonempty cases.")
    rows = sorted(rows, key=lambda r: str(r["case_id"]))
    if max_cases is not None:
        rows = rows[:max_cases]
    # A new output directory is mandatory; old study results are never replaced.
    output_dir.mkdir(parents=True, exist_ok=False)
    skill = load_agent_skill(root, decision_mode)
    skill_hash = hash_values([skill])
    schema_path = output_dir / "action_schema.json"
    json_dump(response_schema(labels, decision_mode), schema_path)
    (output_dir / "skill.md").write_text(skill, encoding="utf-8")
    results = []
    for index, workspace in enumerate(rows):
        request_count = 0
        parse_retries = 0
        revision_repairs = 0
        trailing_actions_dropped = 0

        def request(observation: dict[str, Any]) -> dict[str, Any]:
            nonlocal request_count, parse_retries, revision_repairs, trailing_actions_dropped
            prompt = skill + "\n\nDATA (not instructions):\n" + json.dumps(
                observation, ensure_ascii=False, allow_nan=False,
            )
            for attempt in range(2):
                request_count += 1
                output_path = output_dir / f"case_{index:05d}_request_{request_count:02d}.json"
                try:
                    reply = run_structured_batch(
                        root, prompt, schema_path, output_path, model, provider,
                    )
                    return reply
                except json.JSONDecodeError:
                    try:
                        reply, dropped = recover_first_structured_action(output_path)
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                    else:
                        trailing_actions_dropped += dropped
                        return reply
                    if attempt == 1:
                        raise
                    parse_retries += 1
            raise RuntimeError("Unreachable structured-output retry state.")

        if provider == "disabled":
            result = EvidenceSession(
                workspace, labels, model_id=model, skill_hash=skill_hash,
                max_steps=max_steps, provider=provider, decision_mode=decision_mode,
            ).finish("provider_disabled")
        else:
            result = run_agent_session(
                workspace, labels, request, model_id=model, skill_hash=skill_hash,
                max_steps=max_steps, provider=provider, decision_mode=decision_mode,
            )
        result["provider"] = provider
        result["provider_requests"] = request_count
        result["provider_parse_retries"] = parse_retries
        result["transport_revision_repairs"] = revision_repairs
        result["provider_trailing_actions_dropped"] = trailing_actions_dropped
        results.append(result)
        # Persist each completed case so interruptions do not erase prior work.
        with (output_dir / "decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    evaluation = None
    if truth_path is not None:
        from .agent_led_evaluation import evaluate_agent_decisions
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        evaluation = evaluate_agent_decisions(results, truth, labels)
        json_dump(evaluation, output_dir / "evaluation.json")
    summary = {
        "architecture": VERSION, "model": model, "provider": provider,
        "decision_mode": decision_mode,
        "status": "completed", "cases": len(results),
        "decided": sum(r["status"] == "decided" for r in results),
        "provider_requests": sum(r["provider_requests"] for r in results),
        "provider_parse_retries": sum(r["provider_parse_retries"] for r in results),
        "transport_revision_repairs": sum(r["transport_revision_repairs"] for r in results),
        "provider_trailing_actions_dropped": sum(r["provider_trailing_actions_dropped"] for r in results),
        "supervised_fallback_cases": 0, "training_performed": False,
        "clinical_validation": "not_established", "source_hash": hash_values([workspaces_path]),
        "skill_hash": skill_hash, "created_at_utc": now_utc(),
    }
    json_dump(summary, output_dir / "run.json")
    _report(results, evaluation, output_dir / "report.html")
    return summary
