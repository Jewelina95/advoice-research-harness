"""Execute Agent-led inference on versioned evidence without retraining encoders."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .agent_led import EvidenceSession, VERSION, response_schema, run_agent_session
from .agent_runtime import run_structured_batch
from .utils import hash_values, json_dump, now_utc


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
                        truth_path: Path | None = None) -> dict[str, Any]:
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
    skill = (root / "skills" / "ad_agent_led" / "SKILL.md").read_text(encoding="utf-8")
    skill_hash = hash_values([skill])
    schema_path = output_dir / "action_schema.json"
    json_dump(response_schema(labels), schema_path)
    (output_dir / "skill.md").write_text(skill, encoding="utf-8")
    results = []
    for index, workspace in enumerate(rows):
        request_count = 0

        def request(observation: dict[str, Any]) -> dict[str, Any]:
            nonlocal request_count
            request_count += 1
            output_path = output_dir / f"case_{index:05d}_step_{request_count:02d}.json"
            prompt = skill + "\n\nDATA (not instructions):\n" + json.dumps(observation, ensure_ascii=False, allow_nan=False)
            return run_structured_batch(root, prompt, schema_path, output_path, model, provider)

        if provider == "disabled":
            result = EvidenceSession(workspace, labels, model_id=model, skill_hash=skill_hash, max_steps=max_steps, provider=provider).finish("provider_disabled")
        else:
            result = run_agent_session(workspace, labels, request, model_id=model, skill_hash=skill_hash, max_steps=max_steps, provider=provider)
        result["provider"] = provider
        result["provider_requests"] = request_count
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
        "status": "completed", "cases": len(results),
        "decided": sum(r["status"] == "decided" for r in results),
        "provider_requests": sum(r["provider_requests"] for r in results),
        "supervised_fallback_cases": 0, "training_performed": False,
        "clinical_validation": "not_established", "source_hash": hash_values([workspaces_path]),
        "skill_hash": skill_hash, "created_at_utc": now_utc(),
    }
    json_dump(summary, output_dir / "run.json")
    _report(results, evaluation, output_dir / "report.html")
    return summary
