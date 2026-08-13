from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_runtime import LABELS, output_schema, pseudonym, run_codex_batch
from .utils import json_dump, now_utc


def _sanitized_segments(raw: str) -> list[dict[str, Any]]:
    output = []
    for segment in json.loads(raw):
        source = str(segment.get("segment_id", ""))
        output.append(
            {
                "segment_id": pseudonym(source, prefix="SEG"),
                "start_sec": float(segment["start_sec"]),
                "end_sec": float(segment["end_sec"]),
                "silence_fraction": float(segment["silence_fraction"]),
                "rms_db_mean": float(segment["rms_db_mean"]),
            }
        )
    return output


def run_ours_report_agent(
    root: Path,
    predictions_path: Path,
    state_cards_path: Path,
    agents_config: dict[str, Any],
    provider: str,
    reports_path: Path,
    status_path: Path,
    prompt_path: Path,
) -> None:
    predictions = pd.read_csv(predictions_path, dtype={"subject_id": str})
    cards = pd.read_csv(state_cards_path, dtype={"subject_id": str})
    prompt_path.write_text(agents_config["report_agent_instruction"], encoding="utf-8")
    selected = predictions.sort_values("prediction_confidence", ascending=False).head(int(agents_config.get("max_report_cases", 12)))
    if provider == "disabled":
        pd.DataFrame(columns=["subject_id", "report_zh", "evidence", "uncertainty_zh"]).to_csv(reports_path, index=False)
        json_dump({"condition": "Ours_report_agent", "status": "not_run", "reason": "agent provider disabled; numeric Ours predictions remain valid"}, status_path)
        return
    schema_path = status_path.parent / "ours_report_output_schema.json"
    output_schema(schema_path, include_probability=False)
    cases = []
    for prediction in selected.to_dict("records"):
        subject_cards = cards[cards["subject_id"].eq(prediction["subject_id"])]
        state_payload = []
        for card in subject_cards.to_dict("records"):
            permitted_support = [item for item in json.loads(card["supporting_metrics"]) if bool(item.get("report_permission"))]
            state_payload.append(
                {
                    "state_id": card["state_id"],
                    "state_name_zh": card["state_name_zh"],
                    "category": card["category"],
                    "state_z": round(float(card["state_z"]), 3),
                    "confidence": round(float(card["confidence"]), 3),
                    "supporting_metrics": permitted_support,
                    "evidence_segments": _sanitized_segments(card["evidence_segments"]),
                }
            )
        cases.append(
            {
                "case_id": pseudonym(prediction["subject_id"]),
                "frozen_probabilities": {label: float(prediction[f"prob_{label}"]) for label in LABELS},
                "state_cards": state_payload,
            }
        )
    prompt = agents_config["report_agent_instruction"] + "\n\n输出 predicted_label 必须等于冻结概率最大类别。报告按：筛查结论、主要发现、可核查证据、解释限制、复核建议。\n" + json.dumps(cases, ensure_ascii=False)
    response = run_codex_batch(root, prompt, schema_path, status_path.parent / "ours_report_batch.json", agents_config["model"])
    reverse = {pseudonym(value): value for value in selected["subject_id"]}
    rows = []
    violations = 0
    identifier_leaks = 0
    for case in response["cases"]:
        subject_id = reverse.get(case["case_id"])
        if subject_id is None:
            continue
        expected = selected[selected["subject_id"].eq(subject_id)].iloc[0]["predicted_label"]
        if case["predicted_label"] != expected:
            violations += 1
        report_text = case["report_zh"]
        if any(token in report_text for token in ["AD_F_", "MCI_F_", "HC_F_", "AD_M_", "MCI_M_", "HC_M_"]):
            identifier_leaks += 1
        rows.append({"subject_id": subject_id, "case_id": case["case_id"], "frozen_predicted_label": expected, "agent_predicted_label": case["predicted_label"], "report_zh": report_text, "evidence": json.dumps(case["evidence"], ensure_ascii=False), "uncertainty_zh": case["uncertainty_zh"], "prompt_version": agents_config["report_agent_prompt_version"], "model": agents_config["model"]})
    pd.DataFrame(rows).to_csv(reports_path, index=False)
    completed = bool(rows) and not violations and not identifier_leaks
    json_dump(
        {"condition": "Ours_report_agent", "status": "completed" if completed else "completed_with_violations", "provider": provider, "model": agents_config["model"], "prompt_version": agents_config["report_agent_prompt_version"], "generated_cases": len(rows), "numeric_label_change_violations": violations, "source_identifier_leaks": identifier_leaks, "created_at_utc": now_utc()},
        status_path,
    )

