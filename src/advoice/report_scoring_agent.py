from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_runtime import pseudonym, run_structured_batch
from .utils import json_dump, now_utc


DIMENSIONS = [
    "evidence_completeness",
    "clinical_interpretability",
    "safety_calibration",
    "diagnostic_usefulness",
    "traceability",
]


def _schema(path: Path) -> None:
    score = {
        "type": "object",
        "properties": {
            **{name: {"type": "number", "minimum": 0, "maximum": 5} for name in DIMENSIONS},
            "reason_zh": {"type": "string"},
        },
        "required": [*DIMENSIONS, "reason_zh"],
        "additionalProperties": False,
    }
    json_dump(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string"},
                            "report_A": score,
                            "report_B": score,
                        },
                        "required": ["case_id", "report_A", "report_B"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["cases"],
            "additionalProperties": False,
        },
        path,
    )


def _swap(subject_id: str) -> bool:
    return int(hashlib.sha256(subject_id.encode()).hexdigest()[:8], 16) % 2 == 1


def run_report_scoring_agent(
    root: Path,
    b2_reports_path: Path,
    ours_reports_path: Path,
    agents_config: dict[str, Any],
    provider: str,
    scores_path: Path,
    status_path: Path,
    prompt_path: Path,
) -> None:
    instruction = agents_config["report_scoring_instruction"]
    prompt_path.write_text(instruction, encoding="utf-8")
    if provider == "disabled":
        pd.DataFrame(columns=["case_id", "condition", *DIMENSIONS, "total_25", "reason_zh"]).to_csv(scores_path, index=False)
        json_dump({"status": "not_run", "reason": "agent provider disabled", "created_at_utc": now_utc()}, status_path)
        return
    b2 = pd.read_csv(b2_reports_path, dtype={"case_id": str}).fillna("")
    ours = pd.read_csv(ours_reports_path, dtype={"case_id": str}).fillna("")
    common = sorted(set(b2["case_id"]) & set(ours["case_id"]))
    if not common:
        pd.DataFrame(columns=["case_id", "condition", *DIMENSIONS, "total_25", "reason_zh"]).to_csv(scores_path, index=False)
        json_dump({"status": "not_run", "reason": "no matched B2/Ours reports", "created_at_utc": now_utc()}, status_path)
        return
    cases, mapping = [], {}
    for report_case_id in common:
        b2_row = b2[b2["case_id"].eq(report_case_id)].iloc[0]
        ours_row = ours[ours["case_id"].eq(report_case_id)].iloc[0]
        report_values = {
            "B2": {"report": b2_row["report_zh"], "uncertainty": b2_row["uncertainty_zh"]},
            "Ours": {"report": ours_row["report_zh"], "uncertainty": ours_row["uncertainty_zh"]},
        }
        order = ["Ours", "B2"] if _swap(report_case_id) else ["B2", "Ours"]
        case_id = pseudonym(report_case_id, prefix="R")
        mapping[case_id] = {"A": order[0], "B": order[1], "report_case_id": report_case_id}
        cases.append({"case_id": case_id, "report_A": report_values[order[0]], "report_B": report_values[order[1]]})
    schema_path = status_path.parent / "report_scoring_schema.json"
    _schema(schema_path)
    response = run_structured_batch(
        root,
        instruction + "\n\n以下 A/B 顺序已逐病例随机化。不得猜测系统名称，只按文本质量独立评分。\n" + json.dumps(cases, ensure_ascii=False),
        schema_path,
        status_path.parent / "report_scoring_output.json",
        agents_config["model"],
        provider,
    )
    rows = []
    for case in response.get("cases", []):
        if case.get("case_id") not in mapping:
            continue
        for blinded in ["A", "B"]:
            score = case[f"report_{blinded}"]
            rows.append(
                {
                    "case_id": mapping[case["case_id"]]["report_case_id"],
                    "condition": mapping[case["case_id"]][blinded],
                    **{name: float(score[name]) for name in DIMENSIONS},
                    "total_25": float(sum(float(score[name]) for name in DIMENSIONS)),
                    "reason_zh": score["reason_zh"],
                }
            )
    pd.DataFrame(rows).to_csv(scores_path, index=False)
    json_dump(
        {
            "status": "completed" if len(rows) == 2 * len(common) else "incomplete",
            "provider": provider,
            "model": agents_config["model"],
            "matched_cases": len(common),
            "scored_reports": len(rows),
            "blinding": "within-case A/B order determined by subject hash; condition names withheld",
            "validation_boundary": "automated rater, not physician validation",
            "created_at_utc": now_utc(),
        },
        status_path,
    )
