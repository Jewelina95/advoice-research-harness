from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_runtime import output_schema, pseudonym, run_structured_batch, select_agent_cohort
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
                "source_spans": segment.get("source_spans", "[]"),
                "silence_fraction": float(segment["silence_fraction"]),
                "rms_db_mean": float(segment["rms_db_mean"]),
            }
        )
    return output


def _retained_state_features(model_metadata: dict[str, Any]) -> set[str]:
    return {
        feature
        for branch in model_metadata.get("branches", [])
        if branch.get("kind") == "clinical_state"
        for feature in branch.get("state_features", [])
    }


def run_ours_report_agent(
    root: Path,
    predictions_path: Path,
    state_cards_path: Path,
    model_metadata_path: Path,
    agents_config: dict[str, Any],
    provider: str,
    reports_path: Path,
    status_path: Path,
    prompt_path: Path,
) -> None:
    labels = [str(label) for label in agents_config["labels"]]
    predictions = pd.read_csv(predictions_path, dtype={"subject_id": str})
    cards = pd.read_csv(state_cards_path, dtype={"subject_id": str})
    model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    retained_state_features = _retained_state_features(model_metadata)
    prompt_path.write_text(agents_config["report_agent_instruction"], encoding="utf-8")
    selected = select_agent_cohort(predictions, int(agents_config.get("max_report_cases", 12)))
    if provider == "disabled":
        pd.DataFrame(columns=["subject_id", "report_zh", "evidence", "uncertainty_zh"]).to_csv(reports_path, index=False)
        json_dump({"condition": "Ours_report_agent", "status": "not_run", "reason": "agent provider disabled; numeric Ours predictions remain valid"}, status_path)
        return
    schema_path = status_path.parent / "ours_report_output_schema.json"
    output_schema(schema_path, labels, include_probability=False)
    cases = []
    for prediction in selected.to_dict("records"):
        subject_cards = cards[cards["subject_id"].eq(prediction["subject_id"])]
        overall_cards = subject_cards[subject_cards["task_scope"].eq("overall")]
        task_cards = subject_cards[~subject_cards["task_scope"].eq("overall")].copy()
        if not task_cards.empty:
            task_cards["report_priority"] = (
                task_cards["state_z"].abs() * task_cards["confidence"].fillna(0.0)
            )
            task_cards = task_cards.nlargest(12, "report_priority")
        subject_cards = pd.concat([overall_cards, task_cards], ignore_index=True)
        state_payload = []
        for card in subject_cards.to_dict("records"):
            permitted_support = [item for item in json.loads(card["supporting_metrics"]) if bool(item.get("report_permission"))]
            model_feature = f"state_{card['state_id']}"
            state_payload.append(
                {
                    "state_id": card["state_id"],
                    "state_base_id": card["state_base_id"],
                    "task_scope": card["task_scope"],
                    "trace_resolution": card["trace_resolution"],
                    "state_name_zh": card["state_name_zh"],
                    "category": card["category"],
                    "state_z": round(float(card["state_z"]), 3),
                    "confidence": round(float(card["confidence"]), 3),
                    "prediction_role": (
                        "model_input"
                        if model_feature in retained_state_features
                        else "context_only_not_used_for_prediction"
                    ),
                    "supporting_metrics": permitted_support,
                    "evidence_segments": _sanitized_segments(card["evidence_segments"]),
                }
            )
        cases.append(
            {
                "case_id": pseudonym(prediction["subject_id"]),
                "frozen_probabilities": {label: float(prediction[f"prob_{label}"]) for label in labels},
                "state_cards": state_payload,
            }
        )
    prompt = (
        agents_config["report_agent_instruction"]
        + f"\n本任务：{agents_config.get('target_description', '认知筛查')}。"
        + "\n任务特异状态表示同一临床状态在特定任务中的估计；不得把总体状态和任务状态当作相互独立的疾病机制重复计数。"
        + "\n只有 prediction_role=model_input 的状态可以描述为冻结风险的模型依据；context_only_not_used_for_prediction 只能描述为任务观察，禁止声称它推动或解释了风险概率。"
        + "\n面向医生成文时不得原样输出 prediction_role、context_only_not_used_for_prediction、robust_z 等内部字段名；分别写成‘进入本次风险模型’、‘仅作为任务观察，未进入本次风险模型’和‘相对训练参考偏离多少个稳健尺度’。"
        + "\n输出 predicted_label 必须等于冻结概率最大类别。报告按：筛查结论、主要发现、可核查证据、解释限制、复核建议。\n"
        + json.dumps(cases, ensure_ascii=False)
    )
    response = run_structured_batch(root, prompt, schema_path, status_path.parent / "ours_report_batch.json", agents_config["model"], provider)
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
