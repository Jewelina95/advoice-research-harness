from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_runtime import case_pseudonym, run_structured_batch, select_agent_cohort
from .utils import json_dump, now_utc


def _schema(path: Path, labels: list[str]) -> None:
    json_dump(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["cases"],
            "properties": {
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "case_id",
                            "predicted_label",
                            "used_evidence_ids",
                            "counterevidence_ids",
                            "quality_evidence_ids",
                            "report_zh",
                            "patient_summary_zh",
                            "uncertainty_zh",
                        ],
                        "properties": {
                            "case_id": {"type": "string"},
                            "predicted_label": {"type": "string", "enum": labels},
                            "used_evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "counterevidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "quality_evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "report_zh": {"type": "string"},
                            "patient_summary_zh": {"type": "string"},
                            "uncertainty_zh": {"type": "string"},
                        },
                    },
                }
            },
        },
        path,
    )


def _segment_alias(value: str) -> str:
    return "SEG-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:10].upper()


def _sanitize_workspace(workspace: dict[str, Any], case_id: str) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(workspace, ensure_ascii=False))
    cleaned["case_id"] = case_id
    cleaned.pop("subject_id", None)
    model_only_count = len(cleaned.pop("model_only_state_observations", []))
    inference_states = cleaned.pop("inference_only_state_observations", [])
    inference_metrics = cleaned.pop("inference_only_metric_observations", [])
    cleaned.pop("cognitive_state_reference", None)
    cleaned.pop("reportable_state_observations", None)
    cleaned.pop("agent_candidate", None)
    cleaned.pop("agent_validation", None)
    inference_ids = {
        str(item.get("evidence_id", ""))
        for item in inference_states + inference_metrics
        if item.get("evidence_id")
    }
    inference_ids.update(
        str(value)
        for state in inference_states
        for value in state.get("metric_evidence_ids", [])
    )
    cleaned["model_only_state_count"] = model_only_count
    cleaned["inference_only_evidence_count"] = len(inference_states) + len(
        inference_metrics
    )
    cleaned["state_observations"] = [
        state
        for state in cleaned.get("state_observations", [])
        if bool(state.get("report_permission", False))
    ]
    cleaned["selected_supporting_evidence"] = [
        item
        for item in cleaned.get("selected_supporting_evidence", [])
        if bool(item.get("report_permission", False))
    ]
    cleaned["selected_counterevidence"] = [
        item
        for item in cleaned.get("selected_counterevidence", [])
        if bool(item.get("report_permission", False))
    ]
    reportable_metric_ids = {
        str(item.get("evidence_id", ""))
        for key in ("selected_supporting_evidence", "selected_counterevidence")
        for item in cleaned.get(key, [])
        if item.get("evidence_id")
    }
    reportable_state_ids = {
        str(state.get("evidence_id", ""))
        for state in cleaned.get("state_observations", [])
        if state.get("evidence_id")
    }
    reportable_segment_ids: set[str] = set()
    for state in cleaned.get("state_observations", []):
        state["metric_evidence_ids"] = [
            str(value)
            for value in state.get("metric_evidence_ids", [])
            if str(value) in reportable_metric_ids
        ]
        for key in ("supporting_metrics", "counter_evidence"):
            state[key] = [
                item
                for item in state.get(key, [])
                if f"metric:{item.get('metric_instance_id', item.get('metric_id', ''))}"
                in reportable_metric_ids
            ]
        reportable_segment_ids.update(
            str(segment.get("segment_id", ""))
            for segment in state.get("evidence_segments", [])
            if segment.get("segment_id")
        )
    quality_ids = {
        str(item.get("evidence_id", ""))
        for item in cleaned.get("quality_observations", [])
        if item.get("evidence_id")
    }
    report_registry_ids = (
        reportable_state_ids
        | reportable_metric_ids
        | reportable_segment_ids
        | quality_ids
    )
    cleaned["evidence_registry"] = [
        item
        for item in cleaned.get("evidence_registry", [])
        if str(item.get("evidence_id", "")) not in inference_ids
        and str(item.get("evidence_id", "")) in report_registry_ids
    ]
    for state in cleaned.get("state_observations", []):
        for segment in state.get("evidence_segments", []):
            if "segment_id" in segment:
                segment["segment_id"] = _segment_alias(str(segment["segment_id"]))
            segment.pop("case_id", None)
    return cleaned


def _fallback_report(workspace: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    final = workspace["final_probabilities"]
    predicted = max(labels, key=lambda label: float(final[label]))
    ordered = sorted(final.items(), key=lambda item: float(item[1]), reverse=True)
    support = workspace.get("selected_supporting_evidence", [])[:3]
    counter = workspace.get("selected_counterevidence", [])[:2]
    quality = workspace.get("quality_observations", [])[:3]
    reportable_states = [
        item
        for item in workspace.get("state_observations", [])
        if item.get("report_permission")
    ]
    reportable_states.sort(
        key=lambda item: abs(float(item.get("state_z", 0.0)))
        * float(item.get("confidence", 0.0)),
        reverse=True,
    )
    main_states = reportable_states[:3]
    support_text = "；".join(
        f"{item['metric_id']}={item['value']:.3g}（可靠度 {item['reliability']:.2f}）"
        for item in support
    ) or "未获得足够的可报告临床指标"
    counter_text = "；".join(item["metric_id"] for item in counter) or "未发现明确反向指标"
    state_lines: list[str] = []
    trace_lines: list[str] = []
    category_zh = {
        "normal": "未见明显异常",
        "borderline": "边界性变化",
        "impaired": "出现异常表现",
        "unreliable": "证据不足，暂不判断",
    }
    for state in main_states:
        scope = str(state.get("task_scope", "overall"))
        scope_text = "总体任务" if scope == "overall" else f"{scope} 任务"
        state_lines.append(
            f"{state.get('state_name_zh') or state.get('state_id')}："
            f"{category_zh.get(str(state.get('category')), '需复核')}（{scope_text}，证据可信度 "
            f"{float(state.get('confidence', 0.0)):.0%}）"
        )
        metrics = state.get("supporting_metrics", [])[:2]
        metric_text = "；".join(
            f"{metric.get('metric_id')}={float(metric.get('value', 0.0)):.3g}，"
            f"训练对照中位数={float(metric.get('reference_median', 0.0)):.3g}"
            for metric in metrics
        )
        segments = state.get("evidence_segments", [])[:1]
        segment_text = ""
        if segments:
            segment = segments[0]
            segment_text = (
                f"；对应片段 {float(segment.get('start_sec', 0.0)):.1f}–"
                f"{float(segment.get('end_sec', 0.0)):.1f} 秒"
            )
        trace_lines.append(
            f"{state.get('state_name_zh') or state.get('state_id')} <- "
            f"{metric_text or '当前无可展示的原始指标'}{segment_text}"
        )
    state_text = "；".join(state_lines) or "本次没有形成可靠、可报告的状态结论"
    trace_text = "\n".join(f"- {line}" for line in trace_lines) or "- 无可发布回溯链"
    report = (
        "筛查结论\n"
        f"本次语音认知筛查结果更接近 {predicted} 类表现，估计概率为 {float(final[predicted]):.1%}；"
        f"次高类别为 {ordered[1][0]}（{float(ordered[1][1]):.1%}）。\n\n"
        "本次观察\n"
        f"主要语言与言语行为观察为：{state_text}。"
        f"支持该结论的补充指标为：{support_text}。"
        f"反向或保留证据为：{counter_text}。\n\n"
        "证据回溯\n"
        f"{trace_text}\n\n"
        "采集与解释限制\n"
        "该结果只反映本次任务中的语言和言语行为，不能替代病史、标准认知量表、神经系统检查或生物标志物。"
        "若录音、转录、任务完成度或说话人区分不足，应优先重复标准化采集。\n\n"
        "临床建议\n"
        "结合日常功能变化、情绪、听力和标准认知量表复核；若仍有认知下降疑虑，转诊记忆门诊或神经认知专科进一步评估。"
    )
    patient_summary = (
        "这次语音筛查发现部分说话表现需要进一步核实。语音筛查不能单独判断是否患病，"
        "也不能确定疾病阶段。建议结合日常记忆变化、听力和情绪情况，由医生安排标准认知量表；"
        "如果本人或家属持续注意到认知或生活能力下降，应进一步接受记忆门诊评估。"
    )
    return {
        "case_id": workspace["case_id"],
        "predicted_label": predicted,
        "used_evidence_ids": [item["evidence_id"] for item in support],
        "counterevidence_ids": [item["evidence_id"] for item in counter],
        "quality_evidence_ids": [item["evidence_id"] for item in quality],
        "report_zh": report,
        "patient_summary_zh": patient_summary,
        "uncertainty_zh": "概率接近或证据门较低时应按不确定结果处理，不据此确诊。",
    }


def _report_validation_errors(
    item: dict[str, Any], workspace: dict[str, Any], source_identifier: str
) -> list[str]:
    errors: list[str] = []
    if str(item.get("predicted_label")) != str(workspace["final_prediction"]):
        errors.append("prediction_changed")
    diagnostic_ids = {
        evidence["evidence_id"]
        for key in ["selected_supporting_evidence", "selected_counterevidence", "state_observations"]
        for evidence in workspace.get(key, [])
    }
    diagnostic_ids.update(
        _segment_alias(str(segment["segment_id"]))
        for state in workspace.get("state_observations", [])
        for segment in state.get("evidence_segments", [])
        if segment.get("segment_id")
    )
    quality_ids = {
        evidence["evidence_id"] for evidence in workspace.get("quality_observations", [])
    }
    used = [str(value) for value in item.get("used_evidence_ids", [])]
    counter = [str(value) for value in item.get("counterevidence_ids", [])]
    quality = [str(value) for value in item.get("quality_evidence_ids", [])]
    if any(value not in diagnostic_ids | quality_ids for value in used + counter + quality):
        errors.append("unknown_evidence")
    if any(value not in diagnostic_ids for value in used + counter):
        errors.append("diagnostic_role_error")
    if any(value not in quality_ids for value in quality):
        errors.append("quality_role_error")
    combined_text = "\n".join(
        str(item.get(key, "")) for key in ["report_zh", "patient_summary_zh", "uncertainty_zh"]
    )
    source_tokens = [source_identifier, "AD_F_", "MCI_F_", "HC_F_", "AD_M_", "MCI_M_", "HC_M_"]
    if any(token and token in combined_text for token in source_tokens):
        errors.append("source_identifier_leak")
    if any(phrase in combined_text for phrase in ["确诊为", "诊断为", "已经患有", "属于中期", "属于晚期"]):
        errors.append("diagnostic_overclaim")
    patient_text = str(item.get("patient_summary_zh", ""))
    evidence_tokens = diagnostic_ids | quality_ids
    if "%" in patient_text or "概率" in patient_text or any(
        token and token in patient_text for token in evidence_tokens
    ):
        errors.append("patient_audience_violation")
    if not any(token in patient_text for token in ["建议", "复核", "评估", "复查"]):
        errors.append("patient_next_step_missing")
    if not str(item.get("report_zh", "")).strip() or not str(item.get("patient_summary_zh", "")).strip():
        errors.append("missing_audience_report")
    return sorted(set(errors))


def run_diagnostic_agent_reports(
    root: Path,
    predictions_path: Path,
    workspaces_path: Path,
    agents_config: dict[str, Any],
    provider: str,
    reports_path: Path,
    status_path: Path,
    prompt_path: Path,
) -> None:
    labels = [str(label) for label in agents_config["labels"]]
    predictions = pd.read_csv(predictions_path, dtype={"subject_id": str})
    selected = select_agent_cohort(
        predictions, int(agents_config.get("max_report_cases", 12))
    )
    by_case = {
        str(item["case_id"]): item
        for item in (
            json.loads(line)
            for line in workspaces_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    skill_root = root / "skills" / "ad_evidence_diagnostic"
    skill_text = "\n\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            skill_root / "SKILL.md",
            skill_root / "STATE_KNOWLEDGE.md",
            skill_root / "REPORT_CONTRACT.md",
        ]
    )
    prompt_path.write_text(skill_text, encoding="utf-8")
    reverse: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
    for subject_id in selected["subject_id"].astype(str):
        stored_case_id = case_pseudonym(subject_id)
        workspace = by_case.get(stored_case_id)
        if workspace is None:
            continue
        case_id = stored_case_id
        reverse[case_id] = subject_id
        cases.append(_sanitize_workspace(workspace, case_id))

    if provider == "disabled":
        response = {"cases": [_fallback_report(case, labels) for case in cases]}
        provider_status = "completed_policy_fallback"
    else:
        schema_path = status_path.parent / "diagnostic_agent_report_schema.json"
        _schema(schema_path, labels)
        prompt = (
            skill_text
            + "\n\n你是同一个证据诊断 Agent 的临床沟通阶段。数值推理阶段已经完成，"
            + "你要复核动作轨迹、支持证据、反证和混杂，再生成医生可读报告。"
            + "不得改变 final_prediction 或 final_probabilities；若发现冲突，只能在不确定性中说明。"
            + "used_evidence_ids 和 counterevidence_ids 只能放临床支持或反证；"
            + "录音时长、信噪比、削波、转录或角色覆盖等质量项只能放 quality_evidence_ids，"
            + "并且只能用于说明采集限制，不能支持疾病风险。"
            + "同时输出医生版 report_zh 与面向患者/家属的 patient_summary_zh。患者版不得显示底层指标编号、"
            + "模型概率或确诊式措辞，只说明筛查发现、限制与下一步。"
            + f"本任务类别为 {labels}，研究目标为：{agents_config.get('target_description', '认知筛查')}。"
            + "只返回结构化结果。病例工作区如下：\n"
            + json.dumps(cases, ensure_ascii=False)
        )
        response = run_structured_batch(
            root,
            prompt,
            schema_path,
            status_path.parent / "diagnostic_agent_report_batch.json",
            agents_config["model"],
            provider,
        )
        provider_status = "completed"

    rows: list[dict[str, Any]] = []
    numeric_violations = 0
    trace_violations = 0
    evidence_role_violations = 0
    identifier_leaks = 0
    diagnostic_overclaims = 0
    replaced_unsafe_reports = 0
    for item in response.get("cases", []):
        subject_id = reverse.get(str(item.get("case_id")))
        if subject_id is None:
            continue
        workspace = by_case[str(item["case_id"])]
        expected = str(workspace["final_prediction"])
        validation_errors = _report_validation_errors(item, workspace, subject_id)
        if "prediction_changed" in validation_errors:
            numeric_violations += 1
        if "unknown_evidence" in validation_errors:
            trace_violations += 1
        if any(error.endswith("role_error") for error in validation_errors):
            evidence_role_violations += 1
        if "source_identifier_leak" in validation_errors:
            identifier_leaks += 1
        if "diagnostic_overclaim" in validation_errors:
            diagnostic_overclaims += 1
        if validation_errors:
            item = _fallback_report(_sanitize_workspace(workspace, str(item["case_id"])), labels)
            replaced_unsafe_reports += 1
        used = [str(value) for value in item.get("used_evidence_ids", [])]
        counter = [str(value) for value in item.get("counterevidence_ids", [])]
        quality = [str(value) for value in item.get("quality_evidence_ids", [])]
        report = str(item.get("report_zh", ""))
        rows.append(
            {
                "case_id": item["case_id"],
                "predicted_label": expected,
                "report_zh": report,
                "clinician_report_zh": report,
                "patient_summary_zh": str(item.get("patient_summary_zh", "")),
                "evidence": json.dumps(
                    {
                        "used_evidence_ids": used,
                        "counterevidence_ids": counter,
                        "quality_evidence_ids": quality,
                    },
                    ensure_ascii=False,
                ),
                "uncertainty_zh": str(item.get("uncertainty_zh", "")),
                "prompt_version": agents_config.get(
                    "diagnostic_agent_prompt_version", "b3-evidence-diagnostic-v1"
                ),
                "model": (
                    "deterministic_policy_fallback"
                    if provider == "disabled"
                    else agents_config.get("model", "policy_runtime")
                ),
                "validation_status": "fallback_replaced" if validation_errors else "validated",
                "validation_errors": json.dumps(validation_errors, ensure_ascii=False),
            }
        )
    pd.DataFrame(
        rows,
        columns=[
            "case_id",
            "predicted_label",
            "report_zh",
            "clinician_report_zh",
            "patient_summary_zh",
            "evidence",
            "uncertainty_zh",
            "prompt_version",
            "model",
            "validation_status",
            "validation_errors",
        ],
    ).to_csv(reports_path, index=False)
    completed = (
        bool(rows)
        and replaced_unsafe_reports == 0
    )
    json_dump(
        {
            "condition": "B3_single_evidence_diagnostic_agent",
            "status": provider_status if completed else "completed_with_violations",
            "provider": provider,
            "generated_cases": len(rows),
            "numeric_label_change_violations": numeric_violations,
            "unknown_evidence_violations": trace_violations,
            "evidence_role_violations": evidence_role_violations,
            "source_identifier_leaks": identifier_leaks,
            "diagnostic_overclaim_violations": diagnostic_overclaims,
            "replaced_unsafe_reports": replaced_unsafe_reports,
            "created_at_utc": now_utc(),
        },
        status_path,
    )
