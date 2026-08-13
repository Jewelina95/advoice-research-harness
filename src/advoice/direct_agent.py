from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_runtime import LABELS, normalize_probabilities, output_schema, pseudonym, run_codex_batch
from .utils import json_dump, now_utc


def run_direct_agent(
    root: Path,
    subject_transcripts_path: Path,
    manifest_path: Path,
    agents_config: dict[str, Any],
    provider: str,
    predictions_path: Path,
    reports_path: Path,
    status_path: Path,
    prompt_path: Path,
) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    truth = manifest[manifest["split"].eq("test")][["dataset_id", "subject_id", "label", "split"]].drop_duplicates()
    if provider == "disabled":
        pd.DataFrame(columns=["dataset_id", "subject_id", "label", "split", "condition"] + [f"prob_{x}" for x in LABELS]).to_csv(predictions_path, index=False)
        pd.DataFrame(columns=["subject_id", "report_zh", "evidence", "uncertainty_zh"]).to_csv(reports_path, index=False)
        json_dump(
            {"condition": "B2", "status": "not_run", "reason": "agent provider disabled; no proxy substituted", "created_at_utc": now_utc()},
            status_path,
        )
        prompt_path.write_text(agents_config["direct_agent_instruction"], encoding="utf-8")
        return
    if provider != "codex_cli":
        raise ValueError(f"Unsupported agent provider: {provider}")
    transcripts = pd.read_csv(subject_transcripts_path, dtype={"subject_id": str}).fillna("")
    prompt_path.write_text(agents_config["direct_agent_instruction"], encoding="utf-8")
    schema_path = status_path.parent / "b2_output_schema.json"
    output_schema(schema_path, include_probability=True)
    outputs: list[dict[str, Any]] = []
    batch_size = int(agents_config.get("batch_size", 8))
    for start in range(0, len(transcripts), batch_size):
        batch = transcripts.iloc[start : start + batch_size]
        cases = [{"case_id": pseudonym(row.subject_id), "transcript": row.transcript} for row in batch.itertuples(index=False)]
        prompt = agents_config["direct_agent_instruction"] + "\n\n以下病例相互独立。必须逐例输出，概率之和必须为1。\n" + json.dumps(cases, ensure_ascii=False)
        response = run_codex_batch(
            root,
            prompt,
            schema_path,
            status_path.parent / f"b2_batch_{start // batch_size + 1:02d}.json",
            agents_config["model"],
        )
        outputs.extend(response["cases"])
    reverse = {pseudonym(value): value for value in transcripts["subject_id"]}
    report_rows, prediction_rows = [], []
    for case in outputs:
        subject_id = reverse.get(case["case_id"])
        if subject_id is None:
            continue
        probabilities = normalize_probabilities(case["probabilities"])
        info = truth[truth["subject_id"].eq(subject_id)].iloc[0].to_dict()
        prediction_rows.append({**info, **{f"prob_{label}": probabilities[label] for label in LABELS}, "predicted_label": max(probabilities, key=probabilities.get), "condition": "B2"})
        report_rows.append({"subject_id": subject_id, "case_id": case["case_id"], "report_zh": case["report_zh"], "evidence": json.dumps(case["evidence"], ensure_ascii=False), "uncertainty_zh": case["uncertainty_zh"], "prompt_version": agents_config["direct_agent_prompt_version"], "model": agents_config["model"]})
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(report_rows).to_csv(reports_path, index=False)
    json_dump(
        {"condition": "B2", "status": "completed" if len(prediction_rows) == len(truth) else "incomplete", "provider": provider, "model": agents_config["model"], "prompt_version": agents_config["direct_agent_prompt_version"], "input": "ASR transcript only; no true labels, MetricEvidence, StateCard, or model output", "expected_cases": len(truth), "completed_cases": len(prediction_rows), "created_at_utc": now_utc()},
        status_path,
    )

