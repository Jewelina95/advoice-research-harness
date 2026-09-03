from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .agent_runtime import case_pseudonym, normalize_probabilities, output_schema, pseudonym, run_structured_batch, select_agent_cohort
from .utils import hash_values, json_dump, json_load, now_utc


def _cached_batch(
    output_path: Path,
    metadata_path: Path,
    request_fingerprint: str,
    expected_case_ids: list[str],
) -> dict[str, Any] | None:
    metadata = json_load(metadata_path, {})
    response = json_load(output_path, {})
    cases = response.get("cases", []) if isinstance(response, dict) else []
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if (
        metadata.get("request_fingerprint") == request_fingerprint
        and case_ids == expected_case_ids
    ):
        return response
    return None


def _canonicalize_case_ids(
    response: dict[str, Any], expected_case_ids: list[str]
) -> list[str] | None:
    cases = response.get("cases", [])
    if len(cases) != len(expected_case_ids):
        return None
    unused = set(expected_case_ids)
    canonical: list[str] = []
    for case in cases:
        returned = str(case.get("case_id", ""))
        candidates = [
            case_id
            for case_id in unused
            if case_id == returned
            or (len(returned) >= 8 and case_id.startswith(returned))
        ]
        if len(candidates) != 1:
            return None
        matched = candidates[0]
        case["case_id"] = matched
        canonical.append(matched)
        unused.remove(matched)
    return canonical


def _run_or_load_batch(
    root: Path,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    metadata_path: Path,
    model: str,
    labels: list[str],
    expected_case_ids: list[str],
    provider: str = "codex_cli",
) -> dict[str, Any]:
    request_fingerprint = hash_values(
        [prompt, model, provider, labels, expected_case_ids]
    )
    cached = _cached_batch(
        output_path,
        metadata_path,
        request_fingerprint,
        expected_case_ids,
    )
    if cached is not None:
        return cached
    response: dict[str, Any] = {}
    for attempt in range(2):
        response = run_structured_batch(
            root, prompt, schema_path, output_path, model, provider
        )
        returned_case_ids = _canonicalize_case_ids(response, expected_case_ids)
        if returned_case_ids == expected_case_ids:
            json_dump(response, output_path)
            break
        output_path.unlink(missing_ok=True)
        if attempt == 1:
            raise RuntimeError(
                "Direct-agent batch returned an incomplete or reordered cohort: "
                f"expected={expected_case_ids}, returned={returned_case_ids}"
            )
    json_dump(
        {
            "request_fingerprint": request_fingerprint,
            "case_ids": expected_case_ids,
            "model": model,
        },
        metadata_path,
    )
    return response


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
    labels = [str(label) for label in agents_config["labels"]]
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    truth = manifest[manifest["split"].eq("test")][["dataset_id", "subject_id", "label", "split"]].drop_duplicates()
    cap = int(agents_config.get("agent_evaluation_cap", len(truth)))
    truth = select_agent_cohort(truth, cap)
    if provider == "disabled":
        pd.DataFrame(columns=["dataset_id", "subject_id", "label", "split", "condition"] + [f"prob_{x}" for x in labels]).to_csv(predictions_path, index=False)
        pd.DataFrame(columns=["case_id", "report_zh", "evidence", "uncertainty_zh"]).to_csv(reports_path, index=False)
        json_dump(
            {"condition": "B2", "status": "not_run", "reason": "agent provider disabled; no proxy substituted", "created_at_utc": now_utc()},
            status_path,
        )
        prompt_path.write_text(agents_config["direct_agent_instruction"], encoding="utf-8")
        return
    if provider not in {"codex_cli", "openai_api"}:
        raise ValueError(f"Unsupported agent provider: {provider}")
    transcripts = pd.read_csv(subject_transcripts_path, dtype={"subject_id": str}).fillna("")
    transcripts = truth[["subject_id"]].merge(transcripts, on="subject_id", how="left").fillna("")
    instruction = (
        agents_config["direct_agent_instruction"]
        + f"\n本任务：{agents_config['target_description']}。唯一允许的类别为：{', '.join(labels)}。"
    )
    prompt_path.write_text(instruction, encoding="utf-8")
    schema_path = status_path.parent / "b2_output_schema.json"
    output_schema(schema_path, labels, include_probability=True)
    batch_size = int(agents_config.get("batch_size", 8))
    batch_workers = max(1, int(agents_config.get("batch_workers", 1)))
    jobs: list[dict[str, Any]] = []
    for start in range(0, len(transcripts), batch_size):
        batch = transcripts.iloc[start : start + batch_size]
        cases = [{"case_id": pseudonym(row.subject_id), "transcript": row.transcript} for row in batch.itertuples(index=False)]
        prompt = instruction + "\n\n以下病例相互独立。必须逐例输出，概率之和必须为1。\n" + json.dumps(cases, ensure_ascii=False)
        batch_number = start // batch_size + 1
        jobs.append(
            {
                "number": batch_number,
                "prompt": prompt,
                "output_path": status_path.parent / f"b2_batch_{batch_number:02d}.json",
                "metadata_path": status_path.parent / f"b2_batch_{batch_number:02d}.meta.json",
                "case_ids": [case["case_id"] for case in cases],
            }
        )
    responses: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        futures = {
            executor.submit(
                _run_or_load_batch,
                root,
                job["prompt"],
                schema_path,
                job["output_path"],
                job["metadata_path"],
                agents_config["model"],
                labels,
                job["case_ids"],
                provider,
            ): job["number"]
            for job in jobs
        }
        for future in as_completed(futures):
            responses[futures[future]] = future.result()
    outputs = [
        case
        for batch_number in sorted(responses)
        for case in responses[batch_number]["cases"]
    ]
    reverse = {pseudonym(value): value for value in transcripts["subject_id"]}
    report_rows, prediction_rows = [], []
    for case in outputs:
        subject_id = reverse.get(case["case_id"])
        if subject_id is None:
            continue
        probabilities = normalize_probabilities(case["probabilities"], labels)
        info = truth[truth["subject_id"].eq(subject_id)].iloc[0].to_dict()
        prediction_rows.append({**info, **{f"prob_{label}": probabilities[label] for label in labels}, "predicted_label": max(probabilities, key=probabilities.get), "condition": "B2"})
        report_rows.append({"case_id": case_pseudonym(subject_id), "report_zh": case["report_zh"], "evidence": json.dumps(case["evidence"], ensure_ascii=False), "uncertainty_zh": case["uncertainty_zh"], "prompt_version": agents_config["direct_agent_prompt_version"], "model": agents_config["model"]})
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(report_rows).to_csv(reports_path, index=False)
    json_dump(
        {"condition": "B2", "status": "completed" if len(prediction_rows) == len(truth) else "incomplete", "provider": provider, "model": agents_config["model"], "prompt_version": agents_config["direct_agent_prompt_version"], "input": "transcript only; no true labels, MetricEvidence, StateCard, or model output", "labels": labels, "full_test_cases": int(manifest.loc[manifest["split"].eq("test"), "subject_id"].nunique()), "matched_agent_cases": len(truth), "expected_cases": len(truth), "completed_cases": len(prediction_rows), "created_at_utc": now_utc()},
        status_path,
    )
