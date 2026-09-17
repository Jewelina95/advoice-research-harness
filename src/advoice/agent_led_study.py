from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .agent_led import evidence_snapshot
from .agent_led_run import run_agent_led_cohort
from .agent_runtime import case_pseudonym
from .evaluation import evaluate_predictions
from .utils import hash_values, json_dump, now_utc, sha256_file


PREDICTION_FILES = {
    "b1": "b1_predictions.csv",
    "b2": "b2_predictions.csv",
    "ours": "ours_predictions.csv",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _workspaces(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or not row.get("case_id") for row in rows):
        raise ValueError("Workspaces must be nonempty objects with case_id.")
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Workspace case IDs must be unique.")
    return rows


def _json_safe(value: Any) -> tuple[Any, int]:
    """Map non-finite legacy values to unobserved nulls before Agent access."""
    if isinstance(value, float) and not math.isfinite(value):
        return None, 1
    if isinstance(value, dict):
        output, replacements = {}, 0
        for key, item in value.items():
            safe, count = _json_safe(item)
            output[key] = safe
            replacements += count
        return output, replacements
    if isinstance(value, list):
        output, replacements = [], 0
        for item in value:
            safe, count = _json_safe(item)
            output.append(safe)
            replacements += count
        return output, replacements
    return value, 0


def _prediction_frame(path: Path, labels: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"subject_id", "label", "predicted_label", *(f"prob_{label}" for label in labels)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file {path.name} is missing columns: {missing}")
    if frame["subject_id"].astype(str).duplicated().any():
        raise ValueError(f"Prediction file {path.name} contains duplicate subject IDs.")
    if not frame["label"].astype(str).isin(labels).all():
        raise ValueError(f"Prediction file {path.name} contains labels outside {labels}.")
    return frame


def _advisor_artifacts(artifact_dir: Path, labels: list[str]) -> dict[str, str]:
    model_path = artifact_dir / "ours_model.joblib"
    metadata_path = artifact_dir / "ours_model.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Agent-led study requires frozen ours_model.joblib and ours_model.json.")
    model_hash = sha256_file(model_path)
    metadata = _load_json(metadata_path)
    stored_labels = metadata.get("labels")
    if stored_labels is not None and stored_labels != labels:
        raise ValueError(
            f"Frozen model label order {stored_labels} does not match requested labels {labels}."
        )
    module_a = {
        key: metadata.get(key) for key in (
            "base_architecture", "base_cv", "base_selected_c", "base_oof_macro_auroc",
            "base_oof_macro_f1", "deep_audio_encoder", "deep_text_encoder",
        ) if key in metadata
    }
    module_b = {
        key: metadata.get(key) for key in (
            "correction_cv", "correction_selected_c", "correction_stability_guard",
            "alpha_cv", "dynamic_gate", "final_probability_temperature",
        ) if key in metadata
    }
    return {
        "module_a": hash_values(["module_a", model_hash, module_a]),
        "module_b": hash_values(["module_b", model_hash, module_b]),
    }


def prepare_agent_led_study(
    artifact_dir: Path,
    study_dir: Path,
    *,
    dataset_id: str,
    labels: list[str],
    max_cases: int | None,
    selection_seed: int,
) -> dict[str, Path]:
    """Freeze a label-blind cohort and keep truth outside the Agent input."""
    if study_dir.exists():
        raise FileExistsError(study_dir)
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive.")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("At least two distinct labels are required.")

    workspace_source = artifact_dir / "diagnostic_agent_workspaces.jsonl"
    rows = _workspaces(workspace_source)
    ours = _prediction_frame(artifact_dir / PREDICTION_FILES["ours"], labels)
    if "dataset_id" in ours and set(ours["dataset_id"].astype(str)) != {dataset_id}:
        raise ValueError("Ours prediction dataset_id does not match the requested study dataset.")
    subject_to_case = {
        str(subject_id): case_pseudonym(str(subject_id)) for subject_id in ours["subject_id"]
    }
    predicted_case_ids = set(subject_to_case.values())
    workspace_case_ids = {str(row["case_id"]) for row in rows}
    if predicted_case_ids != workspace_case_ids:
        missing_workspace = sorted(predicted_case_ids - workspace_case_ids)[:5]
        missing_prediction = sorted(workspace_case_ids - predicted_case_ids)[:5]
        raise ValueError(
            "Workspace and prediction case IDs do not match exactly; "
            f"missing_workspace={missing_workspace}, missing_prediction={missing_prediction}"
        )

    ranked = sorted(workspace_case_ids, key=lambda case_id: hash_values([selection_seed, case_id]))
    selected_ids = ranked[:max_cases] if max_cases is not None else ranked
    selected_set = set(selected_ids)
    by_case = {str(row["case_id"]): row for row in rows}
    artifacts = _advisor_artifacts(artifact_dir, labels)
    selected = []
    nonfinite_replacements = 0
    for case_id in selected_ids:
        workspace, replacements = _json_safe(dict(by_case[case_id]))
        nonfinite_replacements += replacements
        snapshot_hash = hash_values([evidence_snapshot(workspace)])
        workspace["advisor_provenance"] = {
            "evidence_hash": snapshot_hash,
            "artifacts": artifacts,
        }
        selected.append(workspace)

    case_to_truth = {
        subject_to_case[str(row.subject_id)]: str(row.label)
        for row in ours.itertuples(index=False)
        if subject_to_case[str(row.subject_id)] in selected_set
    }
    if set(case_to_truth) != selected_set:
        raise ValueError("Selected truth IDs do not match the frozen cohort.")

    study_dir.mkdir(parents=True, exist_ok=False)
    workspaces_path = study_dir / "selected_workspaces.jsonl"
    workspaces_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    truth_path = study_dir / "evaluation_truth.json"
    json_dump(case_to_truth, truth_path)
    source_hashes = {
        workspace_source.name: sha256_file(workspace_source),
        "ours_model.joblib": sha256_file(artifact_dir / "ours_model.joblib"),
        "ours_model.json": sha256_file(artifact_dir / "ours_model.json"),
    }
    for name in PREDICTION_FILES.values():
        source_hashes[name] = sha256_file(artifact_dir / name)
    json_dump({
        "study_version": "agent-led-study-v1",
        "dataset_id": dataset_id,
        "labels": labels,
        "artifact_dir": str(artifact_dir.resolve()),
        "source_hashes": source_hashes,
        "advisor_artifacts": artifacts,
        "selection": {
            "method": "sha256(seed, pseudonymous_case_id)",
            "seed": selection_seed,
            "uses_labels": False,
            "requested_max_cases": max_cases,
            "available_cases": len(rows),
            "selected_cases": len(selected),
        },
        "input_normalization": {
            "nonfinite_values_replaced_with_null": nonfinite_replacements,
            "interpretation": "legacy non-finite values are unobserved, never numeric evidence",
        },
        "selected_case_ids": selected_ids,
        "truth_access": "evaluation_after_inference_only",
        "created_at_utc": now_utc(),
    }, study_dir / "study_manifest.json")
    return {"workspaces_path": workspaces_path, "truth_path": truth_path}


def _subset_metrics(
    artifact_dir: Path,
    selected_case_ids: list[str],
    labels: list[str],
    dataset_id: str,
) -> dict[str, dict[str, Any]]:
    selected = set(selected_case_ids)
    output = {}
    reference = _prediction_frame(artifact_dir / PREDICTION_FILES["ours"], labels).copy()
    reference["case_id"] = reference["subject_id"].astype(str).map(case_pseudonym)
    reference_truth = {
        str(row.case_id): str(row.label)
        for row in reference.itertuples(index=False)
        if str(row.case_id) in selected
    }
    for condition, filename in PREDICTION_FILES.items():
        frame = _prediction_frame(artifact_dir / filename, labels).copy()
        if "dataset_id" in frame and set(frame["dataset_id"].astype(str)) != {dataset_id}:
            raise ValueError(f"{condition} dataset_id does not match the requested study dataset.")
        frame["case_id"] = frame["subject_id"].astype(str).map(case_pseudonym)
        subset = frame[frame["case_id"].isin(selected)].copy()
        if set(subset["case_id"]) != selected or len(subset) != len(selected):
            raise ValueError(f"{condition} does not contain the exact frozen cohort.")
        condition_truth = {
            str(row.case_id): str(row.label) for row in subset.itertuples(index=False)
        }
        if condition_truth != reference_truth:
            raise ValueError(f"{condition} truth labels do not match the frozen reference cohort.")
        observed = set(subset["label"].astype(str))
        if observed == set(labels):
            output[condition] = evaluate_predictions(
                subset, bins=10, labels=labels, positive_class=labels[-1]
            )
        else:
            y = subset["label"].astype(str)
            predicted = subset["predicted_label"].astype(str)
            output[condition] = {
                "n": int(len(subset)),
                "accuracy": float(accuracy_score(y, predicted)),
                "macro_f1": float(f1_score(
                    y, predicted, labels=labels, average="macro", zero_division=0,
                )),
                "micro_f1": float(f1_score(
                    y, predicted, labels=labels, average="micro", zero_division=0,
                )),
                "macro_auroc_ovr": None,
                "metric_scope": "descriptive_only_missing_true_classes",
                "observed_true_labels": sorted(observed),
                "undefined_true_labels": sorted(set(labels) - observed),
            }
    return output


def _agent_summary(evaluation: dict[str, Any], case_ids: list[str]) -> dict[str, Any]:
    per_class_f1 = [
        row["f1"] for row in evaluation["per_class"].values() if row["f1"] is not None
    ]
    return {
        "case_ids": case_ids,
        "n": evaluation["n_cases"],
        "accuracy": evaluation["accuracy"],
        "coverage": evaluation["coverage"],
        "conditional_accuracy": evaluation["conditional_accuracy"],
        "macro_f1": sum(per_class_f1) / len(per_class_f1) if per_class_f1 else None,
        "ranking_macro_auroc_ovr": evaluation["ranking_metrics"]["macro_auroc_ovr"],
        "ranking_denominator": evaluation["ranking_metrics"]["denominator"],
        "probability_denominator": evaluation["probability_metrics"]["denominator"],
        "status_counts": evaluation["status_counts"],
        "confusion_matrix": evaluation["confusion_matrix"],
    }


def _speechcare_reference(dataset_id: str, n_cases: int, full_cohort: bool) -> dict[str, Any]:
    matched = dataset_id == "PREPARE_DrivenData" and full_cohort and n_cases == 412
    return {
        "benchmark": "SpeechCARE PREPARE official test",
        "reported_micro_auroc_ovr": 0.8683,
        "reported_micro_f1": 0.7211,
        "reported_weighted_auroc_ovr": 0.8067,
        "reported_micro_auprc": 0.7473,
        "reported_weighted_auprc": 0.7350,
        "direct_comparison_valid": matched,
        "reason": (
            "same PREPARE official 412-case test cohort"
            if matched else
            "not the complete PREPARE 412-case official test cohort; benchmark values are context only"
        ),
    }


def _render_report(comparison: dict[str, Any], path: Path) -> None:
    def pct(value: Any) -> str:
        return "NA" if value is None else f"{100 * float(value):.1f}%"

    rows = []
    for name, metrics in comparison["baselines"].items():
        rows.append(
            f"<tr><td>{html.escape(name.upper())}</td><td>{metrics['n']}</td>"
            f"<td>{pct(metrics['accuracy'])}</td><td>{pct(metrics['macro_f1'])}</td>"
            f"<td>{pct(metrics['macro_auroc_ovr'])}</td><td>probability</td></tr>"
        )
    agent = comparison["agent_led"]
    rows.append(
        f"<tr><td>Agent-led</td><td>{agent['n']}</td><td>{pct(agent['accuracy'])}</td>"
        f"<td>{pct(agent['macro_f1'])}</td><td>{pct(agent['ranking_macro_auroc_ovr'])}</td>"
        f"<td>ranking; coverage {pct(agent['coverage'])}</td></tr>"
    )
    speechcare = comparison["speechcare"]
    validity = "VALID" if speechcare["direct_comparison_valid"] else "NOT VALID"
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Agent-led study</title>
<style>body{{font:16px Arial,sans-serif;color:#172026;max-width:1080px;margin:36px auto;line-height:1.5}}
h1,h2{{letter-spacing:0}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #cad3d8;padding:10px;text-align:left}}
.note{{border-left:4px solid #b44;padding:10px 14px;background:#fff7f6}}code{{background:#f1f4f5;padding:2px 4px}}</style></head><body>
<h1>{html.escape(comparison['dataset_id'])}: Agent-led frozen-cohort study</h1>
<p>Selection is label-blind. Truth is attached only after inference. All local baselines use the identical selected cases.</p>
<table><thead><tr><th>Method</th><th>N</th><th>Accuracy</th><th>Macro F1</th><th>AUROC</th><th>Score type</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>SpeechCARE protocol status</h2><p class='note'><strong>{validity}</strong>: {html.escape(speechcare['reason'])}.</p>
<p>Agent AUROC is ranking-only unless complete calibrated probabilities are supplied; it is not interchangeable with SpeechCARE probability AUROC.</p>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def run_agent_led_study(
    root: Path,
    artifact_dir: Path,
    study_dir: Path,
    *,
    dataset_id: str,
    labels: list[str],
    provider: str,
    model: str,
    max_cases: int | None,
    selection_seed: int = 20260917,
    max_steps: int = 16,
    confirm_external_data_permission: bool = False,
) -> dict[str, Any]:
    if provider == "openai_api" and not confirm_external_data_permission:
        raise ValueError("openai_api requires explicit external-data permission for selected evidence.")
    prepared = prepare_agent_led_study(
        artifact_dir, study_dir, dataset_id=dataset_id, labels=labels,
        max_cases=max_cases, selection_seed=selection_seed,
    )
    inference_dir = study_dir / "inference"
    run_agent_led_cohort(
        root, prepared["workspaces_path"], inference_dir, labels,
        provider=provider, model=model, max_steps=max_steps,
        truth_path=prepared["truth_path"],
    )
    manifest = _load_json(study_dir / "study_manifest.json")
    selected_ids = manifest["selected_case_ids"]
    evaluation = _load_json(inference_dir / "evaluation.json")
    baselines = _subset_metrics(artifact_dir, selected_ids, labels, dataset_id)
    comparison = {
        "study_version": "agent-led-study-v1",
        "dataset_id": dataset_id,
        "labels": labels,
        "cohort_case_ids": selected_ids,
        "baselines": baselines,
        "agent_led": _agent_summary(evaluation, selected_ids),
        "speechcare": _speechcare_reference(
            dataset_id, len(selected_ids), max_cases is None,
        ),
        "created_at_utc": now_utc(),
    }
    json_dump(comparison, study_dir / "comparison.json")
    _render_report(comparison, study_dir / "study_report.html")
    summary = {
        "dataset_id": dataset_id,
        "cases": len(selected_ids),
        "provider": provider,
        "model": model,
        "agent_led_accuracy": comparison["agent_led"]["accuracy"],
        "agent_led_coverage": comparison["agent_led"]["coverage"],
        "speechcare_direct_comparison_valid": comparison["speechcare"]["direct_comparison_valid"],
        "study_dir": str(study_dir),
    }
    json_dump(summary, study_dir / "study_summary.json")
    return summary
