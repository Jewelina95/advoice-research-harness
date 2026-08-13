from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _state_category(state_z: float, reliability: float, missing_fraction: float) -> str:
    if reliability < 0.45 or missing_fraction > 0.50:
        return "unreliable"
    if state_z >= 2.0:
        return "impaired"
    if state_z >= 1.0:
        return "borderline"
    return "normal"


def _segment_evidence(
    subject_id: str,
    state_id: str,
    recordings: pd.DataFrame,
    segments: pd.DataFrame,
) -> list[dict[str, Any]]:
    case_ids = recordings.loc[recordings["subject_id"].astype(str).eq(str(subject_id)), "case_id"]
    available = segments[segments["case_id"].isin(case_ids)].copy()
    if available.empty or state_id not in {"S01", "S02", "S03"}:
        return []
    if state_id == "S01":
        chosen = available.nlargest(2, "silence_fraction")
    else:
        chosen = available.nsmallest(2, "rms_db_mean")
    return chosen[
        ["segment_id", "case_id", "start_sec", "end_sec", "silence_fraction", "rms_db_mean"]
    ].to_dict("records")


def build_state_cards(
    evidence_path: Path,
    recording_features_path: Path,
    segments_path: Path,
    states_config: dict[str, Any],
    state_cards_path: Path,
    state_wide_path: Path,
) -> None:
    evidence = pd.read_csv(evidence_path, dtype={"subject_id": str})
    recordings = pd.read_csv(recording_features_path, dtype={"subject_id": str})
    segments = pd.read_csv(segments_path)
    rows: list[dict[str, Any]] = []
    for keys, subject_evidence in evidence.groupby(["dataset_id", "subject_id", "label", "split"]):
        dataset_id, subject_id, label, split = keys
        for definition in states_config["states"]:
            state_id = definition["id"]
            metric_weights = dict(zip(definition["metrics"], definition["weights"], strict=True))
            state = subject_evidence[subject_evidence["metric_id"].isin(definition["metrics"])].copy()
            state["configured_weight"] = state["metric_id"].map(metric_weights).fillna(0.0)
            state["effective_weight"] = (
                state["configured_weight"] * state["reliability"] * (~state["missing"]).astype(float)
            )
            denominator = float(state["effective_weight"].sum())
            if denominator:
                state_z = float((state["directional_z"] * state["effective_weight"]).sum() / denominator)
                reliability = float(
                    (state["reliability"] * state["configured_weight"]).sum()
                    / max(float(state["configured_weight"].sum()), 1e-9)
                )
            else:
                state_z, reliability = 0.0, 0.0
            missing_fraction = float(state["missing"].mean()) if len(state) else 1.0
            state_sorted = state.sort_values("directional_z", ascending=False)
            support = state_sorted.head(3)[
                [
                    "metric_id",
                    "value",
                    "cn_train_median",
                    "robust_z",
                    "directional_z",
                    "reliability",
                    "report_permission",
                    "confound_tags",
                ]
            ].to_dict("records")
            counter = state_sorted.tail(2)[
                ["metric_id", "value", "cn_train_median", "directional_z", "reliability"]
            ].to_dict("records")
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": subject_id,
                    "label": label,
                    "split": split,
                    "state_id": state_id,
                    "state_name_zh": definition["name_zh"],
                    "branch": definition["branch"],
                    "clinical_question": definition["clinical_question"],
                    "state_z": state_z,
                    "severity": float(1.0 / (1.0 + np.exp(-state_z))),
                    "category": _state_category(state_z, reliability, missing_fraction),
                    "confidence": reliability,
                    "missing_fraction": missing_fraction,
                    "supporting_metrics": json.dumps(support, ensure_ascii=False),
                    "counter_evidence": json.dumps(counter, ensure_ascii=False),
                    "evidence_segments": json.dumps(
                        _segment_evidence(subject_id, state_id, recordings, segments), ensure_ascii=False
                    ),
                }
            )
    cards = pd.DataFrame(rows)
    cards.to_csv(state_cards_path, index=False)
    identity = ["dataset_id", "subject_id", "label", "split"]
    score = cards.pivot(index=identity, columns="state_id", values="state_z").add_prefix("state_")
    confidence = cards.pivot(index=identity, columns="state_id", values="confidence").add_prefix("rel_")
    wide = score.join(confidence).reset_index()
    wide.to_csv(state_wide_path, index=False)

