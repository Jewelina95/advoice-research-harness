from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import json_dump


def _robust_reference(values: pd.Series) -> tuple[float, float]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if finite.empty:
        return 0.0, 1.0
    median = float(finite.median())
    mad = float((finite - median).abs().median())
    scale = max(1.4826 * mad, float(finite.std(ddof=0)) * 0.25, 1e-6)
    return median, scale


def build_metric_evidence(
    subject_features_path: Path,
    metrics_config: dict[str, Any],
    evidence_path: Path,
    reference_path: Path,
) -> None:
    subjects = pd.read_csv(subject_features_path, dtype={"subject_id": str})
    metric_defs = metrics_config["metrics"]
    controls = subjects[subjects["split"].eq("train") & subjects["label"].eq("HC")]
    references: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for definition in metric_defs:
        metric = definition["id"]
        if metric not in subjects.columns:
            references[metric] = {"median": 0.0, "scale": 1.0, "available": False}
            continue
        median, scale = _robust_reference(controls[metric])
        references[metric] = {"median": median, "scale": scale, "available": True}
        for subject in subjects.to_dict("records"):
            value = subject.get(metric)
            missing = value is None or not np.isfinite(value)
            z = float((value - median) / scale) if not missing else np.nan
            direction = int(definition["direction"])
            directional_z = float(direction * z) if direction and not missing else 0.0
            reliability = float(definition["reliability"] * subject.get("audio_reliability", 0.0))
            if metric.startswith("f0_"):
                reliability *= float(np.clip(subject.get("f0_valid_fraction", 0.0) / 0.45, 0.0, 1.0))
            rows.append(
                {
                    "dataset_id": subject["dataset_id"],
                    "subject_id": subject["subject_id"],
                    "label": subject["label"],
                    "split": subject["split"],
                    "metric_id": metric,
                    "state_id": definition["state"],
                    "branch": definition["branch"],
                    "value": value,
                    "cn_train_median": median,
                    "cn_train_scale": scale,
                    "robust_z": z,
                    "direction": direction,
                    "directional_z": directional_z,
                    "evidence_role": definition["role"],
                    "reliability": reliability,
                    "missing": bool(missing),
                    "confound_tags": json.dumps(definition.get("confounds", []), ensure_ascii=False),
                    "report_permission": bool(definition["report_permission"]),
                }
            )
    pd.DataFrame(rows).to_csv(evidence_path, index=False)
    json_dump(
        {
            "reference_population": "official training split, HC subjects only",
            "normalization": "median and max(1.4826*MAD, 0.25*SD, 1e-6)",
            "metrics": references,
        },
        reference_path,
    )

