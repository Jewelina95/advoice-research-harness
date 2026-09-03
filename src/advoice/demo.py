from __future__ import annotations

import base64
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .features import extract_audio_file


DEMO_REFERENCES: dict[str, dict[str, float | str]] = {
    "silence_fraction": {"median": 0.34, "scale": 0.12, "direction": 1, "unit": "ratio"},
    "long_pause_rate_min": {"median": 5.0, "scale": 3.0, "direction": 1, "unit": "events/min"},
    "speech_run_mean_sec": {"median": 2.2, "scale": 0.8, "direction": -1, "unit": "s"},
    "voiced_fraction": {"median": 0.66, "scale": 0.12, "direction": -1, "unit": "ratio"},
    "speech_rate_wpm": {"median": 118.0, "scale": 28.0, "direction": -1, "unit": "words/min"},
    "filler_rate_100w": {"median": 2.0, "scale": 2.0, "direction": 1, "unit": "events/100 words"},
    "lexical_ttr": {"median": 0.58, "scale": 0.12, "direction": -1, "unit": "ratio"},
}

STATE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "S01",
        "name": "停顿与连续性",
        "question": "发言中是否出现较多停顿、较短连续发声或启动中断？",
        "metrics": [("silence_fraction", 0.40), ("long_pause_rate_min", 0.35), ("speech_run_mean_sec", 0.25)],
    },
    {
        "id": "S02",
        "name": "输出效率",
        "question": "单位时间内的有效言语输出是否减少？",
        "metrics": [("voiced_fraction", 0.45), ("speech_rate_wpm", 0.55)],
    },
    {
        "id": "S08",
        "name": "词汇提取与多样性",
        "question": "表达中是否出现填充增加或词汇多样性降低？",
        "metrics": [("filler_rate_100w", 0.45), ("lexical_ttr", 0.55)],
    },
]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _evidence(feature: dict[str, Any], metric_id: str) -> dict[str, Any]:
    reference = DEMO_REFERENCES[metric_id]
    value = _finite(feature.get(metric_id))
    missing = value is None
    z = 0.0 if missing else (value - float(reference["median"])) / float(reference["scale"])
    directional_z = float(reference["direction"]) * z
    source = "transcript" if metric_id in {"speech_rate_wpm", "filler_rate_100w", "lexical_ttr"} else "audio"
    reliability = float(feature.get("text_reliability", 0.0) if source == "transcript" else feature.get("audio_reliability", 0.0))
    return {
        "id": f"demo:{metric_id}",
        "metric_id": metric_id,
        "value": value,
        "unit": reference["unit"],
        "reference_median": reference["median"],
        "reference_scale": reference["scale"],
        "directional_z": round(directional_z, 3),
        "reliability": round(reliability, 3),
        "source": source,
        "missing": missing,
        "reference_scope": "illustrative_demo_reference_not_a_clinical_norm",
    }


def analyze_demo_audio(
    audio_path: Path,
    transcript: str,
    *,
    language: str = "en",
    task_type: str = "cookie_theft_picture_description",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="advoice-demo-") as directory:
        transcript_path = Path(directory) / "transcript.txt"
        transcript_path.write_text(transcript.strip(), encoding="utf-8")
        feature, segments = extract_audio_file(
            {
                "dataset_id": "PUBLIC_SYNTHETIC_DEMO",
                "case_id": "synthetic_case_001",
                "subject_id": "synthetic_subject_001",
                "label": "UNLABELED",
                "split": "demo",
                "audio_path": str(audio_path),
                "transcript_path": str(transcript_path),
                "transcript_reliability": 0.95 if transcript.strip() else 0.0,
                "task_type": task_type,
                "language": language,
                "channel": "public_demo",
                "analysis_intervals": "[]",
                "role_filter_required": False,
            }
        )

    evidence = {metric: _evidence(feature, metric) for metric in DEMO_REFERENCES}
    states: list[dict[str, Any]] = []
    for definition in STATE_DEFINITIONS:
        available = [(evidence[metric], weight) for metric, weight in definition["metrics"] if not evidence[metric]["missing"]]
        denominator = sum(weight * item["reliability"] for item, weight in available)
        score = (
            sum(weight * item["reliability"] * item["directional_z"] for item, weight in available) / denominator
            if denominator
            else 0.0
        )
        states.append(
            {
                "id": definition["id"],
                "name": definition["name"],
                "clinical_question": definition["question"],
                "score": round(float(score), 3),
                "confidence": round(min(denominator / max(sum(weight for _, weight in definition["metrics"]), 1e-9), 1.0), 3),
                "evidence_ids": [item["id"] for item, _ in available],
            }
        )

    compact_segments = [
        {
            "segment_id": row.get("segment_id", f"segment_{index + 1}"),
            "start_sec": round(float(row.get("start_sec", 0.0)), 2),
            "end_sec": round(float(row.get("end_sec", 0.0)), 2),
            "rms_db": round(float(row.get("rms_db_mean", row.get("rms_db", -80.0))), 2),
            "voiced_fraction": round(float(row.get("voiced_fraction", 0.0)), 3),
        }
        for index, row in enumerate(segments)
    ]
    return {
        "schema_version": "public-demo-v1",
        "case": {
            "case_id": "synthetic_case_001",
            "task_type": task_type,
            "language": language,
            "duration_sec": round(float(feature["duration_sec"]), 3),
            "transcript": transcript.strip(),
        },
        "quality": {
            "audio_reliability": round(float(feature["audio_reliability"]), 3),
            "text_reliability": round(float(feature["text_reliability"]), 3),
            "snr_proxy_db": round(float(feature["snr_proxy_db"]), 2),
            "clipping_fraction": round(float(feature["clipping_fraction"]), 5),
            "vad_backend": feature["vad_backend"],
        },
        "metric_evidence": list(evidence.values()),
        "state_cards": states,
        "segments": compact_segments,
        "decision": {
            "status": "not_generated",
            "reason": "The public synthetic demo validates the evidence pipeline only; it is not a clinical prediction.",
        },
        "trace": [
            {"from": item["id"], "to": state["id"]}
            for state in states
            for item in evidence.values()
            if item["id"] in state["evidence_ids"]
        ],
        "disclaimer": "Synthetic non-patient demonstration. Illustrative references are not clinical norms and no diagnosis is produced.",
    }


def analyze_base64_wav(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = str(payload.get("audio_base64", ""))
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("Audio payload exceeds the 20 MB demo limit.")
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        handle.write(raw)
        handle.flush()
        return analyze_demo_audio(
            Path(handle.name),
            str(payload.get("transcript", "")),
            language=str(payload.get("language", "en")),
            task_type=str(payload.get("task_type", "cookie_theft_picture_description")),
        )


def write_demo_result(audio_path: Path, transcript_path: Path, output_path: Path) -> dict[str, Any]:
    result = analyze_demo_audio(audio_path, transcript_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
