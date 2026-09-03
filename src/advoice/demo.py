from __future__ import annotations

import base64
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .features import extract_audio_file
from .transcripts import read_transcript


DEMO_REFERENCES: dict[str, dict[str, Any]] = {
    "silence_fraction": {"name": "Silence fraction", "median": 0.34, "scale": 0.12, "direction": 1, "unit": "ratio", "source": "audio", "role": "clinical_state"},
    "long_pause_rate_min": {"name": "Long pauses per minute", "median": 5.0, "scale": 3.0, "direction": 1, "unit": "events/min", "source": "audio", "role": "clinical_state"},
    "speech_run_mean_sec": {"name": "Mean speech-run duration", "median": 2.2, "scale": 0.8, "direction": -1, "unit": "seconds", "source": "audio", "role": "clinical_state"},
    "voiced_fraction": {"name": "Voiced-time fraction", "median": 0.66, "scale": 0.12, "direction": -1, "unit": "ratio", "source": "audio", "role": "clinical_state"},
    "speech_rate_wpm": {"name": "Speech rate", "median": 118.0, "scale": 28.0, "direction": -1, "unit": "words/min", "source": "transcript", "role": "clinical_state"},
    "filler_rate_100w": {"name": "Fillers per 100 words", "median": 2.0, "scale": 2.0, "direction": 1, "unit": "events/100 words", "source": "transcript", "role": "clinical_state"},
    "lexical_ttr": {"name": "Lexical diversity", "median": 0.58, "scale": 0.12, "direction": -1, "unit": "ratio", "source": "transcript", "role": "clinical_state"},
    "patient_turn_share": {"name": "Participant turn share", "median": 0.75, "scale": 0.15, "direction": -1, "unit": "ratio", "source": "dialogue", "role": "clinical_state"},
    "mean_utterance_words": {"name": "Mean utterance length", "median": 10.0, "scale": 4.0, "direction": -1, "unit": "words/turn", "source": "dialogue", "role": "clinical_state"},
    "repair_rate_100w": {"name": "Repairs per 100 words", "median": 2.0, "scale": 2.0, "direction": 1, "unit": "events/100 words", "source": "transcript", "role": "clinical_state"},
    "lexical_mattr50": {"name": "Moving-average lexical diversity", "median": 0.70, "scale": 0.12, "direction": -1, "unit": "ratio", "source": "transcript", "role": "clinical_state"},
    "content_word_ratio": {"name": "Content-word ratio", "median": 0.55, "scale": 0.10, "direction": -1, "unit": "ratio", "source": "transcript", "role": "clinical_state"},
    "picture_content_unit_coverage": {"name": "Picture content-unit coverage", "median": 0.70, "scale": 0.20, "direction": -1, "unit": "ratio", "source": "task_score", "role": "clinical_state"},
    "picture_information_density": {"name": "Picture information density", "median": 8.0, "scale": 4.0, "direction": -1, "unit": "units/100 words", "source": "task_score", "role": "clinical_state"},
    "picture_content_redundancy": {"name": "Content redundancy", "median": 0.15, "scale": 0.10, "direction": 1, "unit": "ratio", "source": "task_score", "role": "clinical_state"},
    "picture_uncertainty_rate_100w": {"name": "Uncertainty expressions", "median": 3.0, "scale": 3.0, "direction": 1, "unit": "events/100 words", "source": "task_score", "role": "clinical_state"},
    "rms_db_std": {"name": "Loudness variability", "median": 8.0, "scale": 4.0, "direction": -1, "unit": "dB", "source": "audio", "role": "model_auxiliary"},
    "f0_iqr_hz": {"name": "Pitch interquartile range", "median": 40.0, "scale": 20.0, "direction": -1, "unit": "Hz", "source": "audio", "role": "model_auxiliary"},
    "snr_proxy_db": {"name": "Signal-to-noise proxy", "median": 15.0, "scale": 5.0, "direction": -1, "unit": "dB", "source": "audio", "role": "quality_control"},
}

BASE_DEMO_METRICS = [
    "silence_fraction",
    "long_pause_rate_min",
    "speech_run_mean_sec",
    "voiced_fraction",
    "speech_rate_wpm",
    "filler_rate_100w",
    "lexical_ttr",
]

CHANNEL_EXTRA_METRICS = {
    "clinical_interview": ["patient_turn_share", "mean_utterance_words", "repair_rate_100w"],
    "picture_description": ["picture_content_unit_coverage", "picture_information_density", "picture_content_redundancy", "picture_uncertainty_rate_100w"],
    "structured_multitask": ["lexical_mattr50", "content_word_ratio", "repair_rate_100w"],
    "public_speech": ["rms_db_std", "f0_iqr_hz", "snr_proxy_db"],
}

STATE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "S01",
        "name": "Pausing and continuity",
        "question": "Are pauses, shortened speech runs, or interrupted starts elevated?",
        "metrics": [("silence_fraction", 0.40), ("long_pause_rate_min", 0.35), ("speech_run_mean_sec", 0.25)],
    },
    {
        "id": "S02",
        "name": "Output efficiency",
        "question": "Is effective spoken output reduced for the available task time?",
        "metrics": [("voiced_fraction", 0.45), ("speech_rate_wpm", 0.55)],
    },
    {
        "id": "S08",
        "name": "Lexical retrieval and diversity",
        "question": "Are fillers increased or lexical diversity reduced?",
        "metrics": [("filler_rate_100w", 0.45), ("lexical_ttr", 0.55)],
    },
    {
        "id": "S10",
        "name": "Task information density",
        "question": "Does the picture description omit content, repeat information, or use uncertain wording?",
        "metrics": [("picture_content_unit_coverage", 0.45), ("picture_information_density", 0.35), ("picture_content_redundancy", 0.10), ("picture_uncertainty_rate_100w", 0.10)],
        "channels": {"picture_description"},
    },
    {
        "id": "S12",
        "name": "Interview interaction burden",
        "question": "Do participant share, response length, and repairs indicate increased interaction burden?",
        "metrics": [("patient_turn_share", 0.50), ("mean_utterance_words", 0.30), ("repair_rate_100w", 0.20)],
        "channels": {"clinical_interview"},
    },
    {
        "id": "S07",
        "name": "Task-specific lexical retrieval",
        "question": "Is lexical retrieval, diversity, or content production reduced within the cognitive task?",
        "metrics": [("lexical_mattr50", 0.45), ("content_word_ratio", 0.35), ("repair_rate_100w", 0.20)],
        "channels": {"structured_multitask"},
    },
]

PUBLIC_DEMO_CASES: dict[str, dict[str, Any]] = {
    "synthetic_clinical_interview": {
        "case_id": "synthetic_clinical_interview",
        "dataset_id": "PUBLIC_SYNTHETIC_INTERVIEW",
        "channel_id": "clinical_interview",
        "channel_name": "Clinical interview",
        "task_name": "Structured participant interview",
        "description": "Demonstrates participant-role routing and dialogue evidence without clinical data.",
        "evidence_focus": ["participant turns", "response length", "repairs", "pausing"],
        "task_type": "structured_clinical_interview",
        "language": "en",
        "research_label": "UNLABELED",
        "audio_file": "synthetic_clinical_interview.wav",
        "transcript_file": "synthetic_clinical_interview.txt",
    },
    "synthetic_picture_description": {
        "case_id": "synthetic_picture_description",
        "dataset_id": "PUBLIC_SYNTHETIC_PICTURE",
        "channel_id": "picture_description",
        "channel_name": "Picture description",
        "task_name": "Cookie Theft-style description",
        "description": "Demonstrates content-unit, information-density, lexical, and speech evidence.",
        "evidence_focus": ["content units", "information density", "lexical diversity", "pausing"],
        "task_type": "cookie_theft_picture_description",
        "language": "en",
        "research_label": "UNLABELED",
        "audio_file": "synthetic_picture_description.wav",
        "transcript_file": "synthetic_picture_description.txt",
    },
    "synthetic_structured_task": {
        "case_id": "synthetic_structured_task",
        "dataset_id": "PUBLIC_SYNTHETIC_TASK",
        "channel_id": "structured_multitask",
        "channel_name": "Structured cognitive task",
        "task_name": "Semantic fluency-style response",
        "description": "Demonstrates task-specific lexical retrieval and state aggregation.",
        "evidence_focus": ["lexical retrieval", "content words", "repairs", "output efficiency"],
        "task_type": "semantic_fluency",
        "language": "en",
        "research_label": "UNLABELED",
        "audio_file": "synthetic_structured_task.wav",
        "transcript_file": "synthetic_structured_task.txt",
    },
    "synthetic_public_speech": {
        "case_id": "synthetic_public_speech",
        "dataset_id": "PUBLIC_SYNTHETIC_NATURAL",
        "channel_id": "public_speech",
        "channel_name": "Natural speech",
        "task_name": "Non-standard spontaneous speech",
        "description": "Demonstrates conservative use of prosody and recording-quality evidence.",
        "evidence_focus": ["speech continuity", "prosody", "signal quality", "report permissions"],
        "task_type": "nonstandard_public_speech",
        "language": "en",
        "research_label": "UNLABELED",
        "audio_file": "synthetic_public_speech.wav",
        "transcript_file": "synthetic_public_speech.txt",
    },
}


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes="):
        return None
    start_text, _, end_text = header.removeprefix("bytes=").partition("-")
    if not start_text:
        length = min(int(end_text), size)
        return size - length, size - 1
    start = int(start_text)
    end = min(int(end_text), size - 1) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("invalid byte range")
    return start, end


def public_case_summaries() -> list[dict[str, Any]]:
    summaries = []
    for case in PUBLIC_DEMO_CASES.values():
        summary = {key: value for key, value in case.items() if key not in {"audio_file", "transcript_file"}}
        summary["data_scope"] = "public_synthetic"
        summaries.append(summary)
    return summaries


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
    source = str(reference["source"])
    reliability = float(feature.get("text_reliability", 0.0) if source == "transcript" else feature.get("audio_reliability", 0.0))
    if source in {"dialogue", "task_score"}:
        reliability = float(feature.get("text_reliability", 0.0))
    return {
        "id": f"demo:{metric_id}",
        "metric_id": metric_id,
        "name": reference["name"],
        "value": value,
        "unit": reference["unit"],
        "reference_median": reference["median"],
        "reference_scale": reference["scale"],
        "directional_z": round(directional_z, 3),
        "reliability": round(reliability, 3),
        "source": source,
        "evidence_role": reference["role"],
        "reportable": reference["role"] == "clinical_state",
        "missing": missing,
        "reference_scope": "illustrative_demo_reference_not_a_clinical_norm",
    }


def _agent_report_preview(
    states: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    quality: dict[str, Any],
    *,
    synthetic: bool,
) -> dict[str, Any]:
    ranked_states = sorted(states, key=lambda item: abs(float(item["score"])), reverse=True)
    reportable = [item for item in evidence if item["reportable"] and not item["missing"]]
    cited_ids = [item["id"] for item in sorted(reportable, key=lambda item: abs(float(item["directional_z"])), reverse=True)[:5]]
    observations = [
        f"{item['id']} {item['name']}: evidence score {item['score']:+.2f}, completeness {item['confidence']:.2f}."
        for item in ranked_states[:3]
    ]
    return {
        "status": "offline_preview",
        "title": "Evidence-constrained Agent report preview",
        "screening_impression": (
            "This synthetic case is unlabeled. The report demonstrates the output contract and does not assign a diagnosis or disease probability."
            if synthetic
            else "This restricted local case is shown for pipeline inspection only. No new cohort prediction is generated by the demo."
        ),
        "observations": observations,
        "evidence_ids": cited_ids,
        "quality_statement": (
            f"Audio reliability {quality['audio_reliability']:.2f}; text reliability {quality['text_reliability']:.2f}. "
            "Quality-control and model-auxiliary measures are not presented as disease evidence."
        ),
        "next_action": "Use the full, frozen research pipeline for cohort-level prediction. Clinical interpretation requires task context and independent assessment.",
        "generation": {
            "mode": "deterministic_offline_preview",
            "live_model_invoked": False,
            "full_pipeline_model": "configured in configs/agents/default.yaml",
        },
    }


def _assemble_result(
    feature: dict[str, Any],
    segments: list[dict[str, Any]],
    transcript: str,
    case: dict[str, Any],
    *,
    synthetic: bool,
) -> dict[str, Any]:
    channel_id = str(case.get("channel_id", ""))
    metric_ids = BASE_DEMO_METRICS + CHANNEL_EXTRA_METRICS.get(channel_id, [])
    evidence = {metric: _evidence(feature, metric) for metric in metric_ids}
    states: list[dict[str, Any]] = []
    for definition in STATE_DEFINITIONS:
        if definition.get("channels") and channel_id not in definition["channels"]:
            continue
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
    if synthetic and compact_segments:
        words = transcript.strip().split()
        chunk_size = max(1, math.ceil(len(words) / len(compact_segments)))
        for index, segment in enumerate(compact_segments):
            segment["text"] = " ".join(words[index * chunk_size : (index + 1) * chunk_size])
            segment["text_alignment"] = "synthetic_demo_partition"
    else:
        for segment in compact_segments:
            segment["text"] = ""
            segment["text_alignment"] = "not_available_in_demo"
    audio_segment_ids = [item["segment_id"] for item in compact_segments[:4]]
    for item in evidence.values():
        item["segment_ids"] = audio_segment_ids if item["source"] == "audio" else []

    quality = {
        "audio_reliability": round(float(feature["audio_reliability"]), 3),
        "text_reliability": round(float(feature["text_reliability"]), 3),
        "snr_proxy_db": round(float(feature["snr_proxy_db"]), 2),
        "clipping_fraction": round(float(feature["clipping_fraction"]), 5),
        "vad_backend": feature["vad_backend"],
    }
    evidence_values = list(evidence.values())
    return {
        "schema_version": "public-demo-v3",
        "case": {
            **case,
            "duration_sec": round(float(feature["duration_sec"]), 3),
            "original_duration_sec": round(float(feature.get("original_duration_sec", feature["duration_sec"])), 3),
            "role_filtered_audio": bool(feature.get("role_filtered_audio", False)),
            "role_coverage_fraction": round(float(feature.get("role_coverage_fraction", 1.0)), 3),
            "transcript": transcript.strip(),
        },
        "quality": quality,
        "metric_evidence": evidence_values,
        "state_cards": states,
        "segments": compact_segments,
        "decision": {
            "status": "not_generated",
            "reason": (
                "The public synthetic demo validates the evidence pipeline only; it is not a clinical prediction."
                if synthetic
                else "This local restricted case demonstrates channel processing only; it is not a cohort prediction."
            ),
        },
        "agent_report": _agent_report_preview(states, evidence_values, quality, synthetic=synthetic),
        "trace": [
            {"from": item["id"], "to": state["id"], "segment_ids": item["segment_ids"]}
            for state in states
            for item in evidence.values()
            if item["id"] in state["evidence_ids"]
        ],
        "execution": {
            "mode": "public_offline_demo" if synthetic else "local_restricted_demo",
            "trained_prediction_loaded": False,
            "diagnostic_agent_invoked": False,
            "full_run_command": "make full DATASET=ADReSS_2020",
        },
        "disclaimer": (
            "Synthetic non-patient demonstration. Illustrative references are not clinical norms and no diagnosis is produced."
            if synthetic
            else "Local restricted-data demonstration. Audio is not copied into the repository; illustrative demo references are not cohort norms."
        ),
    }


def _extract_case(
    audio_path: Path,
    transcript: str,
    case: dict[str, Any],
    *,
    synthetic: bool,
    analysis_intervals: str = "[]",
    role_filter_required: bool = False,
    transcript_reliability: float = 0.95,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="advoice-demo-") as directory:
        transcript_path = Path(directory) / "transcript.txt"
        transcript_path.write_text(transcript.strip(), encoding="utf-8")
        feature, segments = extract_audio_file(
            {
                "dataset_id": str(case["dataset_id"]),
                "case_id": str(case["case_id"]),
                "subject_id": str(case["case_id"]),
                "label": str(case.get("research_label", "UNLABELED")),
                "split": "demo",
                "audio_path": str(audio_path),
                "transcript_path": str(transcript_path),
                "transcript_reliability": transcript_reliability if transcript.strip() else 0.0,
                "task_type": str(case.get("task_type", "")),
                "language": str(case.get("language", "en")),
                "channel": str(case.get("channel_id", "public_demo")),
                "analysis_intervals": analysis_intervals,
                "role_filter_required": role_filter_required,
            }
        )
    return _assemble_result(feature, segments, transcript, case, synthetic=synthetic)


def analyze_public_case(case_id: str, assets_dir: Path) -> dict[str, Any]:
    definition = PUBLIC_DEMO_CASES.get(case_id)
    if definition is None:
        raise KeyError(case_id)
    transcript = (assets_dir / str(definition["transcript_file"])).read_text(encoding="utf-8")
    case = {key: value for key, value in definition.items() if key not in {"audio_file", "transcript_file"}}
    case["data_scope"] = "public_synthetic"
    return _extract_case(assets_dir / str(definition["audio_file"]), transcript, case, synthetic=True)


def analyze_demo_audio(
    audio_path: Path,
    transcript: str,
    *,
    language: str = "en",
    task_type: str = "cookie_theft_picture_description",
) -> dict[str, Any]:
    channel_id = {
        "structured_clinical_interview": "clinical_interview",
        "cookie_theft_picture_description": "picture_description",
        "ctd": "picture_description",
        "semantic_fluency": "structured_multitask",
        "sft": "structured_multitask",
        "pft": "structured_multitask",
        "nonstandard_public_speech": "public_speech",
        "spontaneous_speech": "public_speech",
    }.get(task_type, "picture_description")
    case = {
        "case_id": "uploaded_demo_case",
        "dataset_id": "LOCAL_UPLOAD",
        "channel_id": channel_id,
        "channel_name": "Uploaded local audio",
        "task_name": task_type.replace("_", " ").title(),
        "description": "Locally processed upload. The browser demo does not create a clinical prediction.",
        "evidence_focus": ["audio metrics", "transcript metrics", "state aggregation", "traceability"],
        "task_type": task_type,
        "language": language,
        "research_label": "UNLABELED",
        "data_scope": "local_upload_not_persisted",
    }
    return _extract_case(audio_path, transcript, case, synthetic=True)


def _case_value(case: dict[str, Any], key: str, legacy_key: str, default: Any) -> Any:
    value = case.get(key)
    return value if value is not None else case.get(legacy_key, default)


def analyze_local_manifest_case(case: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(case["audio_path"]))
    transcript_value = str(case.get("transcript_path", "")).strip()
    transcript_path = Path(transcript_value) if transcript_value else None
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    transcript, _, _ = (
        read_transcript(str(transcript_path))
        if transcript_path is not None and transcript_path.is_file()
        else ("", "none", 0.0)
    )
    normalized = {
        "case_id": str(case["demo_case_id"]),
        "dataset_id": str(case["dataset_id"]),
        "channel_id": str(case["channel_id"]),
        "channel_name": str(_case_value(case, "channel_name", "channel_name_zh", case["channel_id"])),
        "task_name": str(_case_value(case, "task_name", "task_name_zh", case.get("task_type", "Task"))),
        "description": str(_case_value(case, "description", "description_zh", "Restricted local case")),
        "evidence_focus": list(_case_value(case, "evidence_focus", "evidence_focus_zh", [])),
        "task_type": str(case.get("task_type", "")),
        "language": str(case.get("language", "")),
        "research_label": str(case.get("research_label", "UNAVAILABLE")),
        "data_scope": "local_restricted_not_for_redistribution",
    }
    return _extract_case(
        audio_path,
        transcript,
        normalized,
        synthetic=False,
        analysis_intervals=str(case.get("analysis_intervals", "[]")),
        role_filter_required=bool(case.get("role_filter_required", False)),
        transcript_reliability=float(case.get("transcript_reliability", 0.0)),
    )


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


def write_public_demo_bundle(assets_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = public_case_summaries()
    for case in cases:
        result = analyze_public_case(str(case["case_id"]), assets_dir)
        (output_dir / f"{case['case_id']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "public_cases.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cases


def write_demo_result(audio_path: Path, transcript_path: Path, output_path: Path) -> dict[str, Any]:
    result = analyze_demo_audio(audio_path, transcript_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
