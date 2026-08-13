from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


_ASR_MODEL: Any = None


def _init_asr(model_name: str, cpu_threads: int) -> None:
    global _ASR_MODEL
    from faster_whisper import WhisperModel

    _ASR_MODEL = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
        num_workers=1,
    )


def _transcribe_recording(payload: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    record, language, model_name = payload
    if _ASR_MODEL is None:
        raise RuntimeError("ASR worker was not initialized")
    segments, info = _ASR_MODEL.transcribe(
        record["audio_path"],
        language=language,
        vad_filter=True,
        beam_size=3,
        condition_on_previous_text=False,
    )
    segment_rows = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            segment_rows.append({"start": float(segment.start), "end": float(segment.end), "text": text})
    return {
        "case_id": record["case_id"],
        "subject_id": record["subject_id"],
        "language": info.language,
        "language_probability": float(info.language_probability),
        "text": " ".join(item["text"] for item in segment_rows),
        "segments_json": json.dumps(segment_rows, ensure_ascii=False),
        "asr_model": model_name,
    }


def transcribe_test_audio(
    manifest_path: Path,
    recording_transcripts_path: Path,
    subject_transcripts_path: Path,
    agents_config: dict[str, Any],
) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    test = manifest[manifest["split"].eq("test")].copy()
    model_name = agents_config["asr_model"]
    language = agents_config.get("asr_language", "zh")
    payloads = [(record, language, model_name) for record in test.to_dict("records")]
    with ProcessPoolExecutor(
        max_workers=int(agents_config.get("asr_workers", 2)),
        initializer=_init_asr,
        initargs=(model_name, int(agents_config.get("asr_cpu_threads", 4))),
    ) as executor:
        rows = list(executor.map(_transcribe_recording, payloads))
    recording = pd.DataFrame(rows)
    recording.to_csv(recording_transcripts_path, index=False)
    subjects = []
    for subject_id, group in recording.groupby("subject_id", sort=True):
        parts = [f"[录音{index + 1}] {text}" for index, text in enumerate(group["text"].fillna(""))]
        subjects.append(
            {
                "subject_id": str(subject_id),
                "transcript": "\n".join(parts),
                "recording_count": len(group),
                "asr_model": model_name,
            }
        )
    pd.DataFrame(subjects).to_csv(subject_transcripts_path, index=False)

