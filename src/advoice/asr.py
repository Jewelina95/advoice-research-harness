from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .features import _analysis_audio
from .transcripts import canonical_language, read_transcript


_ASR_MODEL: Any = None


def _effective_asr_language(record: dict[str, Any], configured_language: str) -> str | None:
    configured = canonical_language(configured_language)
    if configured:
        return configured
    manifest_language = canonical_language(str(record.get("language", "")))
    return manifest_language or None


def _assert_role_filter_available(record: dict[str, Any]) -> None:
    if bool(record.get("role_filter_required", False)) and str(
        record.get("analysis_intervals", "[]")
    ) in {"", "[]", "nan"}:
        raise ValueError(
            "Patient-only ASR was required, but no valid patient time intervals were available."
        )


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
    record, configured_language, model_name = payload
    if _ASR_MODEL is None:
        raise RuntimeError("ASR worker was not initialized")
    language = _effective_asr_language(record, configured_language)
    _assert_role_filter_available(record)
    audio_input: Any = record["audio_path"]
    if str(record.get("analysis_intervals", "[]")) not in {"", "[]", "nan"}:
        import soundfile as sf
        import librosa

        waveform, sample_rate = sf.read(record["audio_path"], dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        waveform, _, _, _ = _analysis_audio(waveform, sample_rate, record.get("analysis_intervals", "[]"))
        audio_input = waveform
    segments, info = _ASR_MODEL.transcribe(
        audio_input,
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


def _transcribe_mlx(record: dict[str, Any], configured_language: str, model_name: str) -> dict[str, Any]:
    import librosa
    import mlx_whisper
    import soundfile as sf

    language = _effective_asr_language(record, configured_language)
    _assert_role_filter_available(record)
    audio_input: Any = record["audio_path"]
    if str(record.get("analysis_intervals", "[]")) not in {"", "[]", "nan"}:
        waveform, sample_rate = sf.read(record["audio_path"], dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        waveform, _, _, _ = _analysis_audio(waveform, sample_rate, record.get("analysis_intervals", "[]"))
        audio_input = waveform
    options = {"path_or_hf_repo": model_name, "verbose": False, "word_timestamps": True}
    if language:
        options["language"] = language
    result = mlx_whisper.transcribe(audio_input, **options)
    segments = [
        {"start": float(item["start"]), "end": float(item["end"]), "text": str(item["text"]).strip()}
        for item in result.get("segments", [])
        if str(item.get("text", "")).strip()
    ]
    return {
        "case_id": record["case_id"],
        "subject_id": record["subject_id"],
        "language": result.get("language", language or record.get("language", "")),
        "language_probability": float(result.get("language_probability", float("nan"))),
        "text": str(result.get("text", "")).strip(),
        "segments_json": json.dumps(segments, ensure_ascii=False),
        "asr_model": model_name,
    }


def _asr_cache_key(record: dict[str, Any], backend: str, model_name: str, language: str) -> str:
    audio = Path(str(record["audio_path"]))
    stat = audio.stat()
    payload = "|".join(
        [
            str(audio.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(record.get("analysis_intervals", "[]")),
            backend,
            model_name,
            _effective_asr_language(record, language) or "auto",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_analysis_transcripts(
    manifest_path: Path,
    analysis_manifest_path: Path,
    recording_transcripts_path: Path,
    subject_transcripts_path: Path,
    agents_config: dict[str, Any],
    generate_missing: bool,
) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    backend = str(agents_config.get("asr_backend", "mlx_whisper"))
    model_name = str(agents_config["asr_model"])
    configured_language = str(agents_config.get("asr_language", "auto"))
    cache_dir = recording_transcripts_path.parent / "asr_cache"
    text_dir = recording_transcripts_path.parent / "asr_transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    generated_paths: dict[str, str] = {}
    transcript_rows: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []

    for record in manifest.to_dict("records"):
        transcript_path = "" if pd.isna(record.get("transcript_path")) else str(record.get("transcript_path", ""))
        text, _, _ = read_transcript(transcript_path)
        if text.strip():
            rows.append(
                {
                    "case_id": record["case_id"], "subject_id": record["subject_id"],
                    "language": record.get("language", ""), "language_probability": 1.0,
                    "text": text, "segments_json": "[]", "asr_model": "distributed_human_transcript",
                    "transcript_origin": "human",
                }
            )
        elif generate_missing:
            pending.append(record)

    faster_rows: list[dict[str, Any]] = []
    faster_payloads: list[tuple[dict[str, Any], str, str]] = []
    faster_cache_paths: list[Path] = []
    for index, record in enumerate(pending, start=1):
        try:
            cache_path = cache_dir / f"{_asr_cache_key(record, backend, model_name, configured_language)}.json"
            if cache_path.exists():
                result = json.loads(cache_path.read_text(encoding="utf-8"))
            elif backend == "mlx_whisper":
                result = _transcribe_mlx(record, configured_language, model_name)
                cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            elif backend == "faster_whisper":
                faster_payloads.append((record, configured_language, model_name))
                faster_cache_paths.append(cache_path)
                continue
            else:
                raise ValueError(f"Unsupported ASR backend: {backend}")
            result["transcript_origin"] = "generated_asr"
            rows.append(result)
        except Exception as exc:
            failures.append({"case_id": record["case_id"], "audio_path": record["audio_path"], "error_type": type(exc).__name__, "error": str(exc)})
        if index % 25 == 0:
            pd.DataFrame(rows).to_csv(recording_transcripts_path, index=False)

    if faster_payloads:
        with ProcessPoolExecutor(
            max_workers=int(agents_config.get("asr_workers", 2)),
            initializer=_init_asr,
            initargs=(model_name, int(agents_config.get("asr_cpu_threads", 4))),
        ) as executor:
            faster_rows = list(executor.map(_transcribe_recording, faster_payloads))
        for result, cache_path in zip(faster_rows, faster_cache_paths, strict=True):
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["transcript_origin"] = "generated_asr"
            rows.append(result)

    for result in rows:
        transcript_rows[str(result["case_id"])] = result
        if result["transcript_origin"] != "generated_asr":
            continue
        safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result["case_id"]))
        text_path = text_dir / f"{safe_case}.txt"
        text_path.write_text(str(result["text"]), encoding="utf-8")
        generated_paths[str(result["case_id"])] = str(text_path)

    analysis_manifest = manifest.copy()
    if "transcript_path" not in analysis_manifest.columns:
        analysis_manifest["transcript_path"] = ""
    analysis_manifest["transcript_path"] = analysis_manifest["transcript_path"].fillna("").astype(str)
    if "transcript_origin" not in analysis_manifest.columns:
        analysis_manifest["transcript_origin"] = "missing"
    else:
        analysis_manifest["transcript_origin"] = analysis_manifest["transcript_origin"].fillna("missing").astype(str)
    analysis_manifest["transcript_reliability"] = 0.0
    analysis_manifest["transcript_quality_tag"] = "missing"
    for row_index, record in analysis_manifest.iterrows():
        case_id = str(record["case_id"])
        if case_id in generated_paths:
            analysis_manifest.at[row_index, "transcript_path"] = generated_paths[case_id]
            analysis_manifest.at[row_index, "transcript_origin"] = "generated_asr"
            detected_language = str(transcript_rows[case_id].get("language", "")).strip()
            manifest_language = str(record.get("language", "")).strip().lower()
            # Distributed language metadata is part of the study protocol and is
            # more trustworthy than a single ASR guess. Only use ASR detection
            # when metadata is absent or explicitly describes a multilingual
            # recording whose per-recording language is unknown.
            metadata_is_unspecified = manifest_language in {
                "",
                "auto",
                "unknown",
                "unspecified",
                "nan",
                "zh-en",
                "multilingual",
            }
            if detected_language and metadata_is_unspecified:
                analysis_manifest.at[row_index, "language"] = detected_language
            probability = transcript_rows[case_id].get("language_probability")
            probability = float(probability) if probability is not None and pd.notna(probability) else 0.75
            analysis_manifest.at[row_index, "transcript_reliability"] = float(min(0.75, max(0.50, probability)))
            analysis_manifest.at[row_index, "transcript_quality_tag"] = "asr_unreviewed"
        elif case_id in transcript_rows and str(transcript_rows[case_id].get("text", "")).strip():
            analysis_manifest.at[row_index, "transcript_origin"] = "human"
            suffix = Path(str(record["transcript_path"])).suffix.lower()
            analysis_manifest.at[row_index, "transcript_reliability"] = 0.95 if suffix == ".cha" else 0.90
            analysis_manifest.at[row_index, "transcript_quality_tag"] = "distributed_human_transcript"
    analysis_manifest.to_csv(analysis_manifest_path, index=False)
    recording_columns = ["case_id", "subject_id", "language", "language_probability", "text", "segments_json", "asr_model", "transcript_origin"]
    recording = pd.DataFrame(rows, columns=recording_columns)
    recording.to_csv(recording_transcripts_path, index=False)
    subjects = []
    for subject_id, group in recording.groupby("subject_id", sort=True):
        parts = [f"[录音{index + 1}] {text}" for index, text in enumerate(group["text"].fillna(""))]
        subjects.append({"subject_id": str(subject_id), "transcript": "\n".join(parts), "recording_count": len(group), "asr_model": ";".join(sorted(set(group["asr_model"])))})
    pd.DataFrame(subjects, columns=["subject_id", "transcript", "recording_count", "asr_model"]).to_csv(subject_transcripts_path, index=False)
    pd.DataFrame(failures, columns=["case_id", "audio_path", "error_type", "error"]).to_csv(
        recording_transcripts_path.with_name("asr_failures.csv"), index=False
    )
    if generate_missing and pending:
        completed_ids = {
            str(item["case_id"])
            for item in rows
            if item.get("transcript_origin") == "generated_asr"
            and str(item.get("text", "")).strip()
        }
        success_fraction = len(completed_ids) / len(pending)
        minimum = float(agents_config.get("asr_min_success_fraction", 0.90))
        if success_fraction < minimum:
            raise RuntimeError(
                "ASR coverage below the configured minimum: "
                f"{len(completed_ids)}/{len(pending)}={success_fraction:.3f} < {minimum:.3f}. "
                "The run is stopped so missing transcripts cannot silently create a weaker Agent baseline."
            )
