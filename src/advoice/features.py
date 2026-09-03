from __future__ import annotations

import json
import math
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

try:
    import webrtcvad
except ImportError:  # pragma: no cover - deterministic energy fallback remains available
    webrtcvad = None

from .transcripts import transcript_metrics
from .utils import hash_values


EPS = 1e-10
FEATURE_EXTRACTOR_SIGNATURE = hash_values(
    [
        Path(__file__),
        Path(__file__).with_name("transcripts.py"),
        f"librosa={librosa.__version__}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"soundfile={sf.__version__}",
    ]
)[:16]
IDENTITY_COLUMNS = [
    "dataset_id",
    "case_id",
    "subject_id",
    "label",
    "split",
    "sex",
    "task_type",
    "language",
    "channel",
]

TEXT_METRIC_COLUMNS = [
    "word_count",
    "speech_rate_wpm",
    "lexical_ttr",
    "lexical_mattr50",
    "filler_rate_100w",
    "repair_rate_100w",
    "pronoun_ratio",
    "content_word_ratio",
    "mean_utterance_words",
    "patient_turn_count",
    "interviewer_turn_count",
    "patient_turn_share",
    "transcript_available",
    "picture_content_unit_coverage",
    "picture_information_density",
    "picture_content_redundancy",
    "picture_uncertainty_rate_100w",
]


def _runs(mask: np.ndarray, frame_sec: float) -> list[float]:
    values: list[float] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            values.append((index - start) * frame_sec)
            start = None
    if start is not None:
        values.append((len(mask) - start) * frame_sec)
    return values


def _voice_activity_mask(
    y: np.ndarray,
    sr: int,
    hop_length: int = 160,
) -> tuple[np.ndarray, str]:
    """Estimate speech activity on a 10 ms grid with a noise-aware fallback.

    WebRTC VAD prevents stationary recording noise from being counted as speech.
    The adaptive energy condition rejects VAD positives at the recording noise floor.
    A short hangover preserves weak syllable boundaries without closing clinical pauses.
    """

    if sr != 16000:
        raise ValueError("Voice activity extraction requires 16 kHz audio.")
    frame_samples = int(0.03 * sr)
    frame_count = len(y) // frame_samples
    if frame_count == 0:
        return np.zeros(max(1, int(np.ceil(len(y) / hop_length))), dtype=bool), "energy"

    usable = np.asarray(y[: frame_count * frame_samples], dtype=np.float32)
    peak = max(float(np.max(np.abs(usable))), EPS)
    normalized = np.clip(usable / peak, -1.0, 1.0)
    frames = normalized.reshape(frame_count, frame_samples)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, EPS))
    energy_threshold = max(
        -52.0,
        float(np.percentile(frame_db, 20) + 6.0),
        float(np.percentile(frame_db, 90) - 28.0),
    )
    energy_speech = frame_db >= energy_threshold

    backend = "energy"
    coarse_speech = energy_speech
    if webrtcvad is not None:
        detector = webrtcvad.Vad(2)
        pcm = (normalized * 32767.0).astype("<i2").tobytes()
        bytes_per_frame = frame_samples * 2
        vad_speech = np.asarray(
            [
                detector.is_speech(
                    pcm[index * bytes_per_frame : (index + 1) * bytes_per_frame],
                    sr,
                )
                for index in range(frame_count)
            ],
            dtype=bool,
        )
        coarse_speech = vad_speech & energy_speech
        backend = "webrtc_energy_hybrid"

    # Ninety milliseconds of hangover keeps weak syllables attached while leaving
    # pauses of at least 0.5 s, the clinical threshold used downstream, intact.
    coarse_speech = (
        np.convolve(coarse_speech.astype(np.int8), np.ones(3, dtype=np.int8), mode="same")
        >= 1
    )
    centers = (np.arange(frame_count) * frame_samples + frame_samples / 2.0) / sr
    target_count = 1 + max(0, (len(y) - 400) // hop_length)
    target_centers = (np.arange(target_count) * hop_length + 200.0) / sr
    nearest = np.clip(
        np.searchsorted(centers, target_centers, side="left"), 0, frame_count - 1
    )
    previous = np.maximum(nearest - 1, 0)
    choose_previous = np.abs(target_centers - centers[previous]) < np.abs(
        target_centers - centers[nearest]
    )
    nearest[choose_previous] = previous[choose_previous]
    return coarse_speech[nearest], backend


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {f"{prefix}_mean": math.nan, f"{prefix}_std": math.nan}
    return {f"{prefix}_mean": float(finite.mean()), f"{prefix}_std": float(finite.std())}


def _analysis_audio(
    y: np.ndarray,
    sr: int,
    raw_intervals: Any,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]], float, bool]:
    try:
        intervals = json.loads(raw_intervals) if isinstance(raw_intervals, str) else raw_intervals
    except (TypeError, json.JSONDecodeError):
        intervals = []
    clips: list[np.ndarray] = []
    mapping: list[tuple[float, float, float, float]] = []
    cursor = 0.0
    for interval in intervals or []:
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            continue
        start = max(0.0, float(interval[0]))
        end = min(len(y) / sr, float(interval[1]))
        if end <= start:
            continue
        clip = y[int(start * sr) : int(end * sr)]
        if not len(clip):
            continue
        clip_duration = len(clip) / sr
        clips.append(clip)
        mapping.append((cursor, cursor + clip_duration, start, end))
        cursor += clip_duration
    if not clips:
        duration = len(y) / sr
        return y, [(0.0, duration, 0.0, duration)], 1.0, False
    cropped = np.concatenate(clips)
    return cropped, mapping, float(len(cropped) / max(len(y), 1)), True


def _source_spans(
    start: float,
    end: float,
    mapping: list[tuple[float, float, float, float]],
) -> list[list[float]]:
    spans: list[list[float]] = []
    for concat_start, concat_end, source_start, _ in mapping:
        overlap_start = max(start, concat_start)
        overlap_end = min(end, concat_end)
        if overlap_end > overlap_start:
            spans.append(
                [
                    source_start + overlap_start - concat_start,
                    source_start + overlap_end - concat_start,
                ]
            )
    return spans


def _segment_rows(
    y: np.ndarray,
    sr: int,
    case_id: str,
    mapping: list[tuple[float, float, float, float]],
    seconds: float = 10.0,
) -> list[dict[str, Any]]:
    size = int(seconds * sr)
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(y), size)):
        segment = y[start : start + size]
        if len(segment) < sr:
            continue
        rms = librosa.feature.rms(y=segment, frame_length=400, hop_length=160)[0]
        db = librosa.amplitude_to_db(np.maximum(rms, EPS), ref=1.0)
        voiced, vad_backend = _voice_activity_mask(segment, sr)
        silent = ~voiced
        db = db[: len(voiced)]
        transition_count = int(np.count_nonzero(np.diff(voiced.astype(np.int8))))
        segment_duration = max(len(segment) / sr, EPS)
        voiced_runs = np.split(voiced, np.flatnonzero(np.diff(voiced.astype(np.int8))) + 1)
        voiced_run_lengths = [len(run) for run in voiced_runs if len(run) and bool(run[0])]
        frame_seconds = 160.0 / sr
        rows.append(
            {
                "case_id": case_id,
                "segment_id": f"{case_id}:S{index + 1:02d}",
                "start_sec": start / sr,
                "end_sec": min(len(y), start + size) / sr,
                "source_spans": json.dumps(_source_spans(start / sr, min(len(y), start + size) / sr, mapping)),
                "silence_fraction": float(silent.mean()),
                "voiced_fraction": float(voiced.mean()),
                "activity_transition_rate_hz": float(transition_count / segment_duration),
                "voiced_run_mean_sec": float(
                    np.mean(voiced_run_lengths) * frame_seconds
                    if voiced_run_lengths
                    else 0.0
                ),
                "rms_db_mean": float(db.mean()),
                "vad_backend": vad_backend,
                "evidence_type": "fixed_window_trace_from_long_audio",
            }
        )
    return rows


def extract_audio_file(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(row["audio_path"])
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    y = np.nan_to_num(y.astype(np.float32))
    original_duration = len(y) / sr
    y, source_mapping, role_coverage, role_filtered = _analysis_audio(y, sr, row.get("analysis_intervals", "[]"))
    if bool(row.get("role_filter_required", False)) and not role_filtered:
        raise ValueError(
            "Patient-only acoustic analysis was required, but no valid patient time intervals were available."
        )
    y = y - float(y.mean())
    duration = len(y) / sr
    frame_length, hop_length = 400, 160
    frame_sec = hop_length / sr

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, EPS), ref=1.0)
    voiced, vad_backend = _voice_activity_mask(y, sr, hop_length=hop_length)
    rms_db = rms_db[: len(voiced)]
    silent = ~voiced
    pause_runs = _runs(silent, frame_sec)
    speech_runs = _runs(voiced, frame_sec)
    long_pauses = [value for value in pause_runs if value >= 0.5]

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=512, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=hop_length)

    normalized = y / max(float(np.max(np.abs(y))), EPS)
    f0 = librosa.yin(normalized, fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=hop_length)
    # librosa's centered RMS and YIN grids can differ by a few boundary frames.
    # Align explicitly before masking so short recordings remain extractable.
    f0_frame_count = min(len(voiced), len(f0))
    usable = voiced[:f0_frame_count] & np.isfinite(f0[:f0_frame_count])
    valid_f0 = f0[:f0_frame_count][usable]

    clipping = float((np.abs(y) >= 0.999).mean())
    speech_rms = rms_db[voiced]
    noise_rms = rms_db[silent]
    snr_proxy = float(np.median(speech_rms) - np.median(noise_rms)) if len(noise_rms) else 30.0
    speech_mean = float(np.mean(speech_runs)) if speech_runs else 0.0
    speech_std = float(np.std(speech_runs)) if speech_runs else 0.0
    reliability = float(
        np.clip(
            0.25
            + 0.25 * min(duration / 45.0, 1.0)
            + 0.25 * np.clip(snr_proxy / 25.0, 0.0, 1.0)
            + 0.15 * np.clip(float(voiced.mean()) / 0.5, 0.0, 1.0)
            + 0.10 * (1.0 - min(clipping / 0.01, 1.0)),
            0.0,
            1.0,
        )
    )
    feature: dict[str, Any] = {
        **{
            key: row.get(key)
            for key in ["dataset_id", "case_id", "subject_id", "label", "split", "sex", "age", "task_type", "language", "channel"]
        },
        "duration_sec": float(duration),
        "original_duration_sec": float(original_duration),
        "role_filtered_audio": float(role_filtered),
        "role_coverage_fraction": role_coverage,
        "clipping_fraction": clipping,
        "snr_proxy_db": snr_proxy,
        "vad_backend": vad_backend,
        "silence_fraction": float(silent.mean()),
        "voiced_fraction": float(voiced.mean()),
        "long_pause_rate_min": float(len(long_pauses) / max(duration / 60.0, EPS)),
        "pause_mean_sec": float(np.mean(pause_runs)) if pause_runs else 0.0,
        "pause_p90_sec": float(np.percentile(pause_runs, 90)) if pause_runs else 0.0,
        "speech_run_mean_sec": speech_mean,
        "speech_run_rate_min": float(len(speech_runs) / max(duration / 60.0, EPS)),
        "speech_run_cv": float(speech_std / max(speech_mean, EPS)),
        "rms_db_mean": float(rms_db.mean()),
        "rms_db_std": float(rms_db.std()),
        "f0_median_hz": float(np.median(valid_f0)) if len(valid_f0) else math.nan,
        "f0_iqr_hz": float(np.subtract(*np.percentile(valid_f0, [75, 25]))) if len(valid_f0) else math.nan,
        "f0_valid_fraction": float(usable.mean()) if len(usable) else 0.0,
        "zcr_mean": float(zcr.mean()),
        "spectral_centroid_mean": float(centroid.mean()),
        "spectral_bandwidth_mean": float(bandwidth.mean()),
        "spectral_rolloff_mean": float(rolloff.mean()),
        "spectral_flatness_mean": float(flatness.mean()),
        "audio_reliability": reliability,
        "text_reliability": float(row.get("transcript_reliability", 0.0))
        if pd.notna(row.get("transcript_reliability"))
        else 0.0,
    }
    feature.update(
        transcript_metrics(
            str(row.get("transcript_path", "")) if pd.notna(row.get("transcript_path", "")) else "",
            str(row.get("language", "en")),
            duration,
            str(row.get("task_type", "")),
        )
    )
    for index in range(mfcc.shape[0]):
        feature[f"mfcc_{index + 1:02d}_mean"] = float(mfcc[index].mean())
        feature[f"mfcc_{index + 1:02d}_std"] = float(mfcc[index].std())
    return feature, _segment_rows(y, sr, row["case_id"], source_mapping)


def _safe_extract_audio_file(
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    try:
        feature, trace = extract_audio_file(row)
        return feature, trace, None
    except Exception as exc:
        return None, [], {
            "dataset_id": row.get("dataset_id", ""),
            "case_id": row.get("case_id", ""),
            "subject_id": row.get("subject_id", ""),
            "audio_path": row.get("audio_path", ""),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _recording_cache_path(cache_dir: Path, row: dict[str, Any]) -> Path:
    transcript = Path(str(row.get("transcript_path", "")))
    transcript_input: Any = transcript if transcript.exists() and transcript.is_file() else ""
    key = hash_values(
        [
            FEATURE_EXTRACTOR_SIGNATURE,
            row.get("audio_sha256", ""),
            row.get("analysis_intervals", "[]"),
            transcript_input,
            row.get("language", ""),
            row.get("transcript_reliability", 0.0),
        ]
    )
    return cache_dir / f"{key}.pkl"


def _rebind_cached_result(
    feature: dict[str, Any],
    trace: list[dict[str, Any]],
    row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rebound_feature = dict(feature)
    for column in IDENTITY_COLUMNS:
        if column in row:
            rebound_feature[column] = row[column]
    if "age" in row:
        rebound_feature["age"] = row["age"]
    rebound_trace = []
    for index, segment in enumerate(trace, start=1):
        rebound = dict(segment)
        rebound["case_id"] = row["case_id"]
        rebound["segment_id"] = f"{row['case_id']}:S{index:02d}"
        rebound_trace.append(rebound)
    return rebound_feature, rebound_trace


def _extract_and_cache(
    row: dict[str, Any],
    cache_path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    feature, trace, failure = _safe_extract_audio_file(row)
    if failure is not None or feature is None:
        return feature, trace, failure
    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"feature": feature, "trace": trace}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    return feature, trace, None


def extract_features(
    manifest_path: Path,
    recording_features_path: Path,
    subject_features_path: Path,
    segments_path: Path,
    workers: int = 4,
) -> None:
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    records = manifest.to_dict("records")
    features: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cache_dir = recording_features_path.parent / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[dict[str, Any], Path]] = []
    for row in records:
        cache_path = _recording_cache_path(cache_dir, row)
        if not cache_path.exists():
            pending.append((row, cache_path))
            continue
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            feature, trace = _rebind_cached_result(cached["feature"], cached["trace"], row)
            features.append(feature)
            segments.extend(trace)
        except (OSError, EOFError, KeyError, pickle.UnpicklingError):
            cache_path.unlink(missing_ok=True)
            pending.append((row, cache_path))

    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_extract_and_cache, row, cache_path): row["case_id"]
                for row, cache_path in pending
            }
            for future in as_completed(futures):
                feature, trace, failure = future.result()
                if failure is not None:
                    failures.append(failure)
                    continue
                if feature is None:
                    continue
                features.append(feature)
                segments.extend(trace)
    failure_path = recording_features_path.with_name("feature_extraction_failures.csv")
    failure_columns = ["dataset_id", "case_id", "subject_id", "audio_path", "error_type", "error"]
    pd.DataFrame(failures, columns=failure_columns).to_csv(failure_path, index=False)
    if not features:
        raise RuntimeError(f"Feature extraction failed for all {len(records)} recordings; see {failure_path}.")
    recording = pd.DataFrame(features).sort_values(["split", "subject_id", "case_id"])
    recording.to_csv(recording_features_path, index=False)
    pd.DataFrame(segments).sort_values(["case_id", "start_sec"]).to_csv(segments_path, index=False)

    aggregate_subject_features(recording, subject_features_path)


def refresh_recording_text_metrics(
    recording_features_path: Path,
    analysis_manifest_path: Path,
) -> dict[str, Any]:
    """Refresh language-sensitive transcript metrics without re-extracting audio."""
    recording = pd.read_csv(recording_features_path, dtype={"subject_id": str, "case_id": str})
    manifest = pd.read_csv(analysis_manifest_path, dtype={"subject_id": str, "case_id": str})
    metadata_columns = [
        column
        for column in ["case_id", "task_type", "language", "channel", "transcript_path"]
        if column in manifest.columns
    ]
    metadata = manifest[metadata_columns].drop_duplicates("case_id", keep="last")
    recording = recording.drop(
        columns=[column for column in ["task_type", "language", "channel", "transcript_path"] if column in recording],
        errors="ignore",
    ).merge(metadata, on="case_id", how="left", validate="one_to_one")
    refreshed = []
    for row in recording.to_dict("records"):
        metrics = transcript_metrics(
            str(row.get("transcript_path", "")),
            str(row.get("language", "unknown")),
            float(row.get("duration_sec", 0.0) or 0.0),
            str(row.get("task_type", "")),
        )
        refreshed.append(metrics)
    refreshed_frame = pd.DataFrame(refreshed, index=recording.index)
    for column in TEXT_METRIC_COLUMNS:
        if column in refreshed_frame:
            recording[column] = refreshed_frame[column]
    recording.to_csv(recording_features_path, index=False)
    return {
        "recordings": int(len(recording)),
        "languages": {
            str(key): int(value)
            for key, value in recording["language"].fillna("unknown").astype(str).value_counts().items()
        },
        "text_metrics_refreshed": [column for column in TEXT_METRIC_COLUMNS if column in recording],
    }


def aggregate_subject_features(recording: pd.DataFrame, subject_features_path: Path) -> None:
    """Aggregate recording features while preserving task and multilingual evidence scopes."""
    identifier = ["dataset_id", "subject_id", "label", "split", "sex"]
    numeric = [column for column in recording.select_dtypes(include=[np.number]).columns]
    subject = recording.groupby(identifier, as_index=False, dropna=False)[numeric].mean()
    counts = recording.groupby(identifier, as_index=False, dropna=False).size().rename(columns={"size": "recording_count"})
    subject = subject.merge(counts, on=identifier, how="left")
    def _scope_value(values: pd.Series) -> str:
        unique = sorted({str(value) for value in values.dropna() if str(value)})
        return unique[0] if len(unique) == 1 else "multilingual" if unique else "unknown"

    subject_metadata = recording.groupby(identifier, as_index=False, dropna=False).agg(
        language=("language", _scope_value), channel=("channel", "first")
    )
    subject = subject.merge(subject_metadata, on=identifier, how="left", validate="one_to_one")
    subject["total_recorded_duration_sec"] = subject["duration_sec"] * subject["recording_count"]

    task_numeric = [column for column in numeric if column not in {"recording_index"}]
    task_frame = recording.groupby(identifier + ["task_type"], as_index=False, dropna=False)[task_numeric].mean()
    task_frame["task_type"] = task_frame["task_type"].map(
        lambda value: re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    )
    for task_name, group in task_frame.groupby("task_type"):
        task_values = group.drop(columns=["task_type"]).rename(
            columns={column: f"task_{task_name}__{column}" for column in task_numeric}
        )
        subject = subject.merge(task_values, on=identifier, how="left")

    # Bilingual subjects need language-specific evidence. Pooled lexical metrics
    # are retained for context but are not calibrated as if they belonged to one language.
    language_counts = recording.groupby(identifier, dropna=False)["language"].nunique()
    if bool((language_counts > 1).any()):
        language_frame = recording.groupby(
            identifier + ["language"], as_index=False, dropna=False
        )[task_numeric].mean()
        language_frame["language"] = language_frame["language"].map(
            lambda value: re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
        )
        for language_name, group in language_frame.groupby("language"):
            language_values = group.drop(columns=["language"]).rename(
                columns={
                    column: f"task_language_{language_name}__{column}"
                    for column in task_numeric
                }
            )
            subject = subject.merge(language_values, on=identifier, how="left")
    subject.to_csv(subject_features_path, index=False)
