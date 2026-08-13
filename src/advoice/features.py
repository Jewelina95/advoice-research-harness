from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


EPS = 1e-10


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


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {f"{prefix}_mean": math.nan, f"{prefix}_std": math.nan}
    return {f"{prefix}_mean": float(finite.mean()), f"{prefix}_std": float(finite.std())}


def _segment_rows(y: np.ndarray, sr: int, case_id: str, seconds: float = 10.0) -> list[dict[str, Any]]:
    size = int(seconds * sr)
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(y), size)):
        segment = y[start : start + size]
        if len(segment) < sr:
            continue
        rms = librosa.feature.rms(y=segment, frame_length=400, hop_length=160)[0]
        db = librosa.amplitude_to_db(np.maximum(rms, EPS), ref=1.0)
        threshold = max(-50.0, float(np.percentile(db, 90) - 35.0))
        silent = db < threshold
        rows.append(
            {
                "case_id": case_id,
                "segment_id": f"{case_id}:S{index + 1:02d}",
                "start_sec": start / sr,
                "end_sec": min(len(y), start + size) / sr,
                "silence_fraction": float(silent.mean()),
                "rms_db_mean": float(db.mean()),
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
    y = y - float(y.mean())
    duration = len(y) / sr
    frame_length, hop_length = 400, 160
    frame_sec = hop_length / sr

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, EPS), ref=1.0)
    threshold = max(-50.0, float(np.percentile(rms_db, 90) - 35.0))
    silent = rms_db < threshold
    voiced = ~silent
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
    usable = voiced[: len(f0)] & np.isfinite(f0)
    valid_f0 = f0[usable]

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
        **{key: row[key] for key in ["dataset_id", "case_id", "subject_id", "label", "split", "sex"]},
        "duration_sec": float(duration),
        "clipping_fraction": clipping,
        "snr_proxy_db": snr_proxy,
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
    }
    for index in range(mfcc.shape[0]):
        feature[f"mfcc_{index + 1:02d}_mean"] = float(mfcc[index].mean())
        feature[f"mfcc_{index + 1:02d}_std"] = float(mfcc[index].std())
    return feature, _segment_rows(y, sr, row["case_id"])


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
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_audio_file, row): row["case_id"] for row in records}
        for future in as_completed(futures):
            feature, trace = future.result()
            features.append(feature)
            segments.extend(trace)
    recording = pd.DataFrame(features).sort_values(["split", "subject_id", "case_id"])
    recording.to_csv(recording_features_path, index=False)
    pd.DataFrame(segments).sort_values(["case_id", "start_sec"]).to_csv(segments_path, index=False)

    identifier = ["dataset_id", "subject_id", "label", "split", "sex"]
    numeric = [column for column in recording.select_dtypes(include=[np.number]).columns]
    subject = recording.groupby(identifier, as_index=False, dropna=False)[numeric].mean()
    counts = recording.groupby(identifier, as_index=False, dropna=False).size().rename(columns={"size": "recording_count"})
    subject = subject.merge(counts, on=identifier, how="left")
    subject["total_recorded_duration_sec"] = subject["duration_sec"] * subject["recording_count"]
    subject.to_csv(subject_features_path, index=False)

