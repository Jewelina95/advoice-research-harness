from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfiltfilt


def _parse_intervals(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return []
    intervals: list[tuple[float, float]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("speaker_role", ""))).lower()
        if role and role not in {"patient", "participant", "subject", "par"}:
            continue
        try:
            start = float(item.get("start_sec", item.get("start", 0.0)))
            end = float(item.get("end_sec", item.get("end", 0.0)))
        except (TypeError, ValueError):
            continue
        if end > start >= 0.0:
            intervals.append((start, end))
    return intervals


def _load_patient_audio(row: pd.Series, target_sr: int, lowpass_hz: float) -> np.ndarray:
    path = Path(str(row["audio_path"]))
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    intervals = _parse_intervals(row.get("analysis_intervals", "[]"))
    if intervals:
        pieces = [
            mono[max(0, int(start * sample_rate)) : min(len(mono), int(end * sample_rate))]
            for start, end in intervals
        ]
        pieces = [piece for piece in pieces if len(piece)]
        if pieces:
            mono = np.concatenate(pieces)
    nyquist = float(sample_rate) / 2.0
    if 0.0 < lowpass_hz < nyquist and len(mono) > 64:
        sos = butter(6, lowpass_hz / nyquist, btype="lowpass", output="sos")
        try:
            mono = sosfiltfilt(sos, mono).astype(np.float32)
        except ValueError:
            mono = sosfilt(sos, mono).astype(np.float32)
    if int(sample_rate) != int(target_sr):
        mono = librosa.resample(
            mono, orig_sr=int(sample_rate), target_sr=int(target_sr), res_type="soxr_hq"
        ).astype(np.float32)
    return mono


def _windows(audio: np.ndarray, sample_rate: int, seconds: float, overlap: float) -> list[np.ndarray]:
    window = max(1, int(round(sample_rate * seconds)))
    hop = max(1, int(round(window * (1.0 - overlap))))
    if len(audio) <= window:
        return [audio]
    starts = list(range(0, max(len(audio) - window + 1, 1), hop))
    if starts[-1] + window < len(audio):
        starts.append(len(audio) - window)
    return [audio[start : min(start + window, len(audio))] for start in starts]


def _fingerprint(
    manifest: pd.DataFrame,
    subject_ids: list[str],
    config: dict[str, Any],
) -> str:
    records = manifest[manifest["subject_id"].astype(str).isin(subject_ids)].copy()
    records = records.sort_values(["subject_id", "audio_path"])
    files = []
    for row in records.itertuples(index=False):
        path = Path(str(row.audio_path))
        files.append(
            {
                "subject_id": str(row.subject_id),
                "audio_path": str(path),
                "audio_sha256": str(getattr(row, "audio_sha256", "")),
                "size": path.stat().st_size if path.exists() else -1,
                "mtime_ns": path.stat().st_mtime_ns if path.exists() else -1,
                "analysis_intervals": str(getattr(row, "analysis_intervals", "[]")),
            }
        )
    payload = {
        "representation_schema": "mhubert-window-token-sequence-v4",
        "model": config["model"],
        "revision": config["revision"],
        "sample_rate": int(config.get("sample_rate", 16000)),
        "lowpass_hz": float(config.get("lowpass_hz", 8000.0)),
        "window_seconds": float(config.get("window_seconds", 5.0)),
        "overlap": float(config.get("overlap", 0.25)),
        "max_subject_seconds": float(config.get("max_subject_seconds", 30.0)),
        "temporal_bins_per_window": int(config.get("temporal_bins_per_window", 1)),
        "token_pooling": str(config.get("token_pooling", "temporal_bins")),
        "subjects": subject_ids,
        "files": files,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def encode_multilingual_audio(
    manifest_path: Path,
    subject_ids: list[str],
    cache_path: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return label-free frozen mHuBERT window embeddings in subject order."""
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    fingerprint = _fingerprint(manifest, subject_ids, config)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        required = {"embeddings", "window_embeddings", "window_offsets", "subject_ids"}
        if str(cached["fingerprint"].item()) == fingerprint and required.issubset(cached.files):
            return np.asarray(cached["embeddings"], dtype=np.float32), {
                "enabled": True,
                "cache_hit": True,
                "fingerprint": fingerprint,
                "model": config["model"],
                "revision": config["revision"],
                "subject_coverage": float(np.isfinite(cached["embeddings"]).any(axis=1).mean()),
                "window_count": int(len(cached["window_embeddings"])),
            }

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    model_name = str(config["model"])
    revision = str(config["revision"])
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision)
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = requested_device
    model = model.eval().to(device)

    sample_rate = int(config.get("sample_rate", 16000))
    lowpass_hz = float(config.get("lowpass_hz", 8000.0))
    window_seconds = float(config.get("window_seconds", 5.0))
    overlap = float(config.get("overlap", 0.25))
    maximum_samples = int(round(float(config.get("max_subject_seconds", 30.0)) * sample_rate))
    batch_size = int(config.get("batch_size", 8))
    grouped = {
        subject_id: group.sort_values("audio_path")
        for subject_id, group in manifest.groupby(manifest["subject_id"].astype(str))
    }
    pending_audio: list[np.ndarray] = []
    pending_subject: list[str] = []
    pooled: dict[str, list[np.ndarray]] = defaultdict(list)
    temporal_pooled: dict[str, list[np.ndarray]] = defaultdict(list)
    failures: list[dict[str, str]] = []

    def flush() -> None:
        if not pending_audio:
            return
        encoded = feature_extractor(
            pending_audio,
            sampling_rate=sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = encoded["input_values"].to(device)
        attention_mask = encoded.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        temporal_bins = max(1, int(config.get("temporal_bins_per_window", 1)))
        token_pooling = str(config.get("token_pooling", "temporal_bins"))
        with torch.inference_mode():
            hidden = model(input_values=input_values, attention_mask=attention_mask).last_hidden_state
            if attention_mask is not None and hasattr(model, "_get_feature_vector_attention_mask"):
                hidden_mask = model._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                ).to(hidden.device)
                denominator = hidden_mask.sum(dim=1, keepdim=True).clamp_min(1)
                vectors = (hidden * hidden_mask.unsqueeze(-1)).sum(dim=1) / denominator
            else:
                hidden_mask = torch.ones(
                    hidden.shape[:2], dtype=torch.bool, device=hidden.device
                )
                vectors = hidden.mean(dim=1)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            temporal_tokens: list[np.ndarray] = []
            for index in range(len(hidden)):
                valid_hidden = hidden[index][hidden_mask[index]]
                if token_pooling == "window_stats":
                    tokens = torch.cat(
                        [
                            valid_hidden.mean(dim=0),
                            valid_hidden.std(dim=0, unbiased=False),
                        ]
                    ).unsqueeze(0)
                elif token_pooling == "temporal_bins":
                    chunks = [
                        chunk
                        for chunk in torch.tensor_split(valid_hidden, temporal_bins, dim=0)
                        if len(chunk)
                    ]
                    tokens = torch.stack([chunk.mean(dim=0) for chunk in chunks])
                else:
                    raise ValueError(f"Unsupported token_pooling={token_pooling!r}")
                tokens = torch.nn.functional.normalize(tokens, p=2, dim=1)
                temporal_tokens.append(tokens.detach().cpu().numpy().astype(np.float32))
        for subject_id, vector, tokens in zip(
            pending_subject,
            vectors.detach().cpu().numpy().astype(np.float32),
            temporal_tokens,
            strict=True,
        ):
            pooled[subject_id].append(vector)
            temporal_pooled[subject_id].append(tokens)
        pending_audio.clear()
        pending_subject.clear()

    for subject_id in subject_ids:
        remaining = maximum_samples
        for _, row in grouped.get(subject_id, pd.DataFrame()).iterrows():
            if remaining <= 0:
                break
            try:
                audio = _load_patient_audio(row, sample_rate, lowpass_hz)[:remaining]
            except Exception as error:  # preserve a structured missing embedding instead of aborting a cohort
                failures.append(
                    {"subject_id": subject_id, "audio_path": str(row.get("audio_path", "")), "error": repr(error)}
                )
                continue
            remaining -= len(audio)
            if not len(audio):
                continue
            for window in _windows(audio, sample_rate, window_seconds, overlap):
                pending_audio.append(window)
                pending_subject.append(subject_id)
                if len(pending_audio) >= batch_size:
                    flush()
        flush()

    hidden_size = int(getattr(model.config, "hidden_size", 768))
    token_dimension = (
        hidden_size * 2
        if str(config.get("token_pooling", "temporal_bins")) == "window_stats"
        else hidden_size
    )
    embeddings = np.full((len(subject_ids), hidden_size * 2), np.nan, dtype=np.float32)
    window_counts: dict[str, int] = {}
    ordered_windows: list[np.ndarray] = []
    window_offsets = [0]
    for index, subject_id in enumerate(subject_ids):
        values = pooled.get(subject_id, [])
        window_counts[subject_id] = len(values)
        if not values:
            window_offsets.append(window_offsets[-1])
            continue
        matrix = np.vstack(values)
        embeddings[index] = np.concatenate([matrix.mean(axis=0), matrix.std(axis=0)])
        token_matrix = np.vstack(temporal_pooled[subject_id]).astype(np.float32)
        ordered_windows.append(token_matrix)
        window_offsets.append(window_offsets[-1] + len(token_matrix))

    window_embeddings = (
        np.vstack(ordered_windows)
        if ordered_windows
        else np.empty((0, token_dimension), dtype=np.float32)
    )

    del model
    if device == "mps":
        torch.mps.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        window_embeddings=window_embeddings,
        window_offsets=np.asarray(window_offsets, dtype=np.int64),
        subject_ids=np.asarray(subject_ids),
        fingerprint=np.asarray(fingerprint),
    )
    metadata = {
        "enabled": True,
        "cache_hit": False,
        "fingerprint": fingerprint,
        "model": model_name,
        "revision": revision,
        "device": device,
        "embedding_dimension": int(embeddings.shape[1]),
        "subject_coverage": float(np.isfinite(embeddings).any(axis=1).mean()),
        "window_count": int(sum(window_counts.values())),
        "sequence_token_count": int(len(window_embeddings)),
        "temporal_bins_per_window": int(config.get("temporal_bins_per_window", 1)),
        "token_pooling": str(config.get("token_pooling", "temporal_bins")),
        "window_seconds": window_seconds,
        "overlap": overlap,
        "max_subject_seconds": float(config.get("max_subject_seconds", 30.0)),
        "lowpass_hz": lowpass_hz,
        "failures": failures[:100],
    }
    return embeddings, metadata


def load_audio_window_sequences(
    cache_path: Path,
    subject_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load padded per-subject mHuBERT window sequences and a validity mask."""
    cached = np.load(cache_path, allow_pickle=False)
    cached_subjects = [str(value) for value in cached["subject_ids"].tolist()]
    if cached_subjects != [str(value) for value in subject_ids]:
        raise ValueError("Audio embedding cache subject order does not match model input.")
    windows = np.asarray(cached["window_embeddings"], dtype=np.float32)
    offsets = np.asarray(cached["window_offsets"], dtype=np.int64)
    maximum = max((int(offsets[i + 1] - offsets[i]) for i in range(len(subject_ids))), default=0)
    dimension = int(windows.shape[1]) if windows.ndim == 2 else 0
    padded = np.zeros((len(subject_ids), maximum, dimension), dtype=np.float32)
    mask = np.zeros((len(subject_ids), maximum), dtype=bool)
    for index in range(len(subject_ids)):
        start, end = int(offsets[index]), int(offsets[index + 1])
        length = end - start
        if length:
            padded[index, :length] = windows[start:end]
            mask[index, :length] = True
    return padded, mask
