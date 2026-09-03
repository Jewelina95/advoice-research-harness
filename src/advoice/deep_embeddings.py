from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _fingerprint(
    texts: list[str],
    subject_ids: list[str],
    config: dict[str, Any],
) -> str:
    payload = {
        "model": config["model"],
        "revision": config["revision"],
        "code_revision": config.get("code_revision"),
        "text_prefix": str(config.get("text_prefix", "")),
        "max_length": int(config.get("max_length", 1024)),
        "subjects": subject_ids,
        "text_sha256": [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def encode_multilingual_text(
    texts: list[str],
    subject_ids: list[str],
    cache_path: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create frozen multilingual text embeddings; no outcome labels enter here."""
    fingerprint = _fingerprint(texts, subject_ids, config)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_fingerprint = str(cached["fingerprint"].item())
        cached_embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        if cached_fingerprint == fingerprint and np.isfinite(cached_embeddings).all():
            return cached_embeddings, {
                "cache_hit": True,
                "fingerprint": fingerprint,
                "model": config["model"],
                "revision": config["revision"],
            }

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    model_name = str(config["model"])
    revision = str(config["revision"])
    trust_remote_code = bool(config.get("trust_remote_code", False))
    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = requested_device
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        code_revision=str(config.get("code_revision", revision)),
        trust_remote_code=trust_remote_code,
        dtype=torch.float32 if device in {"cpu", "mps"} else "auto",
    )
    model = model.eval().to(device)
    batch_size = int(config.get("batch_size", 4))
    max_length = int(config.get("max_length", 1024))
    text_prefix = str(config.get("text_prefix", ""))
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            [text_prefix + text for text in texts[start : start + batch_size]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if bool(config.get("explicit_position_ids", False)):
            sequence_length = int(encoded["input_ids"].shape[1])
            encoded["position_ids"] = torch.arange(
                sequence_length, device=device, dtype=torch.long
            ).unsqueeze(0).expand(encoded["input_ids"].shape[0], -1)
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state[:, 0]
            output = functional.normalize(output, p=2, dim=1)
        batches.append(output.detach().cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(batches, axis=0)
    if not np.isfinite(embeddings).all():
        invalid = int((~np.isfinite(embeddings)).sum())
        raise ValueError(
            f"Text encoder produced {invalid} non-finite values; cache was not written."
        )
    del model
    if device == "mps":
        torch.mps.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        fingerprint=np.asarray(fingerprint),
    )
    return embeddings, {
        "cache_hit": False,
        "fingerprint": fingerprint,
        "model": model_name,
        "revision": revision,
        "code_revision": str(config.get("code_revision", revision)),
        "device": device,
        "max_length": max_length,
        "text_prefix": text_prefix,
        "embedding_dimension": int(embeddings.shape[1]),
    }
