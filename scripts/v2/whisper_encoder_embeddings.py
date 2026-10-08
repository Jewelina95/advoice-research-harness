#!/usr/bin/env python3
"""Frozen Whisper-large-v3 encoder embeddings (mlx), mean||std pooled per layer over voiced length."""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_whisper.audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram, pad_or_trim
from mlx_whisper.load_models import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--layers", default="8,16,24,32")
    ap.add_argument("--frames-layer", type=int, default=0, help="also save pooled frame sequence of this layer")
    ap.add_argument("--frame-pool", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=750)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    model = load_model(a.model, dtype=mx.float16)
    enc = model.encoder
    man = pd.read_csv(a.manifest).sort_values("subject_id")
    out: dict[int, list[np.ndarray]] = {k: [] for k in layers}
    ids = []
    seqs: list[np.ndarray] = []
    for i, row in enumerate(man.itertuples()):
        audio = load_audio(row.audio_path)
        chunks = [audio[s:s + N_SAMPLES] for s in range(0, max(len(audio), 1), N_SAMPLES)][:4]
        per_layer: dict[int, list[np.ndarray]] = {k: [] for k in layers}
        for chunk in chunks:
            n_frames = max(1, min(1500, int(np.ceil(len(chunk) / SAMPLE_RATE * 50))))
            mel = log_mel_spectrogram(pad_or_trim(chunk), n_mels=model.dims.n_mels)[None]
            import mlx.nn as nn
            x = nn.gelu(enc.conv1(mel))
            x = nn.gelu(enc.conv2(x))
            x = x + enc._positional_embedding
            for depth, block in enumerate(enc.blocks, start=1):
                x, _, _ = block(x)
                if depth in per_layer:
                    h = x if depth < len(enc.blocks) else enc.ln_post(x)
                    per_layer[depth].append(np.array(h[0, :n_frames].astype(mx.float32)))
        if a.frames_layer:
            fr = np.concatenate(per_layer[a.frames_layer], 0)
            n = len(fr) // a.frame_pool * a.frame_pool or len(fr)
            pooled = fr[:n].reshape(-1, min(a.frame_pool, n), fr.shape[1]).mean(1) if n >= a.frame_pool else fr
            seqs.append(pooled[: a.max_frames].astype(np.float16))
        for k in layers:
            frames = np.concatenate(per_layer[k], 0)
            out[k].append(np.concatenate([frames.mean(0), frames.std(0)]))
        ids.append(str(row.subject_id))
        if i % 200 == 0:
            print(i, flush=True)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for k in layers:
        np.savez_compressed(od / f"whisper_l{k}.npz", embeddings=np.stack(out[k]).astype(np.float32),
                            subject_ids=np.array(ids))
    if a.frames_layer:
        lengths = np.array([len(x) for x in seqs])
        padded = np.zeros((len(seqs), lengths.max(), seqs[0].shape[1]), np.float16)
        for i, x in enumerate(seqs):
            padded[i, : len(x)] = x
        np.save(od / f"whisper_frames_l{a.frames_layer}.npy", padded)
        np.savez(od / f"whisper_frames_l{a.frames_layer}_meta.npz", lengths=lengths, subject_ids=np.array(ids))
    print("done", len(ids))


if __name__ == "__main__":
    main()
