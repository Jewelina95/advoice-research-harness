#!/usr/bin/env python3
"""Attention-pooling head over frozen Whisper frame sequences, cross-fitted on train only.

Writes OOF train probabilities and fold-averaged test probabilities for use as a stacking branch.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

LABELS = ["HC", "MCI", "AD"]


class Head(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, drop: float = 0.3) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop), nn.Linear(dim, hidden), nn.GELU())
        self.score = nn.Linear(hidden, 1)
        self.out = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, 3))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        w = self.score(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        attn = (w.unsqueeze(-1) * h).sum(1)
        mean = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.out(torch.cat([attn, mean], -1))


def fit_predict(x, lengths, y, tr, pred_sets, device, seed, epochs, lr):
    torch.manual_seed(seed)
    model = Head(x.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    idx = np.array(tr)
    ar = np.arange(x.shape[1])
    for _ in range(epochs):
        model.train()
        np.random.default_rng(seed).shuffle(idx)
        for b in range(0, len(idx), 32):
            sel = idx[b:b + 32]
            xb = torch.from_numpy(x[sel].astype(np.float32)).to(device)
            mb = torch.from_numpy(ar[None] < lengths[sel, None]).to(device)
            loss = nn.functional.cross_entropy(model(xb, mb), torch.from_numpy(y[sel]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        seed += 1
    model.eval()
    outs = []
    with torch.no_grad():
        for sel in pred_sets:
            probs = []
            for b in range(0, len(sel), 64):
                s = sel[b:b + 64]
                xb = torch.from_numpy(x[s].astype(np.float32)).to(device)
                mb = torch.from_numpy(ar[None] < lengths[s, None]).to(device)
                probs.append(model(xb, mb).softmax(-1).cpu().numpy())
            outs.append(np.concatenate(probs))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = np.load(a.frames, mmap_mode="r")
    meta = np.load(a.meta)
    ids = meta["subject_ids"].astype(str)
    lengths = meta["lengths"]
    man = pd.read_csv(a.manifest).set_index("subject_id").loc[ids]
    y = man["label"].map({k: i for i, k in enumerate(LABELS)}).to_numpy()
    train = np.where(man["split"].to_numpy() == "train")[0]
    test = np.where(man["split"].to_numpy() == "test")[0]
    oof = np.zeros((len(ids), 3))
    test_p = np.zeros((len(test), 3))
    for seed in range(a.seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=1000 + seed)
        for f, (i_tr, i_va) in enumerate(skf.split(train, y[train])):
            va, te = fit_predict(x, lengths, y, train[i_tr], [train[i_va], test], device,
                                 seed * 10 + f, a.epochs, a.lr)
            oof[train[i_va]] += va / a.seeds
            test_p += te / (5 * a.seeds)
        print("seed", seed, "OOF acc so far", float((oof[train].argmax(1) == y[train]).mean()), flush=True)
    full = oof.copy()
    full[test] = test_p
    np.savez(a.out, probs=full, subject_ids=ids, is_test=np.isin(np.arange(len(ids)), test))


if __name__ == "__main__":
    main()
