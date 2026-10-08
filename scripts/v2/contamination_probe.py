#!/usr/bin/env python3
"""Q4 memorisation probe: does the LLM continue a public transcript verbatim better than chance?

For each case, give the first half of the participant's words; ask for a verbatim continuation.
Score = token 4-gram overlap with the true continuation, versus overlap with a different case's
continuation (same task, so shared content is the baseline). Run on public ADReSS CHAT transcripts
and on PREPARE ASR transcripts (never published as text) as a negative control.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from advoice.agent_runtime import run_structured_batch  # noqa: E402

PROMPT = ("The following is the beginning of a transcript from a publicly released research corpus of a person describing "
          "a picture. If you have seen this exact transcript before, continue it verbatim; otherwise give your best guess "
          "of the exact continuation. Output only the continuation (about {n} words).\n\nBEGINNING:\n{text}")
SCHEMA = {"type": "object", "properties": {"continuation": {"type": "string"}}, "required": ["continuation"],
          "additionalProperties": False}


def cha_words(path: Path) -> list[str]:
    words = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("*PAR:"):
            t = re.sub(r"\x15.*?\x15|\[.*?\]|<|>|&\S+|\(\.+\)|[^\w' ]", " ", line[5:])
            words += t.lower().split()
    return words


def grams(ws: list[str], n: int = 4) -> set:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def overlap(pred: list[str], ref: list[str]) -> float:
    r = grams(ref)
    return len(grams(pred) & r) / max(len(r), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cha-dir", required=True)
    ap.add_argument("--prepare-transcripts", required=True)
    ap.add_argument("--provider", default="openai_api")
    ap.add_argument("--model", default="gpt-5.5-2026-04-23")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    adress = [cha_words(p) for p in sorted(Path(a.cha_dir).rglob("*.cha"))]
    adress = [w for w in adress if len(w) >= 60]
    tx = pd.read_csv(a.prepare_transcripts)
    prep = [str(t).lower().split() for t in tx["transcript"]]
    prep = [w for w in prep if len(w) >= 60]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema.json").write_text(json.dumps(SCHEMA))
    summary = {}
    for name, pool in (("ADReSS_public_CHAT", adress), ("PREPARE_ASR_control", prep)):
        pick = rng.choice(len(pool), min(a.n, len(pool)), replace=False)

        def run(i: int) -> tuple[float, float]:
            ws = pool[i]
            half = len(ws) // 2
            prompt = PROMPT.format(n=len(ws) - half, text=" ".join(ws[:half]))
            res = run_structured_batch(out, prompt, out / "schema.json", out / f"{name}_{i}.json", a.model, a.provider)
            pred = res["continuation"].lower().split()
            other = pool[(i + 7) % len(pool)]
            return overlap(pred, ws[half:]), overlap(pred, other[len(other) // 2:])

        with ThreadPoolExecutor(6) as ex:
            scores = list(ex.map(run, pick))
        true_o, other_o = np.array(scores).T
        summary[name] = {"n": len(scores), "mean_4gram_overlap_true": round(float(true_o.mean()), 4),
                         "mean_4gram_overlap_other_case": round(float(other_o.mean()), 4),
                         "cases_true_overlap_ge_0.3": int((true_o >= 0.3).sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
