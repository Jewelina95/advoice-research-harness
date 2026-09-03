from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from advoice.demo import analyze_demo_audio


ROOT = Path(__file__).resolve().parents[1]


def test_public_demo_produces_trace_without_diagnosis(tmp_path: Path) -> None:
    sample = ROOT / "demo" / "assets" / "synthetic_picture_description.wav"
    if not sample.exists():
        subprocess.run([sys.executable, str(ROOT / "demo" / "generate_sample.py")], check=True)
    transcript = (ROOT / "demo" / "assets" / "synthetic_picture_description.txt").read_text(encoding="utf-8")
    result = analyze_demo_audio(sample, transcript)

    assert result["decision"]["status"] == "not_generated"
    assert len(result["metric_evidence"]) == 7
    assert {card["id"] for card in result["state_cards"]} == {"S01", "S02", "S08"}
    assert result["trace"]
    assert result["segments"]
    assert result["quality"]["audio_reliability"] > 0
    assert "Synthetic" in result["disclaimer"]
