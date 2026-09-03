from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from advoice.demo import analyze_demo_audio, analyze_local_manifest_case, parse_byte_range


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


def test_local_manifest_case_uses_channel_metadata_without_diagnosis() -> None:
    audio = ROOT / "demo" / "assets" / "synthetic_picture_description.wav"
    transcript = ROOT / "demo" / "assets" / "synthetic_picture_description.txt"
    result = analyze_local_manifest_case(
        {
            "demo_case_id": "local_channel_test",
            "dataset_id": "LOCAL_TEST",
            "channel_id": "picture_description",
            "channel_name_zh": "图片描述",
            "task_name_zh": "测试任务",
            "description_zh": "本地测试案例",
            "evidence_focus_zh": ["内容单元", "停顿"],
            "research_label": "AD",
            "audio_path": str(audio),
            "transcript_path": str(transcript),
            "task_type": "cookie_theft_picture_description",
            "language": "en",
            "channel": "picture_description",
            "analysis_intervals": "[]",
            "role_filter_required": False,
            "transcript_reliability": 0.95,
        }
    )

    assert result["case"]["channel_name_zh"] == "图片描述"
    assert result["case"]["data_scope"] == "local_restricted_not_for_redistribution"
    assert result["decision"]["status"] == "not_generated"
    assert "Local restricted-data" in result["disclaimer"]


def test_audio_byte_ranges_support_browser_seeking() -> None:
    assert parse_byte_range(None, 100) is None
    assert parse_byte_range("bytes=10-19", 100) == (10, 19)
    assert parse_byte_range("bytes=90-", 100) == (90, 99)
    assert parse_byte_range("bytes=-10", 100) == (90, 99)
