from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from advoice.demo import (
    PUBLIC_DEMO_CASES,
    analyze_local_manifest_case,
    analyze_public_case,
    parse_byte_range,
    write_public_demo_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "demo" / "assets"


def ensure_public_assets() -> None:
    if not all((ASSETS / str(case["audio_file"])).exists() for case in PUBLIC_DEMO_CASES.values()):
        subprocess.run([sys.executable, str(ROOT / "demo" / "generate_sample.py")], check=True)


def test_public_demo_bundle_covers_four_channels(tmp_path: Path) -> None:
    ensure_public_assets()
    cases = write_public_demo_bundle(ASSETS, tmp_path)

    assert len(cases) == 4
    assert {case["channel_id"] for case in cases} == {
        "clinical_interview",
        "picture_description",
        "structured_multitask",
        "public_speech",
    }
    assert (tmp_path / "public_cases.json").exists()


def test_public_case_produces_trace_and_report_without_diagnosis() -> None:
    ensure_public_assets()
    result = analyze_public_case("synthetic_picture_description", ASSETS)

    assert result["schema_version"] == "public-demo-v3"
    assert result["decision"]["status"] == "not_generated"
    assert result["execution"]["diagnostic_agent_invoked"] is False
    assert result["agent_report"]["status"] == "offline_preview"
    assert result["agent_report"]["evidence_ids"]
    assert {card["id"] for card in result["state_cards"]} == {"S01", "S02", "S08", "S10"}
    assert result["trace"]
    assert result["segments"]
    assert all("text" in segment for segment in result["segments"])
    assert result["quality"]["audio_reliability"] > 0
    assert "Synthetic" in result["disclaimer"]


def test_channel_routing_changes_evidence_set() -> None:
    ensure_public_assets()
    clinical = analyze_public_case("synthetic_clinical_interview", ASSETS)
    natural = analyze_public_case("synthetic_public_speech", ASSETS)

    clinical_ids = {item["metric_id"] for item in clinical["metric_evidence"]}
    natural_by_id = {item["metric_id"]: item for item in natural["metric_evidence"]}
    assert "patient_turn_share" in clinical_ids
    assert natural_by_id["f0_iqr_hz"]["evidence_role"] == "model_auxiliary"
    assert natural_by_id["snr_proxy_db"]["reportable"] is False


def test_local_manifest_case_uses_channel_metadata_without_diagnosis() -> None:
    ensure_public_assets()
    audio = ASSETS / "synthetic_picture_description.wav"
    transcript = ASSETS / "synthetic_picture_description.txt"
    result = analyze_local_manifest_case(
        {
            "demo_case_id": "local_channel_test",
            "dataset_id": "LOCAL_TEST",
            "channel_id": "picture_description",
            "channel_name": "Picture description",
            "task_name": "Local test task",
            "description": "Restricted local test case",
            "evidence_focus": ["content units", "pausing"],
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

    assert result["case"]["channel_name"] == "Picture description"
    assert result["case"]["data_scope"] == "local_restricted_not_for_redistribution"
    assert result["decision"]["status"] == "not_generated"
    assert "Local restricted-data" in result["disclaimer"]


def test_audio_byte_ranges_support_browser_seeking() -> None:
    assert parse_byte_range(None, 100) is None
    assert parse_byte_range("bytes=10-19", 100) == (10, 19)
    assert parse_byte_range("bytes=90-", 100) == (90, 99)
    assert parse_byte_range("bytes=-10", 100) == (90, 99)
