from __future__ import annotations

from advoice.agent_led import EvidenceSession, evidence_snapshot
from advoice.cognitive_agent import _allowed_ids
from advoice.transcript_sanitization import (
    sanitize_segment_payload,
    sanitize_transcript_payload,
)


def test_english_self_report_is_redacted_but_non_label_speech_survives() -> None:
    item = sanitize_transcript_payload({
        "language": "English",
        "text": "I was diagnosed with Alzheimer's disease. I forget appointments and lose my keys.",
    })
    assert item["diagnostic_disclosure"] == "participant_self_report"
    assert item["prediction_eligible"] is False
    assert "Alzheimer" not in item["text"]
    assert "forget appointments" in item["text"]


def test_spanish_self_report_is_redacted() -> None:
    item = sanitize_transcript_payload({
        "language": "Spanish",
        "text": "Me diagnosticaron demencia. Me cuesta recordar los nombres.",
    })
    assert item["diagnostic_disclosure"] == "participant_self_report"
    assert item["prediction_eligible"] is False
    assert "demencia" not in item["text"].lower()
    assert "recordar los nombres" in item["text"]


def test_chinese_self_report_is_redacted() -> None:
    item = sanitize_transcript_payload({
        "language": "Chinese",
        "text": "我被诊断为阿尔茨海默病。我最近经常忘记约会。",
    })
    assert item["diagnostic_disclosure"] == "participant_self_report"
    assert item["prediction_eligible"] is False
    assert "阿尔茨海默" not in item["text"]
    assert "忘记约会" in item["text"]


def test_interviewer_and_filename_disclosures_force_segment_exclusion() -> None:
    item = sanitize_segment_payload({
        "speaker_role": "interviewer",
        "filename": "participant_AD_001.wav",
        "text": "The patient has dementia, but describes the kitchen clearly.",
    })
    assert item["diagnostic_disclosure"] == "interviewer_or_metadata"
    assert item["prediction_eligible"] is False
    assert "filename" not in item
    assert "dementia" not in item["text"].lower()
    assert "describes the kitchen clearly" in item["text"]


def test_transcript_header_label_is_not_treated_as_speech() -> None:
    item = sanitize_transcript_payload({
        "text": "@ID: sample | AD |\nI forget appointments and lose my keys.",
    })
    assert item["diagnostic_disclosure"] == "interviewer_or_metadata"
    assert item["prediction_eligible"] is False
    assert "@ID" not in item["text"]
    assert "forget appointments" in item["text"]


def test_clean_speech_remains_prediction_eligible() -> None:
    item = sanitize_segment_payload({
        "language": "en",
        "speaker_role": "participant",
        "text": "I worry about my memory and sometimes forget names.",
    })
    assert item["diagnostic_disclosure"] == "none"
    assert item["prediction_eligible"] is True
    assert item["text"].startswith("I worry")


def test_snapshot_sanitizes_nested_segments_before_agent_access() -> None:
    workspace = {
        "case_id": "case_fixture",
        "state_observations": [{
            "evidence_id": "state:S01",
            "state_id": "S01",
            "report_permission": True,
            "evidence_segments": [{
                "segment_id": "segment:leak",
                "text": "I have MCI.",
                "report_permission": True,
            }],
        }],
        "case_transcript": {"language": "en", "text": "I have dementia."},
    }
    snapshot = evidence_snapshot(workspace)
    segment = snapshot["state_observations"][0]["evidence_segments"][0]
    assert segment["prediction_eligible"] is False
    assert segment["diagnostic_disclosure"] == "participant_self_report"
    assert "dementia" not in snapshot["case_transcript"]["text"].lower()
    session = EvidenceSession(workspace, ["HC", "AD"], model_id="test", skill_hash="fixture")
    clinical, _, _ = _allowed_ids(session.workspace)
    assert "segment:leak" not in clinical


def test_empty_first_step_abstention_is_rejected() -> None:
    workspace = {
        "case_id": "empty",
        "case_context": {},
        "case_transcript": {"text": ""},
        "state_observations": [],
        "selected_supporting_evidence": [],
        "selected_counterevidence": [],
        "quality_observations": [],
        "evidence_registry": [],
    }
    session = EvidenceSession(workspace, ["HC", "AD"], model_id="test", skill_hash="fixture")
    reply = {
        "action": "abstain", "revision": session.revision, "target_id": "",
        "state_action": "none", "evidence_ids": [], "counterevidence_ids": [],
        "predicted_label": "AD", "scores": {"HC": 0, "AD": 0},
        "rationale": "No usable evidence.", "limitations": ["Empty transcript."],
    }
    assert session.step(reply)["status"] == "rejected"
