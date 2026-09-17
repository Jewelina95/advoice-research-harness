"""Label-leakage screening for transcript payloads crossing the Agent boundary.

This module deliberately keeps the screening conservative at the boundary and
does not alter feature extraction.  It removes only diagnosis-bearing spans;
unrelated clinical speech remains available for provenance and review.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from .transcripts import canonical_language, repair_utf8_mojibake


DISCLOSURE_VALUES = {
    "none", "participant_self_report", "interviewer_or_metadata", "uncertain",
}
_REDACTION = "[diagnostic disclosure removed]"
_TEXT_KEYS = ("text", "transcript", "utterance")
_METADATA_KEYS = {
    "audio_path", "source_path", "path", "filename", "file_name", "file", "title",
    "transcript_path", "dataset", "dataset_id", "label", "diagnosis", "class",
    "group", "cohort", "condition", "study_label", "recording_name",
}

_DISEASE = {
    "en": r"(?:alzheimer(?:'s)?|dementia|mild\s+cognitive\s+impairment)",
    "es": r"(?:alzheimer|demencia|deterioro\s+cognitivo\s+leve)",
    "zh": r"(?:阿尔茨海默(?:病|氏症)?|老年痴呆|痴呆|轻度认知障碍)",
}
_CLASS = r"(?:\b(?:AD|MCI|HC)\b)"

_SELF_PATTERNS = {
    "en": (
        r"\b(?:i|i'm|i am|i've been|i have been|i was)\s+(?:diagnosed\s+with|a\s+patient\s+with|an?\s+patient\s+with)\s+(?:%s|%s)" % (_DISEASE["en"], _CLASS),
        r"\b(?:i have|i've got|my diagnosis is)\s+(?:%s|%s)" % (_DISEASE["en"], _CLASS),
        r"\b(?:i am|i'm)\s+(?:an?\s+)?(?:%s|%s)(?:\s+patient)?" % (_DISEASE["en"], _CLASS),
        r"\b(?:diagnosis|diagnosed)\s*:\s*(?:%s|%s)" % (_DISEASE["en"], _CLASS),
    ),
    "es": (
        r"\b(?:me\s+(?:han\s+)?diagnosticaron|me\s+han\s+diagnosticado|soy\s+paciente\s+de|mi\s+diagnóstico\s+es)\s+(?:%s|%s)" % (_DISEASE["es"], _CLASS),
        r"\b(?:tengo|padezco)\s+(?:%s|%s)" % (_DISEASE["es"], _CLASS),
        r"\b(?:diagnóstico|diagnostico)\s*:\s*(?:%s|%s)" % (_DISEASE["es"], _CLASS),
    ),
    "zh": (
        r"我(?:已被|被)?诊断为(?:%s|%s)" % (_DISEASE["zh"], _CLASS),
        r"我(?:患有|有)(?:%s|%s)" % (_DISEASE["zh"], _CLASS),
        r"我是(?:%s|%s)患者" % (_DISEASE["zh"], _CLASS),
        r"(?:我的诊断是|诊断[:：])\s*(?:%s|%s)" % (_DISEASE["zh"], _CLASS),
    ),
}

_INTERVIEWER_PATTERNS = {
    "en": (
        r"\b(?:the\s+)?(?:patient|participant)\s+(?:has|had|is|was|was\s+diagnosed\s+with)\s+(?:%s|%s)" % (_DISEASE["en"], _CLASS),
        r"\b(?:diagnosis|diagnosed)\s*:\s*(?:%s|%s)" % (_DISEASE["en"], _CLASS),
    ),
    "es": (
        r"\b(?:el|la)\s+(?:paciente|participante)\s+(?:tiene|padece|fue\s+diagnosticad[oa])\s+(?:%s|%s)" % (_DISEASE["es"], _CLASS),
        r"\b(?:diagnóstico|diagnostico)\s*:\s*(?:%s|%s)" % (_DISEASE["es"], _CLASS),
    ),
    "zh": (
        r"(?:患者|参与者)(?:患有|有|被诊断为)(?:%s|%s)" % (_DISEASE["zh"], _CLASS),
        r"诊断[:：]\s*(?:%s|%s)" % (_DISEASE["zh"], _CLASS),
    ),
}

_ROLE_HINTS = re.compile(r"(?:interviewer|interview|inv|clinician|doctor|researcher|医生|访谈|采访)", re.I)
_HEADER_LABEL = re.compile(
    r"(?i)^\s*(?:[@#].{0,120}|(?:subject|participant|case|group|class|diagnosis|诊断)\s*[:：|]).*"
    r"(?:alzheimer|dementia|demencia|mild[_ -]?cognitive|deterioro|阿尔茨海默|痴呆|认知障碍|\bAD\b|\bMCI\b|\bHC\b)"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")


def _language(value: Any) -> str:
    language = canonical_language(str(value or ""))
    return language if language in _DISEASE else ""


def _metadata_disclosure(payload: Mapping[str, Any]) -> bool:
    for key, value in payload.items():
        if str(key).lower() not in _METADATA_KEYS:
            continue
        text = str(value or "")
        if not text:
            continue
        if re.search(r"(?i)(?:alzheimer|dementia|demencia|mild[_ -]?cognitive|deterioro|阿尔茨海默|痴呆|认知障碍)", text):
            return True
        if re.search(r"(?i)(?:^|[_\s./-])(?:AD|MCI|HC)(?:$|[_\s./-])", text):
            return True
    return False


def _screen_text(text: str, language: str, speaker_role: str = "") -> tuple[str, str, bool]:
    value = repair_utf8_mojibake(text)
    languages = [language] if language in _DISEASE else ["en", "es", "zh"]
    patterns: list[tuple[str, str]] = []
    for lang in languages:
        patterns.extend((pattern, "participant_self_report") for pattern in _SELF_PATTERNS[lang])
        patterns.extend((pattern, "interviewer_or_metadata") for pattern in _INTERVIEWER_PATTERNS[lang])
    found: list[str] = []
    redacted_units: list[str] = []
    for unit in _SENTENCE_SPLIT.split(value):
        current = unit
        unit_disclosure = "none"
        if _HEADER_LABEL.search(current):
            unit_disclosure = "interviewer_or_metadata"
            current = _REDACTION
        for pattern, disclosure in patterns:
            if re.search(pattern, current, flags=re.I):
                unit_disclosure = disclosure
                current = re.sub(pattern, _REDACTION, current, flags=re.I)
        if unit_disclosure != "none":
            found.append(unit_disclosure)
        redacted_units.append(current)
    if not found:
        return value, "none", True
    if "interviewer_or_metadata" in found or _ROLE_HINTS.search(speaker_role or ""):
        disclosure = "interviewer_or_metadata"
    else:
        disclosure = "participant_self_report"
    return " ".join(redacted_units).strip(), disclosure, False


def sanitize_segment_payload(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Return a segment safe for Agent/provider payloads."""
    item = dict(segment)
    language = _language(item.get("language"))
    speaker_role = str(item.get("speaker_role", item.get("role", "")))
    existing = str(item.get("diagnostic_disclosure", "none"))
    if existing not in DISCLOSURE_VALUES:
        existing = "uncertain"
    text_key = next((key for key in _TEXT_KEYS if key in item), None)
    text = str(item.get(text_key, "")) if text_key else ""
    safe_text, text_disclosure, eligible = _screen_text(text, language, speaker_role)
    metadata_disclosure = _metadata_disclosure(item)
    disclosure = existing if existing != "none" else text_disclosure
    if metadata_disclosure:
        disclosure = "interviewer_or_metadata"
    if disclosure == "none" and not eligible:
        disclosure = "uncertain"
    if text_key:
        item[text_key] = safe_text
    for key in list(item):
        if str(key).lower() in _METADATA_KEYS:
            item.pop(key, None)
    item["diagnostic_disclosure"] = disclosure
    item["prediction_eligible"] = bool(disclosure == "none" and item.get("prediction_eligible", True) is not False)
    if disclosure != "none":
        item["prediction_eligible"] = False
    return item


def sanitize_transcript_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize a complete transcript and any transcript-level segments."""
    item = dict(payload)
    language = _language(item.get("language"))
    text_key = next((key for key in _TEXT_KEYS if key in item), None)
    text = str(item.get(text_key, "")) if text_key else ""
    safe_text, text_disclosure, eligible = _screen_text(text, language, str(item.get("speaker_role", "")))
    metadata_disclosure = _metadata_disclosure(item)
    existing = str(item.get("diagnostic_disclosure", "none"))
    if existing not in DISCLOSURE_VALUES:
        existing = "uncertain"
    disclosure = "interviewer_or_metadata" if metadata_disclosure else (existing if existing != "none" else text_disclosure)
    if text_key:
        item[text_key] = safe_text
    if isinstance(item.get("segments"), list):
        item["segments"] = [sanitize_segment_payload(segment) for segment in item["segments"] if isinstance(segment, Mapping)]
        if any(segment["diagnostic_disclosure"] != "none" for segment in item["segments"]):
            disclosure = "interviewer_or_metadata" if any(segment["diagnostic_disclosure"] == "interviewer_or_metadata" for segment in item["segments"]) else "participant_self_report"
    for key in list(item):
        if str(key).lower() in _METADATA_KEYS:
            item.pop(key, None)
    item["diagnostic_disclosure"] = disclosure
    item["prediction_eligible"] = bool(disclosure == "none" and eligible)
    return item


def sanitize_workspace_transcripts(workspace: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the same transcript/segment policy to every workspace view."""
    cleaned = deepcopy(dict(workspace))
    if isinstance(cleaned.get("case_transcript"), Mapping):
        cleaned["case_transcript"] = sanitize_transcript_payload(cleaned["case_transcript"])
    elif isinstance(cleaned.get("case_transcript"), str):
        cleaned["case_transcript"] = sanitize_transcript_payload({
            "text": cleaned["case_transcript"],
        })
    for key, value in list(cleaned.items()):
        if not isinstance(value, list):
            continue
        for parent in value:
            if not isinstance(parent, dict):
                continue
            if isinstance(parent.get("case_transcript"), Mapping):
                parent["case_transcript"] = sanitize_transcript_payload(parent["case_transcript"])
            if isinstance(parent.get("evidence_segments"), list):
                parent["evidence_segments"] = [
                    sanitize_segment_payload(segment)
                    for segment in parent["evidence_segments"]
                    if isinstance(segment, Mapping)
                ]
    return cleaned
