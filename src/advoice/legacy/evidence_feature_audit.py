"""Auditable mapping and feature-set calibration for governed AD-voice evidence.

This module is research-only. It does not change the production predictor or
automatically promote historical metrics into the clinical evidence registry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Alias:
    runtime_id: str
    relation: str
    rationale: str


# Deliberately explicit: similar wording is not enough to establish equivalence.
HISTORICAL_ALIASES: dict[str, tuple[Alias, ...]] = {
    "duration_sec": (Alias("duration_sec", "exact_predecessor_role_changed", "Same duration field; runtime registry now treats it as QC."),),
    "n_words": (Alias("word_count", "renamed_equivalent", "Both count transcript tokens; tokenizer implementation must still be versioned."),),
    "words_per_min": (Alias("speech_rate_wpm", "algorithm_updated_same_construct", "Words divided by duration; runtime implementation defines tokenizer and denominator."),),
    "speech_rate": (Alias("speech_rate_wpm", "construct_overlap_not_equivalent", "Historical term was underspecified; runtime field fixes units to words/minute."),),
    "long_pause_rate_per_min": (Alias("long_pause_rate_min", "algorithm_updated_same_construct", "Same rate construct, but historical and runtime pause thresholds differ."),),
    "pause_mode_mean_sec": (Alias("pause_mean_sec", "construct_overlap_not_equivalent", "Mixture-mode mean was replaced by the mean of detected silent runs."),),
    "MeanUnvoicedSegmentLength": (Alias("pause_mean_sec", "algorithm_updated_same_construct", "Both summarize non-voiced run duration using different extractors."),),
    "egemaps_MeanUnvoicedSegmentLength": (Alias("pause_mean_sec", "duplicate_legacy_measure_replaced", "Duplicate eGeMAPS row; runtime uses one VAD-derived pause summary."),),
    "phonation_time_ratio": (
        Alias("voiced_fraction", "algorithm_updated_same_construct", "Both estimate voiced occupancy."),
        Alias("silence_fraction", "formula_related_complement", "Silence fraction is approximately one minus voiced occupancy under one VAD mask."),
    ),
    "VoicedSegmentsPerSec": (Alias("speech_run_rate_min", "unit_transform_and_algorithm_update", "Events/second maps to events/minute after multiplying by 60."),),
    "egemaps_VoicedSegmentsPerSec": (Alias("speech_run_rate_min", "duplicate_legacy_measure_replaced", "Duplicate eGeMAPS row mapped to the runtime speech-run rate."),),
    "MeanVoicedSegmentLengthSec": (Alias("speech_run_mean_sec", "algorithm_updated_same_construct", "Both summarize contiguous voiced-run duration."),),
    "egemaps_MeanVoicedSegmentLengthSec": (Alias("speech_run_mean_sec", "duplicate_legacy_measure_replaced", "Duplicate eGeMAPS row mapped to the runtime speech-run mean."),),
    "StddevVoicedSegmentLengthSec": (Alias("speech_run_cv", "derived_relation_not_equivalent", "Runtime coefficient of variation additionally divides SD by the mean."),),
    "egemaps_StddevVoicedSegmentLengthSec": (Alias("speech_run_cv", "derived_relation_not_equivalent", "Duplicate SD predecessor contributes to the runtime coefficient of variation."),),
    "equivalentSoundLevel_dBp": (Alias("rms_db_mean", "construct_overlap_not_equivalent", "Both summarize level, but calibration and reference definitions differ."),),
    "loudness_sma3_amean": (Alias("rms_db_mean", "construct_overlap_not_equivalent", "Perceptual loudness was replaced by waveform RMS dB."),),
    "egemaps_loudness_sma3_amean": (Alias("rms_db_mean", "construct_overlap_not_equivalent", "eGeMAPS loudness is related but not numerically equivalent to RMS dB."),),
    "loudness_sma3_stddevNorm": (Alias("rms_db_std", "construct_overlap_not_equivalent", "Both describe level variation using different scales."),),
    "egemaps_loudness_sma3_stddevNorm": (Alias("rms_db_std", "construct_overlap_not_equivalent", "eGeMAPS normalized loudness SD is not the runtime RMS dB SD."),),
    "F0semitoneFrom27.5Hz_sma3nz_amean": (Alias("f0_median_hz", "construct_overlap_not_equivalent", "Mean semitone F0 was replaced by median F0 in Hz."),),
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_percentile50.0": (Alias("f0_median_hz", "monotone_unit_transform", "Median semitone F0 can be transformed to Hz, subject to extractor differences."),),
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": (Alias("f0_iqr_hz", "construct_overlap_not_equivalent", "Historical 20--80% semitone range differs from the runtime Hz IQR."),),
    "pronoun_rate": (Alias("pronoun_ratio", "renamed_equivalent", "Pronoun tokens divided by all transcript tokens."),),
    "content_function_ratio": (Alias("content_word_ratio", "algorithm_updated_same_construct", "Runtime multilingual stopword proxy replaces the historical POS-based ratio."),),
    "lexical_density": (Alias("content_word_ratio", "construct_overlap_not_equivalent", "Both concern lexical content but use different denominators and taggers."),),
    "TTR": (Alias("lexical_ttr", "renamed_equivalent", "Unique tokens divided by token count."),),
    "MATTR": (Alias("lexical_mattr50", "renamed_equivalent", "Runtime implementation fixes the moving window at 50 tokens."),),
    "mean_length_utterance": (Alias("mean_utterance_words", "renamed_equivalent", "Mean patient utterance length measured in tokens."),),
    "mean_segment_words": (Alias("mean_utterance_words", "construct_overlap_not_equivalent", "Segments and patient utterances are not necessarily identical units."),),
    "filler_rate": (Alias("filler_rate_100w", "unit_standardized", "Runtime rate is explicitly per 100 words and language-conditioned."),),
    "revision_rate": (Alias("repair_rate_100w", "algorithm_updated_same_construct", "Runtime counts CHAT repair markers and reports per 100 words."),),
    "patient_speech_ratio": (Alias("patient_turn_share", "construct_overlap_not_equivalent", "Speech-time share was replaced by participant turn share; they cannot be double-counted."),),
    "vague_term_rate": (Alias("picture_uncertainty_rate_100w", "task_conditioned_proxy", "Runtime proxy uses a language-specific picture-description uncertainty lexicon."),),
    "content_information_units": (Alias("picture_content_unit_coverage", "task_conditioned_implementation", "Runtime implements Cookie Theft content-unit coverage."),),
    "core_lexicon_count": (Alias("picture_content_unit_coverage", "task_conditioned_proxy", "Coverage uses a multilingual task lexicon rather than a raw count."),),
    "idea_density": (Alias("picture_information_density", "task_conditioned_proxy", "Runtime uses unique picture content units per 100 words."),),
    "information_units_per_min": (Alias("picture_information_density", "construct_overlap_not_equivalent", "Runtime denominator is words, not time."),),
    "task_completion": (Alias("picture_content_unit_coverage", "task_conditioned_proxy", "Content coverage is one task-performance component, not complete task completion."),),
}


RUNTIME_LINEAGE: dict[str, dict[str, str]] = {
    "duration_sec": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "analysis samples / 16000", "inputs": "role-filtered waveform"},
    "original_duration_sec": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "original samples / sampling rate", "inputs": "original waveform"},
    "clipping_fraction": {"source": "audio_qc", "function": "advoice.features.extract_audio_file", "formula": "mean(abs(sample) >= 0.999)", "inputs": "normalized waveform"},
    "snr_proxy_db": {"source": "audio_qc", "function": "advoice.features.extract_audio_file", "formula": "median voiced RMS dB - median silent RMS dB", "inputs": "waveform; VAD mask"},
    "role_coverage_fraction": {"source": "routing_qc", "function": "advoice.features._analysis_audio", "formula": "patient-role samples / original samples", "inputs": "speaker intervals; waveform"},
    "transcript_available": {"source": "transcript_qc", "function": "advoice.transcripts.transcript_metrics", "formula": "indicator(token_count > 0)", "inputs": "transcript"},
    "word_count": {"source": "transcript_qc", "function": "advoice.transcripts.transcript_metrics", "formula": "number of language-conditioned tokens", "inputs": "transcript; language"},
    "silence_fraction": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean(not VAD_speech)", "inputs": "waveform; hybrid WebRTC/energy VAD"},
    "long_pause_rate_min": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "silent runs >= 0.5 s / duration_min", "inputs": "VAD mask; duration"},
    "pause_mean_sec": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean duration of silent runs", "inputs": "VAD mask"},
    "pause_p90_sec": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "90th percentile duration of silent runs", "inputs": "VAD mask"},
    "voiced_fraction": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean(VAD_speech)", "inputs": "waveform; hybrid WebRTC/energy VAD"},
    "speech_run_mean_sec": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean duration of voiced runs", "inputs": "VAD mask"},
    "speech_run_rate_min": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "voiced runs / duration_min", "inputs": "VAD mask; duration"},
    "speech_rate_wpm": {"source": "audio_transcript", "function": "advoice.transcripts.transcript_metrics", "formula": "token_count / duration_min", "inputs": "transcript; language; audio duration"},
    "speech_run_cv": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "SD voiced-run duration / mean voiced-run duration", "inputs": "VAD mask"},
    "rms_db_mean": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean frame RMS amplitude in dB", "inputs": "waveform"},
    "rms_db_std": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "SD frame RMS amplitude in dB", "inputs": "waveform"},
    "f0_median_hz": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "median YIN F0 over voiced valid frames", "inputs": "waveform; VAD mask"},
    "f0_iqr_hz": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "P75 - P25 YIN F0 over voiced valid frames", "inputs": "waveform; VAD mask"},
    "zcr_mean": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean frame zero-crossing rate", "inputs": "waveform"},
    "spectral_centroid_mean": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean spectral centroid", "inputs": "waveform"},
    "spectral_bandwidth_mean": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean spectral bandwidth", "inputs": "waveform"},
    "spectral_rolloff_mean": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean spectral rolloff", "inputs": "waveform"},
    "spectral_flatness_mean": {"source": "audio", "function": "advoice.features.extract_audio_file", "formula": "mean spectral flatness", "inputs": "waveform"},
    "pronoun_ratio": {"source": "transcript", "function": "advoice.transcripts.transcript_metrics", "formula": "pronoun lexicon hits / token_count", "inputs": "transcript; language lexicon"},
    "content_word_ratio": {"source": "transcript", "function": "advoice.transcripts.transcript_metrics", "formula": "non-stopword tokens / token_count", "inputs": "transcript; language stopwords"},
    "lexical_mattr50": {"source": "transcript", "function": "advoice.transcripts._mattr", "formula": "mean TTR across 50-token windows", "inputs": "language-conditioned tokens"},
    "lexical_ttr": {"source": "transcript", "function": "advoice.transcripts.transcript_metrics", "formula": "unique tokens / token_count", "inputs": "language-conditioned tokens"},
    "mean_utterance_words": {"source": "transcript", "function": "advoice.transcripts.transcript_metrics", "formula": "mean token count per patient utterance", "inputs": "speaker-resolved transcript"},
    "filler_rate_100w": {"source": "transcript", "function": "advoice.transcripts.transcript_metrics", "formula": "100 * filler lexicon hits / token_count", "inputs": "transcript; language filler lexicon"},
    "repair_rate_100w": {"source": "transcript", "function": "advoice.transcripts.transcript_metrics", "formula": "100 * CHAT repair markers / token_count", "inputs": "CHAT transcript"},
    "patient_turn_share": {"source": "dialogue", "function": "advoice.transcripts.transcript_metrics", "formula": "patient turns / (patient + interviewer turns)", "inputs": "speaker-resolved transcript"},
    "picture_uncertainty_rate_100w": {"source": "task_transcript", "function": "advoice.transcripts._picture_description_metrics", "formula": "100 * uncertainty phrase hits / token_count", "inputs": "picture transcript; language lexicon"},
    "picture_information_density": {"source": "task_transcript", "function": "advoice.transcripts._picture_description_metrics", "formula": "100 * unique content units / token_count", "inputs": "picture transcript; task scoring lexicon"},
    "picture_content_redundancy": {"source": "task_transcript", "function": "advoice.transcripts._picture_description_metrics", "formula": "repeated content-unit mentions / all content-unit mentions", "inputs": "picture transcript; task scoring lexicon"},
    "picture_content_unit_coverage": {"source": "task_transcript", "function": "advoice.transcripts._picture_description_metrics", "formula": "observed content units / 16 expected units", "inputs": "picture transcript; task scoring lexicon"},
}


def _fallback_disposition(status: object) -> str:
    return {
        "include_auxiliary_only": "historical_auxiliary_not_in_governed_registry",
        "exclude_metric_not_available": "not_extracted_in_current_pipeline",
        "exclude_current_not_implemented": "planned_not_implemented",
        "exclude_pending_direction_review": "held_for_direction_or_confound_review",
        "include_current_state_evaluation": "legacy_core_without_confirmed_runtime_successor",
    }.get(str(status), "unresolved_manual_review")


def map_historical_metrics(
    history: pd.DataFrame, metrics: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create row-preserving historical mappings and runtime lineage tables."""
    required = {"state_id", "state_name", "metric_name", "metric_definition", "how_calculated", "current_evaluation_status"}
    missing = required - set(history)
    if missing:
        raise ValueError(f"Historical dictionary missing {sorted(missing)}")
    runtime = {metric["id"]: metric for metric in metrics}
    if set(runtime) - set(RUNTIME_LINEAGE):
        raise ValueError("Runtime lineage must cover every registered metric")

    rows: list[dict[str, Any]] = []
    for number, row in enumerate(history.to_dict("records"), start=1):
        aliases = HISTORICAL_ALIASES.get(str(row["metric_name"]), ())
        unknown = [alias.runtime_id for alias in aliases if alias.runtime_id not in runtime]
        if unknown:
            raise ValueError(f"Unknown runtime aliases: {unknown}")
        rows.append({
            "historical_row": number,
            "state_id": row["state_id"],
            "state_name": row["state_name"],
            "historical_metric": row["metric_name"],
            "historical_columns": row.get("matched_current_columns", ""),
            "historical_modality": row.get("metric_modality", ""),
            "historical_role": row.get("final_role", ""),
            "historical_status": row.get("current_evaluation_status", ""),
            "historical_definition": row.get("metric_definition", ""),
            "historical_calculation": row.get("how_calculated", ""),
            "runtime_metric_ids": ";".join(alias.runtime_id for alias in aliases),
            "mapping_relations": ";".join(alias.relation for alias in aliases),
            "mapping_rationales": " | ".join(alias.rationale for alias in aliases),
            "disposition": "mapped_to_runtime_registry" if aliases else _fallback_disposition(row.get("current_evaluation_status")),
            "mapping_confidence": "reviewed_rule" if aliases else "unresolved",
            "manual_review_required": not bool(aliases),
        })
    mapping = pd.DataFrame(rows)

    historical_names = set(history.metric_name.astype(str))
    reverse: dict[str, list[dict[str, str]]] = {metric_id: [] for metric_id in runtime}
    for historical, aliases in HISTORICAL_ALIASES.items():
        if historical not in historical_names:
            continue
        for alias in aliases:
            if alias.runtime_id in reverse:
                reverse[alias.runtime_id].append({"historical": historical, "relation": alias.relation})
    lineage_rows = []
    for metric_id, definition in runtime.items():
        links = reverse[metric_id]
        lineage_rows.append({
            "runtime_metric_id": metric_id,
            "state": definition.get("state", ""),
            "branch": definition.get("branch", ""),
            "role": definition.get("role", ""),
            "direction": definition.get("direction", ""),
            "configured_reliability": definition.get("reliability", ""),
            "report_permission": bool(definition.get("report_permission", False)),
            "confounds": ";".join(definition.get("confounds", [])),
            **RUNTIME_LINEAGE[metric_id],
            "historical_predecessors": ";".join(link["historical"] for link in links),
            "historical_relations": ";".join(link["relation"] for link in links),
            "provenance_status": "mapped_predecessor" if links else "new_runtime_metric",
        })
    lineage = pd.DataFrame(lineage_rows)

    summary = pd.DataFrame([
        {"section": "historical_rows", "category": key, "count": int(value)}
        for key, value in mapping.disposition.value_counts().items()
    ] + [
        {"section": "runtime_metrics", "category": key, "count": int(value)}
        for key, value in lineage.provenance_status.value_counts().items()
    ] + [
        {"section": "runtime_roles", "category": key, "count": int(value)}
        for key, value in lineage.role.value_counts().items()
    ])
    return mapping, lineage, summary


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    output = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return output
    ranked = valid[np.argsort(values[valid])]
    adjusted = values[ranked] * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output[ranked] = np.clip(adjusted, 0.0, 1.0)
    return output


def feature_inference(
    frame: pd.DataFrame, metric_ids: list[str], *, kruskal_fn: Any
) -> pd.DataFrame:
    """Train-only descriptive group tests with multiplicity correction."""
    required = {"split", "label"}
    if not required.issubset(frame):
        raise ValueError("Feature table requires explicit split and label columns")
    train = frame.loc[frame.split.astype(str).str.lower().eq("train")].copy()
    labels = train.label.astype(str)
    groups = sorted(labels.unique())
    rows = []
    for metric_id in metric_ids:
        if metric_id not in train:
            rows.append({"metric_id": metric_id, "status": "not_extracted"})
            continue
        values = pd.to_numeric(train[metric_id], errors="coerce").replace([np.inf, -np.inf], np.nan)
        samples = [values.loc[labels.eq(group)].dropna().to_numpy() for group in groups]
        observed = values.dropna()
        status = "ok"
        if len(observed) < max(10, 2 * len(groups)):
            status = "insufficient_observations"
        elif observed.nunique() < 2:
            status = "constant"
        elif any(len(sample) < 2 for sample in samples):
            status = "insufficient_group_observations"
        h_stat = p_value = effect = np.nan
        if status == "ok":
            h_stat, p_value = kruskal_fn(*samples)
            denominator = len(observed) - len(groups)
            effect = max(0.0, float((h_stat - len(groups) + 1) / denominator)) if denominator > 0 else np.nan
        rows.append({
            "metric_id": metric_id,
            "status": status,
            "n_train": len(train),
            "n_observed": len(observed),
            "missing_fraction": float(1 - len(observed) / max(len(train), 1)),
            "n_unique": int(observed.nunique()),
            "n_groups": len(groups),
            "group_counts": ";".join(f"{group}:{len(sample)}" for group, sample in zip(groups, samples, strict=True)),
            "kruskal_h": h_stat,
            "p_value": p_value,
            "epsilon_squared": effect,
            "scope": "descriptive_train_only_not_causal_or_clinically_validated",
        })
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = benjamini_hochberg(result.get("p_value", pd.Series(dtype=float)))
    return result


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([
        rng.choice(indices, size=len(indices), replace=True)
        for value in np.unique(labels)
        if len(indices := np.flatnonzero(labels == value))
    ])


def choose_feature_counts(n_features: int) -> list[int]:
    return sorted({value for value in (1, 2, 4, 8, 12, 16, 24, n_features) if value <= n_features})


def candidate_metric_ids(metrics: list[dict[str, Any]]) -> list[str]:
    """QC variables are audited separately and cannot win disease prediction."""
    return [metric["id"] for metric in metrics if metric.get("role") != "qc_only"]
