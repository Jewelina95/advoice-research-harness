from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent

CHANNELS: list[dict[str, Any]] = [
    {
        "dataset_id": "IAEAV",
        "channel_id": "clinical_interview",
        "channel_name": "Clinical interview",
        "task_name": "Spanish structured clinical interview",
        "description": "Preserves participant and interviewer roles, question-response turns, onset latency, pausing, and interaction burden.",
        "evidence_focus": ["participant speech", "response onset", "dialogue turns", "output efficiency"],
        "task_filter": None,
    },
    {
        "dataset_id": "ADReSS_2020",
        "channel_id": "picture_description",
        "channel_name": "Picture description",
        "task_name": "Cookie Theft picture description",
        "description": "Uses a standardized scene and prompt to assess content units, information density, lexical retrieval, syntax, and fluency.",
        "evidence_focus": ["content units", "information density", "lexical retrieval", "pausing"],
        "task_filter": None,
    },
    {
        "dataset_id": "PROCESS_2",
        "channel_id": "structured_multitask",
        "channel_name": "Structured cognitive task",
        "task_name": "PROCESS-2 semantic fluency task",
        "description": "Preserves task identity across picture description, phonemic fluency, and semantic fluency instead of averaging tasks before fusion.",
        "evidence_focus": ["task identity", "timed output", "lexical retrieval", "repairs"],
        "task_filter": "sft",
    },
    {
        "dataset_id": "DementiaNet_PublicFigures",
        "channel_id": "public_speech",
        "channel_name": "Natural speech",
        "task_name": "Non-standard public speech",
        "description": "Uses uncontrolled topics, devices, and settings for external generalization and quality review without imposing standardized task scores.",
        "evidence_focus": ["natural speech", "recording quality", "robust acoustic behavior", "external generalization"],
        "task_filter": None,
    },
]


def _pick_case(frame: pd.DataFrame, task_filter: str | None) -> dict[str, Any]:
    candidates = frame.loc[frame["label"].astype(str).str.upper().eq("AD")].copy()
    if task_filter:
        candidates = candidates.loc[candidates["task_type"].astype(str).str.lower().eq(task_filter)]
    candidates = candidates.loc[
        candidates["audio_path"].map(lambda value: Path(str(value)).exists())
    ]
    if candidates.empty:
        raise ValueError("No eligible local AD-labelled case with an existing audio file was found.")
    train = candidates.loc[candidates["split"].astype(str).str.lower().eq("train")]
    eligible = (train if not train.empty else candidates).copy()
    eligible["_audio_size_bytes"] = eligible["audio_path"].map(
        lambda value: Path(str(value)).stat().st_size
    )
    return eligible.sort_values("_audio_size_bytes").iloc[0].to_dict()


def build(advoice_root: Path, output: Path) -> Path:
    artifacts = advoice_root / "8.27" / "artifacts"
    cases: list[dict[str, Any]] = []
    for index, channel in enumerate(CHANNELS, start=1):
        manifest_path = artifacts / channel["dataset_id"] / "analysis_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        row = _pick_case(pd.read_csv(manifest_path, dtype=str), channel["task_filter"])
        cases.append(
            {
                "demo_case_id": f"local_channel_{index:02d}",
                "dataset_id": channel["dataset_id"],
                "channel_id": channel["channel_id"],
                "channel_name": channel["channel_name"],
                "task_name": channel["task_name"],
                "description": channel["description"],
                "evidence_focus": channel["evidence_focus"],
                "source_case_id": str(row["case_id"]),
                "source_subject_id": str(row["subject_id"]),
                "research_label": str(row["label"]),
                "audio_path": str(row["audio_path"]),
                "transcript_path": str(row.get("transcript_path", "")),
                "task_type": str(row.get("task_type", "")),
                "language": str(row.get("language", "")),
                "channel": str(row.get("channel", channel["channel_id"])),
                "analysis_intervals": str(row.get("analysis_intervals", "[]")),
                "role_filter_required": str(row.get("role_filter_required", "False")).lower() == "true",
                "transcript_reliability": float(row.get("transcript_reliability", 0.0) or 0.0),
                "source_protocol": "8.27 manifest paths; analyzed on demand with current 9.2 feature code",
            }
        )
    output.write_text(json.dumps({"schema_version": "local-cases-v1", "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local-only four-channel case manifest.")
    parser.add_argument("--advoice-root", type=Path, required=True, help="Directory containing the 8.27 and 9.2 project folders.")
    parser.add_argument("--output", type=Path, default=ROOT / "local_cases.json")
    args = parser.parse_args()
    build(args.advoice_root.expanduser().resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
