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
        "channel_name_zh": "临床访谈",
        "task_name_zh": "西班牙语结构化医患访谈",
        "description_zh": "保留患者与访谈者角色以及问答轮次，重点观察回答启动、患者发言占比、停顿和互动负担。",
        "evidence_focus_zh": ["患者语音范围", "回答启动与停顿", "对话轮次", "词汇与输出效率"],
        "task_filter": None,
    },
    {
        "dataset_id": "ADReSS_2020",
        "channel_id": "picture_description",
        "channel_name_zh": "图片描述",
        "task_name_zh": "Cookie Theft 标准图片描述",
        "description_zh": "同一图片和提示条件下比较表达，重点观察内容单元、信息密度、词汇提取、句法和流畅性。",
        "evidence_focus_zh": ["内容单元", "信息密度", "词汇提取", "停顿与连续性"],
        "task_filter": None,
    },
    {
        "dataset_id": "PROCESS_2",
        "channel_id": "structured_multitask",
        "channel_name_zh": "结构化认知任务",
        "task_name_zh": "PROCESS-2 多任务认知语音",
        "description_zh": "数据集同时包含图片描述、语音流畅性等任务；本例播放其中一个任务，系统保留任务身份，避免先平均后融合。",
        "evidence_focus_zh": ["任务身份", "任务内输出", "语义流畅性", "跨任务状态"],
        "task_filter": "ctd",
    },
    {
        "dataset_id": "DementiaNet_PublicFigures",
        "channel_id": "public_speech",
        "channel_name_zh": "自然公开讲话",
        "task_name_zh": "非标准化自然讲话",
        "description_zh": "讲话主题、设备和场景不可控，因此主要用于外部泛化与质量审查，不强行套用标准任务评分。",
        "evidence_focus_zh": ["自然语流", "录音质量", "稳健声学行为", "外部泛化"],
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
                "channel_name_zh": channel["channel_name_zh"],
                "task_name_zh": channel["task_name_zh"],
                "description_zh": channel["description_zh"],
                "evidence_focus_zh": channel["evidence_focus_zh"],
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
