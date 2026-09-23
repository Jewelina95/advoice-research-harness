"""Prepare small frozen cohorts without labels or external model calls.

This smoke check exercises the real artifact loader, train-only Module A fit,
task/language routing, MetricEvidence handoff, StateCard construction, and
transcript coverage.  It deliberately stops before Agent inference so a
broken data channel is detected without API cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from advoice.authority_study_dataset import AuthorityStudyDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--selection-salt", default="evidence-bound-v5-smoke")
    parser.add_argument("--output", type=Path)
    return parser


def run(artifact_dir: Path, *, max_cases: int, selection_salt: str) -> dict[str, object]:
    started = time.monotonic()
    dataset = AuthorityStudyDataset.from_artifact_dir(artifact_dir)
    cases = dataset.prepare_test_cases(
        max_cases=max_cases,
        order="stable_hash",
        selection_salt=selection_salt,
    )
    rows = []
    for item in cases:
        prepared = item.prepared_case
        rows.append({
            "case_id": prepared.case_id,
            "channel": prepared.case_metadata.get("channel"),
            "language": prepared.case_metadata.get("language"),
            "task_type": prepared.case_metadata.get("task_type"),
            "class_order": list(prepared.route.target_route.labels),
            "evidence_count": len(prepared.evidence),
            "module_a_evidence_count": len(prepared.module_a_evidence),
            "state_count": len(prepared.pre_state_cards),
            "transcript_chars": len(item.transcript),
            "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
            "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
            "reviewed_packet_hash": prepared.reviewed_packet_hash,
        })
    return {
        "status": "passed",
        "dataset_id": dataset.advisor.dataset_id,
        "artifact_dir": str(artifact_dir.expanduser().resolve()),
        "case_count": len(rows),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "external_model_calls": 0,
        "cases": rows,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(
            args.artifact_dir,
            max_cases=args.max_cases,
            selection_salt=args.selection_salt,
        )
    except Exception as error:
        result = {
            "status": "failed",
            "artifact_dir": str(args.artifact_dir.expanduser().resolve()),
            "error": f"{type(error).__name__}: {error}",
            "external_model_calls": 0,
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
