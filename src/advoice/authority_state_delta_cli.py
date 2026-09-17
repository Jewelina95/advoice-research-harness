"""Command-line entry point for resumable authority state-delta cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .authority_state_delta_study import (
    AuthorityStateDeltaStudyConfig,
    build_authority_review_runtime,
    run_authority_state_delta_cohort,
)
from .authority_study_dataset import AuthorityStudyDataset
from .config import paths
from .condition_c_delta import DeltaFusionConfig


SUPPORTED_PROVIDERS = ("codex_cli", "openai_api", "disabled")
SUPPORTED_SELECTION_ORDERS = ("longest_first", "subject_id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="advoice-authority-state-delta",
        description="Run a resumable, label-isolated Agent authority state-delta cohort.",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--selection-order",
        choices=SUPPORTED_SELECTION_ORDERS,
        default="longest_first",
    )
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max-abs-delta", type=float, default=0.75)
    parser.add_argument("--skill-path", type=Path, default=None)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.model).strip():
        raise ValueError("--model must be a non-empty explicit model identifier.")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive when supplied.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1].")
    if not args.max_abs_delta > 0.0:
        raise ValueError("--max-abs-delta must be greater than zero.")


def _summary(result: Any, *, provider: str, model: str) -> dict[str, Any]:
    return {
        "status": "completed" if not result.failed_case_ids else "failed",
        "provider": provider,
        "model": model,
        "study_hash": result.study_hash,
        "attempted_cases": len(result.attempted_case_ids),
        "completed_cases": len(result.completed_case_ids),
        "failed_cases": len(result.failed_case_ids),
        "frozen_metrics": result.frozen_metrics,
        "fused_metrics": result.fused_metrics,
        "paired_counts": dict(result.paired_counts),
        "artifacts": {
            "audit_jsonl": str(result.audit_jsonl_path),
            "audit_json": str(result.audit_json_path),
            "aggregate_json": str(result.aggregate_json_path),
        },
    }


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Run one cohort and return a process status plus a JSON-safe summary."""

    _validate_args(args)
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"--artifact-dir is not a directory: {artifact_dir}")
    dataset = AuthorityStudyDataset.from_artifact_dir(artifact_dir)
    runtime = build_authority_review_runtime(
        root=paths().root,
        provider=args.provider,
        model=args.model,
        skill_path=None if args.skill_path is None else args.skill_path.expanduser().resolve(),
        cache_dir=cache_dir,
    )
    result = run_authority_state_delta_cohort(
        dataset,
        runtime,
        output_dir=output_dir,
        cache_dir=cache_dir,
        config=AuthorityStateDeltaStudyConfig(
            delta_fusion=DeltaFusionConfig(
                alpha=args.alpha,
                max_abs_delta=args.max_abs_delta,
            ),
            max_cases=args.max_cases,
            selection_order=args.selection_order,
        ),
    )
    summary = _summary(result, provider=args.provider, model=args.model)
    return (0 if not result.failed_case_ids else 1), summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        status, summary = run(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run"]
