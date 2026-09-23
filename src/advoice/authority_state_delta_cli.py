"""Command-line entry point for resumable authority joint-fusion cohorts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .authority_state_delta_study import (
    AuthorityStateDeltaStudyConfig,
    build_authority_review_runtime,
    run_authority_state_delta_cohort,
)
from .authority_joint_fusion import AuthorityJointFusionConfig
from .authority_review_runtime import (
    REVIEW_MODE_LEGACY_TWO_PASS,
    REVIEW_MODE_SINGLE_BLIND,
)
from .authority_study_dataset import AuthorityStudyDataset
from .config import paths
from .decision_lock import hash_artifact


SUPPORTED_PROVIDERS = ("codex_cli", "openai_api", "disabled")
SUPPORTED_SELECTION_ORDERS = ("longest_first", "subject_id", "stable_hash")
SUPPORTED_REVIEW_MODES = (REVIEW_MODE_SINGLE_BLIND, REVIEW_MODE_LEGACY_TWO_PASS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="advoice-authority-state-delta",
        description="Run a resumable, label-isolated Agent authority joint-fusion cohort.",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--review-mode", choices=SUPPORTED_REVIEW_MODES,
        default=REVIEW_MODE_SINGLE_BLIND,
        help="One blind evidence decision by default; legacy two-pass is reproduction-only.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--selection-order",
        choices=SUPPORTED_SELECTION_ORDERS,
        default="longest_first",
    )
    parser.add_argument("--selection-salt", default="authority-pilot-v1")
    parser.add_argument(
        "--state-strength", type=float, default=0.0,
        help="Development-validated screening correction strength; zero by default.",
    )
    parser.add_argument(
        "--agent-strength", type=float, default=0.0,
        help="Development-validated fallback screening strength; zero by default.",
    )
    parser.add_argument(
        "--staging-strength", type=float, default=0.0,
        help="Separately validated MCI-versus-AD evidence strength; zero until calibrated.",
    )
    parser.add_argument("--ordinal-temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-abs-delta", type=float, default=0.75)
    parser.add_argument("--skill-path", type=Path, default=None)
    parser.add_argument(
        "--calibration-artifact", type=Path, default=None,
        help=(
            "Validated development-set agent_correction_calibration.json. Screening, staging, "
            "and optional state-replay strengths retain separate calibration identities."
        ),
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--decision-only", dest="decision_only", action="store_true", default=True,
        help="Classification and audit only (default); clinician report generation stays off.",
    )
    output_mode.add_argument(
        "--report", dest="decision_only", action="store_false",
        help="Request clinician reporting; currently rejected because reporting is deferred until formal testing.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.decision_only is not True:
        raise ValueError("Clinician report generation is deferred until formal testing; use --decision-only.")
    if not str(args.model).strip():
        raise ValueError("--model must be a non-empty explicit model identifier.")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive when supplied.")
    if not str(args.selection_salt).strip():
        raise ValueError("--selection-salt must be non-empty.")
    if args.alpha is not None and args.state_strength != 0.0:
        raise ValueError("Use --state-strength or legacy --alpha, not both.")
    state_strength = args.state_strength if args.alpha is None else args.alpha
    if not state_strength >= 0.0:
        raise ValueError("--state-strength must be non-negative.")
    if not args.agent_strength >= 0.0:
        raise ValueError("--agent-strength must be non-negative.")
    if not args.staging_strength >= 0.0:
        raise ValueError("--staging-strength must be non-negative.")
    if not args.max_abs_delta > 0.0:
        raise ValueError("--max-abs-delta must be greater than zero.")
    if not args.ordinal_temperature > 0.0:
        raise ValueError("--ordinal-temperature must be greater than zero.")
    if args.calibration_artifact is not None and any(
        value != 0.0 for value in (state_strength, args.agent_strength, args.staging_strength)
    ):
        raise ValueError(
            "Use --calibration-artifact or manual strength flags, not both."
        )
    if args.calibration_artifact is None and any(
        value != 0.0 for value in (state_strength, args.agent_strength, args.staging_strength)
    ):
        raise ValueError(
            "Nonzero fusion strengths require --calibration-artifact; manual test-time strengths are disabled."
        )


def _resolved_strengths(
    args: argparse.Namespace,
) -> tuple[float, float, float, str | None, dict[str, Any] | None]:
    """Resolve only frozen development-set strengths; reject test-time tuning."""

    manual_state = args.state_strength if args.alpha is None else args.alpha
    if args.calibration_artifact is None:
        return (
            float(manual_state), float(args.agent_strength), float(args.staging_strength), None, None,
        )
    path = args.calibration_artifact.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration artifact must be a JSON object.")
    if payload.get("selection_status") != "validated_joint_gain":
        raise ValueError(
            "Calibration artifact must have selection_status=validated_joint_gain."
        )
    screening = payload.get("selected_screening_strength")
    staging = payload.get("selected_staging_strength")
    for name, value in (("selected_screening_strength", screening), ("selected_staging_strength", staging)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Calibration artifact requires numeric {name}.")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"Calibration artifact {name} must be finite and non-negative.")
    state = 0.0
    if payload.get("state_selection_status") == "validated_joint_gain":
        state_value = payload.get("selected_state_strength")
        if isinstance(state_value, bool) or not isinstance(state_value, (int, float)):
            raise ValueError(
                "State calibration requires numeric selected_state_strength."
            )
        if not math.isfinite(float(state_value)) or float(state_value) < 0.0:
            raise ValueError(
                "Calibration artifact selected_state_strength must be finite and non-negative."
            )
        state = float(state_value)
    # Replayed state deltas and ordinal screening scores use different numeric
    # scales. They remain mutually exclusive at inference, but they cannot share
    # a coefficient unless each route was calibrated on development data.
    return state, float(screening), float(staging), hash_artifact(payload), payload


def _summary(result: Any, *, provider: str, model: str, review_mode: str) -> dict[str, Any]:
    return {
        "status": "completed" if not result.failed_case_ids else "failed",
        "provider": provider,
        "model": model,
        "review_mode": review_mode,
        "decision_only": True,
        "report_generation": "deferred",
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
        review_mode=args.review_mode,
    )
    (
        state_strength,
        agent_strength,
        staging_strength,
        calibration_hash,
        calibration_artifact,
    ) = _resolved_strengths(args)
    result = run_authority_state_delta_cohort(
        dataset,
        runtime,
        output_dir=output_dir,
        cache_dir=cache_dir,
        decision_only=args.decision_only,
        config=AuthorityStateDeltaStudyConfig(
            joint_fusion=AuthorityJointFusionConfig(
                state_strength=state_strength,
                agent_strength=agent_strength,
                staging_strength=staging_strength,
                max_abs_state_delta=args.max_abs_delta,
                ordinal_temperature=args.ordinal_temperature,
            ),
            max_cases=args.max_cases,
            selection_order=args.selection_order,
            selection_salt=args.selection_salt,
            calibration_artifact_hash=calibration_hash,
            calibration_artifact=calibration_artifact,
        ),
    )
    summary = _summary(
        result, provider=args.provider, model=args.model, review_mode=args.review_mode,
    )
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
