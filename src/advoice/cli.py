from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .aggregate_reporting import build_aggregate_report
from .config import load_all, paths
from .workspace import LOCK_TOKEN_ENV, workspace_lock
from .pipeline import (
    clean_cache,
    rebuild_latest_report,
    reevaluate_all,
    reevaluate_dataset,
    run_all_pipelines,
    run_all_processed_pipelines,
    run_pipeline,
    run_processed_pipeline,
    validate_dataset,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="advoice", description="ADvoice reproducible research harness")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset", default="NCMMSC2021_AD")
    run = commands.add_parser("run")
    run.add_argument("--dataset", default="NCMMSC2021_AD")
    run.add_argument("--mode", choices=["quick", "full"], default="quick")
    run.add_argument("--agent-provider", choices=["disabled", "codex_cli", "openai_api"], default="disabled")
    run.add_argument("--force", action="store_true")
    run_all = commands.add_parser("run-all")
    run_all.add_argument("--mode", choices=["quick", "full"], default="quick")
    run_all.add_argument("--agent-provider", choices=["disabled", "codex_cli", "openai_api"], default="disabled")
    run_all.add_argument("--force", action="store_true")
    run_all.add_argument("--datasets", nargs="*")
    processed = commands.add_parser("run-processed")
    processed.add_argument("--dataset", default="NCMMSC2021_AD")
    processed.add_argument("--agent-provider", choices=["disabled", "codex_cli", "openai_api"], default="disabled")
    processed.add_argument("--force", action="store_true")
    processed.add_argument("--source-root", type=Path, help="Directory containing frozen per-dataset artifacts")
    processed_all = commands.add_parser("run-all-processed")
    processed_all.add_argument("--agent-provider", choices=["disabled", "codex_cli", "openai_api"], default="disabled")
    processed_all.add_argument("--force", action="store_true")
    processed_all.add_argument("--datasets", nargs="*")
    processed_all.add_argument("--source-root", type=Path)
    aggregate = commands.add_parser("aggregate-report")
    aggregate.add_argument("--datasets", nargs="+")
    experiment = commands.add_parser("experiment", help="Run a versioned recipe with mandatory Layer A/B reports")
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--check", action="store_true", help="Validate input presence without running models")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--dataset", default="NCMMSC2021_AD")
    evaluate_all = commands.add_parser("evaluate-all")
    evaluate_all.add_argument("--datasets", nargs="*")
    report = commands.add_parser("report")
    report.add_argument("--dataset", default="NCMMSC2021_AD")
    clean = commands.add_parser("clean-cache")
    clean.add_argument("--dataset", default="NCMMSC2021_AD")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "experiment":
        _dispatch(args)
        return
    with workspace_lock(paths().workspace) as token:
        previous = os.environ.get(LOCK_TOKEN_ENV)
        os.environ[LOCK_TOKEN_ENV] = token
        try:
            _dispatch(args)
        finally:
            if previous is None:
                os.environ.pop(LOCK_TOKEN_ENV, None)
            else:
                os.environ[LOCK_TOKEN_ENV] = previous


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "experiment":
        from .experiments import run_experiment
        result = run_experiment(args.config, args.check)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] == "failed":
            raise SystemExit(1)
        return
    if args.command == "validate":
        result = validate_dataset(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if args.command == "run":
        output = run_pipeline(args.dataset, args.mode, args.agent_provider, args.force)
        print(output)
        return
    if args.command == "run-all":
        print(run_all_pipelines(args.mode, args.agent_provider, args.force, args.datasets))
        return
    if args.command == "run-processed":
        print(run_processed_pipeline(args.dataset, args.agent_provider, args.force, args.source_root))
        return
    if args.command == "run-all-processed":
        print(run_all_processed_pipelines(args.agent_provider, args.force, args.datasets, args.source_root))
        return
    if args.command == "aggregate-report":
        dataset_ids = args.datasets or [str(value) for value in load_all("NCMMSC2021_AD")["project"]["default_datasets"]]
        print(build_aggregate_report(paths(), dataset_ids))
        return
    if args.command == "evaluate":
        print(reevaluate_dataset(args.dataset))
        return
    if args.command == "evaluate-all":
        print(reevaluate_all(args.datasets))
        return
    if args.command == "report":
        print(rebuild_latest_report(args.dataset))
        return
    if args.command == "clean-cache":
        clean_cache(args.dataset)
        return
