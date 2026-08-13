from __future__ import annotations

import argparse
import json

from .pipeline import clean_cache, rebuild_latest_report, run_pipeline, validate_dataset


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="advoice", description="ADvoice reproducible research harness")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset", default="NCMMSC2021_AD")
    run = commands.add_parser("run")
    run.add_argument("--dataset", default="NCMMSC2021_AD")
    run.add_argument("--mode", choices=["quick", "full"], default="quick")
    run.add_argument("--agent-provider", choices=["disabled", "codex_cli"], default="disabled")
    run.add_argument("--force", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--dataset", default="NCMMSC2021_AD")
    clean = commands.add_parser("clean-cache")
    clean.add_argument("--dataset", default="NCMMSC2021_AD")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate":
        result = validate_dataset(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if args.command == "run":
        output = run_pipeline(args.dataset, args.mode, args.agent_provider, args.force)
        print(output)
        return
    if args.command == "report":
        print(rebuild_latest_report(args.dataset))
        return
    if args.command == "clean-cache":
        clean_cache(args.dataset)
        return

