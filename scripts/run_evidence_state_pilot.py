#!/usr/bin/env python3
"""CLI for the bounded evidence-state pilot runner.

This command defaults to no paid execution. ``--dry-run`` creates only
structural stage records and stamps fake-provider records as test-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from advoice.pilot.runner import PilotRunner, RunnerError, STAGES


def _config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RunnerError(f"Cannot load runner config: {path}") from exc
    if not isinstance(value, dict):
        raise RunnerError("Runner config must be a YAML mapping.")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded evidence-state pilot.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", choices=("fake", "configured"), default="configured")
    parser.add_argument("--allow-paid", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        runner = PilotRunner(
            _config(args.config), args.run_dir, provider=args.provider,
            allow_paid=args.allow_paid, dry_run=args.dry_run, resume=args.resume,
        )
        records = runner.run(args.stage)
    except RunnerError as exc:
        print(f"pilot runner: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"run_dir": str(args.run_dir), "stages": [
        {"stage": record["stage"], "status": record["status"], "output_hash": record["output_hash"]}
        for record in records
    ]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
