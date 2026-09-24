#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from advoice.pilot.data import ManifestError, freeze_actual_pilot_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze evidence-state pilot data artifacts.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = freeze_actual_pilot_data(raw_root=args.raw_root, historical_root=args.historical_root,
                                         output_dir=args.output_dir)
    except ManifestError as exc:
        print(f"pilot data freeze blocked: {exc}", file=sys.stderr)
        return 2
    print(paths["sha256_manifest.json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
