from __future__ import annotations

import argparse
from pathlib import Path

from advoice.demo import write_demo_result


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public, synthetic ADvoice evidence demo.")
    parser.add_argument("--audio", type=Path, default=ROOT / "assets" / "synthetic_picture_description.wav")
    parser.add_argument("--transcript", type=Path, default=ROOT / "assets" / "synthetic_picture_description.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "synthetic_case_result.json")
    args = parser.parse_args()
    result = write_demo_result(args.audio, args.transcript, args.output)
    print(f"Wrote {args.output}")
    print(f"Evidence objects: {len(result['metric_evidence'])}; state cards: {len(result['state_cards'])}")
    print("No clinical diagnosis was generated because this is a synthetic public demo.")


if __name__ == "__main__":
    main()
