from __future__ import annotations

from pathlib import Path

from advoice.demo import write_public_demo_bundle


ROOT = Path(__file__).resolve().parent


def main() -> None:
    cases = write_public_demo_bundle(ROOT / "assets", ROOT / "output")
    print(f"Wrote {len(cases)} public synthetic case results to {ROOT / 'output'}")
    print("No clinical diagnosis was generated because the public fixtures are synthetic and unlabeled.")


if __name__ == "__main__":
    main()
