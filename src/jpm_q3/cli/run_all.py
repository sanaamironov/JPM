from __future__ import annotations

import argparse
from pathlib import Path

from jpm_q3.lu25.experiments.replicate_section4 import main as lu25_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all Q3 Part 2 experiments (smoke by default).")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny, fast smoke test.")
    parser.add_argument("--out", type=str, default="results", help="Output directory (default: results/).")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # For now, run the Lu(25) Section 4 replication driver.
    # Hybrid experiments can be added here later.
    lu25_args = ["--out", str(out_dir / "lu25")]
    if args.smoke:
        lu25_args.append("--smoke")
    return int(lu25_main(lu25_args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
