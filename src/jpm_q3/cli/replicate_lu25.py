from __future__ import annotations

import argparse
from pathlib import Path

from jpm_q3.lu25.experiments.replicate_section4 import main as lu25_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replicate Lu(25) Section 4 simulation study.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny, fast smoke test.")
    parser.add_argument("--out", type=str, default="results/lu25", help="Output directory.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lu25_args = ["--out", str(out_dir)]
    if args.smoke:
        lu25_args.append("--smoke")
    return int(lu25_main(lu25_args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
