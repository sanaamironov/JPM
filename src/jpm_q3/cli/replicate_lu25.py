from __future__ import annotations

import argparse
from pathlib import Path

from jpm_q3.lu25.experiments.replicate_section4 import main as lu25_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replicate Lu(25) Section 4 simulation study (wrapper).",
        add_help=True,
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny, fast smoke test.")
    parser.add_argument(
        "--out",
        type=str,
        default="results/part2/lu25_section4",
        help="Output directory (default: results/part2/lu25_section4).",
    )

    # Forward any additional args to the underlying runner, so we don't duplicate flags here.
    args, passthrough = parser.parse_known_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lu25_args = ["--out", str(out_dir)]
    if args.smoke:
        lu25_args.append("--smoke")

    # forward everything else (e.g., --R-mc, --seed, --grid, shrinkage knobs)
    lu25_args.extend(passthrough)

    return int(lu25_main(lu25_args))


if __name__ == "__main__":
    raise SystemExit(main())