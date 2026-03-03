"""
CLI wrapper for Lu & Shimizu (2025) Section 4 replication.

What this file is:
- A thin, reviewer-friendly command-line entry point.
- It delegates all work to the canonical replication driver:
    jpm_q3.lu25.experiments.replicate_section4

What it does:
- Creates the output directory (unless overridden downstream).
- Optionally runs a small smoke test via --smoke.
- Forwards any additional arguments to the underlying runner (e.g., --R-mc, --n-reps, --seed,
  shrinkage knobs, multiprocessing knobs).
- Prints high-level progress messages so it’s obvious the program is doing work.

Noise control:
- Suppresses TensorFlow INFO/WARN device logs by default (e.g., Metal device banner).
  If you want full TF logs, set environment variable JPM_TF_LOG_LEVEL=0 before running.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from jpm_q3.lu25.experiments.replicate_section4 import main as lu25_main


def _configure_tensorflow_logging() -> None:
    """
    Reduce TensorFlow's startup/device log spam (Metal device lines, etc.).

    TF uses:
      TF_CPP_MIN_LOG_LEVEL = 0 (all), 1 (filter INFO), 2 (filter INFO+WARNING), 3 (filter all except ERROR)
    We default to 2 unless user overrides via JPM_TF_LOG_LEVEL.
    """
    if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        return  # user already set it

    # Reviewer-friendly default: suppress INFO + WARNING device banners.
    # Allow override via JPM_TF_LOG_LEVEL, e.g. JPM_TF_LOG_LEVEL=0 to see everything.
    lvl = os.environ.get("JPM_TF_LOG_LEVEL", "2").strip()
    if lvl not in {"0", "1", "2", "3"}:
        lvl = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = lvl

    # Optional: reduce absl logging verbosity sometimes used by TF.
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")


def main(argv: list[str] | None = None) -> int:
    _configure_tensorflow_logging()

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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra wrapper-level progress messages.",
    )

    args, passthrough = parser.parse_known_args(argv)

    # If user passed --out through passthrough, don't inject our own.
    passthrough_has_out = any(x == "--out" for x in passthrough)

    out_dir = Path(args.out)
    if not passthrough_has_out:
        out_dir.mkdir(parents=True, exist_ok=True)

    lu25_args: list[str] = []
    if not passthrough_has_out:
        lu25_args.extend(["--out", str(out_dir)])

    if args.smoke and "--smoke" not in passthrough:
        lu25_args.append("--smoke")

    lu25_args.extend(passthrough)

    start = time.time()
    print(f"[lu25] starting replication driver", file=sys.stderr)
    if not passthrough_has_out:
        print(f"[lu25] output dir: {out_dir}", file=sys.stderr)
    if args.verbose:
        print(f"[lu25] forwarded args: {' '.join(lu25_args)}", file=sys.stderr)

    rc = int(lu25_main(lu25_args))

    elapsed = time.time() - start
    print(f"[lu25] done (exit={rc}) in {elapsed:.1f}s", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
