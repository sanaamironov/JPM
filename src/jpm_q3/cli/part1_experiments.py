from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Experiment:
    name: str
    module: str
    args: Optional[List[str]] = None


DEFAULT_EXPERIMENTS: List[Experiment] = [
    Experiment("reproduce_table1", "jpm_q3.zhang25.experiments.reproduce_table1"),
    Experiment("influence_map", "jpm_q3.zhang25.experiments.influence_map"),
    Experiment(
        "attraction_effect_tf", "jpm_q3.zhang25.experiments.attraction_effect_tf"
    ),
    Experiment("decoy_effect", "jpm_q3.zhang25.experiments.decoy_effect"),
    Experiment(
        "compromise_effect_tf", "jpm_q3.zhang25.experiments.compromise_effect_tf"
    ),
    # Optional/disabled by default:
    # Experiment("attraction_effect_torch", "jpm_q3.zhang25.experiments.attraction_effect_torch"),
]


def _default_out_dir() -> Path:
    # Use CWD, because reviewers will run from repo root. Still works elsewhere.
    return Path("results") / "part1"


def _ensure_out_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)


def _run_experiment(
    exp: Experiment,
    *,
    log_file: Path,
    env: dict,
    python_exe: str,
    cwd: Path,
) -> int:
    """
    Runs one experiment module as a subprocess and appends stdout/stderr to log_file.
    Returns the process return code.
    """
    cmd = [python_exe, "-m", exp.module] + (exp.args or [])

    header = (
        f"\n\n{'=' * 90}\n"
        f"RUN:    {exp.name}\n"
        f"MODULE: {exp.module}\n"
        f"CMD:    {' '.join(cmd)}\n"
        f"TIME:   {datetime.now().isoformat()}\n"
        f"{'=' * 90}\n"
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            env=env,
            text=True,
        )
        f.write(f"\n[EXIT CODE] {proc.returncode}\n")
        f.flush()

    return proc.returncode


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Part 1 experiments (Zhang 2025 / DeepHalo) and log outputs.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(_default_out_dir()),
        help="Output directory (default: results/part1).",
    )
    p.add_argument(
        "--log",
        type=str,
        default="run_part1_experiments.log",
        help="Log filename written under --out (default: run_part1_experiments.log).",
    )
    p.add_argument(
        "--cwd",
        type=str,
        default=".",
        help="Working directory to run experiments from (default: current directory).",
    )
    p.add_argument(
        "--tf-log-level",
        type=str,
        default="2",
        choices=["0", "1", "2", "3"],
        help="TF_CPP_MIN_LOG_LEVEL (0=all, 2=warnings+errors). Default: 2.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a minimal subset suitable for quick validation.",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of experiment names to run (matches names in the report).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    out_dir = Path(args.out)
    _ensure_out_dirs(out_dir)
    log_file = out_dir / args.log

    cwd = Path(args.cwd).resolve()

    experiments = list(DEFAULT_EXPERIMENTS)

    if args.smoke:
        # Keep this short; pick 1–2 representative experiments.
        # Adjust if you prefer different smoke coverage.
        experiments = [
            Experiment(
                "reproduce_table1", "jpm_q3.zhang25.experiments.reproduce_table1"
            ),
        ]

    if args.only.strip():
        want = {t.strip() for t in args.only.split(",") if t.strip()}
        experiments = [e for e in experiments if e.name in want]
        missing = want - {e.name for e in experiments}
        if missing:
            print(
                f"[WARN] --only requested unknown experiments: {sorted(missing)}",
                file=sys.stderr,
            )

    if not experiments:
        print("[ERROR] No experiments selected.", file=sys.stderr)
        return 2

    python_exe = sys.executable

    env = dict(os.environ)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", args.tf_log_level)

    # Fresh log file
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            "Part 1 experiments log\n"
            f"CWD: {cwd}\n"
            f"PYTHON: {python_exe}\n"
            f"OUT: {out_dir}\n"
            f"TIME: {datetime.now().isoformat()}\n"
        )

    print(f"[part1] logging to: {log_file}")
    print(f"[part1] running {len(experiments)} experiments (cwd={cwd})")

    failures = []
    for exp in experiments:
        print(f"[part1] -> {exp.name} ...", flush=True)
        rc = _run_experiment(
            exp, log_file=log_file, env=env, python_exe=python_exe, cwd=cwd
        )
        if rc != 0:
            failures.append((exp.name, rc))
            print(f"[part1]    FAILED (rc={rc})")
        else:
            print("[part1]    OK")

    print(f"\n[part1] outputs under: {out_dir}")
    print(f"[part1] figures under: {out_dir / 'figures'}")

    if failures:
        print("\n[part1] some experiments failed:", file=sys.stderr)
        for name, rc in failures:
            print(f"  - {name}: rc={rc}", file=sys.stderr)
        print(f"[part1] see log: {log_file}", file=sys.stderr)
        return 1

    print(f"\n[part1] all experiments completed successfully. see log: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
