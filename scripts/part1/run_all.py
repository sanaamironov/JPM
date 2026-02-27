from __future__ import annotations

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


def repo_root() -> Path:
    """Find repo root by walking upward to pyproject.toml, starting from this file."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # fallback: assume scripts/part1 is under repo_root/scripts/part1
    return here.parents[2]


def results_dir() -> Path:
    d = repo_root() / "results" / "part1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def run_experiment(exp: Experiment, log_file: Path, env: dict, python_exe: str) -> int:
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
            cwd=str(repo_root()),  # run from repo root
            env=env,
            text=True,
        )
        f.write(f"\n[EXIT CODE] {proc.returncode}\n")
        f.flush()

    return proc.returncode


def main() -> int:
    root = repo_root()
    log_file = results_dir() / "run_all.log"

    # IMPORTANT: these must match your new src layout modules.
    experiments: List[Experiment] = [
        Experiment("reproduce_table1", "jpm_q3.zhang25.experiments.reproduce_table1"),
        Experiment("influence_map", "jpm_q3.zhang25.experiments.influence_map"),
        Experiment("attraction_effect_tf", "jpm_q3.zhang25.experiments.attraction_effect_tf"),
        Experiment("decoy_effect", "jpm_q3.zhang25.experiments.decoy_effect"),
        Experiment("compromise_effect_tf", "jpm_q3.zhang25.experiments.compromise_effect_tf"),

        # checking the authors code compare to mine
        # Experiment("attraction_effect_torch", "jpm_q3.zhang25.experiments.attraction_effect_torch"),
    ]

    python_exe = sys.executable

    env = dict(os.environ)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all logs, 2=warnings+errors

    # Start fresh log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(
            "Part 1 run_all log\n"
            f"ROOT: {root}\n"
            f"PYTHON: {python_exe}\n"
            f"TIME: {datetime.now().isoformat()}\n"
        )

    print(f"Logging to: {log_file}")
    print(f"Running {len(experiments)} experiments from repo root: {root}")

    failures = []
    for exp in experiments:
        print(f"-> {exp.name} ...", flush=True)
        rc = run_experiment(exp, log_file=log_file, env=env, python_exe=python_exe)
        if rc != 0:
            failures.append((exp.name, rc))
            print(f"   FAILED (rc={rc})")
        else:
            print("   OK")

    print("\nOutputs are written under:")
    print(f"  {results_dir()}")
    print(f"  {results_dir() / 'figures'}")

    if failures:
        print("\nSome experiments failed:")
        for name, rc in failures:
            print(f"  - {name}: rc={rc}")
        print(f"\nSee log: {log_file}")
        return 1

    print("\nAll experiments completed successfully.")
    print(f"See log: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())