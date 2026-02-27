from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """
    Best-effort project root for a source checkout.

    Priority:
    1) JPM_Q3_ROOT env var
    2) find pyproject.toml by walking up from CWD (not from this file)
    3) fallback to CWD
    """
    env = os.getenv("JPM_Q3_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def results_dir(out_dir: str | Path | None = None) -> Path:
    """
    Results directory.

    Priority:
    1) explicit out_dir argument (recommended: pass from CLI)
    2) JPM_Q3_RESULTS_DIR env var
    3) <project_root>/results
    """
    if out_dir is not None:
        base = Path(out_dir).expanduser().resolve()
    else:
        env = os.getenv("JPM_Q3_RESULTS_DIR")
        base = Path(env).expanduser().resolve() if env else (project_root() / "results")

    base.mkdir(parents=True, exist_ok=True)
    return base


def figures_dir(out_dir: str | Path | None = None) -> Path:
    d = results_dir(out_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return dc