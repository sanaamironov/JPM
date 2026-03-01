"""Dataset simulator for Lu & Shimizu (2025) replication harness."""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .market import simulate_market


def simulate_dataset(
    dgp: str,
    T: int,
    J: int,
    cfg: SimConfig,
    seed: int = 123,
) -> list[dict]:
    """
    Simulate a list of T markets.

    Contract:
      Each returned market dict must contain the observed objects needed by estimators:
        - "s": observed shares, shape (J,)
        - "p": prices, shape (J,)
        - "w": observed characteristic, shape (J,)
      Additionally, for paper-style reporting in Section 4 we preserve (if provided):
        - "xi_true" (or "xi"): true demand shock xi_{jt}, shape (J,)
        - "is_signal": indicator for sparse signal positions, shape (J,), values in {0,1}

    Notes:
      - simulate_market is responsible for generating and attaching xi_true/is_signal when available.
      - We attach "t" market index for easier stacking/debugging.
    """
    rng = np.random.default_rng(seed)
    markets: list[dict] = []

    for t in range(T):
        m = simulate_market(dgp=dgp, J=J, cfg=cfg, rng=rng)

        # Defensive checks for required keys (fail fast if simulator breaks)
        for k in ("s", "p", "w"):
            if k not in m:
                raise KeyError(f"simulate_market must return key '{k}'. Got keys: {list(m.keys())}")

        # Attach market index for downstream reshaping / debugging
        m["t"] = t

        # Normalize naming for reporting:
        # prefer m["xi_true"]; allow legacy m["xi"]
        if "xi_true" not in m and "xi" in m:
            m["xi_true"] = m["xi"]

        # Ensure is_signal is int array if present
        if "is_signal" in m:
            m["is_signal"] = np.asarray(m["is_signal"], dtype=int)

        markets.append(m)

    return markets