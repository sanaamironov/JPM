from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Market:
    """
    One market with J products (inside goods only) + outside option implicit.

    Required:
      - s: (J,) inside shares
      - p: (J,) prices
      - w: (J,) observed characteristic

    Optional (only needed for BLP+CostIV benchmark):
      - u: (J,) cost shock (instrument)
    Optional truth/diagnostics:
      - xi_true: (J,) true demand shocks (if simulated)
      - is_signal: (J,) 1 if product is true signal (sparse nonzero eta), else 0
    """

    s: np.ndarray
    p: np.ndarray
    w: np.ndarray
    u: Optional[np.ndarray] = None

    xi_true: Optional[np.ndarray] = None
    is_signal: Optional[np.ndarray] = None

    def __post_init__(self):
        # Normalize shapes
        object.__setattr__(self, "s", np.asarray(self.s, dtype=float).reshape(-1))
        object.__setattr__(self, "p", np.asarray(self.p, dtype=float).reshape(-1))
        object.__setattr__(self, "w", np.asarray(self.w, dtype=float).reshape(-1))

        if self.u is not None:
            object.__setattr__(self, "u", np.asarray(self.u, dtype=float).reshape(-1))
        if self.xi_true is not None:
            object.__setattr__(self, "xi_true", np.asarray(self.xi_true, dtype=float).reshape(-1))
        if self.is_signal is not None:
            object.__setattr__(self, "is_signal", np.asarray(self.is_signal, dtype=int).reshape(-1))

        J = self.s.shape[0]
        if self.p.shape[0] != J or self.w.shape[0] != J:
            raise ValueError("Market arrays must have same length J: s, p, w.")
        if self.u is not None and self.u.shape[0] != J:
            raise ValueError("If provided, u must have length J.")
        if self.xi_true is not None and self.xi_true.shape[0] != J:
            raise ValueError("If provided, xi_true must have length J.")
        if self.is_signal is not None and self.is_signal.shape[0] != J:
            raise ValueError("If provided, is_signal must have length J.")

        if np.any(self.s < 0) or np.any(self.s > 1):
            raise ValueError("Shares s must be in [0,1].")
        if float(np.sum(self.s)) >= 1.0:
            # Outside option share must be positive for log share ratio initialization
            # We allow equality only up to numerical tolerance.
            if float(np.sum(self.s)) > 1.0 - 1e-10:
                raise ValueError("Sum of inside shares must be < 1 (outside option share must be > 0).")


@dataclass(frozen=True)
class MarketShareDataset:
    """
    Collection of markets.
    This is the primary choice-learn integration data container for Lu(25).
    """
    markets: Sequence[Market]

    @property
    def T(self) -> int:
        return len(self.markets)

    @property
    def J(self) -> int:
        if not self.markets:
            return 0
        return int(self.markets[0].s.shape[0])

    def to_markets_dicts(self) -> List[dict]:
        """
        Convert to the dict-of-arrays structure used by your existing jpm_q3.lu25 estimators.
        """
        out: List[dict] = []
        for m in self.markets:
            d = {"s": m.s, "p": m.p, "w": m.w}
            if m.u is not None:
                d["u"] = m.u
            if m.xi_true is not None:
                d["xi_true"] = m.xi_true
            if m.is_signal is not None:
                d["is_signal"] = m.is_signal
            out.append(d)
        return out

    @staticmethod
    def from_markets_dicts(markets: Iterable[dict]) -> "MarketShareDataset":
        """
        Build from list[dict] markets (e.g., produced by jpm_q3.lu25.simulation.simulate_dataset).
        """
        ms: List[Market] = []
        for d in markets:
            ms.append(
                Market(
                    s=d["s"],
                    p=d["p"],
                    w=d["w"],
                    u=d.get("u", None),
                    xi_true=d.get("xi_true", None),
                    is_signal=d.get("is_signal", None),
                )
            )
        return MarketShareDataset(ms)