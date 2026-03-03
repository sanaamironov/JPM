from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import numpy as np

from .data_io import MarketShareDataset

# Reuse your existing working routines (single source of truth)
from jpm_q3.lu25.estimators.blp import compute_delta_vec, iv_2sls_beta
from jpm_q3.lu25.estimators.shrinkage import shrinkage_fit_beta_given_sigma


@dataclass
class Lu25FitResult:
    sigma_hat: float
    beta_hat: np.ndarray           # [2] for [beta_p, beta_w] under paper-aligned X=[p,w]
    gamma_prob: Optional[np.ndarray]  # [N] posterior inclusion probs (shrinkage), else None
    score_hat: Optional[float]        # shrinkage score at sigma_hat
    acc_rate: Optional[float]         # MCMC acceptance rate at sigma_hat
    extras: Dict[str, object]         # delta_hat, X, xi_hat, etc.


class Lu25SparseShocksEstimator:
    """
    Choice-learn compatible estimator for Lu & Shimizu (2025) sparse market-product shocks,
    operating on aggregated market shares.

    Design notes:
    - Lu(25) Section 4 is formulated on aggregate shares; this estimator accepts market-level
      shares/prices/covariates.
    - Internally, we:
        1) invert deltas via Berry contraction mapping (through compute_delta_vec)
        2) estimate BLP beta via IV-2SLS (benchmark)
        3) estimate shrinkage regression via TFP-MCMC core (Lu-style alternative)
      and select sigma by grid search.

    This class is intentionally lightweight: it provides fit() and stores results.
    """

    def __init__(
        self,
        *,
        sigma_grid: Optional[np.ndarray] = None,
        R_mc: int = 200,
        base_seed: int = 123,
        # shrinkage defaults (you can override via fit(...))
        shrink_n_iter: int = 800,
        shrink_burn: int = 400,
        shrink_thin: int = 1,
        # RWM proposal scales etc can be passed through fit via shrink_kwargs
    ):
        self.sigma_grid = sigma_grid if sigma_grid is not None else np.linspace(0.05, 4.0, 40)
        self.R_mc = int(R_mc)
        self.base_seed = int(base_seed)

        self.shrink_n_iter = int(shrink_n_iter)
        self.shrink_burn = int(shrink_burn)
        self.shrink_thin = int(shrink_thin)

        self.result_: Optional[Lu25FitResult] = None

    @staticmethod
    def _build_matrices_paper(markets: List[dict], iv_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Paper-aligned stacking:
          X = [p, w] (no constant)
          Z (no constant)
            - cost:   [w, w^2, u, u^2]
            - nocost: [w, w^2, w^3, w^4]
        """
        Xs, Zs = [], []
        for m in markets:
            p = np.asarray(m["p"], dtype=float).reshape(-1)
            w = np.asarray(m["w"], dtype=float).reshape(-1)
            X = np.column_stack([p, w])

            if iv_type == "cost":
                if "u" not in m:
                    raise KeyError("iv_type='cost' requires market['u'].")
                u = np.asarray(m["u"], dtype=float).reshape(-1)
                Z = np.column_stack([w, w**2, u, u**2])
            elif iv_type == "nocost":
                Z = np.column_stack([w, w**2, w**3, w**4])
            else:
                raise ValueError("iv_type must be 'cost' or 'nocost'.")

            Xs.append(X)
            Zs.append(Z)

        return np.vstack(Xs), np.vstack(Zs)

    def fit(
        self,
        data: Union[MarketShareDataset, List[dict]],
        *,
        mode: str = "shrinkage",
        iv_type: str = "cost",
        R_mc: Optional[int] = None,
        sigma_grid: Optional[np.ndarray] = None,
        shrink_kwargs: Optional[Dict[str, object]] = None,
    ) -> Lu25FitResult:
        """
        Fit the model.

        Args:
          data: MarketShareDataset or list-of-market dicts with keys s,p,w,(u)
          mode:
            - "blp": run BLP benchmark only (IV_type required), selects sigma by GMM objective
            - "shrinkage": run shrinkage sigma search (Lu-style), returns gamma_prob
          iv_type: "cost" or "nocost" for BLP benchmark (ignored for shrinkage)
          R_mc: number of simulation draws used in delta inversion
          sigma_grid: override sigma grid
          shrink_kwargs: extra kwargs forwarded to shrinkage_fit_beta_given_sigma
            e.g. v0, v1, a_pi, b_pi, beta_rw_scale, pi_rw_scale, seed, etc.

        Returns:
          Lu25FitResult
        """
        if isinstance(data, MarketShareDataset):
            markets = data.to_markets_dicts()
        else:
            markets = data

        R = int(R_mc) if R_mc is not None else self.R_mc
        grid = np.asarray(sigma_grid, dtype=float) if sigma_grid is not None else np.asarray(self.sigma_grid, dtype=float)

        shrink_kwargs = dict(shrink_kwargs or {})
        # defaults
        shrink_kwargs.setdefault("n_iter", self.shrink_n_iter)
        shrink_kwargs.setdefault("burn", self.shrink_burn)
        shrink_kwargs.setdefault("thin", self.shrink_thin)
        shrink_kwargs.setdefault("seed", self.base_seed)

        if mode not in {"blp", "shrinkage"}:
            raise ValueError("mode must be 'blp' or 'shrinkage'.")

        if mode == "blp":
            return self._fit_blp(markets, iv_type=iv_type, R=R, grid=grid)

        return self._fit_shrinkage(markets, R=R, grid=grid, shrink_kwargs=shrink_kwargs)

    def _fit_blp(self, markets: List[dict], *, iv_type: str, R: int, grid: np.ndarray) -> Lu25FitResult:
        best = None  # (obj, sigma, beta, delta, X, Z, xi_hat)
        for s in grid:
            delta_vec = compute_delta_vec(markets, float(s), R=R)
            X, Z = self._build_matrices_paper(markets, iv_type=iv_type)
            beta = iv_2sls_beta(delta_vec, X, Z).numpy()
            xi_hat = delta_vec - X @ beta

            N = xi_hat.shape[0]
            g = (Z.T @ xi_hat) / N
            W = np.linalg.inv((Z.T @ Z) / N)
            obj = float(g.T @ W @ g)

            if best is None or obj < best[0]:
                best = (obj, float(s), np.asarray(beta, dtype=float), delta_vec, X, Z, xi_hat)

        assert best is not None
        obj_hat, sigma_hat, beta_hat, delta_hat, X, Z, xi_hat = best
        extras = {
            "obj_hat": float(obj_hat),
            "delta_hat": np.asarray(delta_hat, dtype=float),
            "X": np.asarray(X, dtype=float),
            "Z": np.asarray(Z, dtype=float),
            "xi_hat": np.asarray(xi_hat, dtype=float),
        }
        res = Lu25FitResult(
            sigma_hat=float(sigma_hat),
            beta_hat=np.asarray(beta_hat, dtype=float),
            gamma_prob=None,
            score_hat=None,
            acc_rate=None,
            extras=extras,
        )
        self.result_ = res
        return res

    def _fit_shrinkage(
        self,
        markets: List[dict],
        *,
        R: int,
        grid: np.ndarray,
        shrink_kwargs: Dict[str, object],
    ) -> Lu25FitResult:
        best = None  # (score, sigma, beta, gamma_prob, acc_rate, delta, X, xi_hat)

        for s in grid:
            delta_vec = compute_delta_vec(markets, float(s), R=R)
            X, _ = self._build_matrices_paper(markets, iv_type="nocost")  # X only (paper-aligned)

            beta_hat, gamma_prob, score, acc_rate = shrinkage_fit_beta_given_sigma(delta_vec, X, **shrink_kwargs)
            xi_hat = delta_vec - X @ beta_hat

            if best is None or float(score) > best[0]:
                best = (float(score), float(s), np.asarray(beta_hat, float), np.asarray(gamma_prob, float), float(acc_rate), delta_vec, X, xi_hat)

        assert best is not None
        score_hat, sigma_hat, beta_hat, gamma_prob, acc_rate, delta_hat, X, xi_hat = best

        extras = {
            "score_hat": float(score_hat),
            "acc_rate": float(acc_rate),
            "delta_hat": np.asarray(delta_hat, dtype=float),
            "X": np.asarray(X, dtype=float),
            "xi_hat": np.asarray(xi_hat, dtype=float),
        }

        res = Lu25FitResult(
            sigma_hat=float(sigma_hat),
            beta_hat=np.asarray(beta_hat, dtype=float),
            gamma_prob=np.asarray(gamma_prob, dtype=float),
            score_hat=float(score_hat),
            acc_rate=float(acc_rate),
            extras=extras,
        )
        self.result_ = res
        return res