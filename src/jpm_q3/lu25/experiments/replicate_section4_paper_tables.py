"""
replicate_section4_paper_tables.py

Paper-aligned replication driver for Lu(25) simulation study (Section 4),
with outputs formatted to match the paper tables as closely as possible.

Key additions vs replicate_section4.py:
- Always tracks intercept ("Int") along with beta_p, beta_w, sigma.
- Attempts to compute xi metrics and posterior Prob(signal/noise) if simulation/estimators
  expose the necessary quantities.
- Writes both:
    (1) summary_long.csv  (wide/long hybrid, one row per method+param+stat)
    (2) paper_table_like.csv (Bias/SD rows matching the paper table layout)

This file is robust: if xi/prob inputs are unavailable, it writes NaN (does not crash).

Usage:
  python -m jpm_q3.lu25.experiments.replicate_section4_paper_tables --cpu --out results/part2/section4_full
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..estimators.blp import estimate_blp_sigma
from ..estimators.shrinkage import estimate_shrinkage_sigma
from ..simulation.config import SimConfig
from ..simulation.simulate import simulate_dataset


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class StudyConfig:
    R_mc: int = 50
    seed: int = 123

    shrink_n_iter: int = 800
    shrink_burn: int = 400
    shrink_v0: float = 1e-4
    shrink_v1: float = 1.0


DEFAULT_GRID: List[Tuple[str, int, int]] = [
    ("DGP1", 25, 15),
    ("DGP2", 25, 15),
    ("DGP3", 25, 15),
    ("DGP4", 25, 15),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


class TeeLogger:
    def __init__(self, log_path: Path):
        self._fh = open(log_path, "w", encoding="utf-8")

    def write(self, msg: str) -> None:
        print(msg, end="")
        self._fh.write(msg)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def inject_market_size(markets: list[dict], cfg: SimConfig) -> None:
    Nt = getattr(cfg, "Nt", None)
    if Nt is None:
        return
    for m in markets:
        m["N"] = int(Nt)


def _warn(prefix: str, e: Exception) -> None:
    print(f"\n[WARN] {prefix}: {type(e).__name__}: {e}")
    print(traceback.format_exc(limit=3))


def init_storage(R: int) -> Dict[str, np.ndarray]:
    """
    Stores scalar results per replication.
    """
    return {
        "int": np.full(R, np.nan),
        "beta_p": np.full(R, np.nan),
        "beta_w": np.full(R, np.nan),
        "sigma": np.full(R, np.nan),

        # xi metrics (paper reports bias/SD of xi_{jt}; we store per-rep scalar summaries)
        "xi_abs_bias": np.full(R, np.nan),
        "xi_sd": np.full(R, np.nan),

        # paper's "Prob." column typically is two numbers: P(gamma=1|signal) and P(gamma=1|noise)
        "prob_signal": np.full(R, np.nan),
        "prob_noise": np.full(R, np.nan),

        "fail": np.zeros(R, dtype=int),
    }


def summarize_param(x: np.ndarray, true_val: float) -> Dict[str, float]:
    """
    mean, bias, sd, rmse, n for a parameter.
    """
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": np.nan, "bias": np.nan, "sd": np.nan, "rmse": np.nan, "n": 0}
    mean = float(np.mean(x))
    bias = float(mean - true_val)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    rmse = float(np.sqrt(bias * bias + sd * sd))
    return {"mean": mean, "bias": bias, "sd": sd, "rmse": rmse, "n": int(x.size)}


def summarize_metric(x: np.ndarray) -> Dict[str, float]:
    """
    mean and sd for a scalar metric already defined as an error summary (e.g., xi_abs_bias).
    """
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": np.nan, "sd": np.nan, "n": 0}
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": int(x.size)}


def print_table(title: str, summ: Dict[str, Dict[str, float]], true: Dict[str, float]) -> None:
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)
    print(f"{'Param':<8} {'True':>10} {'Mean':>10} {'Bias':>10} {'SD':>10} {'RMSE':>10} {'n':>6}")
    print("-" * 90)
    order = [("int", "Int"), ("beta_p", "β_p"), ("beta_w", "β_w"), ("sigma", "σ")]
    for key, label in order:
        s = summ[key]
        tv = true[key]
        print(
            f"{label:<8} {tv:>10.4f} {s['mean']:>10.4f} {s['bias']:>10.4f} {s['sd']:>10.4f} "
            f"{s['rmse']:>10.4f} {s['n']:>6d}"
        )


def write_summary_long(path: Path, cell: str, method: str, param: str, stats: Dict[str, float]) -> None:
    """
    Long-ish format. One row per (cell, method, param).
    """
    header = "cell,method,param,mean,bias,sd,rmse,n\n"
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(
            f"{cell},{method},{param},"
            f"{stats.get('mean', np.nan)},{stats.get('bias', np.nan)},{stats.get('sd', np.nan)},"
            f"{stats.get('rmse', np.nan)},{stats.get('n', 0)}\n"
        )


def write_paper_table_like(
    path: Path,
    cell: str,
    method: str,
    bias_row: Dict[str, float],
    sd_row: Dict[str, float],
    xi_bias: float,
    xi_sd: float,
    prob_signal: float,
    prob_noise: float,
) -> None:
    """
    Paper-like format: 2 rows per (cell, method): Bias and SD.
    Columns: Int, beta_p, beta_w, sigma, xi, Prob_signal, Prob_noise
    """
    header = "cell,method,row,Int,beta_p,beta_w,sigma,xi,Prob_signal,Prob_noise\n"
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(
            f"{cell},{method},Bias,"
            f"{bias_row['int']},{bias_row['beta_p']},{bias_row['beta_w']},{bias_row['sigma']},"
            f"{xi_bias},{prob_signal},{prob_noise}\n"
        )
        f.write(
            f"{cell},{method},SD,"
            f"{sd_row['int']},{sd_row['beta_p']},{sd_row['beta_w']},{sd_row['sigma']},"
            f"{xi_sd},{np.nan},{np.nan}\n"
        )


# -----------------------------------------------------------------------------
# Estimator runners
# -----------------------------------------------------------------------------

def run_blp(markets: list[dict], cfg: SimConfig, iv_type: str):
    # expected: (sigma_hat, beta_hat, extras)
    return estimate_blp_sigma(markets, iv_type=iv_type, R=cfg.R0)


def run_shrinkage(markets: list[dict], study: StudyConfig, cfg: SimConfig):
    # expected: (sigma_hat, beta_hat, score_hat, gamma_prob, extras) since return_extras=True
    return estimate_shrinkage_sigma(
        markets,
        R=cfg.R0,
        n_iter=study.shrink_n_iter,
        burn=study.shrink_burn,
        v0=study.shrink_v0,
        v1=study.shrink_v1,
        return_extras=True,
    )


# -----------------------------------------------------------------------------
# Core helpers for xi/prob metrics
# -----------------------------------------------------------------------------

def _extract_true_xi_and_signal_mask(markets: list[dict], T: int, J: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Try to extract:
      - xi_true: stacked shape (T*J,)
      - is_signal: stacked shape (T*J,) boolean/int mask for sparse cases

    Expected optional keys per market dict:
      - "xi_true" or "xi" as shape (J,)
      - "is_signal" as shape (J,) (1 if true nonzero shock, else 0)
    """
    xi_list = []
    sig_list = []
    have_xi = True
    have_sig = True

    for m in markets:
        if "xi_true" in m:
            xi_list.append(np.asarray(m["xi_true"], dtype=float))
        elif "xi" in m:
            xi_list.append(np.asarray(m["xi"], dtype=float))
        else:
            have_xi = False

        if "is_signal" in m:
            sig_list.append(np.asarray(m["is_signal"], dtype=int))
        else:
            have_sig = False

    xi_true = None
    is_signal = None

    if have_xi and len(xi_list) == T:
        xi_true = np.concatenate(xi_list, axis=0)
        if xi_true.shape != (T * J,):
            xi_true = None

    if have_sig and len(sig_list) == T:
        is_signal = np.concatenate(sig_list, axis=0)
        if is_signal.shape != (T * J,):
            is_signal = None

    return xi_true, is_signal


def _compute_xi_metrics(extras: Dict[str, Any], beta_hat: np.ndarray, xi_true: Optional[np.ndarray]) -> Tuple[float, float]:
    """
    xi_hat = delta_hat - X @ beta_hat. Requires extras to contain delta_hat and X.
    Returns (mean_abs_error, sd_error).
    """
    if xi_true is None:
        return np.nan, np.nan
    if ("delta_hat" not in extras) or ("X" not in extras):
        return np.nan, np.nan

    delta_hat = np.asarray(extras["delta_hat"], dtype=float).reshape(-1)
    X = np.asarray(extras["X"], dtype=float)
    if X.ndim != 2 or delta_hat.ndim != 1:
        return np.nan, np.nan

    xi_hat = delta_hat - X @ beta_hat
    if xi_hat.shape != xi_true.shape:
        return np.nan, np.nan

    err = xi_hat - xi_true
    return float(np.mean(np.abs(err))), float(np.std(err, ddof=1))


def _compute_prob_metrics(gamma_prob: Optional[np.ndarray], is_signal: Optional[np.ndarray]) -> Tuple[float, float]:
    """
    Paper-style Prob columns:
      Prob_signal = E[P(gamma=1) | true signal]
      Prob_noise  = E[P(gamma=1) | true noise]
    """
    if gamma_prob is None or is_signal is None:
        return np.nan, np.nan

    g = np.asarray(gamma_prob, dtype=float).reshape(-1)
    s = np.asarray(is_signal, dtype=int).reshape(-1)
    if g.shape != s.shape:
        return np.nan, np.nan

    if np.any(s == 1):
        ps = float(np.mean(g[s == 1]))
    else:
        ps = np.nan
    if np.any(s == 0):
        pn = float(np.mean(g[s == 0]))
    else:
        pn = np.nan
    return ps, pn


# -----------------------------------------------------------------------------
# One cell runner
# -----------------------------------------------------------------------------

def run_cell(dgp: str, T: int, J: int, study: StudyConfig, cfg: SimConfig) -> Dict[str, Any]:
    # True parameters (set Int true to 0.0 unless your SimConfig defines otherwise)
    true = {
        "int": float(getattr(cfg, "int_star", 0.0)),
        "beta_p": float(cfg.beta_p_star),
        "beta_w": float(cfg.beta_w_star),
        "sigma": float(cfg.sigma_star),
    }

    blp_cost = init_storage(study.R_mc)
    blp_nocost = init_storage(study.R_mc)
    shrink = init_storage(study.R_mc)

    print("\n" + "=" * 90)
    print(f"Cell: {dgp}, T={T}, J={J}, Nt={getattr(cfg, 'Nt', 'NA')}, R0={cfg.R0}, R_mc={study.R_mc}")
    print("=" * 90)

    # Warn once if simulation doesn't provide xi_true / is_signal (needed for paper columns)
    warned_xi = False
    warned_sig = False
    warned_delta = False

    for r in range(study.R_mc):
        rep_seed = int(study.seed + r)
        np.random.seed(rep_seed)

        markets = simulate_dataset(dgp, T=T, J=J, cfg=cfg)
        inject_market_size(markets, cfg)

        xi_true, is_signal = _extract_true_xi_and_signal_mask(markets, T=T, J=J)
        if (xi_true is None) and (not warned_xi):
            print("[INFO] xi_true not found in simulation markets. ξ column will be NaN until you add m['xi_true'] or m['xi'].")
            warned_xi = True
        if (is_signal is None) and (not warned_sig) and dgp in ["DGP1", "DGP2"]:
            print("[INFO] is_signal not found in simulation markets. Prob(signal/noise) will be NaN until you add m['is_signal'].")
            warned_sig = True

        # --- BLP + cost IV ---
        try:
            sigma_hat, beta_hat, extras = run_blp(markets, cfg, iv_type="cost")
            beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)

            blp_cost["sigma"][r] = float(sigma_hat)
            blp_cost["int"][r] = float(beta_hat[0])
            blp_cost["beta_p"][r] = float(beta_hat[1])
            blp_cost["beta_w"][r] = float(beta_hat[2])

            xi_abs, xi_sd = _compute_xi_metrics(extras, beta_hat, xi_true)
            if (np.isnan(xi_abs) or np.isnan(xi_sd)) and (not warned_delta):
                if (xi_true is not None) and (("delta_hat" not in extras) or ("X" not in extras)):
                    print("[INFO] BLP extras missing delta_hat/X. ξ column will be NaN until estimate_blp_sigma returns extras['delta_hat'] and extras['X'].")
                    warned_delta = True
            blp_cost["xi_abs_bias"][r] = xi_abs
            blp_cost["xi_sd"][r] = xi_sd

        except Exception as e:
            blp_cost["fail"][r] = 1
            _warn(f"BLP+CostIV failed (rep={r}, seed={rep_seed})", e)

        # --- BLP no-cost IV ---
        try:
            sigma_hat, beta_hat, extras = run_blp(markets, cfg, iv_type="nocost")
            beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)

            blp_nocost["sigma"][r] = float(sigma_hat)
            blp_nocost["int"][r] = float(beta_hat[0])
            blp_nocost["beta_p"][r] = float(beta_hat[1])
            blp_nocost["beta_w"][r] = float(beta_hat[2])

            xi_abs, xi_sd = _compute_xi_metrics(extras, beta_hat, xi_true)
            blp_nocost["xi_abs_bias"][r] = xi_abs
            blp_nocost["xi_sd"][r] = xi_sd

        except Exception as e:
            blp_nocost["fail"][r] = 1
            _warn(f"BLP-NoCostIV failed (rep={r}, seed={rep_seed})", e)

        # --- Shrinkage ---
        try:
            ret = run_shrinkage(markets, study, cfg)

            # allow either 4- or 5-tuple depending on implementation
            if len(ret) == 4:
                sigma_s, beta_s, score_s, gamma_prob = ret
                extras_s = {}
            else:
                sigma_s, beta_s, score_s, gamma_prob, extras_s = ret

            beta_s = np.asarray(beta_s, dtype=float).reshape(-1)

            shrink["sigma"][r] = float(sigma_s)
            shrink["int"][r] = float(beta_s[0])
            shrink["beta_p"][r] = float(beta_s[1])
            shrink["beta_w"][r] = float(beta_s[2])

            # xi metrics for shrinkage: requires extras with delta_hat/X OR implement it in shrinkage estimator similarly
            xi_abs, xi_sd = _compute_xi_metrics(extras_s, beta_s, xi_true)
            shrink["xi_abs_bias"][r] = xi_abs
            shrink["xi_sd"][r] = xi_sd

            # paper Prob columns
            ps, pn = _compute_prob_metrics(gamma_prob, is_signal)
            shrink["prob_signal"][r] = ps
            shrink["prob_noise"][r] = pn

        except Exception as e:
            shrink["fail"][r] = 1
            _warn(f"Shrinkage failed (rep={r}, seed={rep_seed})", e)

        if (r + 1) % max(1, study.R_mc // 10) == 0 or (r + 1) == study.R_mc:
            print(f"  progress: {r + 1}/{study.R_mc}", flush=True)

    # Summaries for parameters
    summaries = {
        "BLP+CostIV": {k: summarize_param(blp_cost[k], true[k]) for k in ["int", "beta_p", "beta_w", "sigma"]},
        "BLP-NoCostIV": {k: summarize_param(blp_nocost[k], true[k]) for k in ["int", "beta_p", "beta_w", "sigma"]},
        "Shrinkage": {k: summarize_param(shrink[k], true[k]) for k in ["int", "beta_p", "beta_w", "sigma"]},
    }

    # Summaries for xi/prob metrics (already error summaries)
    metrics = {
        "BLP+CostIV": {
            "xi_abs_bias": summarize_metric(blp_cost["xi_abs_bias"]),
            "xi_sd": summarize_metric(blp_cost["xi_sd"]),
            "prob_signal": summarize_metric(blp_cost["prob_signal"]),
            "prob_noise": summarize_metric(blp_cost["prob_noise"]),
        },
        "BLP-NoCostIV": {
            "xi_abs_bias": summarize_metric(blp_nocost["xi_abs_bias"]),
            "xi_sd": summarize_metric(blp_nocost["xi_sd"]),
            "prob_signal": summarize_metric(blp_nocost["prob_signal"]),
            "prob_noise": summarize_metric(blp_nocost["prob_noise"]),
        },
        "Shrinkage": {
            "xi_abs_bias": summarize_metric(shrink["xi_abs_bias"]),
            "xi_sd": summarize_metric(shrink["xi_sd"]),
            "prob_signal": summarize_metric(shrink["prob_signal"]),
            "prob_noise": summarize_metric(shrink["prob_noise"]),
        },
    }

    # Print main parameter tables
    print_table("BLP + Cost IV", summaries["BLP+CostIV"], true)
    print_table("BLP − Cost IV", summaries["BLP-NoCostIV"], true)
    print_table("Shrinkage", summaries["Shrinkage"], true)

    # Print paper-style extras (if available)
    print("\nPaper-style extras (mean over reps):")
    for method in ["BLP+CostIV", "BLP-NoCostIV", "Shrinkage"]:
        xi_b = metrics[method]["xi_abs_bias"]["mean"]
        xi_s = metrics[method]["xi_sd"]["mean"]
        ps = metrics[method]["prob_signal"]["mean"]
        pn = metrics[method]["prob_noise"]["mean"]
        print(f"  {method:12s}  xi_abs_bias={xi_b:.4f}  xi_sd={xi_s:.4f}  Prob_signal={ps:.3f}  Prob_noise={pn:.3f}")

    return {"true": true, "summaries": summaries, "metrics": metrics}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replicate Lu(25) Section 4 simulation study (paper table outputs).")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")

    p.add_argument("--R-mc", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--shrink-n-iter", type=int, default=None)
    p.add_argument("--shrink-burn", type=int, default=None)
    p.add_argument("--shrink-v0", type=float, default=None)
    p.add_argument("--shrink-v1", type=float, default=None)

    p.add_argument("--grid", type=str, default=None, help="e.g. DGP1:25:15,DGP2:25:15")
    return p


def parse_grid(s: str) -> List[Tuple[str, int, int]]:
    grid: List[Tuple[str, int, int]] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        dgp, T, J = token.split(":")
        grid.append((dgp, int(T), int(J)))
    return grid


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.cpu:
        os.environ["TF_METAL_DEVICE_DISABLED"] = "1"
        try:
            import tensorflow as tf  # noqa: F401
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    out_dir = Path(args.out) if args.out else (Path("results") / "part2" / f"lu25_section4_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(out_dir)

    logger = TeeLogger(out_dir / "run.log")
    try:
        logger.write("Lu(25) Section 4 replication (paper-table outputs)\n")
        logger.write(f"TIME: {datetime.now().isoformat()}\n")
        logger.write(f"OUT:  {out_dir.resolve()}\n")
        logger.write(f"CPU_ONLY: {bool(args.cpu)}\n\n")

        study = StudyConfig()
        cfg = SimConfig()

        grid = DEFAULT_GRID
        if args.smoke:
            study.R_mc = 3
            study.seed = 0
            study.shrink_n_iter = 100
            study.shrink_burn = 50
            grid = [("DGP1", 5, 10)]

        if args.R_mc is not None:
            study.R_mc = int(args.R_mc)
        if args.seed is not None:
            study.seed = int(args.seed)

        if args.shrink_n_iter is not None:
            study.shrink_n_iter = int(args.shrink_n_iter)
        if args.shrink_burn is not None:
            study.shrink_burn = int(args.shrink_burn)
        if args.shrink_v0 is not None:
            study.shrink_v0 = float(args.shrink_v0)
        if args.shrink_v1 is not None:
            study.shrink_v1 = float(args.shrink_v1)

        if args.grid:
            grid = parse_grid(args.grid)

        # Save config snapshot
        sim_keys = ["R0", "Nt", "sparse_frac", "sigma_star", "beta_p_star", "beta_w_star", "int_star"]
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "study": study.__dict__,
                    "sim": {k: getattr(cfg, k, None) for k in sim_keys},
                    "grid": grid,
                    "cpu_only": bool(args.cpu),
                },
                f,
                indent=2,
                default=str,
            )

        # Output files
        long_csv = out_dir / "summary_long.csv"
        paper_csv = out_dir / "paper_table_like.csv"
        if long_csv.exists():
            long_csv.unlink()
        if paper_csv.exists():
            paper_csv.unlink()

        for (dgp, T, J) in grid:
            cell = f"{dgp}_T{T}_J{J}"
            logger.write(f"\n=== RUN CELL {cell} ===\n")

            out = run_cell(dgp=dgp, T=T, J=J, study=study, cfg=cfg)
            true = out["true"]
            summaries = out["summaries"]
            metrics = out["metrics"]

            # Write long format for parameters
            for method in ["BLP+CostIV", "BLP-NoCostIV", "Shrinkage"]:
                for param in ["int", "beta_p", "beta_w", "sigma"]:
                    write_summary_long(long_csv, cell, method, param, summaries[method][param])

            # Write paper-like table rows: Bias + SD
            for method in ["BLP+CostIV", "BLP-NoCostIV", "Shrinkage"]:
                bias_row = {k: summaries[method][k]["bias"] for k in ["int", "beta_p", "beta_w", "sigma"]}
                sd_row = {k: summaries[method][k]["sd"] for k in ["int", "beta_p", "beta_w", "sigma"]}

                xi_bias = metrics[method]["xi_abs_bias"]["mean"]
                xi_sd = metrics[method]["xi_sd"]["mean"]
                ps = metrics[method]["prob_signal"]["mean"]
                pn = metrics[method]["prob_noise"]["mean"]

                write_paper_table_like(
                    paper_csv,
                    cell=cell,
                    method=method,
                    bias_row=bias_row,
                    sd_row=sd_row,
                    xi_bias=xi_bias,
                    xi_sd=xi_sd,
                    prob_signal=ps,
                    prob_noise=pn,
                )

        logger.write("\nDONE.\n")
        logger.write(f"summary_long.csv: {long_csv.resolve()}\n")
        logger.write(f"paper_table_like.csv: {paper_csv.resolve()}\n")
        return 0

    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())