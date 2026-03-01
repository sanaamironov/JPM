"""
replicate_section4.py

Paper-aligned replication driver for Lu(25) simulation study (Section 4).

This version is a clean replacement that:
- Keeps your existing CLI and output structure.
- Adds paper-table reporting hooks:
  - Int (intercept) tracked for all methods.
  - xi metrics (mean|xi_hat - xi_true|, sd(xi_hat - xi_true)) for BLP methods
    once estimate_blp_sigma returns extras["xi_hat"].
  - Prob(signal/noise) for Shrinkage in DGP1/DGP2 using markets' is_signal mask
    (requires simulate_market to provide m["is_signal"]).
- Writes:
  - summary.csv (long format; now includes Int and xi metrics)
  - paper_table_like.csv (Bias/SD rows like the paper, with NaN where unavailable)

Preconditions (recommended):
- simulate_market returns keys: xi_true and is_signal (see provided simulate_market.py).
- estimate_blp_sigma returns (sigma_hat, beta_hat, extras) with extras["xi_hat"].
- estimate_shrinkage_sigma already returns gamma_prob; prob metrics computed from it.

Usage:
  python -m jpm_q3.lu25.experiments.replicate_section4 --cpu --out results/part2/section4_full
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..estimators.blp import estimate_blp_sigma
from ..estimators.shrinkage import estimate_shrinkage_sigma
from ..simulation.config import SimConfig
from ..simulation.simulate import simulate_dataset

# Optional Lu25 MAP
HAS_LU25_MAP = False
try:
    from ..estimators.lu25_map import Lu25MapConfig, estimate_lu25_map  # type: ignore

    HAS_LU25_MAP = True
except Exception:
    HAS_LU25_MAP = False


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class StudyConfig:
    """Monte Carlo study config (replications + estimator settings)."""
    R_mc: int = 50
    seed: int = 123

    # Shrinkage (TFP MCMC) settings
    shrink_n_iter: int = 200
    shrink_burn: int = 100
    shrink_v0: float = 1e-4
    shrink_v1: float = 1.0

    # Lu25 MAP (optional)
    lu_steps: int = 1200
    lu_lr: float = 0.05
    lu_l1_strength: float = 8.0
    lu_mu_sd: float = 2.0
    lu_tau_detect: float = 0.25


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
    """Write to console and to a log file."""
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


def inject_market_size(markets: list, cfg: SimConfig) -> None:
    """Ensure market dicts contain market size N if required by likelihood-style estimators."""
    Nt = getattr(cfg, "Nt", None)
    if Nt is None:
        return
    for m in markets:
        m["N"] = int(Nt)


def _warn_exception(prefix: str, e: Exception) -> None:
    print(f"\n[WARN] {prefix}: {type(e).__name__}: {e}")
    print(traceback.format_exc(limit=3))


def init_storage(R: int) -> Dict[str, np.ndarray]:
    """Per-replication storage for each method."""
    return {
        "int": np.full(R, np.nan),
        "beta_p": np.full(R, np.nan),
        "beta_w": np.full(R, np.nan),
        "sigma": np.full(R, np.nan),

        # xi metrics (paper column) -- filled when xi_hat and xi_true available
        "xi_bias_abs": np.full(R, np.nan),  # mean |xi_hat - xi_true|
        "xi_sd": np.full(R, np.nan),        # sd(xi_hat - xi_true)

        # Prob columns for shrinkage (paper column), only meaningful for sparse DGPs
        "prob_signal": np.full(R, np.nan),
        "prob_noise": np.full(R, np.nan),

        "fail": np.zeros(R, dtype=int),
    }


def summarize_param(x: np.ndarray, true_val: float) -> Dict[str, float]:
    """mean/bias/sd/rmse/n for an estimated parameter."""
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
    mean/sd/n for a scalar metric already representing an error summary (xi/Prob).
    We still populate the same keys to reuse CSV writer (bias=mean, rmse=sqrt(mean^2+sd^2)).
    """
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": np.nan, "bias": np.nan, "sd": np.nan, "rmse": np.nan, "n": 0}
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    rmse = float(np.sqrt(mean * mean + sd * sd))
    return {"mean": mean, "bias": mean, "sd": sd, "rmse": rmse, "n": int(x.size)}


def stack_true_xi(markets: list[dict]) -> np.ndarray:
    return np.concatenate([np.asarray(m["xi_true"], dtype=float) for m in markets], axis=0)


def stack_is_signal(markets: list[dict]) -> np.ndarray:
    return np.concatenate([np.asarray(m["is_signal"], dtype=int) for m in markets], axis=0)


def compute_xi_bias_sd(xi_hat: np.ndarray, xi_true: np.ndarray) -> tuple[float, float]:
    err = np.asarray(xi_hat, dtype=float).reshape(-1) - np.asarray(xi_true, dtype=float).reshape(-1)
    return float(np.mean(np.abs(err))), float(np.std(err, ddof=1))


def compute_prob_signal_noise(gamma_prob: np.ndarray, is_signal: np.ndarray) -> tuple[float, float]:
    g = np.asarray(gamma_prob, dtype=float).reshape(-1)
    s = np.asarray(is_signal, dtype=int).reshape(-1)
    ps = float(np.mean(g[s == 1])) if np.any(s == 1) else np.nan
    pn = float(np.mean(g[s == 0])) if np.any(s == 0) else np.nan
    return ps, pn


def save_summary_csv(path: Path, cell: str, summaries: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Long-format summary. One row per (cell, method, parameter).
    Includes: int, beta_p, beta_w, sigma, xi_bias_abs, xi_sd, prob_signal, prob_noise.
    """
    header = "cell,method,parameter,mean,bias,sd,rmse,n\n"
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        for method, summ in summaries.items():
            for param in ["int", "beta_p", "beta_w", "sigma", "xi_bias_abs", "xi_sd", "prob_signal", "prob_noise"]:
                s = summ[param]
                f.write(
                    f"{cell},{method},{param},"
                    f"{s['mean']:.6f},{s['bias']:.6f},{s['sd']:.6f},{s['rmse']:.6f},{s['n']}\n"
                )


def save_paper_table_like(path: Path, cell: str, method: str, summ: Dict[str, Dict[str, float]]) -> None:
    """
    Paper-like CSV: two rows per method (Bias, SD).
    Columns: Int, beta_p, beta_w, sigma, xi, Prob_signal, Prob_noise
    """
    header = "cell,method,row,Int,beta_p,beta_w,sigma,xi,Prob_signal,Prob_noise\n"
    write_header = not path.exists()
    bias_row = {
        "Int": summ["int"]["bias"],
        "beta_p": summ["beta_p"]["bias"],
        "beta_w": summ["beta_w"]["bias"],
        "sigma": summ["sigma"]["bias"],
    }
    sd_row = {
        "Int": summ["int"]["sd"],
        "beta_p": summ["beta_p"]["sd"],
        "beta_w": summ["beta_w"]["sd"],
        "sigma": summ["sigma"]["sd"],
    }
    xi_bias = summ["xi_bias_abs"]["mean"]
    xi_sd = summ["xi_sd"]["mean"]
    ps = summ["prob_signal"]["mean"]
    pn = summ["prob_noise"]["mean"]

    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(f"{cell},{method},Bias,{bias_row['Int']},{bias_row['beta_p']},{bias_row['beta_w']},{bias_row['sigma']},{xi_bias},{ps},{pn}\n")
        f.write(f"{cell},{method},SD,{sd_row['Int']},{sd_row['beta_p']},{sd_row['beta_w']},{sd_row['sigma']},{xi_sd},,\n")


def print_method_table(title: str, summary: Dict[str, Dict[str, float]], true_params: Dict[str, float]) -> None:
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)
    print(f"{'Param':<8} {'True':>10} {'Mean':>10} {'Bias':>10} {'SD':>10} {'RMSE':>10} {'n':>6}")
    print("-" * 90)
    mapping = [("int", "Int"), ("beta_p", "β_p"), ("beta_w", "β_w"), ("sigma", "σ")]
    for k, sym in mapping:
        s = summary[k]
        tv = true_params[k]
        print(
            f"{sym:<8} {tv:>10.4f} {s['mean']:>10.4f} {s['bias']:>10.4f} {s['sd']:>10.4f} "
            f"{s['rmse']:>10.4f} {s['n']:>6d}"
        )


# -----------------------------------------------------------------------------
# Estimator runners
# -----------------------------------------------------------------------------

def run_blp(markets, cfg: SimConfig, iv_type: str):
    # returns (sigma_hat, beta_hat, extras)
    return estimate_blp_sigma(markets, iv_type=iv_type, R=cfg.R0)


def run_shrinkage(markets, study: StudyConfig, cfg: SimConfig):
    # returns either 4-tuple or 5-tuple if return_extras=True
    return estimate_shrinkage_sigma(
        markets,
        R=cfg.R0,
        n_iter=study.shrink_n_iter,
        burn=study.shrink_burn,
        v0=study.shrink_v0,
        v1=study.shrink_v1,
        return_extras=True,
    )


def run_lu25_map(markets, study: StudyConfig, cfg: SimConfig, rep_seed: int):
    if not HAS_LU25_MAP:
        raise RuntimeError("Lu25 MAP estimator not available.")
    lu_cfg = Lu25MapConfig(
        R=cfg.R0,
        steps=study.lu_steps,
        lr=study.lu_lr,
        l1_strength=study.lu_l1_strength,
        mu_sd=study.lu_mu_sd,
        tau_detect=study.lu_tau_detect,
        default_market_size=int(getattr(cfg, "Nt", 1000)),
        seed=rep_seed,
    )
    return estimate_lu25_map(markets, cfg=lu_cfg)


# -----------------------------------------------------------------------------
# One cell runner
# -----------------------------------------------------------------------------

def run_cell(
    dgp: str,
    T: int,
    J: int,
    study: StudyConfig,
    cfg: SimConfig,
    include_lu25_map: bool,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    true_params = {
        "int": float(getattr(cfg, "int_star", 0.0)),
        "sigma": float(cfg.sigma_star),
        "beta_p": float(cfg.beta_p_star),
        "beta_w": float(cfg.beta_w_star),
    }

    blp_cost = init_storage(study.R_mc)
    blp_nocost = init_storage(study.R_mc)
    shrink = init_storage(study.R_mc)
    lu25 = init_storage(study.R_mc) if include_lu25_map and HAS_LU25_MAP else None

    print("\n" + "=" * 90)
    print(f"Cell: {dgp}, T={T}, J={J}, Nt={getattr(cfg, 'Nt', 'NA')}, R0={cfg.R0}, R_mc={study.R_mc}")
    print("=" * 90)

    for r in range(study.R_mc):
        rep_seed = int(study.seed + r)
        np.random.seed(rep_seed)

        markets = simulate_dataset(dgp, T=T, J=J, cfg=cfg, seed=rep_seed)
        inject_market_size(markets, cfg)

        # True latent objects for paper columns (if present)
        xi_true = None
        is_signal = None
        try:
            xi_true = stack_true_xi(markets)
        except Exception:
            pass
        try:
            is_signal = stack_is_signal(markets)
        except Exception:
            pass

        # 1) BLP + Cost IV
        try:
            sigma_hat, beta_hat, extras = run_blp(markets, cfg, iv_type="cost")
            beta_hat = np.asarray(beta_hat, dtype=float)

            blp_cost["sigma"][r] = float(sigma_hat)
            blp_cost["int"][r] = float(beta_hat[0])
            blp_cost["beta_p"][r] = float(beta_hat[1])
            blp_cost["beta_w"][r] = float(beta_hat[2])

            if xi_true is not None and isinstance(extras, dict) and "xi_hat" in extras:
                xi_abs, xi_sd = compute_xi_bias_sd(extras["xi_hat"], xi_true)
                blp_cost["xi_bias_abs"][r] = xi_abs
                blp_cost["xi_sd"][r] = xi_sd

        except Exception as e:
            blp_cost["fail"][r] = 1
            _warn_exception(f"BLP+CostIV failed (rep={r}, seed={rep_seed})", e)

        # 2) BLP - Cost IV
        try:
            sigma_hat, beta_hat, extras = run_blp(markets, cfg, iv_type="nocost")
            beta_hat = np.asarray(beta_hat, dtype=float)

            blp_nocost["sigma"][r] = float(sigma_hat)
            blp_nocost["int"][r] = float(beta_hat[0])
            blp_nocost["beta_p"][r] = float(beta_hat[1])
            blp_nocost["beta_w"][r] = float(beta_hat[2])

            if xi_true is not None and isinstance(extras, dict) and "xi_hat" in extras:
                xi_abs, xi_sd = compute_xi_bias_sd(extras["xi_hat"], xi_true)
                blp_nocost["xi_bias_abs"][r] = xi_abs
                blp_nocost["xi_sd"][r] = xi_sd

        except Exception as e:
            blp_nocost["fail"][r] = 1
            _warn_exception(f"BLP-NoCostIV failed (rep={r}, seed={rep_seed})", e)

        # 3) Shrinkage
        try:
            ret = run_shrinkage(markets, study, cfg)
            if len(ret) == 4:
                sigma_s, beta_s, _score, gamma_prob = ret
                extras_s = {}
            else:
                sigma_s, beta_s, _score, gamma_prob, extras_s = ret

            beta_s = np.asarray(beta_s, dtype=float)

            shrink["sigma"][r] = float(sigma_s)
            shrink["int"][r] = float(beta_s[0])
            shrink["beta_p"][r] = float(beta_s[1])
            shrink["beta_w"][r] = float(beta_s[2])

            # Prob columns (paper) only meaningful for sparse DGPs
            if gamma_prob is not None and is_signal is not None and dgp in ["DGP1", "DGP2"]:
                ps, pn = compute_prob_signal_noise(gamma_prob, is_signal)
                shrink["prob_signal"][r] = ps
                shrink["prob_noise"][r] = pn

            # shrinkage xi metrics require extras_s["xi_hat"] or delta_hat/X; leave NaN unless provided
            if xi_true is not None and isinstance(extras_s, dict) and "xi_hat" in extras_s:
                xi_abs, xi_sd = compute_xi_bias_sd(extras_s["xi_hat"], xi_true)
                shrink["xi_bias_abs"][r] = xi_abs
                shrink["xi_sd"][r] = xi_sd

        except Exception as e:
            shrink["fail"][r] = 1
            _warn_exception(f"Shrinkage failed (rep={r}, seed={rep_seed})", e)

        # 4) Lu25 MAP (optional)
        if include_lu25_map and HAS_LU25_MAP and lu25 is not None:
            try:
                lu_res = run_lu25_map(markets, study, cfg, rep_seed=rep_seed)
                sigma_lu = float(lu_res["sigma_hat"])
                beta_lu = np.asarray(lu_res["beta_hat"], dtype=float)

                lu25["sigma"][r] = sigma_lu
                lu25["int"][r] = float(beta_lu[0])
                lu25["beta_p"][r] = float(beta_lu[1])
                lu25["beta_w"][r] = float(beta_lu[2])

            except Exception as e:
                lu25["fail"][r] = 1
                _warn_exception(f"Lu25MAP failed (rep={r}, seed={rep_seed})", e)

        if (r + 1) % max(1, study.R_mc // 10) == 0 or (r + 1) == study.R_mc:
            print(f"  progress: {r + 1}/{study.R_mc}", flush=True)

    # Summaries
    def pack(method_store: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        return {
            "int": summarize_param(method_store["int"], true_params["int"]),
            "beta_p": summarize_param(method_store["beta_p"], true_params["beta_p"]),
            "beta_w": summarize_param(method_store["beta_w"], true_params["beta_w"]),
            "sigma": summarize_param(method_store["sigma"], true_params["sigma"]),
            "xi_bias_abs": summarize_metric(method_store["xi_bias_abs"]),
            "xi_sd": summarize_metric(method_store["xi_sd"]),
            "prob_signal": summarize_metric(method_store["prob_signal"]),
            "prob_noise": summarize_metric(method_store["prob_noise"]),
        }

    summaries: Dict[str, Dict[str, Dict[str, float]]] = {
        "BLP+CostIV": pack(blp_cost),
        "BLP-NoCostIV": pack(blp_nocost),
        "Shrinkage": pack(shrink),
    }
    if include_lu25_map and HAS_LU25_MAP and lu25 is not None:
        summaries["Lu25MAP"] = pack(lu25)

    # Print tables (parameters)
    print_method_table("BLP + Cost IV", summaries["BLP+CostIV"], true_params)
    print_method_table("BLP − Cost IV", summaries["BLP-NoCostIV"], true_params)
    print_method_table("Shrinkage", summaries["Shrinkage"], true_params)
    if include_lu25_map and HAS_LU25_MAP and "Lu25MAP" in summaries:
        print_method_table("Lu25 MAP", summaries["Lu25MAP"], true_params)

    # Print paper extras quick view
    print("\nPaper-style extras (means over reps):")
    for m in ["BLP+CostIV", "BLP-NoCostIV", "Shrinkage"]:
        xi_b = summaries[m]["xi_bias_abs"]["mean"]
        xi_s = summaries[m]["xi_sd"]["mean"]
        ps = summaries[m]["prob_signal"]["mean"]
        pn = summaries[m]["prob_noise"]["mean"]
        print(f"  {m:12s}  xi_abs_bias={xi_b:.4f}  xi_sd={xi_s:.4f}  Prob_signal={ps:.3f}  Prob_noise={pn:.3f}")

    return summaries


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replicate Lu(25) Section 4 simulation study.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: results/part2/<timestamp>).")
    p.add_argument("--smoke", action="store_true", help="Run a small configuration quickly (sanity check).")
    p.add_argument("--cpu", action="store_true", help="Force CPU (avoid Metal/GPU RNG differences).")

    p.add_argument("--R-mc", type=int, default=None, help="Monte Carlo replications (overrides default).")
    p.add_argument("--seed", type=int, default=None, help="Base seed (rep r uses seed + r).")

    p.add_argument("--include-lu25-map", action="store_true", help="Also run optional Lu25 MAP estimator (if available).")

    p.add_argument("--shrink-n-iter", type=int, default=None)
    p.add_argument("--shrink-burn", type=int, default=None)
    p.add_argument("--shrink-v0", type=float, default=None)
    p.add_argument("--shrink-v1", type=float, default=None)

    p.add_argument("--grid", type=str, default=None, help="Grid override: 'DGP1:25:15,DGP2:25:15' etc.")
    return p


def parse_grid(s: str) -> List[Tuple[str, int, int]]:
    grid: List[Tuple[str, int, int]] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad grid token '{token}'. Expected DGP:T:J")
        dgp, T, J = parts[0], int(parts[1]), int(parts[2])
        grid.append((dgp, T, J))
    return grid


def main(argv: List[str] | None = None) -> int:
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
        logger.write("Lu(25) Section 4 replication\n")
        logger.write(f"TIME: {datetime.now().isoformat()}\n")
        logger.write(f"OUT:  {out_dir.resolve()}\n")
        logger.write(f"CPU_ONLY: {bool(args.cpu)}\n")
        logger.write(f"HAS_LU25_MAP: {HAS_LU25_MAP}\n\n")

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

        sim_keys = ["R0", "Nt", "sparse_frac", "sigma_star", "beta_p_star", "beta_w_star", "int_star"]
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "study": study.__dict__,
                    "sim": {k: getattr(cfg, k, None) for k in sim_keys},
                    "grid": grid,
                    "include_lu25_map": bool(args.include_lu25_map),
                    "cpu_only": bool(args.cpu),
                },
                f,
                indent=2,
                default=str,
            )

        summary_csv = out_dir / "summary.csv"
        paper_csv = out_dir / "paper_table_like.csv"
        if summary_csv.exists():
            summary_csv.unlink()
        if paper_csv.exists():
            paper_csv.unlink()

        for (dgp, T, J) in grid:
            cell_key = f"{dgp}_T{T}_J{J}"
            logger.write(f"\n=== RUN CELL {cell_key} ===\n")

            summaries = run_cell(
                dgp=dgp,
                T=int(T),
                J=int(J),
                study=study,
                cfg=cfg,
                include_lu25_map=bool(args.include_lu25_map),
            )

            save_summary_csv(summary_csv, cell_key, summaries)

            # paper-like rows
            for method in ["BLP+CostIV", "BLP-NoCostIV", "Shrinkage"] + (["Lu25MAP"] if "Lu25MAP" in summaries else []):
                save_paper_table_like(paper_csv, cell_key, method, summaries[method])

        logger.write("\nDONE.\n")
        logger.write(f"Summary CSV: {summary_csv.resolve()}\n")
        logger.write(f"Paper CSV:   {paper_csv.resolve()}\n")
        return 0

    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())