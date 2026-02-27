"""
replicate_section4.py

Paper-aligned replication driver for Lu(25) simulation study (Section 4).

Design goals:
- Package-safe: no sys.path hacks
- Reproducible: seeded runs, explicit outputs
- Reviewer-friendly: one entrypoint, explicit --out, smoke mode
- Optional CPU-only mode for deterministic behavior on macOS Metal

Typical usage (from repo root, after `pip install -e ".[dev]"`):
    python -m jpm_q3.lu25.experiments.replicate_section4 --smoke --out results/part2/smoke
    python -m jpm_q3.lu25.experiments.replicate_section4 --cpu --grid DGP1:25:15 --R-mc 10 --out results/part2/dev
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


# IMPORTANT: edit this grid if you need to match Table 1 exactly.
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
        self.log_path = log_path
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
    """
    Ensure market dictionaries contain market size N if required by downstream estimators.
    Many likelihood-style estimators need N_t; BLP typically doesn't, but keeping it consistent helps.
    """
    Nt = getattr(cfg, "Nt", None)
    if Nt is None:
        return
    for m in markets:
        m["N"] = int(Nt)


def init_storage(R: int) -> Dict[str, np.ndarray]:
    return {
        "sigma": np.full(R, np.nan),
        "beta_p": np.full(R, np.nan),
        "beta_w": np.full(R, np.nan),
        "fail": np.zeros(R, dtype=int),
    }


def summarize_vec(x: np.ndarray, true_val: float) -> Dict[str, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": np.nan, "bias": np.nan, "sd": np.nan, "rmse": np.nan, "n": 0}
    mean = float(np.mean(x))
    bias = float(mean - true_val)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    rmse = float(np.sqrt(bias * bias + sd * sd))
    return {"mean": mean, "bias": bias, "sd": sd, "rmse": rmse, "n": int(x.size)}


def save_summary_csv(path: Path, cell_key: str, summaries: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    header = "cell,method,parameter,mean,bias,sd,rmse,n\n"
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        for method, summ in summaries.items():
            for param in ["sigma", "beta_p", "beta_w"]:
                s = summ[param]
                f.write(
                    f"{cell_key},{method},{param},"
                    f"{s['mean']:.6f},{s['bias']:.6f},{s['sd']:.6f},{s['rmse']:.6f},{s['n']}\n"
                )


def print_method_table(title: str, summary: Dict[str, Dict[str, float]], true_params: Dict[str, float]) -> None:
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)
    print(f"{'Param':<8} {'True':>10} {'Mean':>10} {'Bias':>10} {'SD':>10} {'RMSE':>10} {'n':>6}")
    print("-" * 90)
    mapping = [("sigma", "σ"), ("beta_p", "β_p"), ("beta_w", "β_w")]
    for k, sym in mapping:
        s = summary[k]
        tv = true_params[k]
        print(
            f"{sym:<8} {tv:>10.4f} {s['mean']:>10.4f} {s['bias']:>10.4f} {s['sd']:>10.4f} "
            f"{s['rmse']:>10.4f} {s['n']:>6d}"
        )


def _warn_exception(prefix: str, e: Exception) -> None:
    print(f"\n[WARN] {prefix}: {type(e).__name__}: {e}")
    tb = traceback.format_exc(limit=3)
    print(tb)


# -----------------------------------------------------------------------------
# Estimator runners
# -----------------------------------------------------------------------------

def run_blp(markets, cfg: SimConfig, iv_type: str):
    # estimate_blp_sigma returns (sigma_hat, beta_hat, extras)
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
        "sigma": float(cfg.sigma_star),
        "beta_p": float(cfg.beta_p_star),
        "beta_w": float(cfg.beta_w_star),
    }

    blp_cost = init_storage(study.R_mc)
    blp_nocost = init_storage(study.R_mc)
    shrink = init_storage(study.R_mc)
    lu25 = init_storage(study.R_mc) if include_lu25_map and HAS_LU25_MAP else None

    shrink_gamma_list: List[np.ndarray] = []
    lu25_gamma_list: List[np.ndarray] = []

    print("\n" + "=" * 90)
    print(f"Cell: {dgp}, T={T}, J={J}, Nt={getattr(cfg, 'Nt', 'NA')}, R0={cfg.R0}, R_mc={study.R_mc}")
    print("=" * 90)

    for r in range(study.R_mc):
        rep_seed = int(study.seed + r)
        np.random.seed(rep_seed)

        markets = simulate_dataset(dgp, T=T, J=J, cfg=cfg)
        inject_market_size(markets, cfg)

        # 1) BLP + Cost IV
        try:
            sigma_hat, beta_hat, _ = run_blp(markets, cfg, iv_type="cost")
            blp_cost["sigma"][r] = float(sigma_hat)
            blp_cost["beta_p"][r] = float(beta_hat[1])
            blp_cost["beta_w"][r] = float(beta_hat[2])
        except Exception as e:
            blp_cost["fail"][r] = 1
            _warn_exception(f"BLP+CostIV failed (rep={r}, seed={rep_seed})", e)

        # 2) BLP - Cost IV
        try:
            sigma_hat, beta_hat, _ = run_blp(markets, cfg, iv_type="nocost")
            blp_nocost["sigma"][r] = float(sigma_hat)
            blp_nocost["beta_p"][r] = float(beta_hat[1])
            blp_nocost["beta_w"][r] = float(beta_hat[2])
        except Exception as e:
            blp_nocost["fail"][r] = 1
            _warn_exception(f"BLP-NoCostIV failed (rep={r}, seed={rep_seed})", e)

        # 3) Shrinkage
        try:
            ret = run_shrinkage(markets, study, cfg)

            if len(ret) == 4:
                sigma_s, beta_s, _score, gamma_prob = ret
                extras = {}
            else:
                sigma_s, beta_s, _score, gamma_prob, extras = ret

            shrink["sigma"][r] = float(sigma_s)
            shrink["beta_p"][r] = float(beta_s[1])
            shrink["beta_w"][r] = float(beta_s[2])
            if gamma_prob is not None:
                shrink_gamma_list.append(np.asarray(gamma_prob))

            acc = extras.get("acc_rate", None)
            if acc is not None and study.R_mc <= 5:
                print(f"  [shrinkage] acc_rate={acc:.3f}")

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
                lu25["beta_p"][r] = float(beta_lu[1])
                lu25["beta_w"][r] = float(beta_lu[2])

                if lu_res.get("gamma_hat") is not None:
                    lu25_gamma_list.append(np.asarray(lu_res["gamma_hat"]))
            except Exception as e:
                lu25["fail"][r] = 1
                _warn_exception(f"Lu25MAP failed (rep={r}, seed={rep_seed})", e)

        if (r + 1) % max(1, study.R_mc // 10) == 0 or (r + 1) == study.R_mc:
            print(f"  progress: {r + 1}/{study.R_mc}", flush=True)

    summaries: Dict[str, Dict[str, Dict[str, float]]] = {
        "BLP+CostIV": {
            "sigma": summarize_vec(blp_cost["sigma"], true_params["sigma"]),
            "beta_p": summarize_vec(blp_cost["beta_p"], true_params["beta_p"]),
            "beta_w": summarize_vec(blp_cost["beta_w"], true_params["beta_w"]),
        },
        "BLP-NoCostIV": {
            "sigma": summarize_vec(blp_nocost["sigma"], true_params["sigma"]),
            "beta_p": summarize_vec(blp_nocost["beta_p"], true_params["beta_p"]),
            "beta_w": summarize_vec(blp_nocost["beta_w"], true_params["beta_w"]),
        },
        "Shrinkage": {
            "sigma": summarize_vec(shrink["sigma"], true_params["sigma"]),
            "beta_p": summarize_vec(shrink["beta_p"], true_params["beta_p"]),
            "beta_w": summarize_vec(shrink["beta_w"], true_params["beta_w"]),
        },
    }

    if include_lu25_map and HAS_LU25_MAP and lu25 is not None:
        summaries["Lu25MAP"] = {
            "sigma": summarize_vec(lu25["sigma"], true_params["sigma"]),
            "beta_p": summarize_vec(lu25["beta_p"], true_params["beta_p"]),
            "beta_w": summarize_vec(lu25["beta_w"], true_params["beta_w"]),
        }

    print_method_table("BLP + Cost IV", summaries["BLP+CostIV"], true_params)
    print_method_table("BLP − Cost IV", summaries["BLP-NoCostIV"], true_params)
    print_method_table("Shrinkage", summaries["Shrinkage"], true_params)
    if include_lu25_map and HAS_LU25_MAP and "Lu25MAP" in summaries:
        print_method_table("Lu25 MAP", summaries["Lu25MAP"], true_params)

    if len(shrink_gamma_list) > 0 and dgp in ["DGP1", "DGP2"]:
        cutoff = int(cfg.sparse_frac * J)
        G = np.stack([g.reshape(T, J) for g in shrink_gamma_list], axis=0)
        gamma_avg = G.mean(axis=(0, 1))  # [J]
        signal = float(gamma_avg[:cutoff].mean()) if cutoff > 0 else np.nan
        noise = float(gamma_avg[cutoff:].mean()) if cutoff < J else np.nan
        print("\nSparsity recovery (Shrinkage; avg gamma over markets+reps):")
        print(f"  cutoff (signal products): {cutoff}/{J}")
        print(f"  avg gamma signal: {signal:.4f}")
        print(f"  avg gamma noise:  {noise:.4f}")

    if include_lu25_map and HAS_LU25_MAP and len(lu25_gamma_list) > 0 and dgp in ["DGP1", "DGP2"]:
        cutoff = int(cfg.sparse_frac * J)
        H = np.stack([g.reshape(T, J) for g in lu25_gamma_list], axis=0)
        gamma_rate = H.mean(axis=0)  # [T, J]
        signal_rate = float(gamma_rate[:, :cutoff].mean()) if cutoff > 0 else np.nan
        noise_rate = float(gamma_rate[:, cutoff:].mean()) if cutoff < J else np.nan
        print("\nSparsity recovery (Lu25MAP; mean detect rate |d|>tau):")
        print(f"  tau_detect: {study.lu_tau_detect:.3f}")
        print(f"  detect rate signal: {signal_rate:.4f}")
        print(f"  detect rate noise:  {noise_rate:.4f}")

    return summaries


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replicate Lu(25) Section 4 simulation study.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: results/part2/<timestamp>).")
    p.add_argument("--smoke", action="store_true", help="Run a small configuration quickly (sanity check).")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (avoid Metal/GPU RNG differences; improves reproducibility).",
    )

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

    # Best-effort CPU-only on macOS Metal:
    # Prefer env var (works even if TF is imported elsewhere at module-import time),
    # then also hide visible GPU devices if TF is available.
    if args.cpu:
        os.environ["TF_METAL_DEVICE_DISABLED"] = "1"

        # If TF hasn't been imported yet, this will reliably prevent GPU usage.
        # If it has, this may raise; we swallow and continue.
        try:
            import tensorflow as tf  # noqa: F401
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    # Output dir
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / "part2" / f"lu25_section4_{ts}"
    else:
        out_dir = Path(args.out)
    ensure_dir(out_dir)

    logger = TeeLogger(out_dir / "run.log")
    try:
        logger.write("Lu(25) Section 4 replication\n")
        logger.write(f"TIME: {datetime.now().isoformat()}\n")
        logger.write(f"OUT:  {out_dir.resolve()}\n")
        logger.write(f"CPU_ONLY: {bool(args.cpu)}\n")
        logger.write(f"HAS_LU25_MAP: {HAS_LU25_MAP}\n\n")

        # Base configs
        study = StudyConfig()
        cfg = SimConfig()

        # Grid / smoke overrides
        grid = DEFAULT_GRID
        if args.smoke:
            # Slightly less degenerate than (T=2,J=3), but still fast.
            study.R_mc = 3
            study.seed = 0
            study.shrink_n_iter = 100
            study.shrink_burn = 50
            grid = [("DGP1", 5, 10)]

        # Apply CLI overrides
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

        # Save run config snapshot (explicit keys only; avoids dir(cfg) noise)
        sim_keys = ["R0", "Nt", "sparse_frac", "sigma_star", "beta_p_star", "beta_w_star"]
        cfg_path = out_dir / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
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

        # Run grid
        summary_csv = out_dir / "summary.csv"
        if summary_csv.exists():
            summary_csv.unlink()

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

        logger.write("\nDONE.\n")
        logger.write(f"Summary CSV: {summary_csv.resolve()}\n")
        return 0

    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())