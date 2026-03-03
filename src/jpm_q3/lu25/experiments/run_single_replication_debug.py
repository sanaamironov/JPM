"""
run_single_replication_debug

Debug tool: run ONE simulated dataset and run the SAME estimators used by the canonical
Section 4 replication pipeline (replicate_section4.py), with optional MAP benchmark.

Purpose
-------
- When Monte Carlo tables don't match expectations, a single-rep run is the fastest way
  to inspect what happened: simulated data diagnostics, point estimates, sparsity recovery,
  and basic sanity checks.
- This script is NOT used to generate paper tables. The canonical runner is:
    jpm_q3.lu25.experiments.replicate_section4

What this script runs (by default)
----------------------------------
1) BLP + Cost IV (paper-aligned X/Z: no constants)
2) BLP - Cost IV (paper-aligned X/Z: no constants)
3) Shrinkage (TFP-MCMC core) with sigma grid search (paper-aligned X=[p,w], no constant)

Optional benchmark
------------------
--include-map enables the Lu25 MAP estimator as an additional benchmark.
This is not required for the paper tables, so it is off by default.

Outputs
-------
Writes into:
  results/part2/lu25_debug/<run_id>/
    - single_replication_summary.json
    - single_replication_table.csv

Example
-------
python -m jpm_q3.lu25.experiments.run_single_replication \
  --dgp DGP2 --T 25 --J 15 --seed 123 --R-mc 50 --out results/part2/lu25_debug/run1

python -m jpm_q3.lu25.experiments.run_single_replication \
  --dgp DGP2 --T 25 --J 15 --seed 123 --R-mc 50 --include-map
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..simulation.config import SimConfig
from ..simulation.simulate import simulate_dataset

# Reuse canonical, paper-aligned estimators from replicate_section4
from .replicate_section4 import (
    build_matrices_paper,
    estimate_blp_sigma_paper,
    estimate_shrinkage_sigma_paper,
    mean_abs_and_sd,
    prob_signal_noise,
    stack_is_signal,
    stack_true_xi,
)

# Optional MAP benchmark
try:
    from ..estimators.lu25_map import Lu25MapConfig, estimate_lu25_map  # type: ignore
except Exception:
    Lu25MapConfig = None  # type: ignore
    estimate_lu25_map = None  # type: ignore


@dataclass(frozen=True)
class SingleRunConfig:
    dgp: str
    T: int
    J: int
    seed: int
    R_mc: int
    include_map: bool


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-replication debug runner for Lu(25) Section 4")

    p.add_argument("--dgp", type=str, default="DGP1", help="DGP name (DGP1..DGP4)")
    p.add_argument("--T", type=int, default=25, help="Number of markets")
    p.add_argument("--J", type=int, default=15, help="Number of products")
    p.add_argument("--seed", type=int, default=123, help="Seed for data simulation")
    p.add_argument("--R-mc", type=int, default=50, help="Num simulation draws R used in delta computation")

    p.add_argument(
        "--out",
        type=str,
        default="results/part2/lu25_debug",
        help="Output directory root (default: results/part2/lu25_debug)",
    )

    # Shrinkage knobs (keep consistent with replicate_section4)
    p.add_argument("--shrink-n-iter", type=int, default=400)
    p.add_argument("--shrink-burn", type=int, default=200)
    p.add_argument("--shrink-thin", type=int, default=1)

    # Optional benchmark
    p.add_argument("--include-map", action="store_true", help="Also run optional Lu25 MAP benchmark (off by default)")

    return p.parse_args(argv)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _run_blp_cost(markets: list[dict], R_mc: int) -> Dict[str, float]:
    sigma_hat, beta_hat, extras = estimate_blp_sigma_paper(markets, iv_type="cost", R=R_mc)
    xi_hat = np.asarray(extras["xi_hat"], dtype=float).reshape(-1)
    return {
        "sigma": float(sigma_hat),
        "beta_p": float(beta_hat[0]),
        "beta_w": float(beta_hat[1]),
        "int": float(np.mean(xi_hat)),
    }


def _run_blp_nocost(markets: list[dict], R_mc: int) -> Dict[str, float]:
    sigma_hat, beta_hat, extras = estimate_blp_sigma_paper(markets, iv_type="nocost", R=R_mc)
    xi_hat = np.asarray(extras["xi_hat"], dtype=float).reshape(-1)
    return {
        "sigma": float(sigma_hat),
        "beta_p": float(beta_hat[0]),
        "beta_w": float(beta_hat[1]),
        "int": float(np.mean(xi_hat)),
    }


def _run_shrinkage(markets: list[dict], R_mc: int, shrink_kwargs: dict) -> Tuple[Dict[str, float], Dict[str, object]]:
    sigma_hat, beta_hat, score_hat, gamma_prob, extras = estimate_shrinkage_sigma_paper(
        markets, R=R_mc, return_extras=True, **shrink_kwargs
    )
    xi_hat = np.asarray(extras["xi_hat"], dtype=float).reshape(-1)
    out = {
        "sigma": float(sigma_hat),
        "beta_p": float(beta_hat[0]),
        "beta_w": float(beta_hat[1]),
        "int": float(np.mean(xi_hat)),
        "score": float(score_hat),
        "acc_rate": float(extras.get("acc_rate", float("nan"))),
    }
    aux = {
        "gamma_prob": np.asarray(gamma_prob, dtype=float).reshape(-1),
        "xi_hat": xi_hat,
    }
    return out, aux


def _run_map_optional(markets: list[dict], cfg: SimConfig, seed: int) -> Dict[str, object]:
    if estimate_lu25_map is None or Lu25MapConfig is None:
        return {"status": "skipped", "reason": "lu25_map module not available"}

    # Provide conservative defaults; treat this as a benchmark, not required for paper.
    lu_cfg = Lu25MapConfig(
        R=getattr(cfg, "R0", 50),
        steps=800,
        lr=0.05,
        l1_strength=8.0,
        tau_detect=0.25,
        mu_sd=2.0,
        default_market_size=getattr(cfg, "default_market_size", 1000),
        seed=int(seed),
    )
    res = estimate_lu25_map(markets, cfg=lu_cfg)

    sigma_hat = float(res["sigma_hat"])
    beta_hat = np.asarray(res["beta_hat"], dtype=float).reshape(-1)

    out = {
        "status": "success",
        "sigma": sigma_hat,
        "beta_p": float(beta_hat[1]) if beta_hat.size > 1 else float("nan"),
        "beta_w": float(beta_hat[2]) if beta_hat.size > 2 else float("nan"),
        "beta_intercept": float(beta_hat[0]) if beta_hat.size > 0 else float("nan"),
        "final_neg_log_post": float(res.get("final_neg_log_post", float("nan"))),
        "sparsity": res.get("sparsity", None),
        "config": res.get("config", None),
    }
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_cfg = SingleRunConfig(
        dgp=args.dgp.strip().upper(),
        T=int(args.T),
        J=int(args.J),
        seed=int(args.seed),
        R_mc=int(args.R_mc),
        include_map=bool(args.include_map),
    )

    out_root = Path(args.out)
    run_id = f"{run_cfg.dgp}_T{run_cfg.T}_J{run_cfg.J}_seed{run_cfg.seed}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig()

    print(f"\n[single] START {run_id}")
    print(f"[single] out_dir: {out_dir}")
    print(f"[single] sim: DGP={run_cfg.dgp} T={run_cfg.T} J={run_cfg.J} seed={run_cfg.seed}")
    print(f"[single] delta inversion R-mc={run_cfg.R_mc}")

    # 1) simulate
    markets = simulate_dataset(run_cfg.dgp, T=run_cfg.T, J=run_cfg.J, cfg=cfg, seed=run_cfg.seed)

    # diagnostics (market 0)
    m0 = markets[0]
    s0 = float(max(1e-12, 1.0 - float(np.sum(m0["s"]))))
    xi0 = m0.get("xi_true", None)
    corr_px = _safe_corr(m0["p"], xi0) if xi0 is not None else float("nan")

    diag = {
        "outside_share_market0": s0,
        "corr_p_xi_market0": corr_px,
    }
    print(f"[single] diag: outside_share(m0)={diag['outside_share_market0']:.4f} corr(p,xi)(m0)={diag['corr_p_xi_market0']:.4f}")

    # ground truth (if available)
    xi_true = None
    is_signal = None
    try:
        xi_true = stack_true_xi(markets)
    except Exception:
        xi_true = None
    try:
        is_signal = stack_is_signal(markets)
    except Exception:
        is_signal = None

    # 2) estimators
    results: Dict[str, object] = {}
    results["run_id"] = run_id
    results["config"] = {
        "dgp": run_cfg.dgp,
        "T": run_cfg.T,
        "J": run_cfg.J,
        "seed": run_cfg.seed,
        "R_mc": run_cfg.R_mc,
        "include_map": run_cfg.include_map,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    results["true_params"] = {
        "int": float(cfg.xi_bar_star),
        "sigma": float(cfg.sigma_star),
        "beta_p": float(cfg.beta_p_star),
        "beta_w": float(cfg.beta_w_star),
    }
    results["diagnostics"] = diag

    print("[single] running: BLP+CostIV ...")
    blp_cost = _run_blp_cost(markets, R_mc=run_cfg.R_mc)
    results["BLP+CostIV"] = blp_cost

    print("[single] running: BLP-NoCostIV ...")
    blp_nocost = _run_blp_nocost(markets, R_mc=run_cfg.R_mc)
    results["BLP-NoCostIV"] = blp_nocost

    print("[single] running: Shrinkage ...")
    shrink_kwargs = dict(
        n_iter=int(args.shrink_n_iter),
        burn=int(args.shrink_burn),
        thin=int(args.shrink_thin),
    )
    shrink_res, shrink_aux = _run_shrinkage(markets, R_mc=run_cfg.R_mc, shrink_kwargs=shrink_kwargs)
    results["Shrinkage"] = shrink_res

    # compute xi metrics if xi_true available
    metrics: Dict[str, object] = {}
    if xi_true is not None:
        # BLP cost/no cost xi_hat isn't returned by _run_blp_*; recompute quickly from extras using paper aligned
        # (cheap enough for single run).
        _, _, extras_cost = estimate_blp_sigma_paper(markets, iv_type="cost", R=run_cfg.R_mc)
        xi_hat_cost = np.asarray(extras_cost["xi_hat"], dtype=float).reshape(-1)
        mae, sd = mean_abs_and_sd(xi_hat_cost - xi_true)
        metrics["BLP+CostIV_xi_mae"] = mae
        metrics["BLP+CostIV_xi_sd"] = sd

        _, _, extras_nc = estimate_blp_sigma_paper(markets, iv_type="nocost", R=run_cfg.R_mc)
        xi_hat_nc = np.asarray(extras_nc["xi_hat"], dtype=float).reshape(-1)
        mae, sd = mean_abs_and_sd(xi_hat_nc - xi_true)
        metrics["BLP-NoCostIV_xi_mae"] = mae
        metrics["BLP-NoCostIV_xi_sd"] = sd

        xi_hat_sh = np.asarray(shrink_aux["xi_hat"], dtype=float).reshape(-1)
        mae, sd = mean_abs_and_sd(xi_hat_sh - xi_true)
        metrics["Shrinkage_xi_mae"] = mae
        metrics["Shrinkage_xi_sd"] = sd

    # sparsity recovery if is_signal available
    if is_signal is not None:
        gamma_prob = np.asarray(shrink_aux["gamma_prob"], dtype=float).reshape(-1)
        ps, pn = prob_signal_noise(gamma_prob, is_signal)
        metrics["Shrinkage_prob_signal"] = ps
        metrics["Shrinkage_prob_noise"] = pn

    results["metrics"] = metrics

    # 3) optional MAP
    if run_cfg.include_map:
        print("[single] running: Lu25 MAP (optional) ...")
        results["Lu25MAP"] = _run_map_optional(markets, cfg=cfg, seed=run_cfg.seed)
    else:
        results["Lu25MAP"] = {"status": "skipped", "reason": "run with --include-map to enable"}

    # 4) write outputs
    summary_path = out_dir / "single_replication_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # write compact CSV table
    csv_path = out_dir / "single_replication_table.csv"
    lines = ["method,parameter,estimate,true_value,bias"]
    true = results["true_params"]

    def _add(method: str, param: str, est: float, tru: float) -> None:
        lines.append(f"{method},{param},{est:.6f},{tru:.6f},{(est-tru):+.6f}")

    # BLP cost
    _add("BLP+CostIV", "sigma", float(blp_cost["sigma"]), float(true["sigma"]))
    _add("BLP+CostIV", "beta_p", float(blp_cost["beta_p"]), float(true["beta_p"]))
    _add("BLP+CostIV", "beta_w", float(blp_cost["beta_w"]), float(true["beta_w"]))
    _add("BLP+CostIV", "int", float(blp_cost["int"]), float(true["int"]))

    # BLP no cost
    _add("BLP-NoCostIV", "sigma", float(blp_nocost["sigma"]), float(true["sigma"]))
    _add("BLP-NoCostIV", "beta_p", float(blp_nocost["beta_p"]), float(true["beta_p"]))
    _add("BLP-NoCostIV", "beta_w", float(blp_nocost["beta_w"]), float(true["beta_w"]))
    _add("BLP-NoCostIV", "int", float(blp_nocost["int"]), float(true["int"]))

    # shrinkage
    _add("Shrinkage", "sigma", float(shrink_res["sigma"]), float(true["sigma"]))
    _add("Shrinkage", "beta_p", float(shrink_res["beta_p"]), float(true["beta_p"]))
    _add("Shrinkage", "beta_w", float(shrink_res["beta_w"]), float(true["beta_w"]))
    _add("Shrinkage", "int", float(shrink_res["int"]), float(true["int"]))

    # optional MAP (biases may not be comparable; include for reference if ran)
    if run_cfg.include_map and isinstance(results["Lu25MAP"], dict) and results["Lu25MAP"].get("status") == "success":
        m = results["Lu25MAP"]
        _add("Lu25MAP", "sigma", float(m["sigma"]), float(true["sigma"]))
        _add("Lu25MAP", "beta_p", float(m["beta_p"]), float(true["beta_p"]))
        _add("Lu25MAP", "beta_w", float(m["beta_w"]), float(true["beta_w"]))

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[single] wrote: {summary_path}")
    print(f"[single] wrote: {csv_path}")
    print(f"[single] DONE  {run_id}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
