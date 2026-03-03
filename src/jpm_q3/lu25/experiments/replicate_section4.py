"""
replicate_section4.py

Paper-aligned replication driver for Lu & Shimizu (2025), Section 4.

Key paper-alignment choices (important):
1) Regressors X exclude the constant:
      X = [p, w]
   This makes the residual equal to the paper's full xi_{jt} = bar_xi + eta_{jt}.
2) Instruments Z also exclude the constant:
   - cost IV:   Z = [w, w^2, u, u^2]
   - no-cost IV Z = [w, w^2, w^3, w^4]
   With bar_xi != 0, including a constant in Z would violate E[Z * xi] = 0.
3) "Int" in the paper is bar_xi. We estimate it as:
      Int_hat = mean(xi_hat) across all (j,t).
4) "xi" column is computed from xi_hat vs xi_true:
      mean_abs_error = mean(|xi_hat - xi_true|)
      sd_error       = sd(xi_hat - xi_true)

Outputs:
- summary.csv          (long format)
- paper_table_like.csv (Bias/SD rows like Table 2)

CLI examples (repo root):
  jpmq3-replicate-lu25 --smoke --cpu --out results/lu25_smoke
  jpmq3-replicate-lu25 --cpu --out results/lu25_section4 --R-mc 50 --seed 0 --shrink-n-iter 800 --shrink-burn 400
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# --- Project imports ---
from ..simulation.config import SimConfig
from ..simulation.simulate import simulate_dataset

# BLP primitives
from ..estimators.blp import compute_delta_vec, iv_2sls_beta  # type: ignore

# Shrinkage primitives
from ..estimators.shrinkage import shrinkage_fit_beta_given_sigma  # type: ignore


# -----------------------------------------------------------------------------
# Argument optional acceptacet for multi core programming 
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--R-mc", type=int, default=200, help="Num simulation draws R used in delta computation")
    p.add_argument("--n-reps", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="Small/fast run")

    # multicore
    p.add_argument("--n-jobs", type=int, default=1, help="Number of worker processes")
    p.add_argument("--threads-per-job", type=int, default=1, help="Threads used inside each worker process")

    # Shrinkage controls
    p.add_argument("--shrink-n-iter", type=int, default=800)
    p.add_argument("--shrink-burn", type=int, default=400)
    p.add_argument("--shrink-thin", type=int, default=2)

    return p.parse_args(argv)



# -----------------------------------------------------------------------------
# Helpers: stacking true objects from simulated markets
# -----------------------------------------------------------------------------


def stack_true_xi(markets: List[dict]) -> np.ndarray:
    xs = []
    for m in markets:
        if "xi_true" not in m:
            raise KeyError("market missing xi_true")
        xs.append(np.asarray(m["xi_true"], dtype=float).reshape(-1))
    return np.concatenate(xs, axis=0)


def stack_is_signal(markets: List[dict]) -> np.ndarray:
    ss = []
    for m in markets:
        if "is_signal" not in m:
            raise KeyError("market missing is_signal")
        ss.append(np.asarray(m["is_signal"], dtype=int).reshape(-1))
    return np.concatenate(ss, axis=0)


# -----------------------------------------------------------------------------
# Paper-aligned matrices (NO constant in X or Z)
# -----------------------------------------------------------------------------


def build_matrices_paper(markets: List[dict], iv_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper-aligned stacking.

    X = [p, w]
    Z = instruments WITHOUT a constant
      cost   : [w, w^2, u, u^2]
      nocost : [w, w^2, w^3, w^4]
    """
    Xs, Zs = [], []
    for m in markets:
        p = np.asarray(m["p"], dtype=float).reshape(-1)
        w = np.asarray(m["w"], dtype=float).reshape(-1)

        X = np.column_stack([p, w])

        if iv_type == "cost":
            if "u" not in m:
                raise KeyError("iv_type='cost' requires market['u'] (cost shock)")
            u = np.asarray(m["u"], dtype=float).reshape(-1)
            Z = np.column_stack([w, w**2, u, u**2])
        elif iv_type == "nocost":
            Z = np.column_stack([w, w**2, w**3, w**4])
        else:
            raise ValueError("iv_type must be 'cost' or 'nocost'")

        Xs.append(X)
        Zs.append(Z)

    return np.vstack(Xs), np.vstack(Zs)


# -----------------------------------------------------------------------------
# Metrics (paper table columns)
# -----------------------------------------------------------------------------


def mean_abs_and_sd(err: np.ndarray) -> Tuple[float, float]:
    e = np.asarray(err, dtype=float).reshape(-1)
    return float(np.mean(np.abs(e))), float(np.std(e, ddof=1))


def prob_signal_noise(gamma_prob: np.ndarray, is_signal: np.ndarray) -> Tuple[float, float]:
    g = np.asarray(gamma_prob, dtype=float).reshape(-1)
    s = np.asarray(is_signal, dtype=int).reshape(-1)
    ps = float(np.mean(g[s == 1])) if np.any(s == 1) else float("nan")
    pn = float(np.mean(g[s == 0])) if np.any(s == 0) else float("nan")
    return ps, pn


# -----------------------------------------------------------------------------
# Paper-aligned BLP estimation (grid + refine), using X/Z above
# -----------------------------------------------------------------------------


def gmm_objective_for_sigma_paper(
    sigma: float, markets: List[dict], iv_type: str, R: int
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-step GMM objective, W = (Z'Z / N)^(-1), paper-aligned X/Z (no constant).

    Returns:
      obj, beta_hat, delta_vec, X, Z, xi_hat
    """
    delta_vec = compute_delta_vec(markets, float(sigma), R=R)
    X, Z = build_matrices_paper(markets, iv_type=iv_type)

    beta_hat = iv_2sls_beta(delta_vec, X, Z).numpy()  # shape (2,)
    xi_hat = delta_vec - X @ beta_hat  # this is full xi = bar_xi + eta

    N = xi_hat.shape[0]
    g = (Z.T @ xi_hat) / N
    W = np.linalg.inv((Z.T @ Z) / N)
    obj = float(g.T @ W @ g)

    return obj, beta_hat, delta_vec, X, Z, xi_hat


def estimate_blp_sigma_paper(markets: List[dict], iv_type: str, R: int) -> Tuple[float, np.ndarray, dict]:
    grid = np.linspace(0.05, 4.0, 40)
    best_obj = None
    best_pack = None

    for s in grid:
        obj, beta, delta_vec, X, Z, xi_hat = gmm_objective_for_sigma_paper(s, markets, iv_type=iv_type, R=R)
        if (best_obj is None) or (obj < best_obj):
            best_obj = obj
            best_pack = (float(s), beta, delta_vec, X, Z, xi_hat, float(obj))

    s0 = float(best_pack[0])
    refine = np.linspace(max(0.01, s0 - 0.25), s0 + 0.25, 30)

    for s in refine:
        obj, beta, delta_vec, X, Z, xi_hat = gmm_objective_for_sigma_paper(s, markets, iv_type=iv_type, R=R)
        if obj < best_obj:
            best_obj = obj
            best_pack = (float(s), beta, delta_vec, X, Z, xi_hat, float(obj))

    sigma_hat, beta_hat, delta_hat, X, Z, xi_hat, obj_hat = best_pack
    extras = {
        "obj_hat": float(obj_hat),
        "delta_hat": np.asarray(delta_hat, dtype=float),
        "X": np.asarray(X, dtype=float),
        "Z": np.asarray(Z, dtype=float),
        "xi_hat": np.asarray(xi_hat, dtype=float),
    }
    return float(sigma_hat), np.asarray(beta_hat, dtype=float), extras


# -----------------------------------------------------------------------------
# Paper-aligned shrinkage sigma search (uses paper X, no constant)
# -----------------------------------------------------------------------------


def shrinkage_objective_for_sigma_paper(
    sigma: float, markets: List[dict], R: int, **kwargs
) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      score, beta_hat, gamma_prob, acc_rate, delta_vec, X, xi_hat
    """
    delta_vec = compute_delta_vec(markets, float(sigma), R=R)
    X, _ = build_matrices_paper(markets, iv_type="nocost")  # shrinkage uses X only; paper uses X=[p,w]

    beta_hat, gamma_prob, score, acc_rate = shrinkage_fit_beta_given_sigma(delta_vec, X, **kwargs)
    xi_hat = delta_vec - X @ beta_hat  # full xi

    return float(score), np.asarray(beta_hat, dtype=float), np.asarray(gamma_prob, dtype=float), float(acc_rate), delta_vec, X, xi_hat


def estimate_shrinkage_sigma_paper(
    markets: List[dict],
    R: int,
    sigma_grid: Optional[np.ndarray] = None,
    return_extras: bool = False,
    **kwargs,
):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.05, 4.0, 40)

    best = None  # (score, sigma, beta, gamma, delta, X, xi_hat)
    best_acc = None

    for s in sigma_grid:
        score, beta_hat, gamma_prob, acc_rate, delta_vec, X, xi_hat = shrinkage_objective_for_sigma_paper(
            s, markets, R=R, **kwargs
        )
        if (best is None) or (score > best[0]):
            best = (score, float(s), beta_hat, gamma_prob, delta_vec, X, xi_hat)
            best_acc = acc_rate

    s0 = float(best[1])
    refine = np.linspace(max(0.01, s0 - 0.25), s0 + 0.25, 25)

    for s in refine:
        score, beta_hat, gamma_prob, acc_rate, delta_vec, X, xi_hat = shrinkage_objective_for_sigma_paper(
            s, markets, R=R, **kwargs
        )
        if score > best[0]:
            best = (score, float(s), beta_hat, gamma_prob, delta_vec, X, xi_hat)
            best_acc = acc_rate

    score_hat, sigma_hat, beta_hat, gamma_prob, delta_hat, X, xi_hat = best

    if not return_extras:
        return float(sigma_hat), np.asarray(beta_hat, dtype=float), float(score_hat), np.asarray(gamma_prob, dtype=float)

    extras = {
        "acc_rate": float(best_acc) if best_acc is not None else float("nan"),
        "delta_hat": np.asarray(delta_hat, dtype=float),
        "X": np.asarray(X, dtype=float),
        "xi_hat": np.asarray(xi_hat, dtype=float),
    }
    return float(sigma_hat), np.asarray(beta_hat, dtype=float), float(score_hat), np.asarray(gamma_prob, dtype=float), extras


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    dgp: str
    T: int
    J: int


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _warn_exception(msg: str, e: Exception) -> None:
    print(f"[WARN] {msg}: {type(e).__name__}: {e}")
    # Keep traceback short (still helpful)
    tb = "".join(traceback.format_tb(e.__traceback__, limit=3))
    print(tb)


def run_one_rep(
    *,
    dgp: str,
    T: int,
    J: int,
    cfg: SimConfig,
    rep_seed: int,
    R_mc: int,
    shrink_kwargs: dict,
) -> Dict[str, dict]:
    """
    Returns dict keyed by method name with paper-style fields:
      int, beta_p, beta_w, sigma, xi_bias_abs, xi_sd, prob_signal, prob_noise, fail
    """
    out = {}

    markets = simulate_dataset(dgp, T=T, J=J, cfg=cfg, seed=rep_seed)

    # true objects (optional in your simulator; if missing, we leave NaN metrics)
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

    # -------------------------
    # 1) BLP + CostIV (paper X/Z)
    # -------------------------
    method = "BLP+CostIV"
    rec = dict(
        int=float("nan"),
        beta_p=float("nan"),
        beta_w=float("nan"),
        sigma=float("nan"),
        xi_bias_abs=float("nan"),
        xi_sd=float("nan"),
        prob_signal=float("nan"),
        prob_noise=float("nan"),
        fail=0,
    )
    try:
        sigma_hat, beta_hat, extras = estimate_blp_sigma_paper(markets, iv_type="cost", R=R_mc)
        xi_hat = np.asarray(extras["xi_hat"], dtype=float).reshape(-1)

        rec["sigma"] = float(sigma_hat)
        rec["beta_p"] = float(beta_hat[0])
        rec["beta_w"] = float(beta_hat[1])
        rec["int"] = float(np.mean(xi_hat))  # paper's bar_xi estimate

        if xi_true is not None:
            mae, sd = mean_abs_and_sd(xi_hat - xi_true)
            rec["xi_bias_abs"] = mae
            rec["xi_sd"] = sd
    except Exception as e:
        rec["fail"] = 1
        _warn_exception(f"{method} failed (rep={rep_seed})", e)
    out[method] = rec

    # -------------------------
    # 2) BLP - CostIV (paper X/Z)
    # -------------------------
    method = "BLP-NoCostIV"
    rec = dict(
        int=float("nan"),
        beta_p=float("nan"),
        beta_w=float("nan"),
        sigma=float("nan"),
        xi_bias_abs=float("nan"),
        xi_sd=float("nan"),
        prob_signal=float("nan"),
        prob_noise=float("nan"),
        fail=0,
    )
    try:
        sigma_hat, beta_hat, extras = estimate_blp_sigma_paper(markets, iv_type="nocost", R=R_mc)
        xi_hat = np.asarray(extras["xi_hat"], dtype=float).reshape(-1)

        rec["sigma"] = float(sigma_hat)
        rec["beta_p"] = float(beta_hat[0])
        rec["beta_w"] = float(beta_hat[1])
        rec["int"] = float(np.mean(xi_hat))

        if xi_true is not None:
            mae, sd = mean_abs_and_sd(xi_hat - xi_true)
            rec["xi_bias_abs"] = mae
            rec["xi_sd"] = sd
    except Exception as e:
        rec["fail"] = 1
        _warn_exception(f"{method} failed (rep={rep_seed})", e)
    out[method] = rec

    # -------------------------
    # 3) Shrinkage (paper X, no constant)
    # -------------------------
    method = "Shrinkage"
    rec = dict(
        int=float("nan"),
        beta_p=float("nan"),
        beta_w=float("nan"),
        sigma=float("nan"),
        xi_bias_abs=float("nan"),
        xi_sd=float("nan"),
        prob_signal=float("nan"),
        prob_noise=float("nan"),
        fail=0,
    )
    try:
        sigma_hat, beta_hat, score_hat, gamma_prob, extras = estimate_shrinkage_sigma_paper(
            markets, R=R_mc, return_extras=True, **shrink_kwargs
        )
        xi_hat = np.asarray(extras["xi_hat"], dtype=float).reshape(-1)

        rec["sigma"] = float(sigma_hat)
        rec["beta_p"] = float(beta_hat[0])
        rec["beta_w"] = float(beta_hat[1])
        rec["int"] = float(np.mean(xi_hat))

        if xi_true is not None:
            mae, sd = mean_abs_and_sd(xi_hat - xi_true)
            rec["xi_bias_abs"] = mae
            rec["xi_sd"] = sd

        if is_signal is not None and gamma_prob is not None:
            ps, pn = prob_signal_noise(gamma_prob, is_signal)
            rec["prob_signal"] = ps
            rec["prob_noise"] = pn
    except Exception as e:
        rec["fail"] = 1
        _warn_exception(f"{method} failed (rep={rep_seed})", e)
    out[method] = rec

    return out


def summarize_methods(method_recs: List[dict], true_params: dict) -> Dict[str, dict]:
    """
    method_recs: list of dicts, each dict is method->fields for one replication.
    Return: method -> summary dict with Bias/SD for each param, and mean xi metrics, etc.
    """
    methods = sorted({m for d in method_recs for m in d.keys()})
    out = {}

    for m in methods:
        rows = [d[m] for d in method_recs if m in d]
        out_m = {}

        # parameters with paper bias/sd
        for k in ["int", "beta_p", "beta_w", "sigma"]:
            vals = np.array([r[k] for r in rows], dtype=float)
            ok = np.isfinite(vals)
            if np.any(ok):
                bias = float(np.mean(vals[ok] - true_params[k]))
                sd = float(np.std(vals[ok], ddof=1))
            else:
                bias, sd = float("nan"), float("nan")
            out_m[f"{k}_bias"] = bias
            out_m[f"{k}_sd"] = sd

        # xi metrics: average across replications
        for k in ["xi_bias_abs", "xi_sd", "prob_signal", "prob_noise"]:
            vals = np.array([r[k] for r in rows], dtype=float)
            ok = np.isfinite(vals)
            out_m[k] = float(np.mean(vals[ok])) if np.any(ok) else float("nan")

        out_m["fail_rate"] = float(np.mean([r["fail"] for r in rows]))
        out[m] = out_m

    return out


def write_outputs(out_dir: Path, grid: GridPoint, summary: Dict[str, dict], true_params: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # summary.csv (long format)
    lines = ["dgp,T,J,method,metric,value"]
    for method, sm in summary.items():
        for k, v in sm.items():
            lines.append(f"{grid.dgp},{grid.T},{grid.J},{method},{k},{v}")
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # paper_table_like.csv (wide-ish, bias/sd rows)
    # rows: Bias, SD
    header = [
        "DGP",
        "T",
        "J",
        "Method",
        "Row",
        "Int",
        "beta_p",
        "beta_w",
        "sigma",
        "xi_mean_abs_error",
        "xi_sd_error",
        "Prob_signal",
        "Prob_noise",
        "FailRate",
    ]
    rows = [",".join(header)]
    for method, sm in summary.items():
        # Bias row
        rows.append(
            ",".join(
                [
                    grid.dgp,
                    str(grid.T),
                    str(grid.J),
                    method,
                    "Bias",
                    str(sm["int_bias"]),
                    str(sm["beta_p_bias"]),
                    str(sm["beta_w_bias"]),
                    str(sm["sigma_bias"]),
                    str(sm["xi_bias_abs"]),
                    str(sm["xi_sd"]),
                    str(sm["prob_signal"]),
                    str(sm["prob_noise"]),
                    str(sm["fail_rate"]),
                ]
            )
        )
        # SD row
        rows.append(
            ",".join(
                [
                    grid.dgp,
                    str(grid.T),
                    str(grid.J),
                    method,
                    "SD",
                    str(sm["int_sd"]),
                    str(sm["beta_p_sd"]),
                    str(sm["beta_w_sd"]),
                    str(sm["sigma_sd"]),
                    "",  # paper lists xi as metrics; SD row for xi columns usually blank; keep blank.
                    "",
                    "",
                    "",
                    str(sm["fail_rate"]),
                ]
            )
        )

    (out_dir / "paper_table_like.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Also dump a small config snapshot for reproducibility
    cfg_dump = {
        "dgp": grid.dgp,
        "T": grid.T,
        "J": grid.J,
        "true_params": true_params,
        "timestamp": _now(),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")


def _set_thread_env(threads: int) -> None:
    # Keep each worker from spawning its own 16 threads.
    # This matters a lot when you run 16 processes.
    t = str(max(1, int(threads)))
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["NUMEXPR_NUM_THREADS"] = t
    os.environ["TF_NUM_INTRAOP_THREADS"] = t
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"


def _worker_run_one_rep(args_tuple):
    """
    Top-level worker function for multiprocessing.
    Must be importable/picklable (macOS spawn).
    """
    dgp, T, J, rep_seed, R_mc, shrink_kwargs, threads_per_job = args_tuple
    _set_thread_env(int(threads_per_job))

    cfg = SimConfig()
    return run_one_rep(
        dgp=dgp,
        T=T,
        J=J,
        cfg=cfg,
        rep_seed=int(rep_seed),
        R_mc=int(R_mc),
        shrink_kwargs=dict(shrink_kwargs),
    )


def _run_reps_parallel(tasks: List[tuple], n_jobs: int) -> List[Dict[str, dict]]:
    results: List[Dict[str, dict]] = [None] * len(tasks)  # type: ignore

    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        fut_to_idx = {ex.submit(_worker_run_one_rep, task): i for i, task in enumerate(tasks)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            results[i] = fut.result()

    return results

#updated code to use multipprocess/mutilthread for high computation.
#warning this is on a mac chip max4
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)

    if args.smoke:
        n_reps = 5
        R_mc = 50
    else:
        n_reps = int(args.n_reps)
        R_mc = int(args.R_mc)

    # macOS safety: use spawn (avoids fork issues with TF/BLAS)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    grid_points = [
        GridPoint("DGP1", 25, 15),
        GridPoint("DGP2", 25, 15),
        GridPoint("DGP3", 25, 15),
        GridPoint("DGP4", 25, 15),
    ]

    shrink_kwargs = dict(
        n_iter=int(args.shrink_n_iter),
        burn=int(args.shrink_burn),
        thin=int(args.shrink_thin),
    )

    for gp in grid_points:
        cfg = SimConfig()
        true_params = {
            "int": float(cfg.xi_bar_star),
            "sigma": float(cfg.sigma_star),
            "beta_p": float(cfg.beta_p_star),
            "beta_w": float(cfg.beta_w_star),
        }

        # seeds per replication
        rep_seeds = [int(args.seed) + 10_000 * r + 123 for r in range(n_reps)]

        # run reps (possibly parallel)
        if int(args.n_jobs) <= 1:
            reps = [
                run_one_rep(
                    dgp=gp.dgp,
                    T=gp.T,
                    J=gp.J,
                    cfg=cfg,
                    rep_seed=s,
                    R_mc=R_mc,
                    shrink_kwargs=shrink_kwargs,
                )
                for s in rep_seeds
            ]
        else:
            # dispatch tasks across processes
            tasks = [
                (gp.dgp, gp.T, gp.J, s, R_mc, shrink_kwargs, int(args.threads_per_job))
                for s in rep_seeds
            ]
            reps = _run_reps_parallel(tasks, n_jobs=int(args.n_jobs))

        summary = summarize_methods(reps, true_params=true_params)
        out_dir = out_root / f"{gp.dgp}_T{gp.T}_J{gp.J}"
        write_outputs(out_dir, gp, summary, true_params)

        print(f"\n=== {gp.dgp}  T={gp.T}  J={gp.J}  reps={n_reps}  R={R_mc}  jobs={args.n_jobs} ===")
        for method, sm in summary.items():
            print(
                f"{method:12s}  "
                f"bias(beta_p)={sm['beta_p_bias']:+.3f} sd={sm['beta_p_sd']:.3f}  "
                f"bias(beta_w)={sm['beta_w_bias']:+.3f} sd={sm['beta_w_sd']:.3f}  "
                f"bias(sig)={sm['sigma_bias']:+.3f} sd={sm['sigma_sd']:.3f}  "
                f"bias(Int)={sm['int_bias']:+.3f} sd={sm['int_sd']:.3f}  "
                f"xi_mae={sm['xi_bias_abs']:.3f} xi_sd={sm['xi_sd']:.3f}  "
                f"fail={sm['fail_rate']:.2%}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

