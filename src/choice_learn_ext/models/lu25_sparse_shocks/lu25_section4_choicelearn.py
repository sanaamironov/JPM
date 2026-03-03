from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np

from jpm_q3.lu25.simulation.config import SimConfig
from jpm_q3.lu25.simulation.simulate import simulate_dataset
from choice_learn_ext.models.lu25_sparse_shocks import MarketShareDataset, Lu25SparseShocksEstimator


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lu(25) Section 4 replication using choice_learn_ext estimator wrapper.")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--R-mc", type=int, default=50)
    p.add_argument("--n-reps", type=int, default=5)
    p.add_argument("--grid", type=str, default="DGP1:25:15,DGP2:25:15,DGP3:25:15,DGP4:25:15")
    return p.parse_args(argv)


def parse_grid(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        dgp, T, J = tok.split(":")
        out.append((dgp.strip().upper(), int(T), int(J)))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig()
    grid = parse_grid(args.grid)

    # Choice-learn integrated estimator
    est = Lu25SparseShocksEstimator(R_mc=int(args.R_mc), base_seed=int(args.seed))

    for dgp, T, J in grid:
        cell_dir = out_root / f"{dgp}_T{T}_J{J}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        rows = []  # collect per-rep estimates (lightweight)
        for r in range(int(args.n_reps)):
            rep_seed = int(args.seed) + 10_000 * r + 123
            markets = simulate_dataset(dgp, T=T, J=J, cfg=cfg, seed=rep_seed)
            ds = MarketShareDataset.from_markets_dicts(markets)

            # Benchmarks (BLP)
            blp_cost = est.fit(ds, mode="blp", iv_type="cost")
            blp_nocost = est.fit(ds, mode="blp", iv_type="nocost")

            # Shrinkage
            sh = est.fit(ds, mode="shrinkage")

            rows.append(
                {
                    "rep": r,
                    "blp_cost_sigma": blp_cost.sigma_hat,
                    "blp_cost_beta_p": blp_cost.beta_hat[0],
                    "blp_cost_beta_w": blp_cost.beta_hat[1],
                    "blp_nocost_sigma": blp_nocost.sigma_hat,
                    "blp_nocost_beta_p": blp_nocost.beta_hat[0],
                    "blp_nocost_beta_w": blp_nocost.beta_hat[1],
                    "sh_sigma": sh.sigma_hat,
                    "sh_beta_p": sh.beta_hat[0],
                    "sh_beta_w": sh.beta_hat[1],
                    "sh_acc": sh.acc_rate if sh.acc_rate is not None else np.nan,
                    "sh_score": sh.score_hat if sh.score_hat is not None else np.nan,
                }
            )

        # Write a simple CSV (this script is a demo runner; your canonical runner remains primary)
        import csv
        out_path = cell_dir / "choicelearn_replications.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"[choicelearn] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())