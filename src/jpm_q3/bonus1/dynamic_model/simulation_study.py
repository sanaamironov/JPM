"""
simulation_study.py — Part 3: Simulation and Parameter Recovery.

Generates data from the known DGP, estimates the model, computes
95% Laplace credible intervals, and reports true vs. estimated values.

Usage:
    python -m jpm_q3.bonus1.dynamic_model.simulation_study
    # or from the CLI entry point defined in pyproject.toml
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .data import simulate_dynamic_panel
from .intervals import compute_laplace_intervals, print_interval_summary
from .model import DynamicContextSparseChoiceModel
from .trainer import DynamicTrainer


def run_simulation_study(
    cfg: DynamicModelConfig | None = None,
    out_dir: str | Path | None = None,
    seed: int = 0,
) -> dict:
    """
    Full simulation study pipeline:
      1. Generate synthetic data from the DGP (with known true parameters).
      2. Estimate model via MAP (DynamicTrainer).
      3. Compute Laplace credible intervals.
      4. Report true vs estimated values.

    The estimator starts from a fresh random initialisation — it does NOT
    receive the true DeepHalo weights or true parameter values.

    Returns a summary dict suitable for JSON serialisation.
    """
    if cfg is None:
        cfg = DynamicModelConfig(
            num_households=150,
            T=20,
            epochs=30,
            batch_size=256,
            compile_train_step=True,
            force_cpu=True,
            seed=seed,
        )

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    if cfg.force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 1: Generate data
    # ------------------------------------------------------------------
    print(f"[simulation_study] Generating panel: I={cfg.num_households}, T={cfg.T}, J={cfg.J}")
    data, meta = simulate_dynamic_panel(cfg, seed=seed)
    print(f"  N observations: {len(data['choice'])}")
    print(f"  True beta_price: {cfg.true_beta_price:.3f}")
    print(f"  True mu range:   [{meta['mu_true'].min():.3f}, {meta['mu_true'].max():.3f}]")
    print(f"  Nonzero d frac:  {meta['gamma_true'].mean():.3f}")

    # ------------------------------------------------------------------
    # Step 2: Estimate — fresh model, NOT given true weights
    # ------------------------------------------------------------------
    print("\n[simulation_study] Training model...")
    model = DynamicContextSparseChoiceModel(cfg)
    # Freeze Halo in first-pass training so sparse shocks explain residuals.
    # A full joint run would be two-stage (freeze → unfreeze).
    model.halo.trainable = False

    trainer = DynamicTrainer(model, cfg)
    trainer.fit(data)

    # ------------------------------------------------------------------
    # Step 3: Laplace credible intervals
    # ------------------------------------------------------------------
    print("\n[simulation_study] Computing Laplace credible intervals...")
    intervals = compute_laplace_intervals(model, data, cfg, confidence=0.95)

    # ------------------------------------------------------------------
    # Step 4: Report
    # ------------------------------------------------------------------
    print_interval_summary(intervals, meta={"true_beta_price": cfg.true_beta_price,
                                             "mu_true": meta["mu_true"]})

    # Coverage: does the CI contain the true value?
    beta_lo = float(intervals["beta_price"]["ci_lower"])
    beta_hi = float(intervals["beta_price"]["ci_upper"])
    beta_covered = bool(beta_lo <= cfg.true_beta_price <= beta_hi)

    mu_covered = float(np.mean(
        (intervals["mu"]["ci_lower"] <= meta["mu_true"]) &
        (meta["mu_true"] <= intervals["mu"]["ci_upper"])
    ))
    d_covered = float(np.mean(
        (intervals["d"]["ci_lower"] <= meta["d_true"]) &
        (meta["d_true"] <= intervals["d"]["ci_upper"])
    ))

    print(f"\n[simulation_study] Coverage summary (95% nominal):")
    print(f"  beta_price  covered: {beta_covered}")
    print(f"  mu          coverage: {mu_covered:.2f}")
    print(f"  d           coverage: {d_covered:.2f}")

    summary = {
        "config": {
            "J": cfg.J,
            "T": cfg.T,
            "num_households": cfg.num_households,
            "epochs": cfg.epochs,
            "true_beta_price": cfg.true_beta_price,
            "gamma_endogeneity": cfg.gamma_endogeneity,
            "sparse_frac": cfg.sparse_frac,
        },
        "true": {
            "beta_price": float(cfg.true_beta_price),
            "mu_mean": float(meta["mu_true"].mean()),
            "mu_std": float(meta["mu_true"].std()),
            "gamma_true_mean": float(meta["gamma_true"].mean()),
        },
        "estimated": {
            "beta_price": float(intervals["beta_price"]["estimate"]),
            "beta_price_se": float(intervals["beta_price"]["se"]),
            "beta_price_ci": [float(intervals["beta_price"]["ci_lower"]),
                              float(intervals["beta_price"]["ci_upper"])],
            "pi_hat": float(tf.math.sigmoid(model.logit_pi).numpy()),
            "mu_rmse": float(np.sqrt(np.mean((model.mu.numpy() - meta["mu_true"])**2))),
        },
        "coverage": {
            "beta_price": beta_covered,
            "mu": mu_covered,
            "d": d_covered,
        },
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "simulation_study.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"\n[simulation_study] Results saved to {out}")

    return summary


def main() -> None:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

    cfg = DynamicModelConfig(
        num_households=150,
        T=20,
        epochs=30,
        batch_size=256,
        compile_train_step=True,
        force_cpu=True,
        seed=0,
    )
    run_simulation_study(cfg, out_dir="results/bonus1/simulation_study")


if __name__ == "__main__":
    main()
