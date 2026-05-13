"""
coverage_study.py — Multi-seed coverage study for Laplace credible intervals.

The single-draw simulation in `simulation_study.py` reports 100% coverage
because the intervals are wide enough to contain almost any point estimate.
Empirical coverage is a frequentist property that requires repeated sampling
of the data-generating process; that is what this module computes.

For each seed:
  1. Generate a fresh synthetic panel from the DGP.
  2. Train the model end-to-end via the two-stage estimator.
  3. Compute exact MAP Hessian credible intervals.
  4. Record whether the truth falls inside each interval.

Aggregate empirical coverage rate per parameter group over the seeds.
"""
from __future__ import annotations

import json
import logging
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .data import simulate_dynamic_panel
from .intervals import compute_laplace_intervals
from .model import DynamicContextSparseChoiceModel
from .simulation_study import _fit_stage   # reuse the two-stage trainer step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_to_dict(cfg: DynamicModelConfig) -> Dict:
    """Serialise the config to a JSON-safe dict, ignoring @property fields."""
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}


def _compute_intervals_for_fresh_model(
    model: DynamicContextSparseChoiceModel,
    data: Dict[str, np.ndarray],
    cfg: DynamicModelConfig,
) -> Dict:
    """
    Compute one graph-mode Hessian interval pass for a newly fitted model.

    The coverage study deliberately fits a fresh model for every seed. Each
    model therefore needs its own Hessian graph, which TensorFlow's global
    frequent-tracing detector can mislabel as retracing when many seeds run in
    one Python process. The interval helper itself still uses a zero-argument
    tf.function with an input signature and traces once per fitted model.
    """
    tf_logger = tf.get_logger()
    previous_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)
    try:
        return compute_laplace_intervals(model, data, cfg, confidence=0.95)
    finally:
        tf_logger.setLevel(previous_level)


def _train_two_stages(
    cfg: DynamicModelConfig,
    data: Dict[str, np.ndarray],
) -> DynamicContextSparseChoiceModel:
    """Run the two-stage estimator on a single dataset."""
    model = DynamicContextSparseChoiceModel(cfg)

    # Stage 1: freeze Halo + value head, train econometric params with static NLL.
    model.halo.trainable = False
    model.market_embed.trainable = False
    model.value_head.trainable = False
    econometric_vars = [
        model.beta_price, model.lambda_control,
        model.mu, model.d, model.eta, model.logit_pi,
    ]
    _fit_stage(
        model, cfg, data, econometric_vars,
        epochs=cfg.epochs, lr=cfg.lr, label="S1", use_static_nll=True,
    )

    # Stage 2: unfreeze all, joint fine-tuning.
    model.halo.trainable = True
    model.market_embed.trainable = True
    model.value_head.trainable = True
    _fit_stage(
        model, cfg, data, model.trainable_variables,
        epochs=max(5, cfg.epochs // 3), lr=cfg.lr * 0.3, label="S2",
    )
    return model


def _evaluate_one_seed(
    model: DynamicContextSparseChoiceModel,
    data: Dict[str, np.ndarray],
    meta: Dict,
    cfg: DynamicModelConfig,
) -> Dict:
    """Compute coverage indicators and point estimates for one seed."""
    intervals = _compute_intervals_for_fresh_model(model, data, cfg)

    # Scalar coverage: is the truth inside the CI?
    beta_lo = float(intervals["beta_price"]["ci_lower"])
    beta_hi = float(intervals["beta_price"]["ci_upper"])
    beta_covered = bool(beta_lo <= cfg.true_beta_price <= beta_hi)

    # Vector coverage: proportion of element-wise intervals containing the truth.
    mu_in_ci = (
        (intervals["mu"]["ci_lower"] <= meta["mu_true"])
        & (meta["mu_true"] <= intervals["mu"]["ci_upper"])
    )
    mu_coverage = float(mu_in_ci.mean())

    d_in_ci = (
        (intervals["d"]["ci_lower"] <= meta["d_true"])
        & (meta["d_true"] <= intervals["d"]["ci_upper"])
    )
    d_coverage = float(d_in_ci.mean())

    # Direction (correlation) and scale metrics for mu.
    mu_hat = model.mu.numpy()
    mu_corr = float(np.corrcoef(mu_hat, meta["mu_true"])[0, 1])

    return {
        "beta_price_hat":   float(intervals["beta_price"]["estimate"]),
        "beta_price_se":    float(intervals["beta_price"]["se"]),
        "beta_price_ci":    [beta_lo, beta_hi],
        "beta_covered":     beta_covered,
        "lambda_control":   float(model.lambda_control.numpy()),
        "mu_coverage":      mu_coverage,
        "mu_corr_true":     mu_corr,
        "mu_std_hat":       float(mu_hat.std()),
        "d_coverage":       d_coverage,
        "eta_rmse":         float(np.sqrt(np.mean(
                                  (model.eta.numpy() - meta["eta_true"]) ** 2))),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_coverage_study(
    base_cfg: DynamicModelConfig | None = None,
    n_seeds: int = 20,
    out_dir: str | Path | None = None,
) -> Dict:
    """
    Run a multi-seed coverage study.

    For each seed in 0..n_seeds-1, generate data, train the model, compute
    intervals, and record coverage indicators. Aggregate empirical coverage
    rates across seeds.
    """
    if base_cfg is None:
        # Lightweight config for tractable multi-seed runs.
        base_cfg = DynamicModelConfig(
            num_households=50,
            T=10,
            epochs=15,
            batch_size=128,
            compile_train_step=True,
            force_cpu=True,
            seed=0,
        )

    if base_cfg.force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    per_seed: List[Dict] = []
    for seed in range(n_seeds):
        cfg = replace(base_cfg, seed=seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        print(f"\n========== Coverage seed {seed + 1}/{n_seeds} ==========")
        data, meta = simulate_dynamic_panel(cfg, seed=seed)
        model = _train_two_stages(cfg, data)
        result = _evaluate_one_seed(model, data, meta, cfg)
        result["seed"] = seed
        per_seed.append(result)

        print(f"  seed {seed}: beta_covered={result['beta_covered']}  "
              f"mu_cov={result['mu_coverage']:.2f}  "
              f"d_cov={result['d_coverage']:.2f}")

    # Aggregate
    beta_covered_rate = float(np.mean([r["beta_covered"] for r in per_seed]))
    mu_coverage_mean = float(np.mean([r["mu_coverage"] for r in per_seed]))
    d_coverage_mean = float(np.mean([r["d_coverage"] for r in per_seed]))

    beta_hat_mean = float(np.mean([r["beta_price_hat"] for r in per_seed]))
    beta_hat_std = float(np.std([r["beta_price_hat"] for r in per_seed]))
    lambda_mean = float(np.mean([r["lambda_control"] for r in per_seed]))
    mu_corr_mean = float(np.mean([r["mu_corr_true"] for r in per_seed]))
    eta_rmse_mean = float(np.mean([r["eta_rmse"] for r in per_seed]))

    summary = {
        "n_seeds": n_seeds,
        "config": _cfg_to_dict(base_cfg),
        "empirical_coverage_95": {
            "beta_price": beta_covered_rate,
            "mu":         mu_coverage_mean,
            "d":          d_coverage_mean,
        },
        "point_estimate_distribution": {
            "beta_price_mean":     beta_hat_mean,
            "beta_price_std":      beta_hat_std,
            "beta_price_true":     base_cfg.true_beta_price,
            "lambda_control_mean": lambda_mean,
            "mu_corr_true_mean":   mu_corr_mean,
            "eta_rmse_mean":       eta_rmse_mean,
        },
        "per_seed": per_seed,
    }

    print("\n========== Coverage Summary ==========")
    print(f"  Seeds:                {n_seeds}")
    print(f"  beta_price coverage:  {beta_covered_rate:.2f}")
    print(f"  mu coverage (mean):   {mu_coverage_mean:.2f}")
    print(f"  d coverage (mean):    {d_coverage_mean:.2f}")
    print(f"  beta_price mean:      {beta_hat_mean:.3f}  (std: {beta_hat_std:.3f})")
    print(f"  beta_price true:      {base_cfg.true_beta_price:.3f}")
    print(f"  lambda_control mean:  {lambda_mean:.3f}")
    print(f"  corr(mu_hat, true):   {mu_corr_mean:.3f}")
    print(f"  eta RMSE mean:        {eta_rmse_mean:.3f}")

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "coverage_study.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"\nSaved to: {out}")

    return summary


def main() -> None:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

    # Lightweight config for a 20-seed sweep.
    base_cfg = DynamicModelConfig(
        num_households=50,
        T=10,
        epochs=15,
        batch_size=128,
        compile_train_step=True,
        force_cpu=True,
        seed=0,
    )
    run_coverage_study(
        base_cfg=base_cfg,
        n_seeds=20,
        out_dir="results/bonus1/coverage_study",
    )


if __name__ == "__main__":
    main()
