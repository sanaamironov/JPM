"""Smoke-test entry point for the revised Bonus 1 dynamic model."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .counterfactual import price_promotion_analysis, print_counterfactual_summary
from .data import simulate_dynamic_panel
from .model import DynamicContextSparseChoiceModel
from .simulation_study import _fit_stage


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

    cfg = DynamicModelConfig()

    if cfg.force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    print("Simulating synthetic dynamic panel (revised Bonus 1 DGP)...")
    data, meta = simulate_dynamic_panel(cfg)
    print(
        f"  Households: {cfg.num_households}, T: {cfg.T}, J: {cfg.J}, "
        f"N obs: {len(data['choice'])}"
    )
    print(f"  True beta_price: {cfg.true_beta_price:.3f}")
    print(f"  Gamma endogeneity: {cfg.gamma_endogeneity:.3f}")

    model = DynamicContextSparseChoiceModel(cfg)

    # Stage 1: freeze Halo + value head, train econometric params with static NLL.
    print("\nStage 1: econometric params (Halo + value head frozen)...")
    model.halo.trainable = False
    model.market_embed.trainable = False
    model.value_head.trainable = False
    econometric_vars = [
        model.beta_price, model.lambda_control,
        model.mu, model.d, model.eta, model.logit_pi,
    ]
    _fit_stage(model, cfg, data, econometric_vars,
               epochs=cfg.epochs, lr=cfg.lr, label="S1", use_static_nll=True)

    # Stage 2: unfreeze all, joint fine-tuning.
    print("\nStage 2: joint fine-tuning (all params)...")
    model.halo.trainable = True
    model.market_embed.trainable = True
    model.value_head.trainable = True
    _fit_stage(model, cfg, data, model.trainable_variables,
               epochs=max(10, cfg.epochs // 3), lr=cfg.lr * 0.3, label="S2")

    # Diagnostics
    pi_hat = float(tf.math.sigmoid(model.logit_pi).numpy())
    beta_hat = float(model.beta_price.numpy())
    mu_sd_hat = float(tf.math.reduce_std(model.mu).numpy())
    mean_abs_d = float(tf.reduce_mean(tf.abs(model.d)).numpy())

    print(f"\nEstimated parameters:")
    print(f"  beta_price: {beta_hat:.4f}  (true: {cfg.true_beta_price:.3f})")
    print(f"  pi_hat:     {pi_hat:.4f}")
    print(f"  std(mu):    {mu_sd_hat:.4f}")
    print(f"  mean|d|:    {mean_abs_d:.4f}")

    # Support recovery
    d_hat = model.d.numpy()
    gamma_true = meta["gamma_true"]
    true_nz = gamma_true == 1
    true_z = gamma_true == 0
    print(f"\nSparsity support recovery (tau=0.15):")
    tau = 0.15
    gamma_hat = (np.abs(d_hat) > tau).astype(np.int32)
    sens = (
        float((gamma_hat[true_nz] == 1).mean()) if true_nz.sum() > 0 else float("nan")
    )
    spec = float((gamma_hat[true_z] == 0).mean()) if true_z.sum() > 0 else float("nan")
    print(f"  sensitivity={sens:.3f}  specificity={spec:.3f}")

    # Counterfactual
    cf = price_promotion_analysis(model, data, cfg, brand_x=1, discount_pct=10.0)
    print_counterfactual_summary(cf)

    # Save
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "J": cfg.J,
            "T": cfg.T,
            "num_households": cfg.num_households,
            "true_beta_price": cfg.true_beta_price,
        },
        "estimated": {
            "beta_price": beta_hat,
            "pi_hat": pi_hat,
            "std_mu": mu_sd_hat,
            "mean_abs_d": mean_abs_d,
            "support_sensitivity": sens,
            "support_specificity": spec,
        },
        "counterfactual": cf,
    }

    out_dir = Path("results/bonus1/dynamic_model") / f"demo_{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
