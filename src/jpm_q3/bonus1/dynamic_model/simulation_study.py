"""
simulation_study.py — Part 3: Simulation and Parameter Recovery.

Training strategy (two-stage):
  Stage 1 — econometric estimation:
    Freeze Halo AND value head. Train only beta_price, mu, d, logit_pi.
    Objective: NLL + priors only (no TD loss).
    This isolates the econometric parameters from the value function
    approximation that would otherwise compete for the same signal.

  Stage 2 — joint fine-tuning:
    Unfreeze all parameters. Train jointly (NLL + TD + priors).
    The econometric parameters are now initialised near their true values,
    so the Halo and value head can refine rather than confound.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .data import simulate_dynamic_panel
from .intervals import compute_laplace_intervals, print_interval_summary
from .model import DynamicContextSparseChoiceModel


# ---------------------------------------------------------------------------
# Two-stage trainer
# ---------------------------------------------------------------------------

def _make_stage_trainer(
    model: DynamicContextSparseChoiceModel,
    cfg: DynamicModelConfig,
    train_vars: list,
    epochs: int,
    lr: float = 1e-3,
) -> None:
    """Train for `epochs` epochs updating only `train_vars`."""
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    tensors = None  # lazy — built on first call

    def _step(batch):
        cur = {k: batch[k] for k in
               ["item_ids", "available", "price", "market_id", "inventory", "choice"]}
        with tf.GradientTape() as tape:
            nll = model.choice_nll(cur, training=True)
            prior = model.sparse_shock_prior_penalty()
            loss = nll + float(cfg.prior_weight) * prior
        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        opt.apply_gradients(pairs)
        return loss, nll

    return _step


def _fit_stage(
    model: DynamicContextSparseChoiceModel,
    cfg: DynamicModelConfig,
    data: Dict[str, np.ndarray],
    train_vars: list,
    epochs: int,
    lr: float = 1e-3,
    label: str = "Stage",
    use_static_nll: bool = False,
) -> None:
    """
    Train `train_vars` for `epochs` epochs.

    If `use_static_nll=True`, the NLL is computed using only the static utility
    components (no continuation value). This is appropriate for Stage 1 where the
    value head is frozen at its random initialisation — including the random
    continuation value would otherwise bias mu_t toward zero via competition with
    the value head's market_embed.
    """
    tensors = {k: tf.constant(v) for k, v in data.items()}
    ds = (
        tf.data.Dataset.from_tensor_slices(tensors)
        .shuffle(4096, seed=cfg.seed)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    nll_fn = model.static_choice_nll if use_static_nll else model.choice_nll

    @tf.function
    def step(batch):
        cur = {k: batch[k] for k in
               ["item_ids", "available", "price", "price_residual",
                "market_id", "household_id", "inventory", "choice"]}
        with tf.GradientTape() as tape:
            nll = nll_fn(cur, training=True)
            prior = model.sparse_shock_prior_penalty()
            loss = nll + float(cfg.prior_weight) * prior
        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        opt.apply_gradients(pairs)
        return loss, nll

    for ep in range(1, epochs + 1):
        m_loss = tf.keras.metrics.Mean()
        m_nll = tf.keras.metrics.Mean()
        for batch in ds:
            loss, nll = step(batch)
            m_loss.update_state(loss)
            m_nll.update_state(nll)
        print(f"  {label} Epoch {ep:03d} | loss={m_loss.result():.4f}  nll={m_nll.result():.4f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_simulation_study(
    cfg: DynamicModelConfig | None = None,
    out_dir: str | Path | None = None,
    seed: int = 0,
) -> dict:
    """
    Full simulation study:
      1. Generate data from the known DGP.
      2. Estimate via two-stage MAP.
      3. Compute exact MAP Hessian credible intervals.
      4. Report true vs. estimated values and coverage.
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
    print(f"  N observations:  {len(data['choice'])}")
    print(f"  True beta_price: {cfg.true_beta_price:.3f}")
    print(f"  True mu range:   [{meta['mu_true'].min():.3f}, {meta['mu_true'].max():.3f}]")
    print(f"  Nonzero d frac:  {meta['gamma_true'].mean():.3f}")

    # ------------------------------------------------------------------
    # Step 2: Two-stage estimation
    # ------------------------------------------------------------------
    model = DynamicContextSparseChoiceModel(cfg)

    # Stage 1: Freeze Halo + value head. Train econometric params only.
    # This prevents the value head's market_embed from absorbing mu_t signal.
    print("\n[simulation_study] Stage 1: econometric params (Halo + value head frozen)...")
    model.halo.trainable = False
    model.market_embed.trainable = False
    model.value_head.trainable = False

    econometric_vars = [
        model.beta_price, model.lambda_control,
        model.mu, model.d, model.eta, model.logit_pi,
    ]
    _fit_stage(model, cfg, data, econometric_vars,
               epochs=cfg.epochs, lr=cfg.lr, label="S1",
               use_static_nll=True)

    print(f"  beta_price     after Stage 1: {float(model.beta_price.numpy()):.4f}  "
          f"(true: {cfg.true_beta_price:.3f})")
    print(f"  lambda_control after Stage 1: {float(model.lambda_control.numpy()):.4f}  "
          f"(Petrin-Train residual coef)")
    print(f"  std(mu)        after Stage 1: {float(tf.math.reduce_std(model.mu).numpy()):.4f}")

    # Stage 2: Unfreeze everything, fine-tune jointly.
    print("\n[simulation_study] Stage 2: joint fine-tuning (all params)...")
    model.halo.trainable = True
    model.market_embed.trainable = True
    model.value_head.trainable = True

    all_vars = model.trainable_variables
    _fit_stage(model, cfg, data, all_vars,
               epochs=max(10, cfg.epochs // 3), lr=cfg.lr * 0.3, label="S2")

    print(f"\n  beta_price final:    {float(model.beta_price.numpy()):.4f}  "
          f"(true: {cfg.true_beta_price:.3f})")
    print(f"  lambda_control final: {float(model.lambda_control.numpy()):.4f}")
    print(f"  std(mu) final:       {float(tf.math.reduce_std(model.mu).numpy()):.4f}  "
          f"(true: {cfg.mu_true_sd:.3f})")
    mu_hat = model.mu.numpy()
    mu_corr = float(np.corrcoef(mu_hat, meta["mu_true"])[0, 1])
    print(f"  corr(mu_hat,true):   {mu_corr:.4f}  (closer to 1 = better direction)")
    print(f"  mean|d| final:       {float(tf.reduce_mean(tf.abs(model.d)).numpy()):.4f}")
    eta_rmse = float(np.sqrt(np.mean((model.eta.numpy() - meta["eta_true"]) ** 2)))
    print(f"  eta RMSE final:      {eta_rmse:.4f}  (true sd: {cfg.sigma_eta:.3f})")

    # ------------------------------------------------------------------
    # Step 3: Exact MAP Hessian credible intervals
    # ------------------------------------------------------------------
    print("\n[simulation_study] Computing exact MAP Hessian intervals...")
    intervals = compute_laplace_intervals(model, data, cfg, confidence=0.95)

    # ------------------------------------------------------------------
    # Step 4: Report
    # ------------------------------------------------------------------
    print_interval_summary(intervals, meta={"true_beta_price": cfg.true_beta_price,
                                             "mu_true": meta["mu_true"]})

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

    print(f"\n[simulation_study] Coverage (95% nominal):")
    print(f"  beta_price  covered: {beta_covered}")
    print(f"  mu          coverage: {mu_covered:.2f}")
    print(f"  d           coverage: {d_covered:.2f}")

    summary = {
        "config": {
            "J": cfg.J, "T": cfg.T, "num_households": cfg.num_households,
            "epochs": cfg.epochs, "true_beta_price": cfg.true_beta_price,
            "gamma_endogeneity": cfg.gamma_endogeneity, "sparse_frac": cfg.sparse_frac,
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
            "lambda_control": float(model.lambda_control.numpy()),
            "pi_hat": float(tf.math.sigmoid(model.logit_pi).numpy()),
            "mu_rmse": float(np.sqrt(np.mean(
                (model.mu.numpy() - meta["mu_true"]) ** 2
            ))),
            "mu_corr_true": float(np.corrcoef(
                model.mu.numpy(), meta["mu_true"]
            )[0, 1]),
            "mu_std_hat": float(model.mu.numpy().std()),
            "mu_std_true": float(meta["mu_true"].std()),
            "eta_rmse": float(np.sqrt(np.mean(
                (model.eta.numpy() - meta["eta_true"]) ** 2
            ))),
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
