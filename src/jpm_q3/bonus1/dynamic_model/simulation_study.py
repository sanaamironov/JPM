"""
simulation_study.py — Part 3: Simulation and Parameter Recovery.

Training strategy (two-stage):
  Stage 1 — econometric estimation:
    Freeze Halo AND value head. Train only beta_price, lambda_control,
    mu, d, eta, and logit_pi.
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
    # Only keep keys consumed by the step function to minimise the input spec.
    _step_keys = [
        "item_ids", "available", "price", "price_residual",
        "market_id", "household_id", "inventory", "choice",
        "reward", "done", "delta_i",
        "next_item_ids", "next_available", "next_price", "next_price_residual",
        "next_market_id", "next_household_id", "next_inventory",
    ]
    tensors = {k: tf.constant(data[k]) for k in _step_keys}
    ds = (
        tf.data.Dataset.from_tensor_slices(tensors)
        .shuffle(4096, seed=cfg.seed)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    nll_fn = model.static_choice_nll if use_static_nll else model.choice_nll

    J = cfg.num_items
    _step_sig = {
        "item_ids":       tf.TensorSpec([None, J], tf.int32),
        "available":      tf.TensorSpec([None, J], tf.float32),
        "price":          tf.TensorSpec([None, J], tf.float32),
        "price_residual": tf.TensorSpec([None, J], tf.float32),
        "market_id":      tf.TensorSpec([None],    tf.int32),
        "household_id":   tf.TensorSpec([None],    tf.int32),
        "inventory":      tf.TensorSpec([None],    tf.float32),
        "choice":         tf.TensorSpec([None],    tf.int32),
        "reward":         tf.TensorSpec([None],    tf.float32),
        "done":           tf.TensorSpec([None],    tf.float32),
        "delta_i":        tf.TensorSpec([None],    tf.float32),
        "next_item_ids":       tf.TensorSpec([None, J], tf.int32),
        "next_available":      tf.TensorSpec([None, J], tf.float32),
        "next_price":          tf.TensorSpec([None, J], tf.float32),
        "next_price_residual": tf.TensorSpec([None, J], tf.float32),
        "next_market_id":      tf.TensorSpec([None],    tf.int32),
        "next_household_id":   tf.TensorSpec([None],    tf.int32),
        "next_inventory":      tf.TensorSpec([None],    tf.float32),
    }

    @tf.function(input_signature=[_step_sig])
    def step(batch):
        cur = {
            "item_ids":       batch["item_ids"],
            "available":      batch["available"],
            "price":          batch["price"],
            "price_residual": batch["price_residual"],
            "market_id":      batch["market_id"],
            "household_id":   batch["household_id"],
            "inventory":      batch["inventory"],
            "choice":         batch["choice"],
            "delta_i":        batch["delta_i"],
        }
        nxt = {
            "item_ids":       batch["next_item_ids"],
            "available":      batch["next_available"],
            "price":          batch["next_price"],
            "price_residual": batch["next_price_residual"],
            "market_id":      batch["next_market_id"],
            "household_id":   batch["next_household_id"],
            "inventory":      batch["next_inventory"],
            "delta_i":        batch["delta_i"],   # same household, same discount
        }
        with tf.GradientTape() as tape:
            nll = nll_fn(cur, training=True)
            td = (
                tf.constant(0.0, dtype=tf.float32)
                if use_static_nll
                else model.td_error_loss(
                    cur,
                    nxt,
                    reward=batch["reward"],
                    done=batch["done"],
                    training=True,
                )
            )
            prior = model.sparse_shock_prior_penalty()
            loss = nll + float(cfg.td_weight) * td + float(cfg.prior_weight) * prior
        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        opt.apply_gradients(pairs)
        return loss, nll, td

    for ep in range(1, epochs + 1):
        m_loss = tf.keras.metrics.Mean()
        m_nll = tf.keras.metrics.Mean()
        m_td = tf.keras.metrics.Mean()
        for batch in ds:
            loss, nll, td = step(batch)
            m_loss.update_state(loss)
            m_nll.update_state(nll)
            m_td.update_state(td)
        print(
            f"  {label} Epoch {ep:03d} | loss={m_loss.result():.4f}  "
            f"nll={m_nll.result():.4f}  td={m_td.result():.4f}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_simulation_study(
    cfg: DynamicModelConfig | None = None,
    out_dir: str | Path | None = None,
    seed: int = 0,
    with_cf: bool = True,
) -> dict:
    """
    Full simulation study:
      1. Generate data from the known DGP.
      2. Estimate via two-stage MAP.
      3. Compute exact MAP Hessian credible intervals.
      4. Report true vs. estimated values and coverage.

    with_cf: if False, freeze lambda_control=0 (no Petrin-Train control function).
    Used to produce the Without-CF baseline in Table 3 of the report.
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

    if not with_cf:
        # Disable Petrin-Train control function: fix lambda_control=0.
        model.lambda_control.assign(0.0)
        model.lambda_control.trainable = False
        print("[simulation_study] Control function DISABLED (with_cf=False)")

    # Stage 1: Freeze Halo + value head. Train econometric params only.
    # This prevents the value head's market_embed from absorbing mu_t signal.
    print("\n[simulation_study] Stage 1: econometric params (Halo + value head frozen)...")
    model.halo.trainable = False
    model.market_embed.trainable = False
    model.value_head.trainable = False

    econometric_vars = [
        v for v in [
            model.beta_price, model.lambda_control,
            model.mu, model.d, model.eta, model.logit_pi, model.kappa_0,
        ] if v.trainable
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
    # Note: in TF 2.16/Keras 3, integer-index embedding lookups (halo item embedding,
    # market_embed) receive zero gradients through @tf.function. Only value_head (MLP)
    # and scalar/vector econometric params actually update. See CLAUDE.md for details.
    print("\n[simulation_study] Stage 2: joint fine-tuning (value head + econometric params)...")
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

    # -- mu CI mean width (average 95% CI half-width across T elements) --
    mu_ci_half_width_mean = float(np.mean(
        intervals["mu"]["ci_upper"] - intervals["mu"]["ci_lower"]
    ) / 2.0)

    summary = {
        "config": {
            "J": cfg.J, "T": cfg.T, "num_households": cfg.num_households,
            "epochs": cfg.epochs, "true_beta_price": cfg.true_beta_price,
            "gamma_endogeneity": cfg.gamma_endogeneity, "sparse_frac": cfg.sparse_frac,
            "with_cf": with_cf,
        },
        "true": {
            "beta_price": float(cfg.true_beta_price),
            "kappa_0": float(cfg.kappa_stockout),
            "mu_mean": float(meta["mu_true"].mean()),
            "mu_std": float(meta["mu_true"].std()),
            "gamma_true_mean": float(meta["gamma_true"].mean()),
        },
        "estimated": {
            # beta_price
            "beta_price": float(intervals["beta_price"]["estimate"]),
            "beta_price_se": float(intervals["beta_price"]["se"]),
            "beta_price_ci": [float(intervals["beta_price"]["ci_lower"]),
                              float(intervals["beta_price"]["ci_upper"])],
            # lambda_control (CF coefficient)
            "lambda_control": float(intervals["lambda_control"]["estimate"]) if with_cf else None,
            "lambda_control_se": float(intervals["lambda_control"]["se"]) if with_cf else None,
            "lambda_control_ci": [float(intervals["lambda_control"]["ci_lower"]),
                                  float(intervals["lambda_control"]["ci_upper"])] if with_cf else None,
            # pi (sparsity) — probability space
            "pi_hat": float(intervals["pi"]["estimate"]),
            "pi_se": float(intervals["pi"]["se"]),
            "pi_ci": [float(intervals["pi"]["ci_lower"]),
                      float(intervals["pi"]["ci_upper"])],
            # kappa_0 (stockout penalty)
            "kappa_0_hat": float(intervals["kappa_0"]["estimate"]),
            "kappa_0_se": float(intervals["kappa_0"]["se"]),
            "kappa_0_ci": [float(intervals["kappa_0"]["ci_lower"]),
                           float(intervals["kappa_0"]["ci_upper"])],
            # mu (market shocks)
            "mu_rmse": float(np.sqrt(np.mean(
                (model.mu.numpy() - meta["mu_true"]) ** 2
            ))),
            "mu_corr_true": float(np.corrcoef(
                model.mu.numpy(), meta["mu_true"]
            )[0, 1]),
            "mu_std_hat": float(model.mu.numpy().std()),
            "mu_std_true": float(meta["mu_true"].std()),
            "mu_ci_half_width_mean": mu_ci_half_width_mean,
            # eta (consumer tastes)
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
        fname = "simulation_study.json" if with_cf else "simulation_study_no_cf.json"
        (out / fname).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"\n[simulation_study] Results saved to {out / fname}")

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
    out = "results/bonus1/simulation_study"
    # With CF: skip if artifact already exists (pre-computed for the report).
    # To force a fresh run, delete results/bonus1/simulation_study/simulation_study.json first.
    if not os.path.exists(os.path.join(out, "simulation_study.json")):
        cf_summary = run_simulation_study(cfg, out_dir=out, with_cf=True)
    else:
        print("[main] simulation_study.json already exists — skipping with-CF run.")
        print("       Delete it and re-run to regenerate.")
        import json as _json
        cf_summary = _json.loads(
            open(os.path.join(out, "simulation_study.json")).read()
        )
    # Without CF: always produces simulation_study_no_cf.json (Table 3 baseline).
    no_cf_summary = run_simulation_study(cfg, out_dir=out, with_cf=False)

    # Report the correct bias-reduction fraction (not the recovery fraction).
    beta_cf    = cf_summary["estimated"]["beta_price"]
    beta_no_cf = no_cf_summary["estimated"]["beta_price"]
    beta_true  = cfg.true_beta_price
    gap_no_cf  = abs(beta_true) - abs(beta_no_cf)   # bias without CF
    bias_removed = abs(beta_cf) - abs(beta_no_cf)   # improvement due to CF
    if abs(gap_no_cf) > 1e-8:
        bias_reduction_pct = 100.0 * bias_removed / gap_no_cf
    else:
        bias_reduction_pct = float("nan")
    recovery_pct = 100.0 * abs(beta_cf) / abs(beta_true)
    print("\n[main] CF comparison:")
    print(f"  beta_true        = {beta_true:.3f}")
    print(f"  beta without CF  = {beta_no_cf:.3f}  (recovery {100*abs(beta_no_cf)/abs(beta_true):.1f}%)")
    print(f"  beta with CF     = {beta_cf:.3f}  (recovery {recovery_pct:.1f}%)")
    print(f"  Bias reduction   = {bias_reduction_pct:.1f}%  (= improvement / gap-without-CF)")


if __name__ == "__main__":
    main()
