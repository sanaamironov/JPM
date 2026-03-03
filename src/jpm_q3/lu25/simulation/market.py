"""
market.py

Single-market simulator for Lu & Shimizu (2025) Monte Carlo designs.

This is a full replacement file.

What changed and why
--------------------
1) Added `xi_true`, `eta_true`, and `is_signal` to the returned market dict.
   - The paper's Section 4 tables report an xi column and Prob(signal/noise) for shrinkage.
   - `xi_true` is the true unobserved demand shock used in utility.
   - `eta_true` is the true sparse deviation component (eta*).
   - `is_signal = 1{eta_true != 0}` matches the paper's definition of signal vs noise
     (condition on eta* != 0 vs eta* == 0).

2) Preserved backward-compatible keys (`xi`, `eta`) so existing code does not break.

No estimation logic is changed; this only exposes ground truth needed for reporting.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .config import SimConfig
from .dgp import generate_eta_alpha

tfd = tfp.distributions


def _simulate_shares_rc_logit(
    p: np.ndarray,
    w: np.ndarray,
    xi_true: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate inside-good shares for one market under RC logit with an outside option."""
    p_tf = tf.convert_to_tensor(p, dtype=tf.float64)         # (J,)
    w_tf = tf.convert_to_tensor(w, dtype=tf.float64)         # (J,)
    xi_tf = tf.convert_to_tensor(xi_true, dtype=tf.float64)  # (J,)

    # Draw individual-specific price coefficients: beta_p_i ~ N(beta_p_star, sigma_star^2)
    beta_p_draws = tf.cast(
        tfd.Normal(cfg.beta_p_star, cfg.sigma_star).sample(
            cfg.R0, seed=int(rng.integers(1, 2**31 - 1))
        ),
        tf.float64,
    )  # (R0,)

    # Mean utility: beta_p_star * p + beta_w_star * w + xi_true
    delta = cfg.beta_p_star * p_tf + cfg.beta_w_star * w_tf + xi_tf  # (J,)

    # Heterogeneous component: (beta_p_i - beta_p_star) * p
    mu = tf.expand_dims(beta_p_draws - cfg.beta_p_star, 1) * tf.expand_dims(p_tf, 0)  # (R0, J)

    util = tf.expand_dims(delta, 0) + mu  # (R0, J)
    expu = tf.exp(util)
    denom = 1.0 + tf.reduce_sum(expu, axis=1, keepdims=True)  # outside option included
    s_r = expu / denom
    s = tf.reduce_mean(s_r, axis=0)  # (J,)

    return s.numpy().astype(float)


def simulate_market(dgp: str, J: int, cfg: SimConfig, rng: np.random.Generator) -> dict:
    """Simulate a single market.

    Returns a dict with keys:

    Observed objects (used by estimators):
      - w: observed product characteristic, shape (J,)
      - p: price, shape (J,)
      - s: inside-good shares, shape (J,)

    Simulation design objects (valid IV etc.):
      - u: cost shock (valid cost IV), shape (J,)
      - alpha: endogenous price component from DGP, shape (J,)

    Ground-truth latent objects (needed for paper-style reporting):
      - xi_true: true demand shock xi_{jt} used in utility, shape (J,)
      - eta_true: true sparse deviation eta*_{jt}, shape (J,)
      - is_signal: indicator 1{eta_true != 0}, shape (J,)

    Backward-compatible aliases:
      - xi: same as xi_true
      - eta: same as eta_true
    """
    # Observed product characteristic w
    w = rng.uniform(cfg.w_low, cfg.w_high, size=J).astype(float)

    # Cost shock (valid IV in the simulation design)
    u = rng.normal(loc=0.0, scale=cfg.cost_sd, size=J).astype(float)

    # Demand shock components (eta_star) and endogenous price component (alpha_star)
    eta_star, alpha_star = generate_eta_alpha(dgp, J, cfg, rng)

    # Total demand shock level: xi = xi_bar + eta_star
    xi_true = (cfg.xi_bar_star + eta_star).astype(float)

    # Price equation: p = alpha_star + 0.3*w + u
    p = (alpha_star + 0.3 * w + u).astype(float)

    # Shares under RC logit with outside option
    s = _simulate_shares_rc_logit(p, w, xi_true, cfg, rng)

    # Paper signal mask: eta* != 0
    is_signal = (eta_star != 0.0).astype(int)

    return {
        # observed
        "w": w,
        "p": p,
        "s": s,

        # instruments / price components
        "u": u,
        "alpha": alpha_star.astype(float),

        # ground truth for reporting
        "xi_true": xi_true,
        "eta_true": eta_star.astype(float),
        "is_signal": is_signal,

        # backward-compatible aliases
        "xi": xi_true,
        "eta": eta_star.astype(float),
    }