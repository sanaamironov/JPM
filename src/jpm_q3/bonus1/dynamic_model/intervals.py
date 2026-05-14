"""
intervals.py — Laplace credible intervals for the dynamic storable-goods model.

Method
------
At the MAP estimate we approximate the posterior by a Gaussian:

    p(theta | data) ≈ N(theta_hat, H^{-1})

where H = d²L_MAP/dtheta² is the Hessian of the full MAP objective
(NLL + TD + prior terms) evaluated at theta_hat.

We compute the diagonal of H via automatic differentiation:
  - Scalars (beta_price, logit_pi): exact d²L/dθ² via double GradientTape.
  - Vectors (mu, shape T): exact diagonal via tf.jacobian of the gradient.
  - Matrices (d, shape T×J): exact diagonal of the flattened Hessian via
    tf.jacobian, reshaped back.

Using the full MAP Hessian (not just the NLL Fisher) is critical:
  - The prior adds curvature 1/mu_sd² per mu element and 1/v0 or 1/v1 per d
    element even when the likelihood gradient is near zero.
  - Fisher-only approximations give SE → ∞ when likelihood gradients vanish
    at the MAP point, producing degenerate intervals.

All operations are graph-mode compatible.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .model import DynamicContextSparseChoiceModel


# ---------------------------------------------------------------------------
# MAP loss (NLL + TD + priors) — the objective whose Hessian we compute
# ---------------------------------------------------------------------------

def _build_map_loss_fn(
    model: DynamicContextSparseChoiceModel,
    tensors: Dict[str, tf.Tensor],
    cfg: DynamicModelConfig,
) -> Callable[[], tf.Tensor]:
    """Return a zero-argument callable that evaluates the MAP loss."""
    cur = {
        "item_ids":       tensors["item_ids"],
        "available":      tensors["available"],
        "price":          tensors["price"],
        "price_residual": tensors["price_residual"],
        "market_id":      tensors["market_id"],
        "household_id":   tensors["household_id"],
        "inventory":      tensors["inventory"],
        "choice":         tensors["choice"],
    }
    nxt = {
        "item_ids":       tensors["next_item_ids"],
        "available":      tensors["next_available"],
        "price":          tensors["next_price"],
        "price_residual": tensors["next_price_residual"],
        "market_id":      tensors["next_market_id"],
        "household_id":   tensors["next_household_id"],
        "inventory":      tensors["next_inventory"],
    }

    def loss_fn() -> tf.Tensor:
        parts = model.compute_loss(
            inputs=cur,
            next_inputs=nxt,
            reward=tensors["reward"],
            done=tensors["done"],
            training=False,
        )
        return parts["total"]

    return loss_fn


# ---------------------------------------------------------------------------
# Exact Hessian diagonal via automatic differentiation
# ---------------------------------------------------------------------------

def _hessian_diag_scalar(
    loss_fn: Callable[[], tf.Tensor],
    param: tf.Variable,
) -> tf.Tensor:
    """Exact d²L/dθ² for a scalar parameter θ."""
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            loss = loss_fn()
        g = t1.gradient(loss, param)
    h = t2.gradient(g, param)
    return h  # scalar


def _hessian_diag_vector(
    loss_fn: Callable[[], tf.Tensor],
    param: tf.Variable,
) -> tf.Tensor:
    """
    Exact diagonal of d²L/dθ² for a 1-D vector parameter θ of shape (n,).

    Uses tf.jacobian to compute the full (n, n) Hessian and extracts the
    diagonal. Safe for n ≤ ~100.
    """
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            loss = loss_fn()
        g = t1.gradient(loss, param)          # (n,)
    H = t2.jacobian(g, param)                  # (n, n)
    return tf.linalg.diag_part(H)              # (n,)


def _hessian_diag_matrix(
    loss_fn: Callable[[], tf.Tensor],
    param: tf.Variable,
) -> tf.Tensor:
    """
    Exact diagonal of d²L/dθ² for a 2-D matrix parameter θ of shape (m, n).

    Flattens to (m*n,), computes the Jacobian of the gradient, extracts the
    diagonal, and reshapes back to (m, n). Safe for m*n ≤ ~200.
    """
    shape = param.shape
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            loss = loss_fn()
        g = t1.gradient(loss, param)          # (m, n) — inside t2 so t2 tracks it
        g_flat = tf.reshape(g, [-1])           # (m*n,) — must also be inside t2
    H_flat = t2.jacobian(g_flat, param)        # (m*n, m, n)
    n_flat = g_flat.shape[0]
    H_sq = tf.reshape(H_flat, [n_flat, n_flat])  # (m*n, m*n)
    diag = tf.linalg.diag_part(H_sq)           # (m*n,)
    return tf.reshape(diag, shape)             # (m, n)


# ---------------------------------------------------------------------------
# SE and CI from Hessian diagonal
# ---------------------------------------------------------------------------

def _se_from_hessian_diag(h_diag: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """
    SE = 1 / sqrt(max(|H_ii|, floor)).

    The floor prevents division by zero when a parameter is poorly identified.
    A floor of 1e-6 gives SE ≤ 1000, which is honest about poor identification
    rather than producing the infinite SEs of the Fisher approximation.
    """
    return 1.0 / np.sqrt(np.maximum(np.abs(h_diag), floor))


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_laplace_intervals(
    model: DynamicContextSparseChoiceModel,
    data: Dict[str, np.ndarray],
    cfg: DynamicModelConfig,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute Laplace credible intervals for key model parameters.

    Parameters of interest:
        beta_price  — scalar price coefficient
        logit_pi    — scalar sparsity logit
        mu          — (T,) market mean shocks
        d           — (T, J) sparse deviations

    Returns a dict keyed by parameter name, each containing:
        estimate  : numpy array  (MAP estimate)
        se        : numpy array  (standard error = 1/sqrt(|H_ii|))
        ci_lower  : numpy array  (lower credible bound)
        ci_upper  : numpy array  (upper credible bound)
    """
    z = float(
        tf.math.erfinv(tf.constant(confidence, tf.float32)).numpy() * np.sqrt(2)
    )

    # Subsample data for Hessian computation — full data is expensive but correct.
    # Use all data (N ≤ ~3000 is fine for this model size).
    tensors = {k: tf.constant(v) for k, v in data.items()}
    loss_fn = _build_map_loss_fn(model, tensors, cfg)

    # ------------------------------------------------------------------
    # Compile all Hessian computations into a single tf.function graph.
    #
    # The zero-argument wrapper closes over loss_fn and all model parameters.
    # It traces exactly once and runs the double-differentiation in graph mode,
    # satisfying the JPM constraint that all computationally intensive operations
    # must be executable without retracing.
    # ------------------------------------------------------------------
    @tf.function(input_signature=[], reduce_retracing=True)
    def _compute_all_hessians():
        h_beta    = _hessian_diag_scalar(loss_fn, model.beta_price)
        h_pi      = _hessian_diag_scalar(loss_fn, model.logit_pi)
        h_lambda  = _hessian_diag_scalar(loss_fn, model.lambda_control)
        h_kappa   = _hessian_diag_scalar(loss_fn, model.kappa_0)
        h_mu      = _hessian_diag_vector(loss_fn, model.mu)
        h_d       = _hessian_diag_matrix(loss_fn, model.d)
        return h_beta, h_pi, h_lambda, h_kappa, h_mu, h_d

    h_beta_t, h_pi_t, h_lambda_t, h_kappa_t, h_mu_t, h_d_t = _compute_all_hessians()

    h_beta   = h_beta_t.numpy()
    h_pi     = h_pi_t.numpy()
    h_lambda = h_lambda_t.numpy()
    h_kappa  = h_kappa_t.numpy()
    h_mu     = h_mu_t.numpy()   # (T,)
    h_d      = h_d_t.numpy()    # (T, J)

    results: Dict[str, Dict[str, np.ndarray]] = {}

    # --- beta_price (scalar) ---
    se_beta = _se_from_hessian_diag(np.array(h_beta))
    est_beta = float(model.beta_price.numpy())
    results["beta_price"] = {
        "estimate": np.array(est_beta),
        "se":       se_beta,
        "ci_lower": np.array(est_beta - z * float(se_beta)),
        "ci_upper": np.array(est_beta + z * float(se_beta)),
    }

    # --- logit_pi (scalar) —
    # Also provide CI in probability space via sigmoid transform (delta method).
    se_pi = _se_from_hessian_diag(np.array(h_pi))
    est_lpi = float(model.logit_pi.numpy())
    results["logit_pi"] = {
        "estimate": np.array(est_lpi),
        "se":       se_pi,
        "ci_lower": np.array(est_lpi - z * float(se_pi)),
        "ci_upper": np.array(est_lpi + z * float(se_pi)),
    }
    # Probability-space CI (sigmoid of logit CI bounds — asymmetric but valid).
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))
    est_pi_prob = _sigmoid(est_lpi)
    results["pi"] = {
        "estimate": np.array(est_pi_prob),
        "se":       np.array(se_pi * est_pi_prob * (1.0 - est_pi_prob)),  # delta method
        "ci_lower": np.array(_sigmoid(est_lpi - z * float(se_pi))),
        "ci_upper": np.array(_sigmoid(est_lpi + z * float(se_pi))),
    }

    # --- lambda_control (scalar) ---
    se_lambda = _se_from_hessian_diag(np.array(h_lambda))
    est_lambda = float(model.lambda_control.numpy())
    results["lambda_control"] = {
        "estimate": np.array(est_lambda),
        "se":       se_lambda,
        "ci_lower": np.array(est_lambda - z * float(se_lambda)),
        "ci_upper": np.array(est_lambda + z * float(se_lambda)),
    }

    # --- kappa_0 (scalar) ---
    se_kappa = _se_from_hessian_diag(np.array(h_kappa))
    est_kappa = float(model.kappa_0.numpy())
    results["kappa_0"] = {
        "estimate": np.array(est_kappa),
        "se":       se_kappa,
        "ci_lower": np.array(est_kappa - z * float(se_kappa)),
        "ci_upper": np.array(est_kappa + z * float(se_kappa)),
    }

    # --- mu (T-vector) ---
    se_mu  = _se_from_hessian_diag(h_mu)
    est_mu = model.mu.numpy()
    results["mu"] = {
        "estimate": est_mu,
        "se":       se_mu,
        "ci_lower": est_mu - z * se_mu,
        "ci_upper": est_mu + z * se_mu,
    }

    # --- d (T×J matrix) ---
    se_d  = _se_from_hessian_diag(h_d)
    est_d = model.d.numpy()
    results["d"] = {
        "estimate": est_d,
        "se":       se_d,
        "ci_lower": est_d - z * se_d,
        "ci_upper": est_d + z * se_d,
    }

    return results


def print_interval_summary(
    results: Dict[str, Dict[str, np.ndarray]],
    meta: Dict | None = None,
) -> None:
    """Print a concise table of estimates and credible intervals."""
    print(f"\n{'Parameter':<18} {'True':>8} {'Estimate':>10} {'SE':>8} {'95% CI':>22}")
    print("-" * 72)

    def _row(name: str, true_val: float | None, r: Dict) -> None:
        est = float(np.mean(r["estimate"]))
        se = float(np.mean(r["se"]))
        lo = float(np.mean(r["ci_lower"]))
        hi = float(np.mean(r["ci_upper"]))
        true_str = f"{true_val:8.3f}" if true_val is not None else "       —"
        print(f"{name:<18} {true_str} {est:10.3f} {se:8.3f} [{lo:8.3f}, {hi:8.3f}]")

    true_beta = float(meta.get("true_beta_price", float("nan"))) if meta else None
    _row("beta_price", true_beta, results["beta_price"])

    pi_hat = float(tf.math.sigmoid(
        tf.constant(float(results["logit_pi"]["estimate"]))
    ).numpy())
    pi_se = float(results["logit_pi"]["se"])
    _row("pi (sparsity)", None, {
        "estimate": np.array(pi_hat),
        "se":       np.array(pi_se),
        "ci_lower": np.array(pi_hat - 1.96 * pi_se),
        "ci_upper": np.array(pi_hat + 1.96 * pi_se),
    })

    mu_true = meta.get("mu_true") if meta else None
    for t in range(len(results["mu"]["estimate"])):
        tv = float(mu_true[t]) if mu_true is not None else None
        _row(f"mu[{t}]", tv, {k: results["mu"][k][t:t+1] for k in results["mu"]})

    print("-" * 72)
    print("(Intervals use exact MAP Hessian diagonal via automatic differentiation)")
