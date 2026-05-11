"""
intervals.py — Laplace credible intervals for the dynamic storable-goods model.

Method
------
At the MAP estimate, we approximate the posterior by a Gaussian:

    p(theta | data) ≈ N(theta_hat, H^{-1})

where H = d²L/dtheta² is the Hessian of the MAP objective evaluated at theta_hat.

For scalars (beta_price, logit_pi) we compute the exact second derivative via
double GradientTape.  For vectors (mu, d) we use the diagonal of the empirical
Fisher information — the outer product of per-observation log-likelihood
gradients — as a computationally tractable approximation.  This is standard
practice in neural-network posteriors and is explicitly stated as an
approximation in the report.

All operations are graph-mode compatible.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .model import DynamicContextSparseChoiceModel


# ---------------------------------------------------------------------------
# MAP loss (NLL + priors) — used for Hessian computation
# ---------------------------------------------------------------------------

def _map_loss(
    model: DynamicContextSparseChoiceModel,
    tensors: Dict[str, tf.Tensor],
    cfg: DynamicModelConfig,
) -> tf.Tensor:
    """Full MAP objective evaluated on a tensor dict."""
    cur = {
        "item_ids":  tensors["item_ids"],
        "available": tensors["available"],
        "price":     tensors["price"],
        "market_id": tensors["market_id"],
        "inventory": tensors["inventory"],
        "choice":    tensors["choice"],
    }
    nxt = {
        "item_ids":  tensors["next_item_ids"],
        "available": tensors["next_available"],
        "price":     tensors["next_price"],
        "market_id": tensors["next_market_id"],
        "inventory": tensors["next_inventory"],
    }
    parts = model.compute_loss(
        inputs=cur,
        next_inputs=nxt,
        reward=tensors["reward"],
        done=tensors["done"],
        training=False,
    )
    return parts["total"]


# ---------------------------------------------------------------------------
# Scalar Hessian (exact second derivative via double GradientTape)
# ---------------------------------------------------------------------------

def _scalar_hessian(
    model: DynamicContextSparseChoiceModel,
    tensors: Dict[str, tf.Tensor],
    cfg: DynamicModelConfig,
    param: tf.Variable,
) -> tf.Tensor:
    """Exact d²L/dθ² for a scalar parameter θ."""
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            loss = _map_loss(model, tensors, cfg)
        g = t1.gradient(loss, param)
    h = t2.gradient(g, param)
    return h


# ---------------------------------------------------------------------------
# Diagonal Fisher approximation for vector parameters
# ---------------------------------------------------------------------------

def _diag_fisher(
    model: DynamicContextSparseChoiceModel,
    data: Dict[str, np.ndarray],
    cfg: DynamicModelConfig,
    param: tf.Variable,
    batch_size: int = 512,
) -> tf.Tensor:
    """
    Diagonal empirical Fisher: I_ii = mean_n (d log p(y_n|theta) / d theta_i)^2.

    Computed by summing squared per-batch gradients of the NLL.
    Approximation note: treats observations within each batch as independent
    and ignores the prior curvature (which is small relative to the likelihood
    at large N).
    """
    N = len(data["choice"])
    tensors_all = {k: tf.constant(v) for k, v in data.items()}
    ds = tf.data.Dataset.from_tensor_slices(tensors_all).batch(batch_size)

    accum = tf.zeros_like(param)
    count = 0

    for batch in ds:
        cur = {
            "item_ids":  batch["item_ids"],
            "available": batch["available"],
            "price":     batch["price"],
            "market_id": batch["market_id"],
            "inventory": batch["inventory"],
            "choice":    batch["choice"],
        }
        with tf.GradientTape() as tape:
            nll = model.choice_nll(cur, training=False)
        g = tape.gradient(nll, param)
        if g is not None:
            accum = accum + tf.square(g) * float(batch["choice"].shape[0])
            count += int(batch["choice"].shape[0])

    return accum / float(max(count, 1))


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
        se        : numpy array  (standard error)
        ci_lower  : numpy array  (lower credible bound)
        ci_upper  : numpy array  (upper credible bound)
    """
    z = float(tf.math.erfinv(tf.constant(confidence, tf.float32)).numpy() * np.sqrt(2))

    tensors = {k: tf.constant(v) for k, v in data.items()}

    results: Dict[str, Dict[str, np.ndarray]] = {}

    # --- beta_price (scalar, exact Hessian) ---
    h_beta = _scalar_hessian(model, tensors, cfg, model.beta_price)
    se_beta = float(1.0 / (tf.sqrt(tf.abs(h_beta) + 1e-8)).numpy())
    est_beta = float(model.beta_price.numpy())
    results["beta_price"] = {
        "estimate": np.array(est_beta),
        "se":       np.array(se_beta),
        "ci_lower": np.array(est_beta - z * se_beta),
        "ci_upper": np.array(est_beta + z * se_beta),
    }

    # --- logit_pi (scalar, exact Hessian) ---
    h_pi = _scalar_hessian(model, tensors, cfg, model.logit_pi)
    se_pi = float(1.0 / (tf.sqrt(tf.abs(h_pi) + 1e-8)).numpy())
    est_pi = float(model.logit_pi.numpy())
    results["logit_pi"] = {
        "estimate": np.array(est_pi),
        "se":       np.array(se_pi),
        "ci_lower": np.array(est_pi - z * se_pi),
        "ci_upper": np.array(est_pi + z * se_pi),
    }

    # --- mu (vector, diagonal Fisher) ---
    f_mu = _diag_fisher(model, data, cfg, model.mu).numpy()
    se_mu = 1.0 / np.sqrt(np.abs(f_mu) + 1e-8)
    est_mu = model.mu.numpy()
    results["mu"] = {
        "estimate": est_mu,
        "se":       se_mu,
        "ci_lower": est_mu - z * se_mu,
        "ci_upper": est_mu + z * se_mu,
    }

    # --- d (matrix, diagonal Fisher — flattened then reshaped) ---
    f_d = _diag_fisher(model, data, cfg, model.d).numpy()
    se_d = 1.0 / np.sqrt(np.abs(f_d) + 1e-8)
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

    true_beta = float(meta["true_beta_price"]) if meta and "true_beta_price" in meta else None
    _row("beta_price", true_beta, results["beta_price"])

    pi_hat = float(tf.math.sigmoid(results["logit_pi"]["estimate"]).numpy())
    _row("pi (sparsity)", None, {"estimate": np.array(pi_hat),
                                  "se": results["logit_pi"]["se"],
                                  "ci_lower": np.array(pi_hat - 1.96 * float(results["logit_pi"]["se"])),
                                  "ci_upper": np.array(pi_hat + 1.96 * float(results["logit_pi"]["se"]))})

    mu_true = meta.get("mu_true") if meta else None
    for t in range(len(results["mu"]["estimate"])):
        tv = float(mu_true[t]) if mu_true is not None else None
        _row(f"mu[{t}]", tv, {k: results["mu"][k][t:t+1] for k in results["mu"]})

    print("-" * 72)
    print(f"(mu and d intervals use diagonal Fisher approximation)")
