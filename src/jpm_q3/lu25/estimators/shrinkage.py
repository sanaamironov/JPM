"""
shrinkage.py

TFP-MCMC shrinkage estimator for Lu & Shimizu (2025) Section 4 replication.

This is a full replacement file.

What changed and why
--------------------
1) Adds paper-reporting-friendly outputs in `extras` when `return_extras=True`:
   - delta_hat: stacked delta at selected sigma
   - X: stacked X matrix
   - xi_hat: delta_hat - X @ beta_hat
   - acc_rate: acceptance rate at selected sigma
   This is required to fill the paper's xi column for the shrinkage estimator.

2) Keeps the estimator logic identical:
   - Still uses the collapsed mixture likelihood integrating out gamma.
   - Still samples (beta, logit(pi)) using tfp.mcmc.RandomWalkMetropolis.
   - Still selects sigma by grid search maximizing the posterior score.

3) Avoids tf.function retracing:
   - `_sample_beta_pi_chain` is compiled once with reduce_retracing=True.

Expected inputs
---------------
- markets: list of dicts with keys "s", "p", "w" (and optionally "u")
- Uses build_matrices from blp.py (X = [1, p, w])

Returns
-------
- estimate_shrinkage_sigma(..., return_extras=False):
    (sigma_hat, beta_hat, score_hat, gamma_prob)
- estimate_shrinkage_sigma(..., return_extras=True):
    (sigma_hat, beta_hat, score_hat, gamma_prob, extras)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .blp import build_matrices, invert_delta_contraction

tfd = tfp.distributions
tfm = tfp.mcmc


def _log_norm_pdf(x: np.ndarray, var: float) -> np.ndarray:
    """log N(x;0,var) elementwise."""
    return -0.5 * (np.log(2.0 * np.pi * var) + (x * x) / var)


@tf.function(reduce_retracing=True)
def _sample_beta_pi_chain(
    X_tf: tf.Tensor,
    y_tf: tf.Tensor,
    beta_init_tf: tf.Tensor,
    logit_pi_init_tf: tf.Tensor,
    v0_tf: tf.Tensor,
    v1_tf: tf.Tensor,
    beta_var_tf: tf.Tensor,
    a_pi_tf: tf.Tensor,
    b_pi_tf: tf.Tensor,
    beta_rw_scale_tf: tf.Tensor,
    pi_rw_scale_tf: tf.Tensor,
    num_results: tf.Tensor,
    num_burnin: tf.Tensor,
    seed: tf.Tensor,
):
    """
    Run TFP RandomWalkMetropolis for (beta, logit_pi) with collapsed mixture likelihood.
    Compiled once and reused to avoid retracing overhead.
    """
    eps_tf = tf.constant(1e-12, dtype=tf.float64)
    log2pi_tf = tf.constant(np.log(2.0 * np.pi), dtype=tf.float64)

    def target_log_prob_fn(beta, logit_pi):
        pi = tf.math.sigmoid(logit_pi)
        r = y_tf - tf.linalg.matvec(X_tf, beta)  # [N]

        logn0 = -0.5 * (tf.math.log(v0_tf) + log2pi_tf + tf.square(r) / v0_tf)
        logn1 = -0.5 * (tf.math.log(v1_tf) + log2pi_tf + tf.square(r) / v1_tf)

        # log[(1-pi)N(r;0,v0) + pi N(r;0,v1)]
        log_mix = tf.reduce_logsumexp(
            tf.stack(
                [
                    tf.math.log1p(-pi + eps_tf) + logn0,
                    tf.math.log(pi + eps_tf) + logn1,
                ],
                axis=0,
            ),
            axis=0,
        )
        ll = tf.reduce_sum(log_mix)

        # beta prior: N(0, beta_var I)
        beta_lp = tf.reduce_sum(
            -0.5 * (tf.square(beta) / beta_var_tf + tf.math.log(beta_var_tf) + log2pi_tf)
        )

        # pi prior via transformed variable logit(pi)
        pi_lp = tfd.Beta(concentration1=a_pi_tf, concentration0=b_pi_tf).log_prob(pi)
        log_jac = tf.math.log(pi + eps_tf) + tf.math.log1p(-pi + eps_tf)

        return ll + beta_lp + pi_lp + log_jac

    rw_fn = tfm.random_walk_normal_fn(scale=[beta_rw_scale_tf, pi_rw_scale_tf])
    kernel = tfm.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=rw_fn,
    )

    states, trace = tfm.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin,
        current_state=[beta_init_tf, logit_pi_init_tf],
        kernel=kernel,
        trace_fn=lambda _cs, kr: (kr.accepted_results.target_log_prob, kr.is_accepted),
        seed=seed,
    )

    beta_draws, logit_pi_draws = states
    logpost_draws, is_accepted = trace
    acc_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float64))
    return beta_draws, logit_pi_draws, logpost_draws, acc_rate


def shrinkage_fit_beta_given_sigma(
    delta_vec,
    X,
    n_iter=200,
    burn=100,
    v0=0.05,   # spike variance
    v1=1.0,    # slab variance
    a_pi=1.0,
    b_pi=9.0,  # prior mean pi ~ 0.1 (sparse)
    beta_var=1e6,
    seed=123,
    beta_rw_scale=0.05,
    pi_rw_scale=0.20,
):
    """
    Collapsed spike-and-slab regression:
        delta = X beta + xi
        xi_n | gamma_n ~ N(0,v0) if gamma_n=0 else N(0,v1)
        gamma_n ~ Bern(pi), pi ~ Beta(a_pi,b_pi)

    We integrate out gamma:
        p(r_n | beta, pi) = (1-pi)N(0,v0) + pi N(0,v1)

    Returns:
      beta_mean: posterior mean of beta after burn-in
      gamma_prob: posterior inclusion probabilities (E[gamma_n | data])
      score: mean log posterior (diagnostic objective)
      acc_rate: MCMC acceptance rate
    """
    tf.random.set_seed(seed)

    X = np.asarray(X, dtype=np.float64)
    delta_vec = np.asarray(delta_vec, dtype=np.float64)

    X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
    y_tf = tf.convert_to_tensor(delta_vec, dtype=tf.float64)

    beta_init = np.linalg.lstsq(X, delta_vec, rcond=None)[0].astype(np.float64)
    pi_init = np.float64(a_pi / (a_pi + b_pi))
    logit_pi_init = np.log(pi_init) - np.log(1.0 - pi_init)

    v0_tf = tf.constant(v0, dtype=tf.float64)
    v1_tf = tf.constant(v1, dtype=tf.float64)
    beta_var_tf = tf.constant(beta_var, dtype=tf.float64)
    a_pi_tf = tf.constant(a_pi, dtype=tf.float64)
    b_pi_tf = tf.constant(b_pi, dtype=tf.float64)

    beta_rw_scale_tf = tf.constant(beta_rw_scale, dtype=tf.float64)
    pi_rw_scale_tf = tf.constant(pi_rw_scale, dtype=tf.float64)

    num_results = int(max(1, int(n_iter - burn)))
    num_burnin = int(max(0, int(burn)))

    beta_draws, logit_pi_draws, logpost_draws, acc_rate_tf = _sample_beta_pi_chain(
        X_tf=X_tf,
        y_tf=y_tf,
        beta_init_tf=tf.convert_to_tensor(beta_init, dtype=tf.float64),
        logit_pi_init_tf=tf.convert_to_tensor(logit_pi_init, dtype=tf.float64),
        v0_tf=v0_tf,
        v1_tf=v1_tf,
        beta_var_tf=beta_var_tf,
        a_pi_tf=a_pi_tf,
        b_pi_tf=b_pi_tf,
        beta_rw_scale_tf=beta_rw_scale_tf,
        pi_rw_scale_tf=pi_rw_scale_tf,
        num_results=tf.convert_to_tensor(num_results, dtype=tf.int32),
        num_burnin=tf.convert_to_tensor(num_burnin, dtype=tf.int32),
        seed=tf.convert_to_tensor(seed, dtype=tf.int32),
    )

    beta_samples = beta_draws.numpy()  # [S, k]
    pi_samples = tf.math.sigmoid(logit_pi_draws).numpy()  # [S]
    score_mean = float(np.mean(logpost_draws.numpy()))
    acc_rate = float(acc_rate_tf.numpy())

    # Posterior inclusion probabilities:
    # P(gamma_n=1 | r_n, pi) per draw, then average over draws.
    XB = beta_samples @ X.T  # [S, N]
    R = delta_vec[None, :] - XB  # [S, N]
    logn1 = _log_norm_pdf(R, v1)
    logn0 = _log_norm_pdf(R, v0)

    logp1 = np.log(pi_samples[:, None] + 1e-12) + logn1
    logp0 = np.log(1.0 - pi_samples[:, None] + 1e-12) + logn0

    m = np.maximum(logp1, logp0)
    p1 = np.exp(logp1 - m) / (np.exp(logp1 - m) + np.exp(logp0 - m))

    beta_mean = beta_samples.mean(axis=0)
    gamma_prob = p1.mean(axis=0)
    return beta_mean, gamma_prob, score_mean, acc_rate


def _invert_all_markets_delta(markets, sigma, R=200, base_seed=123) -> np.ndarray:
    """Compute stacked delta_vec across all markets for a given sigma."""
    delta_list = []
    for t, m in enumerate(markets):
        delta_t = invert_delta_contraction(m["s"], m["p"], sigma, R=R, seed=base_seed + t)
        delta_list.append(delta_t.numpy())
    return np.concatenate(delta_list, axis=0)  # (T*J,)


def shrinkage_objective_for_sigma(sigma, markets, R=200, **kwargs):
    """
    Returns a tuple that includes all objects needed to report paper columns.
    """
    delta_vec = _invert_all_markets_delta(markets, sigma, R=R)

    # X = [1, p, w] (same for all iv_type)
    X, _ = build_matrices(markets, iv_type="nocost")

    beta_hat, gamma_prob, score, acc_rate = shrinkage_fit_beta_given_sigma(delta_vec, X, **kwargs)
    xi_hat = delta_vec - X @ beta_hat

    # maximize score
    return score, beta_hat, gamma_prob, acc_rate, delta_vec, X, xi_hat


def estimate_shrinkage_sigma(markets, R=200, sigma_grid=None, return_extras: bool = False, **kwargs):
    """
    Grid-search sigma by maximizing (collapsed) log posterior score.

    Returns:
      - if return_extras=False:
          (sigma_hat, beta_hat, score_hat, gamma_prob)
      - if return_extras=True:
          (sigma_hat, beta_hat, score_hat, gamma_prob, extras)

    extras includes:
      - acc_rate: acceptance rate at selected sigma
      - delta_hat: stacked delta at selected sigma
      - X: stacked X matrix
      - xi_hat: stacked implied xi_hat = delta_hat - X @ beta_hat
    """
    if sigma_grid is None:
        sigma_grid = np.linspace(0.05, 4.0, 40)

    best = None  # (score, sigma, beta, gamma, delta, X, xi_hat)
    best_acc = None

    for s in sigma_grid:
        score, beta_hat, gamma_prob, acc_rate, delta_vec, X, xi_hat = shrinkage_objective_for_sigma(
            s, markets, R=R, **kwargs
        )
        if (best is None) or (score > best[0]):
            best = (score, s, beta_hat, gamma_prob, delta_vec, X, xi_hat)
            best_acc = acc_rate

    # local refine
    s0 = float(best[1])
    refine = np.linspace(max(0.01, s0 - 0.25), s0 + 0.25, 25)
    for s in refine:
        score, beta_hat, gamma_prob, acc_rate, delta_vec, X, xi_hat = shrinkage_objective_for_sigma(
            s, markets, R=R, **kwargs
        )
        if score > best[0]:
            best = (score, s, beta_hat, gamma_prob, delta_vec, X, xi_hat)
            best_acc = acc_rate

    score_hat, sigma_hat, beta_hat, gamma_prob, delta_hat, X, xi_hat = best

    if return_extras:
        extras = {
            "acc_rate": float(best_acc) if best_acc is not None else np.nan,
            "delta_hat": np.asarray(delta_hat, dtype=float),
            "X": np.asarray(X, dtype=float),
            "xi_hat": np.asarray(xi_hat, dtype=float),
        }
        return float(sigma_hat), np.asarray(beta_hat, dtype=float), float(score_hat), gamma_prob, extras

    return float(sigma_hat), np.asarray(beta_hat, dtype=float), float(score_hat), gamma_prob