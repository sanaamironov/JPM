"""
shrinkage.py

TFP-MCMC shrinkage regression core for Lu & Shimizu (2025), Section 4 replication.

What this module does
---------------------
Given:
  - delta_vec: stacked mean utilities (delta) for all markets/products at a fixed sigma
  - X: stacked regressor matrix (paper-aligned caller passes X=[p,w], no constant)

We estimate:
  - beta (posterior mean)
  - gamma_prob: posterior inclusion probability per observation n (proxy for sparsity)

Model (collapsed spike-and-slab)
--------------------------------
delta = X beta + xi
xi_n | gamma_n ~ N(0, v0) if gamma_n=0  (spike)
xi_n | gamma_n ~ N(0, v1) if gamma_n=1  (slab)
gamma_n ~ Bern(pi),  pi ~ Beta(a_pi, b_pi)

We integrate out gamma_n:
  p(r_n | beta, pi) = (1-pi)N(r_n;0,v0) + pi N(r_n;0,v1),
where r_n = delta_n - x_n' beta.

We sample (beta, logit(pi)) using tfp.mcmc.RandomWalkMetropolis.

Changes made (relative to earlier version)
-----------------------------------------------
1) Adds `thin` to shrinkage_fit_beta_given_sigma so the replication driver can pass
   --shrink-thin without crashing.
2) Removes sigma-search and delta-inversion helpers from this module to avoid
   two competing implementations. Sigma search remains in replicate_section4.py.
3) Removes tf.random.set_seed(...) to avoid global RNG side effects (especially
   problematic under multiprocessing). Determinism is controlled via the seed passed
   into tfp.mcmc.sample_chain.
4) Updates documentation to avoid implying X includes a constant. The caller decides X.

Practical notes for reviewers
-----------------------------
- Use tfp.mcmc.RandomWalkMetropolis rather than a bespoke Gibbs sampler because
  TFP provides general-purpose MCMC kernels. This still satisfies the requirement
  to implement the MCMC using tfp.mcmc and yields stable posterior summaries for
  the simulation study.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfm = tfp.mcmc


def _log_norm_pdf(x: np.ndarray, var: float) -> np.ndarray:
    """Elementwise log N(x; 0, var). x can be any shape."""
    x = np.asarray(x, dtype=np.float64)
    var = float(var)
    return -0.5 * (np.log(2.0 * np.pi * var) + (x * x) / var)


@tf.function(reduce_retracing=True)
def _sample_beta_pi_chain(
    *,
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
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Run RandomWalkMetropolis for (beta, logit_pi) with collapsed mixture likelihood.

    Returns:
      beta_draws: [S, k]
      logit_pi_draws: [S]
      logpost_draws: [S]
      acc_rate: scalar float64
    """
    eps_tf = tf.constant(1e-12, dtype=tf.float64)
    log2pi_tf = tf.constant(np.log(2.0 * np.pi), dtype=tf.float64)

    def target_log_prob_fn(beta, logit_pi):
        # Transform pi in (0,1)
        pi = tf.math.sigmoid(logit_pi)

        # Residuals
        r = y_tf - tf.linalg.matvec(X_tf, beta)  # [N]

        # log N(r;0,v)
        logn0 = -0.5 * (tf.math.log(v0_tf) + log2pi_tf + tf.square(r) / v0_tf)
        logn1 = -0.5 * (tf.math.log(v1_tf) + log2pi_tf + tf.square(r) / v1_tf)

        # log[(1-pi)N0 + pi N1]
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
            -0.5
            * (tf.square(beta) / beta_var_tf + tf.math.log(beta_var_tf) + log2pi_tf)
        )

        # pi prior: Beta(a_pi, b_pi) on pi, with logistic transform jacobian
        pi_lp = tfd.Beta(concentration1=a_pi_tf, concentration0=b_pi_tf).log_prob(pi)
        # log|d pi / d logit_pi| = log(pi) + log(1-pi)
        log_abs_det_jac = tf.math.log(pi + eps_tf) + tf.math.log1p(-pi + eps_tf)

        return ll + beta_lp + pi_lp + log_abs_det_jac

    # Independent RW steps for beta and logit_pi
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
    delta_vec: np.ndarray,
    X: np.ndarray,
    *,
    n_iter: int = 800,
    burn: int = 400,
    thin: int = 1,
    v0: float = 0.05,  # spike variance
    v1: float = 1.0,  # slab variance
    a_pi: float = 1.0,
    b_pi: float = 9.0,  # prior mean pi ~ 0.1 (sparse)
    beta_var: float = 1e6,
    seed: int = 123,
    beta_rw_scale: float = 0.05,
    pi_rw_scale: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Collapsed spike-and-slab regression with MCMC over (beta, pi).

    Args:
      delta_vec: shape [N]
      X: shape [N, k]
      n_iter: total MCMC iterations (including burn-in)
      burn: burn-in steps discarded
      thin: keep every thin-th draw after burn-in (thin>=1)
      v0, v1: spike/slab variances
      a_pi, b_pi: Beta prior for pi
      beta_var: variance of diffuse N(0, beta_var I) prior on beta
      seed: seed passed into TFP MCMC (controls determinism)
      beta_rw_scale, pi_rw_scale: RW proposal scales for beta and logit(pi)

    Returns:
      beta_mean: posterior mean of beta after burn-in/thinning, shape [k]
      gamma_prob: posterior inclusion prob per obs, shape [N]
      score: mean log posterior over retained draws (diagnostic objective)
      acc_rate: acceptance rate (over all post-burnin draws before thinning)
    """
    X = np.asarray(X, dtype=np.float64)
    delta_vec = np.asarray(delta_vec, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array [N,k].")
    if X.shape[0] != delta_vec.shape[0]:
        raise ValueError(
            f"X and delta_vec must have same N. Got {X.shape[0]} vs {delta_vec.shape[0]}."
        )

    n_iter = int(n_iter)
    burn = int(burn)
    thin = max(1, int(thin))
    if burn >= n_iter:
        raise ValueError(f"burn must be < n_iter. Got burn={burn}, n_iter={n_iter}.")

    # TFP uses num_results AFTER burn-in; we keep n_iter-burn draws before thinning.
    num_results = int(n_iter - burn)
    num_burnin = int(burn)

    # Convert to TF tensors
    X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
    y_tf = tf.convert_to_tensor(delta_vec, dtype=tf.float64)

    # Init beta via least squares (stable)
    beta_init = np.linalg.lstsq(X, delta_vec, rcond=None)[0].astype(np.float64)

    # Init pi at prior mean
    pi_init = float(a_pi / (a_pi + b_pi))
    pi_init = min(max(pi_init, 1e-6), 1.0 - 1e-6)
    logit_pi_init = np.log(pi_init) - np.log(1.0 - pi_init)

    # Scalars
    v0_tf = tf.constant(float(v0), dtype=tf.float64)
    v1_tf = tf.constant(float(v1), dtype=tf.float64)
    beta_var_tf = tf.constant(float(beta_var), dtype=tf.float64)
    a_pi_tf = tf.constant(float(a_pi), dtype=tf.float64)
    b_pi_tf = tf.constant(float(b_pi), dtype=tf.float64)
    beta_rw_scale_tf = tf.constant(float(beta_rw_scale), dtype=tf.float64)
    pi_rw_scale_tf = tf.constant(float(pi_rw_scale), dtype=tf.float64)

    beta_draws_tf, logit_pi_draws_tf, logpost_draws_tf, acc_rate_tf = (
        _sample_beta_pi_chain(
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
            seed=tf.constant([int(seed), int(seed) ^ 0x9E3779B9], dtype=tf.int32),
            # seed=tf.convert_to_tensor(int(seed), dtype=tf.int32),
        )
    )

    # Convert to numpy
    beta_samples = beta_draws_tf.numpy()  # [S,k], S=num_results
    pi_samples = tf.math.sigmoid(logit_pi_draws_tf).numpy()  # [S]
    logpost = logpost_draws_tf.numpy()  # [S]
    acc_rate = float(acc_rate_tf.numpy())

    # Apply thinning consistently to posterior summaries.
    beta_samples_th = beta_samples[::thin]
    pi_samples_th = pi_samples[::thin]
    logpost_th = logpost[::thin]

    # Diagnostic score: mean log posterior of retained draws
    score = float(np.mean(logpost_th)) if logpost_th.size > 0 else float("nan")

    # Posterior inclusion probabilities:
    # For each retained draw s: P(gamma_n=1 | r_n, pi_s) = [pi_s N(r_n;0,v1)] / [(1-pi_s)N(r_n;0,v0) + pi_s N(r_n;0,v1)]
    XB = beta_samples_th @ X.T  # [S_th, N]
    R = delta_vec[None, :] - XB  # [S_th, N]

    logn1 = _log_norm_pdf(R, v1)
    logn0 = _log_norm_pdf(R, v0)

    logp1 = np.log(pi_samples_th[:, None] + 1e-12) + logn1
    logp0 = np.log(1.0 - pi_samples_th[:, None] + 1e-12) + logn0

    m = np.maximum(logp1, logp0)
    p1 = np.exp(logp1 - m) / (np.exp(logp1 - m) + np.exp(logp0 - m))

    beta_mean = beta_samples_th.mean(axis=0)
    gamma_prob = p1.mean(axis=0)

    return beta_mean, gamma_prob, score, acc_rate
