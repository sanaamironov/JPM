# shrinkage.py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .blp import build_matrices, invert_delta_contraction

tfd = tfp.distributions


def _log_norm_pdf(x, var):
    # log N(x;0,var)
    return -0.5 * (np.log(2 * np.pi * var) + (x * x) / var)


def shrinkage_fit_beta_given_sigma(
    delta_vec,
    X,
    n_iter=200,
    burn=100,
    v0=0.05,  # spike variance (tuned to typical inversion noise scale)
    v1=1.0,  # slab variance (large)
    a_pi=1.0,
    b_pi=9.0,  # prior mean pi ~ 0.1 (sparse)
    beta_var=1e6,  # weak prior on beta
    seed=123,
    beta_rw_scale=0.05,
    pi_rw_scale=0.20,
):
    """
    Bayesian sparse-errors regression (TFP MCMC, collapsed over gamma):
        delta = X beta + xi
        xi_n ~ N(0, v0) if gamma_n=0 else N(0,v1)
        gamma_n ~ Bern(pi), pi ~ Beta(a_pi,b_pi)

    Implementation note:
    We sample only continuous states (beta, pi) via tfp.mcmc and integrate
    gamma out in the likelihood:
        p(r_n | beta, pi) = (1-pi)N(0,v0) + pi N(0,v1)
    Posterior inclusion probabilities are computed after sampling:
        E[ P(gamma_n=1 | r_n, pi) | data ]

    Returns:
      beta_mean: posterior mean of beta after burn-in
      gamma_prob: posterior inclusion probabilities (mean gamma)
      score: average target log posterior over kept draws
    """
    tf.random.set_seed(seed)

    X = np.asarray(X, dtype=np.float64)
    delta_vec = np.asarray(delta_vec, dtype=np.float64)
    _, k = X.shape

    X_tf = tf.convert_to_tensor(X, dtype=tf.float64)
    y_tf = tf.convert_to_tensor(delta_vec, dtype=tf.float64)

    beta_init = np.linalg.lstsq(X, delta_vec, rcond=None)[0].astype(np.float64)
    pi_init = np.float64(a_pi / (a_pi + b_pi))
    logit_pi_init = np.log(pi_init) - np.log(1.0 - pi_init)

    # fixed constants as float64 to avoid dtype mismatch in tf.function
    v0_tf = tf.constant(v0, dtype=tf.float64)
    v1_tf = tf.constant(v1, dtype=tf.float64)
    beta_var_tf = tf.constant(beta_var, dtype=tf.float64)
    a_pi_tf = tf.constant(a_pi, dtype=tf.float64)
    b_pi_tf = tf.constant(b_pi, dtype=tf.float64)
    eps_tf = tf.constant(1e-12, dtype=tf.float64)
    log2pi_tf = tf.constant(np.log(2.0 * np.pi), dtype=tf.float64)

    def target_log_prob_fn(beta, logit_pi):
        pi = tf.math.sigmoid(logit_pi)
        r = y_tf - tf.linalg.matvec(X_tf, beta)  # [N]

        # log mixture per observation:
        # log[(1-pi)N(r;0,v0) + pi N(r;0,v1)]
        logn0 = -0.5 * (tf.math.log(v0_tf) + log2pi_tf + tf.square(r) / v0_tf)
        logn1 = -0.5 * (tf.math.log(v1_tf) + log2pi_tf + tf.square(r) / v1_tf)
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

        # pi prior via transformed variable logit_pi:
        # pi ~ Beta(a_pi, b_pi)
        # p(logit_pi) = p(pi) * |dpi/dlogit_pi|, with dpi/dz = pi(1-pi)
        beta_prior = tfd.Beta(
            concentration1=a_pi_tf,
            concentration0=b_pi_tf,
        ).log_prob(pi)
        log_jac = tf.math.log(pi + eps_tf) + tf.math.log1p(-pi + eps_tf)

        return ll + beta_lp + beta_prior + log_jac

    num_results = max(1, int(n_iter - burn))
    num_burnin = max(0, int(burn))

    rw_fn = tfp.mcmc.random_walk_normal_fn(
        scale=[tf.constant(beta_rw_scale, tf.float64), tf.constant(pi_rw_scale, tf.float64)]
    )
    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=rw_fn,
    )

    @tf.function
    def run_chain():
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin,
            current_state=[
                tf.convert_to_tensor(beta_init, dtype=tf.float64),
                tf.convert_to_tensor(logit_pi_init, dtype=tf.float64),
            ],
            kernel=kernel,
            trace_fn=lambda _, kr: (kr.accepted_results.target_log_prob, kr.is_accepted),
            seed=seed,
        )

    states, traces = run_chain()
    beta_draws, logit_pi_draws = states
    logpost_draws, _ = traces

    beta_samples = beta_draws.numpy()  # [S, k]
    pi_samples = tf.math.sigmoid(logit_pi_draws).numpy()  # [S]
    score_mean = float(np.mean(logpost_draws.numpy()))

    # posterior inclusion probabilities:
    # p(gamma_n=1 | r_n, pi) for each draw, then average over draws
    XB = beta_samples @ X.T  # [S, N]
    R = delta_vec[None, :] - XB  # [S, N]
    logn1 = _log_norm_pdf(R, v1)  # [S, N]
    logn0 = _log_norm_pdf(R, v0)  # [S, N]
    logp1 = np.log(pi_samples[:, None] + 1e-12) + logn1
    logp0 = np.log(1.0 - pi_samples[:, None] + 1e-12) + logn0
    m = np.maximum(logp1, logp0)
    p1 = np.exp(logp1 - m) / (np.exp(logp1 - m) + np.exp(logp0 - m))

    beta_mean = beta_samples.mean(axis=0)
    gamma_prob = p1.mean(axis=0)
    return beta_mean, gamma_prob, float(score_mean)


def shrinkage_objective_for_sigma(sigma, markets, R=200, **kwargs):
    # 1) invert all markets to get delta_vec(sigma)
    delta_list = []
    for t, m in enumerate(markets):
        delta_t = invert_delta_contraction(m["s"], m["p"], sigma, R=R, seed=123 + t)
        delta_list.append(delta_t.numpy())
    delta_vec = np.concatenate(delta_list, axis=0)

    # 2) build X only (no Z needed)
    X, _ = build_matrices(markets, iv_type="nocost")  # X is same regardless
    beta_hat, gamma_prob, score = shrinkage_fit_beta_given_sigma(delta_vec, X, **kwargs)

    # We *maximize* score, so return it
    return score, beta_hat, gamma_prob


def estimate_shrinkage_sigma(markets, R=200, sigma_grid=None, **kwargs):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.05, 4.0, 40)

    best = None
    for s in sigma_grid:
        score, beta_hat, gamma_prob = shrinkage_objective_for_sigma(
            s, markets, R=R, **kwargs
        )
        if (best is None) or (score > best[0]):
            best = (score, s, beta_hat, gamma_prob)

    # local refine
    s0 = best[1]
    refine = np.linspace(max(0.01, s0 - 0.25), s0 + 0.25, 25)
    for s in refine:
        score, beta_hat, gamma_prob = shrinkage_objective_for_sigma(
            s, markets, R=R, **kwargs
        )
        if score > best[0]:
            best = (score, s, beta_hat, gamma_prob)

    score_hat, sigma_hat, beta_hat, gamma_prob = best
    return sigma_hat, beta_hat, score_hat, gamma_prob
