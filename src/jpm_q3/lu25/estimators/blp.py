"""
Implementation of BLP estimation for a random-coefficients logit demand model.

This version adds paper-reporting-friendly extras:
- delta_hat: stacked mean utilities at sigma_hat
- X: stacked regressor matrix
- Z: stacked instrument matrix
- xi_hat: delta_hat - X @ beta_hat
- obj_hat: final GMM objective value
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _simulate_shares_given_delta(delta, p, sigma, nu_draws):
    """Random-coeff logit shares for one market, inside goods only, with outside option."""
    mu = tf.expand_dims(nu_draws, 1) * tf.expand_dims(p, 0) * sigma  # [R, J]
    util = tf.expand_dims(delta, 0) + mu  # [R, J]

    expu = tf.exp(util)
    denom = 1.0 + tf.reduce_sum(expu, axis=1, keepdims=True)
    s_r = expu / denom
    return tf.reduce_mean(s_r, axis=0)  # [J]


def invert_delta_contraction(
    s_obs, p, sigma, R=200, max_iter=2000, tol=1e-10, seed=123
):
    """Berry contraction mapping to find delta such that shares match s_obs."""
    s_obs = tf.convert_to_tensor(s_obs, dtype=tf.float64)
    p = tf.convert_to_tensor(p, dtype=tf.float64)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float64)

    nu_draws = tfd.Normal(0.0, 1.0).sample(R, seed=int(seed))
    nu_draws = tf.cast(nu_draws, tf.float64)

    s0 = tf.maximum(1.0 - tf.reduce_sum(s_obs), 1e-12)
    delta = tf.math.log(tf.maximum(s_obs, 1e-12)) - tf.math.log(s0)

    for _ in range(max_iter):
        s_hat = _simulate_shares_given_delta(delta, p, sigma, nu_draws)
        delta_new = delta + (
            tf.math.log(tf.maximum(s_obs, 1e-12))
            - tf.math.log(tf.maximum(s_hat, 1e-12))
        )
        if tf.reduce_max(tf.abs(delta_new - delta)) < tol:
            delta = delta_new
            break
        delta = delta_new

    return delta  # [J]


def build_matrices(markets, iv_type="cost"):
    """
    Build stacked X and Z across all markets/products.
    X = [1, p, w]
    Z depends on iv_type.
    """
    Xs, Zs = [], []
    for m in markets:
        p = np.asarray(m["p"], dtype=float)
        w = np.asarray(m["w"], dtype=float)
        u = m.get("u", None)

        X = np.column_stack([np.ones_like(p), p, w])

        if iv_type == "cost":
            if u is None:
                raise ValueError("iv_type='cost' requires market['u'] (cost shock).")
            u = np.asarray(u, dtype=float)
            Z = np.column_stack([np.ones_like(p), w, w**2, u, u**2])
        elif iv_type == "nocost":
            Z = np.column_stack([np.ones_like(p), w, w**2, w**3, w**4])
        else:
            raise ValueError("iv_type must be 'cost' or 'nocost'.")

        Xs.append(X)
        Zs.append(Z)

    return np.vstack(Xs), np.vstack(Zs)


def iv_2sls_beta(delta_vec, X, Z):
    """2SLS: beta = (X' Pz X)^(-1) X' Pz delta."""
    delta_vec = tf.convert_to_tensor(delta_vec, dtype=tf.float64)  # [N]
    X = tf.convert_to_tensor(X, dtype=tf.float64)  # [N,k]
    Z = tf.convert_to_tensor(Z, dtype=tf.float64)  # [N,l]

    ZTZ_inv = tf.linalg.inv(tf.matmul(Z, Z, transpose_a=True))

    # PzX = Z (Z'Z)^-1 Z' X
    PzX = tf.matmul(Z, tf.matmul(ZTZ_inv, tf.matmul(Z, X, transpose_a=True)))  # [N,k]

    XTPzX = tf.matmul(X, PzX, transpose_a=True)  # [k,k]
    XTPzY = tf.matmul(
        X,
        tf.matmul(Z, tf.matmul(ZTZ_inv, tf.matmul(Z, delta_vec[:, None], transpose_a=True))),
        transpose_a=True,
    )  # [k,1]

    beta = tf.linalg.solve(XTPzX, XTPzY)[:, 0]  # [k]
    return beta


def compute_delta_vec(markets, sigma, R=200, base_seed=123):
    """Stack delta across markets for a given sigma."""
    delta_list = []
    for t, m in enumerate(markets):
        delta_t = invert_delta_contraction(m["s"], m["p"], sigma, R=R, seed=base_seed + t)
        delta_list.append(delta_t.numpy())
    return np.concatenate(delta_list, axis=0)  # [N=T*J]


def gmm_objective_for_sigma(sigma, markets, iv_type="cost", R=200):
    """One-step GMM objective with W = (Z'Z)^(-1)."""
    delta_vec = compute_delta_vec(markets, sigma, R=R)
    X, Z = build_matrices(markets, iv_type=iv_type)

    beta = iv_2sls_beta(delta_vec, X, Z).numpy()
    xi_hat = delta_vec - X @ beta

    N = xi_hat.shape[0]
    g = (Z.T @ xi_hat) / N
    W = np.linalg.inv(Z.T @ Z / N)
    obj = float(g.T @ W @ g)

    return obj, beta, delta_vec, X, Z, xi_hat


def estimate_blp_sigma(markets, iv_type="cost", R=200):
    """
    Estimate BLP sigma via grid search + local refine.
    Returns:
      (sigma_hat, beta_hat, extras)
    extras includes: obj_hat, delta_hat, X, Z, xi_hat.
    """
    grid = np.linspace(0.05, 4.0, 40)
    best = None
    best_pack = None

    for s in grid:
        obj, beta, delta_vec, X, Z, xi_hat = gmm_objective_for_sigma(s, markets, iv_type=iv_type, R=R)
        if (best is None) or (obj < best):
            best = obj
            best_pack = (s, beta, delta_vec, X, Z, xi_hat, obj)

    s0 = float(best_pack[0])
    refine = np.linspace(max(0.01, s0 - 0.25), s0 + 0.25, 30)

    for s in refine:
        obj, beta, delta_vec, X, Z, xi_hat = gmm_objective_for_sigma(s, markets, iv_type=iv_type, R=R)
        if obj < best:
            best = obj
            best_pack = (s, beta, delta_vec, X, Z, xi_hat, obj)

    sigma_hat, beta_hat, delta_hat, X, Z, xi_hat, obj_hat = best_pack

    extras = {
        "obj_hat": float(obj_hat),
        "delta_hat": np.asarray(delta_hat, dtype=float),
        "X": np.asarray(X, dtype=float),
        "Z": np.asarray(Z, dtype=float),
        "xi_hat": np.asarray(xi_hat, dtype=float),
    }
    return float(sigma_hat), np.asarray(beta_hat, dtype=float), extras