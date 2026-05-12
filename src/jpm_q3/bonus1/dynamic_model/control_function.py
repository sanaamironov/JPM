"""
control_function.py — Petrin & Train (2010) control function for price endogeneity.

Reference
---------
Petrin, A. and Train, K. (2010). "A Control Function Approach to Endogeneity
in Consumer Choice Models." Journal of Marketing Research, 47(1):3-13.

Method
------
1. First stage: regress the endogenous price p_jt on observable exogenous
   instruments z_jt. Here z = [brand dummies, w_jt, w_jt^2] where w_jt is an
   exogenous cost shifter generated in the DGP.

2. Compute the residual ê_jt = p_jt - p̂_jt — the part of price NOT explained
   by exogenous variation. Under the assumption that the unobserved shock xi
   is linear in this residual, ê_jt is a sufficient statistic for the
   endogenous component.

3. Include ê_jt in the second-stage utility:
        u_ijt = ... + beta * p_jt + lambda_control * ê_jt + ...
   The estimated lambda_control absorbs the endogeneity; beta becomes
   consistent for the structural price effect.

Implementation
--------------
The first stage is a closed-form OLS on stacked (j, t) observations.  All
math runs in pure numpy — the residuals are precomputed once after data
generation and stored as `price_residual` in the data dict, so the model
sees them as an exogenous input.
"""
from __future__ import annotations

import numpy as np


def compute_price_residuals(
    price_inside: np.ndarray,   # (T, J)  observed inside-good prices
    w_obs: np.ndarray,           # (T, J)  observable cost shifter w_jt
) -> np.ndarray:
    """
    First-stage OLS to obtain the Petrin-Train control function residuals.

    Regressors per (j, t):
        - one-hot brand dummies (J columns)
        - w_jt
        - w_jt^2  (allows a quadratic relationship between cost and price)

    Returns:
        residuals : (T, J) float32 — ê_jt = p_jt - p̂_jt
    """
    T, J = price_inside.shape
    N = T * J

    # Stack (j, t) into a long format.
    p_long = price_inside.reshape(N).astype(np.float64)
    w_long = w_obs.reshape(N).astype(np.float64)
    brand_idx = np.tile(np.arange(J), T)   # (N,) brand id 0..J-1

    # Build regressor matrix: [brand dummies | w | w^2]
    brand_dummies = np.eye(J, dtype=np.float64)[brand_idx]   # (N, J)
    X = np.column_stack([brand_dummies, w_long, w_long ** 2])  # (N, J+2)

    # OLS via normal equations (small system, no need for QR).
    XtX = X.T @ X
    Xtp = X.T @ p_long
    pi_hat = np.linalg.solve(XtX, Xtp)   # (J+2,)

    # Fitted values and residuals
    p_hat = X @ pi_hat
    e_long = p_long - p_hat

    residuals = e_long.reshape(T, J).astype(np.float32)
    return residuals


def build_residual_array_for_training(
    price_residual_inside: np.ndarray,   # (T, J)
    num_items: int,                       # J + 1
) -> np.ndarray:
    """
    Pad the inside-good residual array with a zero column for the outside
    option, producing shape (T, num_items) that can be indexed by market_id.
    """
    T, J = price_residual_inside.shape
    full = np.zeros((T, num_items), dtype=np.float32)
    full[:, 1:] = price_residual_inside
    return full
