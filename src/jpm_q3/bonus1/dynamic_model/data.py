from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import DynamicModelConfig
from .control_function import (
    build_residual_array_for_training,
    compute_price_residuals,
)


# ---------------------------------------------------------------------------
# Halo-effect precomputation (fixed DeepHalo weights = "true" context effects)
# ---------------------------------------------------------------------------

def _compute_halo_effects(
    avail_true: np.ndarray,
    cfg: DynamicModelConfig,
    seed: int,
) -> Tuple[np.ndarray, list]:
    """
    Compute context-dependent Halo utilities for each (t, j) pair using a
    randomly-initialised (but fixed) DeepHalo model.

    The weights are fixed before data generation and saved in meta so the
    simulation study can verify they are not used to initialise the estimator.

    Returns:
        halo_effects : (T, num_items) float32
        halo_weights : list of numpy arrays (saved to meta for reference)
    """
    import tensorflow as tf
    from choice_learn_ext.models.deep_context.config import DeepHaloConfig
    from choice_learn_ext.models.deep_context.deep_halo_core import DeepHalo

    tf.random.set_seed(seed)
    halo_cfg = DeepHaloConfig(
        d_embed=cfg.d_embed,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_blocks,
        residual_variant=cfg.residual_variant,
        featureless=True,
        vocab_size=cfg.num_items,
        dropout=0.0,
    )
    halo = DeepHalo(halo_cfg)

    T = avail_true.shape[0]
    item_ids = np.tile(
        np.arange(cfg.num_items, dtype=np.int32)[None, :], (T, 1)
    )
    out = halo(
        {
            "available": tf.constant(avail_true),
            "item_ids": tf.constant(item_ids),
        },
        training=False,
    )
    halo_effects = out["utilities"].numpy().astype(np.float32)  # (T, num_items)
    return halo_effects, halo.get_weights()


# ---------------------------------------------------------------------------
# Exact backward induction (pure numpy — no TF)
# ---------------------------------------------------------------------------

def _backward_induction(
    T: int,
    S_max: int,
    num_items: int,
    avail_t: np.ndarray,         # (T, num_items) float32
    price_t: np.ndarray,         # (T, num_items) float32, price[t,0]=0 (outside)
    halo_t: np.ndarray,          # (T, num_items) float32
    alpha: np.ndarray,           # (J,) float32  brand fixed effects for items 1..J
    xi_t: np.ndarray,            # (T, J) float32  xi_jt = mu_t + d_jt (inside goods)
    eta_i: np.ndarray,           # (J,) float32   consumer i's brand-specific tastes
    kappa: float,
    delta_i: float,
    beta_price: float,
) -> np.ndarray:
    """
    Compute exact value function V[t, s] for t = 0..T, s = 0..S_max.

    Transition (fixed consumption = 1 per Ching 2020):
        s_consumed = max(0, s - 1)
        j = 0 (outside): s_next = s_consumed
        j > 0 (buy 1):   s_next = min(S_max, s_consumed + 1)

    Bellman (GEV Type-I errors → exact log-sum-exp):
        V[t, s] = logsumexp_{j ∈ A_t} ( ū_jt(s) + delta_i * V[t+1, s'(j,s)] )

    Terminal:
        V[T, s] = 0 for all s.
    """
    V = np.zeros((T + 1, S_max + 1), dtype=np.float64)

    s_arr = np.arange(S_max + 1)
    s_consumed = np.maximum(s_arr - 1, 0)           # (S_max+1,)
    s_next_out = s_consumed                          # outside option
    s_next_in = np.minimum(s_consumed + 1, S_max)   # inside options

    for t in range(T - 1, -1, -1):
        avail = avail_t[t]   # (num_items,)
        price = price_t[t]   # (num_items,)
        halo = halo_t[t]     # (num_items,)
        xi = xi_t[t]         # (J,)

        for s in range(S_max + 1):
            util_vec = []
            for j in range(num_items):
                if avail[j] < 0.5:
                    continue
                if j == 0:
                    u = -kappa * float(s == 0)
                    v_nxt = V[t + 1, s_next_out[s]]
                else:
                    u = (
                        float(alpha[j - 1])
                        + beta_price * float(price[j])
                        + float(halo[j])
                        + float(xi[j - 1])
                        + float(eta_i[j - 1])     # consumer-specific brand taste
                    )
                    v_nxt = V[t + 1, s_next_in[s]]
                util_vec.append(u + delta_i * v_nxt)

            if util_vec:
                u_arr = np.array(util_vec, dtype=np.float64)
                u_max = u_arr.max()
                V[t, s] = np.log(np.sum(np.exp(u_arr - u_max))) + u_max

    return V  # (T+1, S_max+1)


# ---------------------------------------------------------------------------
# Forward simulation (pure numpy — no TF)
# ---------------------------------------------------------------------------

def _forward_simulate(
    V: np.ndarray,               # (T+1, S_max+1)
    T: int,
    S_max: int,
    num_items: int,
    avail_t: np.ndarray,
    price_t: np.ndarray,
    halo_t: np.ndarray,
    alpha: np.ndarray,
    xi_t: np.ndarray,
    eta_i: np.ndarray,           # (J,) consumer's brand-specific tastes
    kappa: float,
    delta_i: float,
    beta_price: float,
    init_inv: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample choices for one consumer given their value function."""
    choices = np.zeros(T, dtype=np.int32)
    inventories = np.zeros(T, dtype=np.int32)
    next_inventories = np.zeros(T, dtype=np.int32)

    s_arr = np.arange(S_max + 1)
    s_consumed = np.maximum(s_arr - 1, 0)
    s_next_out = s_consumed
    s_next_in = np.minimum(s_consumed + 1, S_max)

    s = int(init_inv)
    for t in range(T):
        avail = avail_t[t]
        price = price_t[t]
        halo = halo_t[t]
        xi = xi_t[t]

        inventories[t] = s
        j_avail = []
        util_vec = []

        for j in range(num_items):
            if avail[j] < 0.5:
                continue
            j_avail.append(j)
            if j == 0:
                u = -kappa * float(s == 0)
                v_nxt = V[t + 1, s_next_out[s]]
            else:
                u = (
                    float(alpha[j - 1])
                    + beta_price * float(price[j])
                    + float(halo[j])
                    + float(xi[j - 1])
                    + float(eta_i[j - 1])     # consumer-specific brand taste
                )
                v_nxt = V[t + 1, s_next_in[s]]
            util_vec.append(u + delta_i * v_nxt)

        u_arr = np.array(util_vec, dtype=np.float64)
        u_arr -= u_arr.max()
        probs = np.exp(u_arr)
        probs /= probs.sum()

        chosen_local = rng.choice(len(j_avail), p=probs)
        j_chosen = j_avail[chosen_local]
        choices[t] = j_chosen

        s_next = int(s_next_in[s] if j_chosen > 0 else s_next_out[s])
        next_inventories[t] = s_next
        s = s_next

    return choices, inventories, next_inventories


# ---------------------------------------------------------------------------
# Main DGP entry point
# ---------------------------------------------------------------------------

def simulate_dynamic_panel(
    cfg: DynamicModelConfig,
    seed: int | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Generate synthetic dynamic panel data matching the revised Bonus 1 question.

    DGP assumptions (explicitly stated per the question):
    - J=cfg.J brands; item 0 is the outside option (never excluded from A_t).
    - Choice set A_t is common to all consumers at time t; |A_t| >= cfg.min_avail.
    - Prices are endogenous: p_jt = c_j + gamma * xi_jt + pi_w * w_jt + nu_jt.
    - xi_jt = mu_t + d_jt where d_jt is sparse (Lu 2025 style).
    - Inventory: s in {0, ..., S_max}; fixed consumption of 1 per period (Ching 2020).
    - Each consumer has a personal discount factor delta_i ~ Uniform(delta_min, delta_max).
    - Consumers see the full price path {p_t} and solve by exact backward induction.
    - Halo effect: utility of brand j depends on the composition of A_t via DeepHalo.
      True DeepHalo weights are fixed once before data generation.

    Additional assumptions (for tractability):
    - Initial inventory: floor(S_max / 2) for all consumers.
    - Brand cost shifters c_j ~ Uniform(0.5, 2.0) are fixed across time.

    Returns:
        data : training dict (batch-ready numpy arrays)
        meta : ground-truth parameters dict
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    J = cfg.J
    T = cfg.T
    I_h = cfg.num_households
    S_max = cfg.S_max
    n_items = cfg.num_items   # J + 1

    # 1. Brand fixed effects alpha_j for j = 1..J
    alpha_true = rng.normal(0.0, cfg.sigma_alpha, size=J).astype(np.float32)

    # 2. Sparse market-product shocks xi_jt = mu_t + d_jt
    mu_true = rng.normal(0.0, cfg.mu_true_sd, size=T).astype(np.float32)
    d_true = np.zeros((T, J), dtype=np.float32)
    gamma_true = np.zeros((T, J), dtype=np.int32)
    k_nz = max(1, int(cfg.sparse_frac * J))
    for t in range(T):
        nz_idx = rng.choice(J, size=k_nz, replace=False)
        d_true[t, nz_idx] = rng.normal(0.0, cfg.d_true_sd, size=k_nz).astype(np.float32)
        gamma_true[t, nz_idx] = 1
    xi_true = mu_true[:, None] + d_true   # (T, J) — for inside goods only

    # 3. Brand cost shifters (fixed across time, unobserved to econometrician)
    c_brand = rng.uniform(0.5, 2.0, size=J).astype(np.float32)

    # 3b. Observable brand-time cost shifter w_jt — exogenous instrument for the
    # Petrin & Train (2010) control function correction of price endogeneity.
    # Economically: brand-specific input cost shocks (e.g., commodity prices,
    # exchange rates, supplier markups). Observable to the econometrician,
    # excluded from utility by construction.
    w_obs = rng.normal(0.0, cfg.sigma_w_cost, size=(T, J)).astype(np.float32)

    # 4. Endogenous prices including the observable cost shifter w_jt:
    #     p_jt = c_j + gamma * xi_jt + pi_w * w_jt + price_noise_jt
    price_noise = rng.normal(0.0, cfg.sigma_price_noise, size=(T, J)).astype(np.float32)
    price_inside = (
        c_brand[None, :]
        + cfg.gamma_endogeneity * xi_true
        + cfg.pi_w_true * w_obs
        + price_noise
    )
    price_inside = np.clip(price_inside, 0.1, None).astype(np.float32)

    # Full price array including outside option (price = 0)
    price_all = np.zeros((T, n_items), dtype=np.float32)
    price_all[:, 1:] = price_inside

    # Full w array including outside option (w = 0)
    w_all = np.zeros((T, n_items), dtype=np.float32)
    w_all[:, 1:] = w_obs

    # 4b. Petrin & Train (2010) control function: compute price residuals from
    # first-stage OLS of p on [brand dummies, w, w^2]. The residual ê_jt is the
    # part of price not explained by the exogenous instruments — it absorbs the
    # endogenous correlation with xi_jt when used as a control in the utility.
    price_residual_inside = compute_price_residuals(price_inside, w_obs)
    price_residual_all = build_residual_array_for_training(
        price_residual_inside, num_items=n_items
    )   # (T, J+1), column 0 (outside) = 0

    # 5. Common choice sets A_t per time period (same across consumers).
    # Question: A_t ⊆ {1,...,J} with |A_t| ≥ 3 BRANDS (inside goods only;
    # outside option is always available in addition).
    avail_all = np.zeros((T, n_items), dtype=np.float32)
    avail_all[:, 0] = 1.0   # outside option always available
    for t in range(T):
        while True:
            inside_mask = rng.random(J) < 0.80
            if int(inside_mask.sum()) >= cfg.min_avail:
                break
        avail_all[t, 1:] = inside_mask.astype(np.float32)

    # 6. Halo effects for all (t, j) pairs using a fixed random DeepHalo
    halo_effects, halo_weights = _compute_halo_effects(avail_all, cfg, seed=cfg.seed)
    # halo_effects: (T, n_items) float32

    # 7. Consumer-specific discount factors
    delta_true = rng.uniform(cfg.delta_min, cfg.delta_max, size=I_h).astype(np.float32)

    # 7b. Consumer-specific brand tastes eta_ij (revised question item (2)):
    # heterogeneous across consumers, homogeneous across time.
    eta_true = rng.normal(0.0, cfg.sigma_eta, size=(I_h, J)).astype(np.float32)

    # 8. Per-consumer backward induction then forward simulation
    N = I_h * T
    item_ids_row = np.arange(n_items, dtype=np.int32)
    init_inv = S_max // 2

    all_choices = np.empty(N, dtype=np.int32)
    all_inv = np.empty(N, dtype=np.int32)
    all_next_inv = np.empty(N, dtype=np.int32)
    all_market_id = np.empty(N, dtype=np.int32)
    all_hh_id = np.empty(N, dtype=np.int32)

    for i in range(I_h):
        delta_i = float(delta_true[i])
        eta_i = eta_true[i]    # (J,) consumer i's brand-specific tastes

        V = _backward_induction(
            T, S_max, n_items,
            avail_all, price_all, halo_effects,
            alpha_true, xi_true, eta_i,
            cfg.kappa_stockout, delta_i, cfg.true_beta_price,
        )

        ch, inv, nxt = _forward_simulate(
            V, T, S_max, n_items,
            avail_all, price_all, halo_effects,
            alpha_true, xi_true, eta_i,
            cfg.kappa_stockout, delta_i, cfg.true_beta_price,
            init_inv, rng,
        )

        sl = slice(i * T, (i + 1) * T)
        all_choices[sl] = ch
        all_inv[sl] = inv
        all_next_inv[sl] = nxt
        all_market_id[sl] = np.arange(T, dtype=np.int32)
        all_hh_id[sl] = i

    # 9. Build training arrays
    # Expand common arrays to per-observation using market_id as index
    avail_obs = avail_all[all_market_id]                       # (N, n_items)
    price_obs = price_all[all_market_id]                       # (N, n_items)
    w_obs_obs = w_all[all_market_id]                            # (N, n_items)
    price_residual_obs = price_residual_all[all_market_id]      # (N, n_items)
    item_ids_obs = np.tile(item_ids_row, (N, 1))               # (N, n_items)

    next_market_id = np.minimum(all_market_id + 1, T - 1)
    done = (all_market_id == T - 1).astype(np.float32)
    next_avail = avail_all[next_market_id]
    next_price = price_all[next_market_id]
    next_w = w_all[next_market_id]
    next_price_residual = price_residual_all[next_market_id]

    # Reward proxy: value of purchase (|beta| * price of chosen brand)
    reward = np.zeros(N, dtype=np.float32)
    for n in range(N):
        j = int(all_choices[n])
        if j > 0:
            reward[n] = abs(cfg.true_beta_price) * float(price_obs[n, j])

    data: Dict[str, np.ndarray] = {
        "item_ids":            item_ids_obs.astype(np.int32),
        "available":           avail_obs.astype(np.float32),
        "price":               price_obs.astype(np.float32),
        "w_obs":               w_obs_obs.astype(np.float32),
        "price_residual":      price_residual_obs.astype(np.float32),
        "inventory":           all_inv.astype(np.float32),
        "choice":              all_choices.astype(np.int32),
        "market_id":           all_market_id.astype(np.int32),
        "household_id":        all_hh_id.astype(np.int32),
        "reward":              reward.astype(np.float32),
        "done":                done.astype(np.float32),
        "next_item_ids":       item_ids_obs.astype(np.int32),
        "next_available":      next_avail.astype(np.float32),
        "next_price":          next_price.astype(np.float32),
        "next_w_obs":          next_w.astype(np.float32),
        "next_price_residual": next_price_residual.astype(np.float32),
        "next_market_id":      next_market_id.astype(np.int32),
        "next_household_id":   all_hh_id.astype(np.int32),
        "next_inventory":      all_next_inv.astype(np.float32),
    }

    meta: Dict[str, object] = {
        "alpha_true":             alpha_true,
        "mu_true":                mu_true,
        "d_true":                 d_true,
        "gamma_true":             gamma_true,
        "xi_true":                xi_true,
        "eta_true":               eta_true,            # (I, J) consumer-brand tastes
        "price_inside":           price_inside,
        "w_obs":                  w_obs,                # (T, J) observable cost shifter
        "price_residual_inside":  price_residual_inside, # (T, J) Petrin-Train residual
        "avail_true":             avail_all,
        "delta_true":             delta_true,
        "c_brand":                c_brand,              # unobserved by econometrician
        "halo_weights":           halo_weights,         # NOT given to estimator
    }

    return data, meta
