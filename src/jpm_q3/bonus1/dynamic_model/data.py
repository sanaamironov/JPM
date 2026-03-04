from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import DynamicModelConfig


def simulate_dynamic_panel(
    cfg: DynamicModelConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Synthetic dynamic panel generator (storable goods + sparse unobservables + context variation)."""
    rng = np.random.default_rng(cfg.seed)
    H = int(cfg.households)
    T = int(cfg.periods)
    J = int(cfg.num_items)
    N = H * T

    # Household -> market assignment (fixed over time for simplicity)
    household_market = rng.integers(0, cfg.num_markets, size=H, endpoint=False)

    # IDs and availability
    item_ids = np.tile(np.arange(J, dtype=np.int32)[None, :], (N, 1))
    available = np.ones((N, J), dtype=np.float32)

    # Random availability creates context-dependent choice sets (outside always available)
    for n in range(N):
        mask = rng.random(J) < float(cfg.availability_prob)
        mask[0] = True
        # guarantee at least 2 available items
        if mask.sum() < 2:
            mask[1] = True
        available[n] = mask.astype(np.float32)

    market_id = np.repeat(household_market, T).astype(np.int32)

    # True latent shocks (sparse d and market fixed effect mu)
    mu_true = rng.normal(0.0, 0.8, size=cfg.num_markets).astype(np.float32)
    d_true = np.zeros((cfg.num_markets, J - 1), dtype=np.float32)
    gamma_true = np.zeros((cfg.num_markets, J - 1), dtype=np.int32)

    k = max(1, (J - 1) // 3)
    for m in range(cfg.num_markets):
        nz = rng.choice(J - 1, size=k, replace=False)
        d_true[m, nz] = rng.normal(0.0, 0.9, size=k).astype(np.float32)
        gamma_true[m, nz] = 1

    base_item_u = rng.normal(0.0, 0.5, size=J).astype(np.float32)
    base_item_u[0] = 0.0

    inventory = np.zeros(N, dtype=np.float32)
    next_inventory = np.zeros(N, dtype=np.float32)
    choices = np.zeros(N, dtype=np.int32)
    reward = np.zeros(N, dtype=np.float32)
    done = np.zeros(N, dtype=np.float32)

    for h in range(H):
        inv = float(cfg.init_inventory)
        m = int(household_market[h])
        for t in range(T):
            n = h * T + t
            inventory[n] = inv

            # inventory motive (encourage purchase when low)
            inv_term = 1.0 / (1.0 + inv)

            # utility baseline
            u = base_item_u.copy()

            # Lu-style shocks: inside goods
            u[1:] += mu_true[m] + d_true[m]

            # inventory motive to inside goods
            u[1:] += 0.8 * inv_term

            # mask unavailable options
            u = np.where(available[n] > 0.5, u, -1e9)

            # dynamic continuation: discourage high next inventory (holding cost proxy)
            # inv_next(j) = max(0, inv + purchase_qty*1(j>0) - mean_consumption)
            inv_next_all = np.maximum(
                inv
                + (np.arange(J) > 0).astype(np.float32) * float(cfg.purchase_qty)
                - float(cfg.mean_consumption),
                0.0,
            )
            u = u - float(cfg.discount) * float(cfg.holding_cost) * inv_next_all

            # small noise
            u = u + rng.normal(0.0, 0.1, size=J).astype(np.float32)

            # choice
            u_shift = u - np.max(u)
            p = np.exp(u_shift)
            p = p / p.sum()
            c = int(rng.choice(J, p=p))
            choices[n] = c

            purchase = float(cfg.purchase_qty) if c > 0 else 0.0
            cons = float(rng.poisson(cfg.mean_consumption))
            inv_next = max(0.0, inv + purchase - cons)
            next_inventory[n] = inv_next

            # reward proxy
            reward[n] = float(u[c])
            done[n] = 1.0 if (t == T - 1) else 0.0
            inv = inv_next

    next_item_ids = item_ids.copy()
    next_available = available.copy()
    next_market_id = market_id.copy()

    data = {
        "item_ids": item_ids,
        "available": available,
        "market_id": market_id.astype(np.int32),
        "inventory": inventory.astype(np.float32),
        "choice": choices.astype(np.int32),
        "reward": reward.astype(np.float32),
        "done": done.astype(np.float32),
        "next_item_ids": next_item_ids,
        "next_available": next_available,
        "next_market_id": next_market_id.astype(np.int32),
        "next_inventory": next_inventory.astype(np.float32),
    }

    meta = {
        "mu_true": mu_true,
        "d_true": d_true,
        "gamma_true": gamma_true,
    }
    return data, meta
