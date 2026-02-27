from __future__ import annotations

from typing import Dict

import numpy as np

from .config import DynamicModelConfig


def simulate_dynamic_panel(cfg: DynamicModelConfig) -> Dict[str, np.ndarray]:
    """Synthetic panel generator for storable-goods dynamic choice."""
    rng = np.random.default_rng(cfg.seed)
    H = cfg.households
    T = cfg.periods
    J = cfg.num_items
    N = H * T

    # Household-specific market assignment (fixed over time for simplicity).
    household_market = rng.integers(0, cfg.num_markets, size=H, endpoint=False)

    item_ids = np.tile(np.arange(J, dtype=np.int32)[None, :], (N, 1))
    available = np.ones((N, J), dtype=np.float32)
    market_id = np.repeat(np.repeat(household_market[:, None], T, axis=1), 1, axis=0).reshape(-1)

    # True latent shocks (sparse d and market fixed effect mu).
    mu_true = rng.normal(0.0, 0.8, size=cfg.num_markets).astype(np.float32)
    d_true = np.zeros((cfg.num_markets, J - 1), dtype=np.float32)
    for m in range(cfg.num_markets):
        k = max(1, (J - 1) // 3)
        nz = rng.choice(J - 1, size=k, replace=False)
        d_true[m, nz] = rng.normal(0.0, 0.9, size=k).astype(np.float32)

    base_item_u = rng.normal(0.0, 0.5, size=J).astype(np.float32)
    base_item_u[0] = 0.0  # outside option baseline

    inventory = np.zeros(N, dtype=np.float32)
    next_inventory = np.zeros(N, dtype=np.float32)
    choices = np.zeros(N, dtype=np.int32)
    reward = np.zeros(N, dtype=np.float32)
    done = np.zeros(N, dtype=np.float32)

    for h in range(H):
        inv = float(cfg.init_inventory)
        m = household_market[h]
        for t in range(T):
            n = h * T + t
            inventory[n] = inv

            # Context-independent synthetic utility with inventory motive:
            # low inventory increases purchase propensity of inside goods.
            inv_term = 1.0 / (1.0 + inv)
            u = base_item_u.copy()
            u[1:] += mu_true[m] + d_true[m]
            u[1:] += 0.8 * inv_term
            u += rng.normal(0.0, 0.1, size=J).astype(np.float32)

            # Softmax choice.
            u_shift = u - np.max(u)
            p = np.exp(u_shift)
            p /= p.sum()
            c = int(rng.choice(J, p=p))
            choices[n] = c

            purchase = cfg.purchase_qty if c > 0 else 0.0
            cons = float(rng.poisson(cfg.mean_consumption))
            inv_next = max(0.0, inv + purchase - cons)
            next_inventory[n] = inv_next

            # Reward: utility proxy minus simple stockout pressure.
            reward[n] = float(u[c] - 0.05 * max(0.0, 1.0 - inv))
            done[n] = 1.0 if (t == T - 1) else 0.0
            inv = inv_next

    # Next-state placeholders for choice-set fields (same static menu here).
    next_item_ids = item_ids.copy()
    next_available = available.copy()
    next_market_id = market_id.copy()

    return {
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

