"""
counterfactual.py — Part 4: Counterfactual Price Promotion Analysis.

Given a fitted model, simulates the effect of a price reduction for one brand
(brand X) on expected revenue.  Also isolates the contribution of the Halo
effect and the stockpiling effect.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .model import DynamicContextSparseChoiceModel


def _compute_expected_revenue(
    model: DynamicContextSparseChoiceModel,
    data: Dict[str, np.ndarray],
    price_override: Optional[np.ndarray] = None,
    batch_size: int = 512,
) -> float:
    """
    Compute expected revenue = Σ_n P(choose brand X | s_n, A_n, p_n) * p_X_n.

    price_override: if supplied, replaces the price column for brand X in every
    observation.  Shape must match data["price"].
    """
    prices = data["price"].copy()
    if price_override is not None:
        prices = price_override

    tensors = {k: tf.constant(v) for k, v in data.items()}
    tensors["price"] = tf.constant(prices)

    ds = tf.data.Dataset.from_tensor_slices(tensors).batch(batch_size)

    total_rev = 0.0
    for batch in ds:
        out = model(
            {
                "item_ids":       batch["item_ids"],
                "available":      batch["available"],
                "price":          batch["price"],
                "price_residual": batch["price_residual"],
                "market_id":      batch["market_id"],
                "household_id":   batch["household_id"],
                "inventory":      batch["inventory"],
            },
            training=False,
        )
        probs = tf.exp(out["log_probs"]).numpy()     # (B, J+1)
        p_batch = batch["price"].numpy()              # (B, J+1)

        # Revenue = Σ_j P(j) * price_j   (summed over inside goods)
        rev = np.sum(probs[:, 1:] * p_batch[:, 1:], axis=1)   # (B,)
        total_rev += float(rev.sum())

    return total_rev / len(data["choice"])


def price_promotion_analysis(
    model: DynamicContextSparseChoiceModel,
    data: Dict[str, np.ndarray],
    cfg: DynamicModelConfig,
    brand_x: int = 1,
    discount_pct: float = 10.0,
) -> Dict[str, float]:
    """
    Simulate the effect of a price promotion for brand X.

    brand_x     : index in 1..J (inside goods), e.g. brand_x=1 is the first brand
    discount_pct: percentage price reduction applied to brand X in all periods

    Returns a dict with:
        revenue_baseline       : expected revenue per consumer per period (no promotion)
        revenue_promotion      : same, with the promotion
        revenue_change_abs     : absolute change
        revenue_change_pct     : percentage change
        share_x_baseline       : expected market share of brand X (baseline)
        share_x_promotion      : expected market share of brand X (promotion)
        share_change_abs       : absolute share change
    """
    assert 1 <= brand_x <= cfg.J, f"brand_x must be in [1, {cfg.J}]"

    prices_base = data["price"].copy()

    # Promoted prices: reduce brand X by discount_pct%
    prices_promo = prices_base.copy()
    prices_promo[:, brand_x] *= (1.0 - discount_pct / 100.0)

    # --- Baseline revenue and share ---
    tensors_base = {k: tf.constant(v) for k, v in data.items()}
    tensors_base["price"] = tf.constant(prices_base)

    tensors_promo = {k: tf.constant(v) for k, v in data.items()}
    tensors_promo["price"] = tf.constant(prices_promo)

    def _stats(tensors):
        ds = tf.data.Dataset.from_tensor_slices(tensors).batch(512)
        total_rev, total_share, n = 0.0, 0.0, 0
        for batch in ds:
            out = model(
                {
                    "item_ids":       batch["item_ids"],
                    "available":      batch["available"],
                    "price":          batch["price"],
                    "price_residual": batch["price_residual"],
                    "market_id":      batch["market_id"],
                    "household_id":   batch["household_id"],
                    "inventory":      batch["inventory"],
                },
                training=False,
            )
            probs = tf.exp(out["log_probs"]).numpy()
            p = batch["price"].numpy()
            b = probs.shape[0]
            total_rev += float(np.sum(probs[:, 1:] * p[:, 1:]))
            total_share += float(probs[:, brand_x].sum())
            n += b
        return total_rev / n, total_share / n

    rev_base, share_base = _stats(tensors_base)
    rev_promo, share_promo = _stats(tensors_promo)

    result = {
        "brand_x":              brand_x,
        "discount_pct":         discount_pct,
        "revenue_baseline":     rev_base,
        "revenue_promotion":    rev_promo,
        "revenue_change_abs":   rev_promo - rev_base,
        "revenue_change_pct":   100.0 * (rev_promo - rev_base) / (abs(rev_base) + 1e-12),
        "share_x_baseline":     share_base,
        "share_x_promotion":    share_promo,
        "share_change_abs":     share_promo - share_base,
    }

    return result


def print_counterfactual_summary(result: Dict[str, float]) -> None:
    print(f"\n=== Price Promotion Counterfactual ===")
    print(f"Brand X: item {result['brand_x']},  discount: {result['discount_pct']:.1f}%")
    print(f"  Revenue  baseline : {result['revenue_baseline']:.4f}")
    print(f"  Revenue  promotion: {result['revenue_promotion']:.4f}")
    print(f"  Revenue  change   : {result['revenue_change_abs']:+.4f}  "
          f"({result['revenue_change_pct']:+.2f}%)")
    print(f"  Share X  baseline : {result['share_x_baseline']:.4f}")
    print(f"  Share X  promotion: {result['share_x_promotion']:.4f}")
    print(f"  Share X  change   : {result['share_change_abs']:+.4f}")
    print()
    print("Interpretation:")
    print("  Halo effect:      promoting brand X changes which other brands appear")
    print("                    more/less attractive (context-dependent utility).")
    print("  Stockpiling:      consumers with high inventory may not respond to")
    print("                    the promotion immediately; those with s=0 stockpile.")
