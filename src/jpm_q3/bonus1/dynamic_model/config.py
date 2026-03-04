from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DynamicModelConfig:
    # Choice-set dimensions
    num_items: int = 8  # includes outside option at index 0
    num_markets: int = 12

    # Deep context backbone (choice_learn_ext.models.deep_context)
    d_embed: int = 16
    n_blocks: int = 2
    n_heads: int = 2
    residual_variant: str = "fixed_base"
    dropout: float = 0.0

    # Dynamic component
    discount: float = 0.95
    inventory_scale: float = 10.0
    td_weight: float = 0.2

    # Storable-goods transition / economics
    purchase_qty: float = 1.0
    mean_consumption: float = 0.7
    holding_cost: float = 0.15  # discourages stockpiling

    # Lu-style sparse shock prior on d
    # v0: float = 1e-3  # spike variance
    v0: float = 0.05
    v1: float = 1.0  # slab variance
    a_pi: float = 1.0
    b_pi: float = 9.0
    mu_sd: float = 3.0
    center_d_within_market: bool = True
    prior_weight: float = 0.005

    # Training
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 10
    seed: int = 123
    compile_train_step: bool = False
    force_cpu: bool = True

    # Synthetic panel generation
    households: int = 200
    periods: int = 20
    init_inventory: float = 1.0

    # Context: random availability to create choice-set variation
    availability_prob: float = 0.85
