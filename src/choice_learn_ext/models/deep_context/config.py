# src/jpm/question_3/choice_learn_ext/models/deep_context/config.py

"""
Configuration object for the DeepHalo (Deep Context-Dependent Choice) model.
This dataclass specifies all architectural and training-related hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepHaloConfig:
    # Embedding dimension for item representations
    d_embed: int = 32

    # Number of phi-interaction heads and stacked halo layers
    n_heads: int = 4
    n_layers: int = 2

    # Residual behavior:
    #   "standard":  phi receives z^{l-1}
    #   "fixed_base": phi receives z^{0}
    residual_variant: str = "standard"

    # Dropout rate for phi MLPs
    dropout: float = 0.0

    # Input mode:
    #   featureless=True  → item_ids are embedded using vocab_size
    #   featureless=False → raw item features X with dimension d_x
    featureless: bool = True
    vocab_size: Optional[int] = None

    # Required if featureless=False (feature-based items)
    d_x: Optional[int] = None

    # Authors replication mode:
    #   If True, use TensorFlow implementations that mirror the authors' PyTorch reference blocks
    #   (part_1/authors/Featureless.py and FeatureBased.py).
    #   If False, use the simplified halo-block implementation in layers.HaloBlock.
    authors_mode: bool = False

    # Featureless authors-network hyperparameters (used only when featureless=True and authors_mode=True).
    # The authors' "MainNetwork" uses an input projection (opt_size -> resnet_width),
    # a stack of residual blocks (depth-1 of them), and an output projection back to opt_size.
    authors_resnet_width: int = 128

    # Comma-separated list of residual block types for the featureless authors network.
    # Each entry must be "exa" or "qua". The list length should equal n_layers.
    # Example: "exa,qua,qua"
    authors_block_types: str = "exa"
