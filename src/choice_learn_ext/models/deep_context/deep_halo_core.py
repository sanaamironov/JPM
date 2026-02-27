# src/jpm/question_3/choice_learn_ext/models/deep_context/model.py

from __future__ import annotations

from typing import Dict

import tensorflow as tf

from .config import DeepHaloConfig
from .layers import BaseEncoder, HaloBlock, AuthorsFeaturelessNetTF, AuthorsFeatureBasedNetTF
from .utils import masked_log_softmax


class DeepHalo(tf.keras.Model):
    """
    Deep context-dependent choice model.

    Inputs dict must contain:
        - "available": (B, J)  float32/bool
        - if cfg.featureless: "item_ids": (B, J) int32
        - else: "X": (B, J, d_x) float32
        - "choice": (B,) int32  (only needed for nll)
    """

    def __init__(self, cfg: DeepHaloConfig, name: str = "DeepHalo"):
        super().__init__(name=name)
        self.cfg = cfg

        self.authors_net = None

        if cfg.authors_mode:
            if cfg.featureless:
                self.authors_net = AuthorsFeaturelessNetTF(cfg, name="authors_featureless")
            else:
                self.authors_net = AuthorsFeatureBasedNetTF(cfg, name="authors_feature_based")

        self.base_encoder = BaseEncoder(cfg, name="base_encoder")
        self.blocks = [
            HaloBlock(cfg, name=f"halo_block_l{layer_idx}")
            for layer_idx in range(cfg.n_layers)
        ]
        self.beta = tf.keras.layers.Dense(1, use_bias=False, name="beta_final")

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> Dict[str, tf.Tensor]:
        avail = tf.cast(inputs["available"], tf.float32)  # (B, J)

        # Authors replication path (mirrors part_1/authors/*.py)
        if self.cfg.authors_mode and self.authors_net is not None:
            if self.cfg.featureless:
                out = self.authors_net(avail, training=training)
                u = out["logits"]
                log_probs = out["log_probs"]
                return {"utilities": u, "log_probs": log_probs}
            else:
                X = tf.convert_to_tensor(inputs["X"])
                out = self.authors_net(X, avail, training=training)
                u = out["logits"]
                log_probs = out["log_probs"]
                return {"utilities": u, "log_probs": log_probs}

        # Simplified / ablation path
        z0 = self.base_encoder(inputs, training=training)  # (B, J, d)
        z = z0

        for block in self.blocks:
            z = block(z_prev=z, z_base=z0, avail=avail, training=training)

        u = self.beta(z)  # (B, J, 1)
        u = tf.squeeze(u, axis=-1)  # (B, J)

        log_probs = masked_log_softmax(u, mask=avail, axis=1)

        return {"utilities": u, "log_probs": log_probs}

    def nll(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Mean negative log-likelihood of observed choices.
        """
        outputs = self.call(inputs, training=training)
        logP = outputs["log_probs"]  # (B, J)
        choices = tf.cast(inputs["choice"], tf.int32)
        B = tf.shape(logP)[0]
        idx = tf.stack([tf.range(B, dtype=tf.int32), choices], axis=1)
        chosen_logp = tf.gather_nd(logP, idx)  # (B,)
        return -tf.reduce_mean(chosen_logp)


class DeepContextChoiceModel(DeepHalo):
    """
    Convenience wrapper used in tests and in the public wrapper.

    Notes:
      - If featureless=True, num_items is used as vocab_size for item_id embeddings.
      - If featureless=False, you must provide d_x (feature dimension).
    """

    def __init__(
        self,
        num_items: int,
        d_embed: int = 16,
        n_blocks: int = 2,
        n_heads: int = 2,
        residual_variant: str = "standard",
        dropout: float = 0.0,
        featureless: bool = True,
        d_x: int | None = None,
        authors_mode: bool = False,
        authors_resnet_width: int = 128,
        authors_block_types: str = "exa",
        name: str = "DeepContext",
    ):
        if not featureless and d_x is None:
            raise ValueError("d_x must be provided when featureless=False.")
        cfg = DeepHaloConfig(
            d_embed=d_embed,
            n_heads=n_heads,
            n_layers=n_blocks,
            residual_variant=residual_variant,
            featureless=bool(featureless),
            vocab_size=int(num_items) if featureless else None,
            d_x=int(d_x) if (d_x is not None) else None,
            dropout=dropout,
            authors_mode=authors_mode,
            authors_resnet_width=authors_resnet_width,
            authors_block_types=authors_block_types,
        )
        super().__init__(cfg, name=name)
