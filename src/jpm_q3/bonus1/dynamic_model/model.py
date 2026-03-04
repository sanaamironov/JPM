from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf
from choice_learn_ext.models.deep_context.config import DeepHaloConfig
from choice_learn_ext.models.deep_context.deep_halo_core import DeepHalo

from .config import DynamicModelConfig


class DynamicContextSparseChoiceModel(tf.keras.Model):
    """
    Dynamic discrete choice with:
      (1) context-dependent utility backbone (DeepHalo),
      (2) Lu-style unobservables: xi_{t j} = mu_t + d_{t j} (d sparse),
      (3) storable-goods dynamics via continuation values V(market, inventory).

    IMPORTANT: Do not define a method named `loss` on a Keras Model (Keras uses model.loss).
    We expose `compute_loss(...)` instead.
    """

    def __init__(self, cfg: DynamicModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.num_items < 2:
            raise ValueError("num_items must be >= 2 (outside + >=1 inside item).")

        halo_cfg = DeepHaloConfig(
            d_embed=cfg.d_embed,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_blocks,
            residual_variant=cfg.residual_variant,
            featureless=True,
            vocab_size=cfg.num_items,
            dropout=cfg.dropout,
        )
        self.halo = DeepHalo(halo_cfg)

        # ---------------------------------------------------------------------
        # Lu-style shocks: inside goods only
        #
        # IMPORTANT (Keras 3): use add_weight so these parameters are tracked as
        # trainable model weights. Plain tf.Variable can fail to be tracked and
        # will not be updated by the optimizer in some setups.
        # ---------------------------------------------------------------------
        self.mu = self.add_weight(
            name="mu",
            shape=(cfg.num_markets,),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        self.d = self.add_weight(
            name="d",
            shape=(cfg.num_markets, cfg.num_items - 1),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        # Global mixture weight for spike-and-slab-inspired prior on d
        pi0 = float(cfg.a_pi / (cfg.a_pi + cfg.b_pi))
        pi0 = min(max(pi0, 1e-6), 1.0 - 1e-6)
        logit_pi0 = float(np.log(pi0) - np.log(1.0 - pi0))

        self.logit_pi = self.add_weight(
            name="logit_pi",
            shape=(),
            initializer=tf.keras.initializers.Constant(logit_pi0),
            trainable=True,
            dtype=tf.float32,
        )

        # Value baseline V(market, inventory)
        self.market_embed = tf.keras.layers.Embedding(cfg.num_markets, cfg.d_embed)
        self.value_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg.d_embed, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )

    # -----------------------
    # Components
    # -----------------------
    def _augmented_utilities(
        self, inputs: Dict[str, tf.Tensor], training: bool
    ) -> tf.Tensor:
        """u_halo + inside(mu + d)."""
        out = self.halo(inputs, training=training)
        u_halo = out["utilities"]  # [B, J]

        market_id = tf.cast(inputs["market_id"], tf.int32)  # [B]
        mu_t = tf.gather(self.mu, market_id)  # [B]
        d_t = tf.gather(self.d, market_id)  # [B, J-1]

        if self.cfg.center_d_within_market:
            d_t = d_t - tf.reduce_mean(d_t, axis=1, keepdims=True)

        d_pad = tf.concat([tf.zeros_like(d_t[:, :1]), d_t], axis=1)  # [B, J]
        inside_mask = tf.concat(
            [tf.zeros_like(d_pad[:, :1]), tf.ones_like(d_pad[:, 1:])], axis=1
        )  # [B,J]
        return u_halo + inside_mask * mu_t[:, None] + d_pad

    def _value_from(
        self, market_id: tf.Tensor, inventory: tf.Tensor, training: bool
    ) -> tf.Tensor:
        inv = tf.cast(inventory, tf.float32)
        inv_scaled = inv[:, None] / tf.constant(self.cfg.inventory_scale, tf.float32)
        mid = tf.cast(market_id, tf.int32)
        m_emb = self.market_embed(mid)
        x = tf.concat([inv_scaled, m_emb], axis=1)
        return tf.squeeze(self.value_head(x, training=training), axis=1)  # [B]

    def _expected_next_inventory_by_action(self, inventory: tf.Tensor) -> tf.Tensor:
        """
        Deterministic expected next inventory for each action j:
            inv_next(j) = max(0, inv + purchase_qty * 1{j>0} - mean_consumption)
        Returns [B, J]
        """
        inv = tf.cast(inventory, tf.float32)  # [B]
        B = tf.shape(inv)[0]
        J = self.cfg.num_items

        inside = tf.concat(
            [tf.zeros([1], tf.float32), tf.ones([J - 1], tf.float32)], axis=0
        )[None, :]  # [1,J]
        add = float(self.cfg.purchase_qty) * tf.tile(inside, [B, 1])
        inv_next = inv[:, None] + add - float(self.cfg.mean_consumption)
        return tf.maximum(inv_next, 0.0)

    # -----------------------
    # Forward
    # -----------------------
    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Dynamic utility:
            u_dyn = u0 + discount * V(market, inv_next(j))

        where u0 includes DeepHalo + Lu-shocks + an inventory-motive term.
        """
        u0 = self._augmented_utilities(inputs, training=training)  # [B,J]

        inv = tf.cast(inputs["inventory"], tf.float32)  # [B]
        market_id = tf.cast(inputs["market_id"], tf.int32)  # [B]

        # Simple inventory motive: encourages buying when inventory is low
        inv_term = 1.0 / (1.0 + inv)  # [B]
        B = tf.shape(inv)[0]
        inside_mask = tf.concat(
            [
                tf.zeros([B, 1], tf.float32),
                tf.ones([B, self.cfg.num_items - 1], tf.float32),
            ],
            axis=1,
        )
        u0 = u0 + inside_mask * (0.8 * inv_term[:, None])

        # Continuation value per action via expected next inventory
        inv_next = self._expected_next_inventory_by_action(inv)  # [B,J]
        J = self.cfg.num_items
        inv_next_flat = tf.reshape(inv_next, [-1])  # [B*J]
        market_rep = tf.repeat(market_id, repeats=J)  # [B*J]
        v_next_flat = self._value_from(market_rep, inv_next_flat, training=training)
        v_next = tf.reshape(v_next_flat, [B, J])  # [B,J]

        u_dyn = u0 + float(self.cfg.discount) * v_next

        # Availability mask
        avail = tf.cast(inputs["available"], tf.float32)
        u_masked = tf.where(avail > 0.5, u_dyn, tf.cast(-1e9, u_dyn.dtype))
        log_probs = tf.nn.log_softmax(u_masked, axis=1)

        v_cur = self._value_from(market_id, inv, training=training)

        return {"utilities": u_dyn, "log_probs": log_probs, "value": v_cur}

    # -----------------------
    # Loss pieces
    # -----------------------
    def choice_nll(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        out = self.call(inputs, training=training)
        y = tf.cast(inputs["choice"], tf.int32)
        idx = tf.stack([tf.range(tf.shape(y)[0], dtype=tf.int32), y], axis=1)
        return -tf.reduce_mean(tf.gather_nd(out["log_probs"], idx))

    def td_error_loss(
        self,
        inputs: Dict[str, tf.Tensor],
        next_inputs: Dict[str, tf.Tensor],
        reward: tf.Tensor,
        done: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        v = self.call(inputs, training=training)["value"]
        v_next = self.call(next_inputs, training=False)["value"]
        target = tf.cast(reward, tf.float32) + float(self.cfg.discount) * (
            1.0 - tf.cast(done, tf.float32)
        ) * tf.stop_gradient(v_next)
        return tf.reduce_mean(tf.square(v - target))

    def sparse_shock_prior_penalty(self) -> tf.Tensor:
        """Spike-and-slab-inspired negative log prior (MAP)."""
        d_flat = tf.reshape(self.d, [-1])
        pi = tf.math.sigmoid(self.logit_pi)
        eps = tf.constant(1e-7, dtype=tf.float32)
        pi = tf.clip_by_value(pi, eps, 1.0 - eps)

        v0 = tf.constant(self.cfg.v0, dtype=tf.float32)
        v1 = tf.constant(self.cfg.v1, dtype=tf.float32)
        log2pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)

        logn0 = -0.5 * (tf.math.log(v0) + log2pi + tf.square(d_flat) / v0)
        logn1 = -0.5 * (tf.math.log(v1) + log2pi + tf.square(d_flat) / v1)

        log_mix = tf.reduce_logsumexp(
            tf.stack([tf.math.log1p(-pi) + logn0, tf.math.log(pi) + logn1], axis=0),
            axis=0,
        )

        # Beta prior on pi with logit Jacobian
        a = tf.constant(self.cfg.a_pi, dtype=tf.float32)
        b = tf.constant(self.cfg.b_pi, dtype=tf.float32)
        log_beta = tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b)
        beta_lp = (
            (a - 1.0) * tf.math.log(pi) + (b - 1.0) * tf.math.log1p(-pi) - log_beta
        )
        log_jac = tf.math.log(pi) + tf.math.log1p(-pi)

        mu_ridge = 0.5 * tf.reduce_mean(tf.square(self.mu)) / (self.cfg.mu_sd**2)

        return -tf.reduce_mean(log_mix) - (beta_lp + log_jac) + mu_ridge

    def compute_loss(
        self,
        inputs: Dict[str, tf.Tensor],
        next_inputs: Dict[str, tf.Tensor],
        reward: tf.Tensor,
        done: tf.Tensor,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        nll = self.choice_nll(inputs, training=training)
        td = self.td_error_loss(inputs, next_inputs, reward, done, training=training)
        prior = self.sparse_shock_prior_penalty()
        total = (
            nll + float(self.cfg.td_weight) * td + float(self.cfg.prior_weight) * prior
        )
        return {"total": total, "nll": nll, "td": td, "prior": prior}
