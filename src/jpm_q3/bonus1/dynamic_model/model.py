from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from choice_learn_ext.models.deep_context.config import DeepHaloConfig
from choice_learn_ext.models.deep_context.deep_halo_core import DeepHalo
from .config import DynamicModelConfig


class DynamicContextSparseChoiceModel(tf.keras.Model):
    """Deep context choice + Lu-style sparse shocks + dynamic inventory value."""

    def __init__(self, cfg: DynamicModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.num_items < 2:
            raise ValueError("num_items must be >= 2 (outside + at least one inside).")

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

        # Lu-style decomposition xi_tj = mu_t + d_tj for inside items only.
        self.mu = tf.Variable(tf.zeros([cfg.num_markets], dtype=tf.float32), name="mu")
        self.d = tf.Variable(
            tf.zeros([cfg.num_markets, cfg.num_items - 1], dtype=tf.float32), name="d"
        )
        # Global inclusion probability for spike-and-slab prior on d.
        pi0 = cfg.a_pi / (cfg.a_pi + cfg.b_pi)
        self.logit_pi = tf.Variable(
            np.log(pi0) - np.log(1.0 - pi0), dtype=tf.float32, name="logit_pi"
        )

        self.market_embed = tf.keras.layers.Embedding(cfg.num_markets, cfg.d_embed)
        self.value_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg.d_embed, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ]
        )

    def _augmented_utilities(self, inputs: Dict[str, tf.Tensor], training: bool) -> tf.Tensor:
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
        )
        return u_halo + inside_mask * mu_t[:, None] + d_pad

    def _value(self, inputs: Dict[str, tf.Tensor], training: bool) -> tf.Tensor:
        inventory = tf.cast(inputs["inventory"], tf.float32)  # [B]
        inv_scaled = inventory[:, None] / tf.constant(self.cfg.inventory_scale, tf.float32)
        market_id = tf.cast(inputs["market_id"], tf.int32)
        m_emb = self.market_embed(market_id)  # [B, d]
        val_in = tf.concat([inv_scaled, m_emb], axis=1)
        return tf.squeeze(self.value_head(val_in, training=training), axis=1)  # [B]

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> Dict[str, tf.Tensor]:
        u = self._augmented_utilities(inputs, training=training)
        avail = tf.cast(inputs["available"], tf.float32)
        u_masked = tf.where(avail > 0.5, u, tf.cast(-1e9, u.dtype))
        log_probs = tf.nn.log_softmax(u_masked, axis=1)
        value = self._value(inputs, training=training)
        return {"utilities": u, "log_probs": log_probs, "value": value}

    def choice_nll(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        out = self.call(inputs, training=training)
        y = tf.cast(inputs["choice"], tf.int32)
        idx = tf.stack([tf.range(tf.shape(y)[0], dtype=tf.int32), y], axis=1)
        chosen = tf.gather_nd(out["log_probs"], idx)
        return -tf.reduce_mean(chosen)

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
        target = tf.cast(reward, tf.float32) + self.cfg.discount * (
            1.0 - tf.cast(done, tf.float32)
        ) * tf.stop_gradient(v_next)
        return tf.reduce_mean(tf.square(v - target))

    def sparse_shock_prior_penalty(self) -> tf.Tensor:
        """Negative log-prior inspired by Lu-style spike-and-slab shrinkage."""
        d_flat = tf.reshape(self.d, [-1])
        pi = tf.math.sigmoid(self.logit_pi)
        eps = tf.constant(1e-7, dtype=tf.float32)
        pi_safe = tf.clip_by_value(pi, eps, 1.0 - eps)

        v0 = tf.constant(self.cfg.v0, dtype=tf.float32)
        v1 = tf.constant(self.cfg.v1, dtype=tf.float32)
        log2pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)

        logn0 = -0.5 * (tf.math.log(v0) + log2pi + tf.square(d_flat) / v0)
        logn1 = -0.5 * (tf.math.log(v1) + log2pi + tf.square(d_flat) / v1)
        log_mix = tf.reduce_logsumexp(
            tf.stack(
                [
                    tf.math.log1p(-pi_safe) + logn0,
                    tf.math.log(pi_safe) + logn1,
                ],
                axis=0,
            ),
            axis=0,
        )

        # Beta prior on pi with Jacobian for logit(pi), implemented directly
        # to avoid hard dependency on tensorflow_probability.
        a = tf.constant(self.cfg.a_pi, dtype=tf.float32)
        b = tf.constant(self.cfg.b_pi, dtype=tf.float32)
        log_beta_fn = tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b)
        beta_lp = (a - 1.0) * tf.math.log(pi_safe) + (b - 1.0) * tf.math.log1p(-pi_safe) - log_beta_fn
        log_jac = tf.math.log(pi_safe) + tf.math.log1p(-pi_safe)

        mu_ridge = 0.5 * tf.reduce_mean(tf.square(self.mu)) / (self.cfg.mu_sd**2)
        return -tf.reduce_mean(log_mix) - (beta_lp + log_jac) + mu_ridge

    def loss(
        self,
        inputs: Dict[str, tf.Tensor],
        next_inputs: Dict[str, tf.Tensor],
        reward: tf.Tensor,
        done: tf.Tensor,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        nll = self.choice_nll(inputs, training=training)
        td = self.td_error_loss(
            inputs=inputs,
            next_inputs=next_inputs,
            reward=reward,
            done=done,
            training=training,
        )
        prior = self.sparse_shock_prior_penalty()
        total = nll + self.cfg.td_weight * td + self.cfg.prior_weight * prior
        return {"total": total, "nll": nll, "td": td, "prior": prior}
