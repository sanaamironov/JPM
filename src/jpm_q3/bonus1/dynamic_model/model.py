from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from choice_learn_ext.models.deep_context.config import DeepHaloConfig
from choice_learn_ext.models.deep_context.deep_halo_core import DeepHalo

tfd = tfp.distributions

from .config import DynamicModelConfig


class DynamicContextSparseChoiceModel(tf.keras.Model):
    """
    Dynamic discrete choice model for storable goods (revised Bonus 1).

    Architecture combines:
      (1) Halo effect — DeepHalo backbone: utility of brand j depends on the
          full composition of the available set A_t (context-dependent).
      (2) Price sensitivity — scalar beta_price * p_jt term.
      (3) Lu-style sparse shocks — xi_jt = mu_t + d_jt added to inside-good utility.
      (4) Inventory dynamics — continuation value V(t+1, s') approximated by a
          neural value head taking (market_id, inventory) as input.

    Estimation uses MAP (NLL + sparse prior + ridge on mu + TD-error on value).
    Common discount factor delta approximates the consumer-specific delta_i
    (stated as a modelling assumption; true delta_i are consumer-specific).

    Input batch keys:
        item_ids    : (B, J+1) int32   — [0, 1, ..., J] for every row
        available   : (B, J+1) float32 — 0/1 availability mask
        price       : (B, J+1) float32 — observed prices; price[:, 0] = 0 (outside)
        inventory   : (B,)     float32 — current inventory s_it (integer-valued in DGP)
        market_id   : (B,)     int32   — time period index t (0-indexed)
        choice      : (B,)     int32   — observed choice (only needed for NLL)
    """

    def __init__(self, cfg: DynamicModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.num_items < 2:
            raise ValueError("num_items must be >= 2 (outside option + >=1 brand).")

        # Freeze Python-level booleans so cfg reassignment never causes retracing.
        self._center_d = bool(cfg.center_d_within_market)

        # --- Halo backbone (featureless: brand identity via embedding) ---
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

        # --- Price sensitivity (scalar, shared across brands) ---
        self.beta_price = self.add_weight(
            name="beta_price",
            shape=(),
            initializer=tf.keras.initializers.Constant(cfg.beta_price_init),
            trainable=True,
            dtype=tf.float32,
        )

        # --- Petrin & Train (2010) control function coefficient ---
        # Absorbs the endogenous component of price via the first-stage residual
        # ê_jt computed in `control_function.compute_price_residuals`.
        self.lambda_control = self.add_weight(
            name="lambda_control",
            shape=(),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        # --- Lu-style sparse shocks: inside goods only ---
        self.mu = self.add_weight(
            name="mu",
            shape=(cfg.num_markets,),   # num_markets = T
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )
        self.d = self.add_weight(
            name="d",
            shape=(cfg.num_markets, cfg.J),   # (T, J) — inside goods
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        # --- Consumer-specific brand tastes (revised question item (2)) ---
        # eta_ij: heterogeneous across consumers, homogeneous across time.
        self.eta = self.add_weight(
            name="eta",
            shape=(cfg.num_households, cfg.J),
            initializer="zeros",
            trainable=True,
            dtype=tf.float32,
        )

        # Sparsity inclusion probability (logit scale)
        pi0 = float(cfg.a_pi / (cfg.a_pi + cfg.b_pi))
        pi0 = min(max(pi0, 1e-6), 1.0 - 1e-6)
        self.logit_pi = self.add_weight(
            name="logit_pi",
            shape=(),
            initializer=tf.keras.initializers.Constant(
                float(np.log(pi0) - np.log(1.0 - pi0))
            ),
            trainable=True,
            dtype=tf.float32,
        )

        # --- Value function approximation V(t, s) ---
        self.market_embed = tf.keras.layers.Embedding(
            cfg.num_markets, cfg.d_embed, name="market_embed"
        )
        self.value_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg.d_embed, activation="relu"),
                tf.keras.layers.Dense(1, activation=None),
            ],
            name="value_head",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augmented_utilities(
        self, inputs: Dict[str, tf.Tensor], training: bool
    ) -> tf.Tensor:
        """
        u_aug = u_halo
              + beta_price * p_jt
              + lambda_control * price_residual_jt     (Petrin-Train control function)
              + 1{j>0} * (mu_t + d_jt + eta_ij)

        Shape: (B, J+1)
        """
        out = self.halo(inputs, training=training)
        u_halo = out["utilities"]   # (B, J+1)

        market_id = tf.cast(inputs["market_id"], tf.int32)        # (B,)
        household_id = tf.cast(inputs["household_id"], tf.int32)  # (B,)

        mu_t = tf.gather(self.mu, market_id)                  # (B,)
        d_t = tf.gather(self.d, market_id)                    # (B, J)
        eta_i = tf.gather(self.eta, household_id)             # (B, J)

        if self._center_d:
            d_t = d_t - tf.reduce_mean(d_t, axis=1, keepdims=True)

        # Pad outside option column (index 0) with zero shock for both d and eta.
        d_pad = tf.concat([tf.zeros_like(d_t[:, :1]), d_t], axis=1)        # (B, J+1)
        eta_pad = tf.concat([tf.zeros_like(eta_i[:, :1]), eta_i], axis=1)  # (B, J+1)
        inside_mask = tf.concat(
            [tf.zeros_like(d_pad[:, :1]), tf.ones_like(d_pad[:, 1:])], axis=1
        )  # (B, J+1)

        prices = tf.cast(inputs["price"], tf.float32)              # (B, J+1)
        price_residual = tf.cast(inputs["price_residual"], tf.float32)  # (B, J+1)

        return (
            u_halo
            + self.beta_price * prices
            + self.lambda_control * price_residual            # Petrin-Train control
            + inside_mask * mu_t[:, None]
            + d_pad
            + eta_pad
        )

    def _value_from(
        self,
        market_id: tf.Tensor,
        inventory: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        """Approximate V(t, s) via the value_head. Shape: (B,)"""
        inv_scaled = tf.cast(inventory, tf.float32)[:, None] / float(self.cfg.S_max)
        m_emb = self.market_embed(tf.cast(market_id, tf.int32))   # (B, d_embed)
        x = tf.concat([inv_scaled, m_emb], axis=1)
        return tf.squeeze(self.value_head(x, training=training), axis=1)  # (B,)

    def _next_inventory_by_action(self, inventory: tf.Tensor) -> tf.Tensor:
        """
        Expected next inventory for each action j (deterministic transition).

        Fixed consumption = 1 per period (Ching 2020):
            s_consumed = max(0, s - 1)
            j = 0 (outside): s_next = s_consumed
            j > 0 (buy 1):   s_next = min(S_max, s_consumed + 1)

        Returns: (B, J+1)
        """
        inv = tf.cast(inventory, tf.float32)   # (B,)
        S_max = float(self.cfg.S_max)

        s_consumed = tf.maximum(inv - 1.0, 0.0)   # (B,)

        # Outside option column
        s_next_out = s_consumed[:, None]   # (B, 1)

        # Inside-option columns (all brands share same transition)
        s_next_in = tf.minimum(
            tf.expand_dims(s_consumed, 1) + 1.0, S_max
        )  # (B, 1) broadcast → tile over J brands
        s_next_in = tf.tile(s_next_in, [1, self.cfg.J])   # (B, J)

        return tf.concat([s_next_out, s_next_in], axis=1)   # (B, J+1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """
        Dynamic utility:
            u_dyn = u_aug + delta * V(t+1, s_next(j))

        where u_aug already contains halo + price + sparse shock terms.
        """
        u_aug = self._augmented_utilities(inputs, training=training)   # (B, J+1)

        inv = tf.cast(inputs["inventory"], tf.float32)          # (B,)
        market_id = tf.cast(inputs["market_id"], tf.int32)      # (B,)

        # Continuation values
        inv_next = self._next_inventory_by_action(inv)          # (B, J+1)
        B = tf.shape(inv)[0]
        n_items = self.cfg.num_items

        inv_next_flat = tf.reshape(inv_next, [-1])              # (B*(J+1),)
        next_t = tf.minimum(market_id + 1, self.cfg.T - 1)
        market_rep = tf.repeat(next_t, repeats=n_items)         # (B*(J+1),)

        v_next_flat = self._value_from(market_rep, inv_next_flat, training=training)
        v_next = tf.reshape(v_next_flat, [B, n_items])          # (B, J+1)

        u_dyn = u_aug + float(self.cfg.discount) * v_next

        # Availability masking
        avail = tf.cast(inputs["available"], tf.float32)
        u_masked = tf.where(avail > 0.5, u_dyn, tf.constant(float('-inf'), dtype=u_dyn.dtype))
        log_probs = tf.nn.log_softmax(u_masked, axis=1)

        v_cur = self._value_from(market_id, inv, training=training)

        return {"utilities": u_dyn, "log_probs": log_probs, "value": v_cur}

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def choice_nll(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        out = self.call(inputs, training=training)
        y = tf.cast(inputs["choice"], tf.int32)
        idx = tf.stack([tf.range(tf.shape(y)[0], dtype=tf.int32), y], axis=1)
        return -tf.reduce_mean(tf.gather_nd(out["log_probs"], idx))

    def static_choice_nll(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        """
        NLL using only static utilities — no continuation value contribution.

        This is the appropriate objective for Stage 1 of the two-stage estimator
        where the value head is frozen at its random initialisation.  Using the
        full `choice_nll` in that setting allows the random `market_embed`
        in the value head to absorb time-period variation that should be
        attributed to mu_t, biasing mu recovery toward zero.
        """
        u_aug = self._augmented_utilities(inputs, training=training)
        avail = tf.cast(inputs["available"], tf.float32)
        u_masked = tf.where(
            avail > 0.5, u_aug, tf.constant(float("-inf"), dtype=u_aug.dtype)
        )
        log_probs = tf.nn.log_softmax(u_masked, axis=1)
        y = tf.cast(inputs["choice"], tf.int32)
        idx = tf.stack([tf.range(tf.shape(y)[0], dtype=tf.int32), y], axis=1)
        return -tf.reduce_mean(tf.gather_nd(log_probs, idx))

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
        """
        Negative log prior (MAP penalty) implemented with TensorFlow Probability.

        Three components, each using a tfd distribution:

        1. Spike-and-slab on d  — tfd.MixtureSameFamily of two Gaussians:
               spike: N(0, sqrt(v0))  weight (1 - pi)
               slab:  N(0, sqrt(v1))  weight pi

        2. Beta prior on pi  — tfd.Beta(a_pi, b_pi) evaluated at sigmoid(logit_pi),
           plus the log-Jacobian for the logit reparameterisation.

        3. Gaussian prior on mu  — tfd.Normal(0, mu_sd).
        """
        d_flat = tf.reshape(self.d, [-1])   # (T*J,)
        pi = tf.clip_by_value(tf.math.sigmoid(self.logit_pi), 1e-7, 1.0 - 1e-7)

        v0 = tf.constant(self.cfg.v0, dtype=tf.float32)
        v1 = tf.constant(self.cfg.v1, dtype=tf.float32)

        # 1. Spike-and-slab prior on d via tfd.MixtureSameFamily
        spike_slab = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=tf.stack([1.0 - pi, pi])   # (2,) — weights for spike, slab
            ),
            components_distribution=tfd.Normal(
                loc=tf.zeros(2, dtype=tf.float32),
                scale=tf.stack([tf.sqrt(v0), tf.sqrt(v1)]),
            ),
        )

        # 2. Beta prior on pi (probability space) + logit-space Jacobian
        beta_prior = tfd.Beta(
            concentration1=tf.constant(self.cfg.a_pi, dtype=tf.float32),
            concentration0=tf.constant(self.cfg.b_pi, dtype=tf.float32),
        )
        log_jac = tf.math.log(pi) + tf.math.log1p(-pi)   # d(pi)/d(logit_pi)

        # 3. Gaussian prior on mu via tfd.Normal
        mu_prior = tfd.Normal(
            loc=tf.constant(0.0, dtype=tf.float32),
            scale=tf.constant(self.cfg.mu_sd, dtype=tf.float32),
        )

        # 4. Gaussian prior on eta_ij (consumer-brand tastes) via tfd.Normal
        eta_prior = tfd.Normal(
            loc=tf.constant(0.0, dtype=tf.float32),
            scale=tf.constant(self.cfg.sigma_eta, dtype=tf.float32),
        )

        return (
            -tf.reduce_mean(spike_slab.log_prob(d_flat))        # spike-and-slab on d
            - (beta_prior.log_prob(pi) + log_jac)               # Beta prior on pi (logit)
            - tf.reduce_mean(mu_prior.log_prob(self.mu))        # Gaussian prior on mu
            - tf.reduce_mean(eta_prior.log_prob(self.eta))      # Gaussian prior on eta_ij
        )

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
            nll
            + float(self.cfg.td_weight) * td
            + float(self.cfg.prior_weight) * prior
        )
        return {"total": total, "nll": nll, "td": td, "prior": prior}
