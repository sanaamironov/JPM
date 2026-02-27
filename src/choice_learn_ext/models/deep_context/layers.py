# src/jpm/question_3/choice_learn_ext/models/deep_context/layers.py

from __future__ import annotations

from typing import Dict

import tensorflow as tf

from .config import DeepHaloConfig
from .utils import masked_mean


class BaseEncoder(tf.keras.layers.Layer):
    """
    Shared encoder χ(x_j) -> z_j^{(0)}.

    - If cfg.featureless: Embedding(vocab_size, d_embed) on item_ids
    - Else: MLP on item features X (B, J, d_x)
    """

    def __init__(self, cfg: DeepHaloConfig, name: str = "base_encoder"):
        super().__init__(name=name)
        self.cfg = cfg

        if cfg.featureless:
            if cfg.vocab_size is None:
                raise ValueError("cfg.vocab_size must be set when featureless=True.")
            self.embedding = tf.keras.layers.Embedding(
                input_dim=cfg.vocab_size,
                output_dim=cfg.d_embed,
                name="item_embedding",
            )
            # identity: embeddings already have dimension d_embed
            self.mlp = tf.keras.layers.Lambda(lambda x: x, name="identity_mlp")
        else:
            if cfg.d_x is None:
                raise ValueError("cfg.d_x must be set when featureless=False.")
            self.embedding = None
            self.mlp = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(cfg.d_embed, activation="gelu"),
                    tf.keras.layers.Dense(cfg.d_embed, activation=None),
                ],
                name="feature_mlp",
            )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Returns z0: (B, J, d_embed)
        """
        if self.cfg.featureless:
            item_ids = tf.cast(inputs["item_ids"], tf.int32)  # (B, J)
            z = self.embedding(item_ids)  # (B, J, d_embed)
            return self.mlp(z, training=training)
        else:
            X = tf.convert_to_tensor(inputs["X"])  # (B, J, d_x)
            B = tf.shape(X)[0]
            J = tf.shape(X)[1]
            X_flat = tf.reshape(X, [B * J, self.cfg.d_x])
            z_flat = self.mlp(X_flat, training=training)  # (B*J, d_embed)
            return tf.reshape(z_flat, [B, J, self.cfg.d_embed])


class HaloBlock(tf.keras.layers.Layer):
    """
    One DeepHalo-style context layer.

    For each layer ℓ:
      - Compute context summary c = masked_mean(z_prev)
      - For each item j, each head h:
          u_{j,h} = φ_{ℓ,h}([z_phi_in_j, c])
      - Aggregate heads and residual:
          z_j_out = z_j_prev + (1/H) * Σ_h u_{j,h}
    """

    def __init__(self, cfg: DeepHaloConfig, name: str = "halo_block"):
        super().__init__(name=name)
        self.cfg = cfg

        self.phi_heads = []
        for h in range(cfg.n_heads):
            mlp = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        cfg.d_embed,
                        activation="gelu",
                        name=f"{name}_h{h}_dense1",
                    ),
                    tf.keras.layers.Dropout(cfg.dropout),
                    tf.keras.layers.Dense(
                        cfg.d_embed,
                        activation=None,
                        name=f"{name}_h{h}_dense2",
                    ),
                ],
                name=f"{name}_phi_head_{h}",
            )
            self.phi_heads.append(mlp)

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name=f"{name}_ln")

    def call(
        self,
        z_prev: tf.Tensor,  # (B, J, d)
        z_base: tf.Tensor,  # (B, J, d)
        avail: tf.Tensor,  # (B, J)
        training: bool = False,
    ) -> tf.Tensor:
        B = tf.shape(z_prev)[0]
        J = tf.shape(z_prev)[1]
        d = self.cfg.d_embed

        # Which representation feeds into φ
        if self.cfg.residual_variant == "fixed_base":
            z_phi_in = z_base  # (B, J, d)
        else:
            z_phi_in = z_prev  # (B, J, d)

        # Global context c: (B, d)
        c = masked_mean(z_prev, mask=avail, axis=1)  # (B, d)

        # Broadcast context to all items: (B, 1, d) -> (B, J, d)
        c_exp = tf.expand_dims(c, axis=1)  # (B, 1, d)
        c_exp = tf.tile(c_exp, [1, J, 1])  # (B, J, d)

        # φ inputs: concat [z_phi_in_j, c] -> (B, J, 2d)
        phi_inputs = tf.concat([z_phi_in, c_exp], axis=-1)

        # Flatten to apply MLPs
        phi_flat = tf.reshape(phi_inputs, [B * J, 2 * d])
        # each item is updated by a nonlinear function of its own embedding
        # plus the context summary.
        upd = 0.0
        for mlp in self.phi_heads:
            upd_flat = mlp(phi_flat, training=training)  # (B*J, d)
            upd_h = tf.reshape(upd_flat, [B, J, d])
            upd += upd_h

        upd = upd / float(self.cfg.n_heads)
        z_next = z_prev + upd
        z_next = self.layer_norm(z_next, training=training)
        return z_next

    """       In standard mode: no difference.
            In fixed_base mode: you’re using current layer representation to
            build context and base representation in φ inputs.
    """



# -----------------------------------------------------------------------------
# Authors-mode implementations (mirrors part_1/authors/*.py)
# -----------------------------------------------------------------------------

def _parse_block_types(spec: str, n_layers: int) -> list[str]:
    """
    Parse a comma-separated block type specification.

    The authors' Featureless MainNetwork expects `len(block_types) == depth - 1`.
    In our config, we interpret `n_layers` as the number of residual blocks, so we require
    `len(block_types) == n_layers`.

    If spec is a single token (e.g. "exa"), it is repeated n_layers times.
    """
    tokens = [t.strip().lower() for t in (spec or "").split(",") if t.strip()]
    if not tokens:
        tokens = ["exa"]
    if len(tokens) == 1 and n_layers > 1:
        tokens = tokens * n_layers
    if len(tokens) != n_layers:
        raise ValueError(
            f"authors_block_types must have length {n_layers} (got {len(tokens)}): {tokens}"
        )
    for t in tokens:
        if t not in ("exa", "qua"):
            raise ValueError(f"Unknown block type '{t}'. Expected 'exa' or 'qua'.")
    return tokens


class ExaResBlockTF(tf.keras.layers.Layer):
    """
    TensorFlow port of authors.Featureless.ExaResBlock.

    forward(z_prev, z0):
        linear_main(z_prev * linear_act(z0)) + z_prev
    """

    def __init__(self, input_dim: int, hidden_dim: int, name: str = "exa_res_block"):
        super().__init__(name=name)
        self.linear_main = tf.keras.layers.Dense(hidden_dim, use_bias=False, name=f"{name}_linear_main")
        self.linear_act = tf.keras.layers.Dense(hidden_dim, use_bias=False, name=f"{name}_linear_act")
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

    def call(self, z_prev: tf.Tensor, z0: tf.Tensor, training: bool = False) -> tf.Tensor:
        # z_prev: (B, hidden_dim), z0: (B, input_dim)
        gate = self.linear_act(z0)
        return self.linear_main(z_prev * gate) + z_prev


class QuaResBlockTF(tf.keras.layers.Layer):
    """
    TensorFlow port of authors.Featureless.QuaResBlock.

    forward(x):
        linear(x^2) + x
    """

    def __init__(self, d: int, name: str = "qua_res_block"):
        super().__init__(name=name)
        self.linear = tf.keras.layers.Dense(d, use_bias=False, name=f"{name}_linear")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.linear(tf.square(x)) + x


class AuthorsFeaturelessNetTF(tf.keras.layers.Layer):
    """
    TensorFlow port of authors.Featureless.MainNetwork.

    Inputs:
        - avail: (B, J) float/bool where 1 indicates item is available.
    Outputs:
        - logits: (B, J) with -inf where unavailable (for masked softmax)
        - log_probs: (B, J) masked log-softmax along axis=1
    """

    def __init__(self, cfg: DeepHaloConfig, name: str = "authors_featureless"):
        super().__init__(name=name)
        if cfg.vocab_size is None:
            raise ValueError("cfg.vocab_size must be set when featureless=True.")
        self.cfg = cfg
        J = int(cfg.vocab_size)
        W = int(cfg.authors_resnet_width)

        self.in_lin = tf.keras.layers.Dense(W, use_bias=False, name=f"{name}_in_lin")
        self.out_lin = tf.keras.layers.Dense(J, use_bias=False, name=f"{name}_out_lin")

        block_types = _parse_block_types(cfg.authors_block_types, cfg.n_layers)
        self.blocks: list[tf.keras.layers.Layer] = []
        for i, t in enumerate(block_types):
            if t == "exa":
                self.blocks.append(ExaResBlockTF(input_dim=J, hidden_dim=W, name=f"{name}_exa_{i}"))
            elif t == "qua":
                self.blocks.append(QuaResBlockTF(d=W, name=f"{name}_qua_{i}"))
            else:
                raise ValueError(f"Unknown block type: {t}")

    def call(self, avail: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        # avail: (B, J) in {0,1}
        mask = tf.cast(avail > 0.5, tf.bool)  # (B, J)
        e0 = tf.cast(avail, tf.float32)       # (B, J)
        z = self.in_lin(e0)                   # (B, W)

        for b in self.blocks:
            if isinstance(b, ExaResBlockTF):
                z = b(z_prev=z, z0=e0, training=training)
            else:
                z = b(z, training=training)

        logits = self.out_lin(z)  # (B, J)
        neg_inf = tf.constant(-1e9, dtype=logits.dtype)
        logits = tf.where(mask, logits, neg_inf)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        return {"logits": logits, "log_probs": log_probs}


class NonlinearTransformationTF(tf.keras.layers.Layer):
    """
    TensorFlow port of authors.FeatureBased.NonlinearTransformation.
    """

    def __init__(self, H: int, embed: int, dropout: float = 0.0, name: str = "nonlinear_transformation"):
        super().__init__(name=name)
        self.H = int(H)
        self.embed = int(embed)
        self.fc1 = tf.keras.layers.Dense(self.embed * self.H, use_bias=True, name=f"{name}_fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed, use_bias=True, name=f"{name}_fc2")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.enc_norm = tf.keras.layers.LayerNormalization(axis=-1, name=f"{name}_ln")

    def call(self, X: tf.Tensor, training: bool = False) -> tf.Tensor:
        # X: (B, J, embed)
        B = tf.shape(X)[0]
        J = tf.shape(X)[1]
        Y = self.fc1(X)  # (B, J, embed*H)
        Y = tf.reshape(Y, [B, J, self.H, self.embed])  # (B, J, H, embed)
        Y = tf.nn.relu(Y)
        Y = self.dropout(Y, training=training)
        # fc2 maps last dim embed->embed, broadcast over heads
        Y = self.fc2(Y)
        Y = self.enc_norm(Y, training=training)
        return Y


class AuthorsFeatureBasedNetTF(tf.keras.layers.Layer):
    """
    TensorFlow port of authors.FeatureBased.DeepHalo, adapted to fixed-universe masking.

    Inputs:
        - X: (B, J, d_x) float32
        - avail: (B, J) float/bool mask
    Outputs:
        - logits: (B, J) masked with -inf
        - log_probs: (B, J) masked log-softmax
    """

    def __init__(self, cfg: DeepHaloConfig, name: str = "authors_feature_based"):
        super().__init__(name=name)
        if cfg.d_x is None:
            raise ValueError("cfg.d_x must be set when featureless=False.")
        self.cfg = cfg
        embed = int(cfg.d_embed)
        H = int(cfg.n_heads)
        L = int(cfg.n_layers)

        self.basic_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(embed, activation="relu", name=f"{name}_enc_fc1"),
                tf.keras.layers.Dropout(cfg.dropout),
                tf.keras.layers.Dense(embed, activation="relu", name=f"{name}_enc_fc2"),
                tf.keras.layers.Dropout(cfg.dropout),
                tf.keras.layers.Dense(embed, activation=None, name=f"{name}_enc_fc3"),
            ],
            name=f"{name}_basic_encoder",
        )
        self.enc_norm = tf.keras.layers.LayerNormalization(axis=-1, name=f"{name}_enc_ln")

        self.aggregate_linear = [
            tf.keras.layers.Dense(H, use_bias=True, name=f"{name}_agg_{i}") for i in range(L)
        ]
        self.nonlinear = [
            NonlinearTransformationTF(H=H, embed=embed, dropout=cfg.dropout, name=f"{name}_nt_{i}")
            for i in range(L)
        ]

        self.final_linear = tf.keras.layers.Dense(1, use_bias=True, name=f"{name}_final")

    def call(self, X: tf.Tensor, avail: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        # X: (B, J, d_x), avail: (B, J)
        valid = tf.cast(avail > 0.5, tf.float32)  # (B, J)
        lengths = tf.reduce_sum(valid, axis=1)    # (B,)
        # Avoid divide-by-zero (shouldn't happen in valid data)
        lengths = tf.maximum(lengths, 1.0)

        Z = self.enc_norm(self.basic_encoder(X, training=training), training=training)  # (B, J, embed)
        X0 = tf.identity(Z)  # fixed copy, like X = Z.clone()

        for fc, nt in zip(self.aggregate_linear, self.nonlinear):
            # fc(Z): (B, J, H) -> masked sum over items, divide by lengths
            Z_bar = fc(Z) * tf.expand_dims(valid, axis=-1)  # (B, J, H)
            Z_bar = tf.reduce_sum(Z_bar, axis=1) / tf.expand_dims(lengths, axis=1)  # (B, H)
            Z_bar = tf.expand_dims(tf.expand_dims(Z_bar, axis=1), axis=-1)  # (B, 1, H, 1)

            phi = nt(X0, training=training)  # (B, J, H, embed)
            phi = phi * tf.expand_dims(tf.expand_dims(valid, axis=-1), axis=-1)  # mask (B,J,1,1)

            # (phi * Z_bar).sum(heads)/H + Z
            Z = tf.reduce_sum(phi * Z_bar, axis=2) / float(self.cfg.n_heads) + Z

        logits = tf.squeeze(self.final_linear(Z, training=training), axis=-1)  # (B, J)
        neg_inf = tf.constant(-1e9, dtype=logits.dtype)
        logits = tf.where(tf.cast(avail > 0.5, tf.bool), logits, neg_inf)
        log_probs = tf.nn.log_softmax(logits, axis=1)
        return {"logits": logits, "log_probs": log_probs}
