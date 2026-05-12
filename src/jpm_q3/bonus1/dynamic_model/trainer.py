from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .model import DynamicContextSparseChoiceModel


class DynamicTrainer:
    """MAP trainer for the dynamic storable-goods model."""

    def __init__(self, model: DynamicContextSparseChoiceModel, cfg: DynamicModelConfig):
        self.model = model
        self.cfg = cfg
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.lr))

        J = cfg.num_items   # J+1, including outside option
        _sig = {
            "item_ids":          tf.TensorSpec([None, J], tf.int32),
            "available":         tf.TensorSpec([None, J], tf.float32),
            "price":             tf.TensorSpec([None, J], tf.float32),
            "market_id":         tf.TensorSpec([None],    tf.int32),
            "household_id":      tf.TensorSpec([None],    tf.int32),
            "inventory":         tf.TensorSpec([None],    tf.float32),
            "choice":            tf.TensorSpec([None],    tf.int32),
            "reward":            tf.TensorSpec([None],    tf.float32),
            "done":              tf.TensorSpec([None],    tf.float32),
            "next_item_ids":     tf.TensorSpec([None, J], tf.int32),
            "next_available":    tf.TensorSpec([None, J], tf.float32),
            "next_price":        tf.TensorSpec([None, J], tf.float32),
            "next_market_id":    tf.TensorSpec([None],    tf.int32),
            "next_household_id": tf.TensorSpec([None],    tf.int32),
            "next_inventory":    tf.TensorSpec([None],    tf.float32),
        }
        self._train_step_fn = (
            tf.function(self._train_step_eager, input_signature=[_sig])
            if cfg.compile_train_step
            else self._train_step_eager
        )

    def _train_step_eager(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        cur = {
            "item_ids":     batch["item_ids"],
            "available":    batch["available"],
            "price":        batch["price"],
            "market_id":    batch["market_id"],
            "household_id": batch["household_id"],
            "inventory":    batch["inventory"],
            "choice":       batch["choice"],
        }
        nxt = {
            "item_ids":     batch["next_item_ids"],
            "available":    batch["next_available"],
            "price":        batch["next_price"],
            "market_id":    batch["next_market_id"],
            "household_id": batch["next_household_id"],
            "inventory":    batch["next_inventory"],
        }
        with tf.GradientTape() as tape:
            parts = self.model.compute_loss(
                inputs=cur,
                next_inputs=nxt,
                reward=batch["reward"],
                done=batch["done"],
                training=True,
            )
        grads = tape.gradient(parts["total"], self.model.trainable_variables)
        pairs = [
            (g, v)
            for g, v in zip(grads, self.model.trainable_variables)
            if g is not None
        ]
        self.opt.apply_gradients(pairs)
        return parts

    def train_step(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return self._train_step_fn(batch)

    def fit(self, data: Dict[str, np.ndarray]) -> None:
        tensors = {k: tf.convert_to_tensor(v) for k, v in data.items()}
        ds = (
            tf.data.Dataset.from_tensor_slices(tensors)
            .shuffle(4096, seed=int(self.cfg.seed))
            .batch(int(self.cfg.batch_size))
            .prefetch(tf.data.AUTOTUNE)
        )
        for ep in range(1, int(self.cfg.epochs) + 1):
            m_total = tf.keras.metrics.Mean()
            m_nll = tf.keras.metrics.Mean()
            m_prior = tf.keras.metrics.Mean()

            for batch in ds:
                parts = self.train_step(batch)
                m_total.update_state(parts["total"])
                m_nll.update_state(parts["nll"])
                m_prior.update_state(parts["prior"])

            print(
                f"Epoch {ep:03d} | total={m_total.result():.4f} "
                f"nll={m_nll.result():.4f} prior={m_prior.result():.4f}"
            )
