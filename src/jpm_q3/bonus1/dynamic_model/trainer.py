from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .model import DynamicContextSparseChoiceModel


class DynamicTrainer:
    def __init__(self, model: DynamicContextSparseChoiceModel, cfg: DynamicModelConfig):
        self.model = model
        self.cfg = cfg
        self.opt = self._make_optimizer(cfg.lr, cfg.use_legacy_adam)
        if cfg.compile_train_step:
            self._train_step_fn = tf.function(self._train_step_eager)
        else:
            self._train_step_fn = self._train_step_eager

    @staticmethod
    def _make_optimizer(lr: float, use_legacy_adam: bool):
        if use_legacy_adam:
            legacy = getattr(tf.keras.optimizers, "legacy", None)
            if legacy is not None and hasattr(legacy, "Adam"):
                return legacy.Adam(learning_rate=lr)
        return tf.keras.optimizers.Adam(learning_rate=lr)

    def _train_step_eager(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        cur = {
            "item_ids": batch["item_ids"],
            "available": batch["available"],
            "market_id": batch["market_id"],
            "inventory": batch["inventory"],
            "choice": batch["choice"],
        }
        nxt = {
            "item_ids": batch["next_item_ids"],
            "available": batch["next_available"],
            "market_id": batch["next_market_id"],
            "inventory": batch["next_inventory"],
        }
        reward = batch["reward"]
        done = batch["done"]

        with tf.GradientTape() as tape:
            parts = self.model.loss(
                inputs=cur,
                next_inputs=nxt,
                reward=reward,
                done=done,
                training=True,
            )
        grads = tape.gradient(parts["total"], self.model.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
        self.opt.apply_gradients(pairs)
        return parts

    def train_step(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return self._train_step_fn(batch)

    def fit(self, data: Dict[str, np.ndarray]) -> None:
        tensors = {k: tf.convert_to_tensor(v) for k, v in data.items()}
        ds = tf.data.Dataset.from_tensor_slices(tensors).shuffle(4096, seed=self.cfg.seed).batch(
            self.cfg.batch_size
        )

        for ep in range(1, self.cfg.epochs + 1):
            m_total = tf.keras.metrics.Mean()
            m_nll = tf.keras.metrics.Mean()
            m_td = tf.keras.metrics.Mean()
            m_prior = tf.keras.metrics.Mean()
            for batch in ds:
                parts = self.train_step(batch)
                m_total.update_state(parts["total"])
                m_nll.update_state(parts["nll"])
                m_td.update_state(parts["td"])
                m_prior.update_state(parts["prior"])
            print(
                f"Epoch {ep:03d} | total={m_total.result():.4f} "
                f"nll={m_nll.result():.4f} td={m_td.result():.4f} prior={m_prior.result():.4f}"
            )
