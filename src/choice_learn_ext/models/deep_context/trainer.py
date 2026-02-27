from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf

from choice_learn_ext.models.deep_context.deep_halo_core import DeepHalo
from choice_learn_ext.models.deep_context.training import make_dataset, train_epochs, predict_proba


class Trainer:
    """Compatibility wrapper around DeepHalo training utilities.

    Notes
    -----
    Review feedback requested decoupling training, data conversion, and interfaces.
    The preferred API is now in `choice_learn_ext.models.deep_context.training`.
    This class remains for backwards compatibility with existing experiments.
    """

    def __init__(self, model: DeepHalo, lr: float = 1e-3):
        self.model = model
        # Keras 3: `tf.keras.optimizers.legacy` is not supported.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(self, batch: Dict[str, tf.Tensor]) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.model.nll(batch, training=True)
            reg = 1e-6 * tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
            loss = loss + reg
        grads = tape.gradient(loss, self.model.trainable_variables)
        pairs = [(g, v) for (g, v) in zip(grads, self.model.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(pairs)
        return loss

    def fit_arrays(
        self,
        available: tf.Tensor,
        choices: tf.Tensor,
        item_ids: Optional[tf.Tensor] = None,
        X: Optional[tf.Tensor] = None,
        batch_size: int = 256,
        epochs: int = 20,
        verbose: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        ds_inputs: Dict[str, tf.Tensor] = {"available": available, "choice": choices}
        if self.model.cfg.featureless:
            if item_ids is None:
                raise ValueError("item_ids is required when featureless=True.")
            ds_inputs["item_ids"] = item_ids
        else:
            if X is None:
                raise ValueError("X is required when featureless=False.")
            ds_inputs["X"] = X

        ds = make_dataset(ds_inputs, batch_size=batch_size, shuffle=shuffle, seed=seed)
        train_epochs(self, ds, epochs=epochs, verbose=verbose)

    def predict_probs(
        self,
        available: tf.Tensor,
        item_ids: Optional[tf.Tensor] = None,
        X: Optional[tf.Tensor] = None,
        batch_size: int = 512,
    ) -> tf.Tensor:
        inputs: Dict[str, tf.Tensor] = {"available": available}
        if self.model.cfg.featureless:
            if item_ids is None:
                raise ValueError("item_ids is required when featureless=True.")
            inputs["item_ids"] = item_ids
        else:
            if X is None:
                raise ValueError("X is required when featureless=False.")
            inputs["X"] = X
        ds = make_dataset(inputs, batch_size=batch_size, shuffle=False)
        return predict_proba(self.model, ds)
