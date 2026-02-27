from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf


def make_dataset(
    tensors: Dict[str, tf.Tensor],
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Create a batched `tf.data.Dataset` from a dict of tensors.

    This is intentionally separate from model/training code to make unit testing easier
    and to avoid hidden assumptions about input schemas.
    """
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if shuffle:
        ds = ds.shuffle(buffer_size=4096, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_epochs(
    trainer_like,
    ds: tf.data.Dataset,
    epochs: int,
    verbose: int = 1,
) -> None:
    """Train for a fixed number of epochs.

    `trainer_like` must expose `train_step(batch) -> loss`.
    """
    for ep in range(1, epochs + 1):
        running = tf.keras.metrics.Mean()
        for batch in ds:
            loss = trainer_like.train_step(batch)
            running.update_state(loss)
        if verbose:
            tf.print("Epoch", f"{ep:03d}", "NLL:", tf.round(running.result() * 1e4) / 1e4)


def predict_proba(model, ds: tf.data.Dataset) -> tf.Tensor:
    """Predict choice probabilities for all batches in `ds`."""
    probs_list = []
    for batch in ds:
        out = model(batch, training=False)
        probs_list.append(tf.exp(out["log_probs"]))
    return tf.concat(probs_list, axis=0)
