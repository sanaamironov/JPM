from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf


def make_dataset(
    tensors: Dict[str, tf.Tensor],
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Create a batched tf.data.Dataset from a dict of tensors."""
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if shuffle:
        ds = ds.shuffle(buffer_size=4096, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_proba(model, ds: tf.data.Dataset) -> tf.Tensor:
    """Predict choice probabilities for all batches in ds."""
    probs_list = []
    for batch in ds:
        out = model(batch, training=False)
        probs_list.append(tf.exp(out["log_probs"]))
    return tf.concat(probs_list, axis=0)
