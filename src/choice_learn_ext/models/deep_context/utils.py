# src/jpm/question_3/choice_learn_ext/models/deep_context/utils.py

from __future__ import annotations

import platform
import sys

import tensorflow as tf


def masked_mean(x: tf.Tensor, mask: tf.Tensor, axis: int = 1) -> tf.Tensor:
    """
    Compute mean along `axis`, only where mask == 1.

    x:    (..., J, d)
    mask: (..., J)
    axis: axis over items (usually 1)

    Returns:
        (..., d)
    """
    x = tf.convert_to_tensor(x)
    mask = tf.cast(mask, x.dtype)  # (..., J)

    # Expand mask so it can multiply x
    while len(mask.shape) < len(x.shape):
        mask = tf.expand_dims(mask, axis=-1)  # (..., J, 1)

    # Sum over items
    num = tf.reduce_sum(x * mask, axis=axis)  # (..., d)
    den = tf.reduce_sum(mask, axis=axis)  # (..., 1)

    den = tf.maximum(den, tf.constant(1.0, dtype=x.dtype))
    # Broadcasting: (..., d) / (..., 1) -> (..., d)
    return num / den


def masked_softmax(logits: tf.Tensor,
                   mask: tf.Tensor,
                   axis: int = -1) -> tf.Tensor:
    """
    Softmax over `logits` while enforcing `mask` (0/1) entries to have prob 0.
    logits: [..., J]
    mask:   [..., J] broadcastable to logits
    """
    logits = tf.convert_to_tensor(logits)
    mask = tf.cast(mask, logits.dtype)

    # Put -inf on masked-out items so softmax gives exactly 0 there.
    neg_inf = tf.constant(-1e9, dtype=logits.dtype)
    masked_logits = tf.where(mask > 0, logits, neg_inf)

    return tf.nn.softmax(masked_logits, axis=axis)

def masked_log_softmax(logits: tf.Tensor,
                       mask: tf.Tensor,
                       axis: int = -1) -> tf.Tensor:
    """
    Log-softmax with a binary availability mask.

    logits: (B, J)
    mask:   (B, J)  1 if item is available, 0 otherwise

    Returns log-probabilities over available items; unavailable items
    effectively have log-prob ~ -inf.
    """

    logits = tf.convert_to_tensor(logits)
    mask = tf.cast(mask, logits.dtype)
    neg_inf = tf.constant(-1e9, dtype=logits.dtype)
    masked_logits = tf.where(mask > 0, logits, neg_inf)
    return tf.nn.log_softmax(masked_logits, axis=axis)
def apple_silicon():
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    return is_apple_silicon



def set_global_determinism(seed: int = 123) -> None:
    """Best-effort reproducibility across Python/NumPy/TensorFlow.

    Notes:
    - Full determinism is not guaranteed across all TF ops / devices.
    - On Apple Silicon / Metal backend, some ops may still be non-deterministic.
    """
    try:
        import os, random
        import numpy as np
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

    try:
        tf.random.set_seed(seed)
        # TF 2.13+ supports enabling deterministic ops; safe to try.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass
