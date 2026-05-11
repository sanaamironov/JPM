import unittest

import numpy as np
import tensorflow as tf

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.training import make_dataset


def _fixed_batch(N=16, J=4, seed=0):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    available = tf.ones([N, J], dtype=tf.float32)
    item_ids = tf.broadcast_to(tf.range(J, dtype=tf.int32)[tf.newaxis], [N, J])
    choices = tf.zeros([N], dtype=tf.int32)
    return {"available": available, "item_ids": item_ids, "choice": choices}


class TestNLL(unittest.TestCase):
    def test_nll_is_nonnegative(self):
        model = DeepContextChoiceModel(num_items=4)
        nll = model.nll(_fixed_batch(), training=False)
        self.assertGreaterEqual(float(nll.numpy()), 0.0)

    def test_nll_is_finite(self):
        model = DeepContextChoiceModel(num_items=4)
        nll = model.nll(_fixed_batch(), training=False)
        self.assertTrue(bool(tf.math.is_finite(nll).numpy()))

    def test_nll_decreases_after_training(self):
        tf.random.set_seed(0)
        J, N = 4, 32
        model = DeepContextChoiceModel(num_items=J)
        batch = _fixed_batch(N=N, J=J)

        nll_before = float(model.nll(batch, training=False).numpy())

        model.compile(optimizer=tf.keras.optimizers.Adam(5e-2))
        ds = make_dataset(batch, batch_size=N, shuffle=False)
        model.fit(ds, epochs=30, verbose=0)

        nll_after = float(model.nll(batch, training=False).numpy())
        self.assertLess(nll_after, nll_before,
                        msg=f"NLL did not decrease: {nll_before:.4f} -> {nll_after:.4f}")

    def test_nll_at_init_bounded_by_log_j(self):
        # At random init, NLL should be in a reasonable range around log(J).
        # We allow a wide window [0, 3*log(J)] to avoid flakiness.
        tf.random.set_seed(42)
        J = 4
        model = DeepContextChoiceModel(num_items=J)
        nll = float(model.nll(_fixed_batch(N=256, J=J), training=False).numpy())
        upper_bound = 3.0 * np.log(J)
        self.assertLess(nll, upper_bound,
                        msg=f"NLL={nll:.4f} unexpectedly large vs 3*log({J})={upper_bound:.4f}")

    def test_perfect_prediction_nll_near_zero(self):
        # Manually set logits so item 0 gets probability ~1.
        # Model output log_probs ≈ [0, -inf, -inf, ...] → NLL ≈ 0.
        model = DeepContextChoiceModel(num_items=4)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-1))
        N, J = 64, 4
        available = tf.ones([N, J], dtype=tf.float32)
        item_ids = tf.broadcast_to(tf.range(J, dtype=tf.int32)[tf.newaxis], [N, J])
        choices = tf.zeros([N], dtype=tf.int32)
        batch = {"available": available, "item_ids": item_ids, "choice": choices}
        ds = make_dataset(batch, batch_size=N, shuffle=False)
        model.fit(ds, epochs=200, verbose=0)
        nll = float(model.nll(batch, training=False).numpy())
        self.assertLess(nll, 0.1)


if __name__ == "__main__":
    unittest.main()
