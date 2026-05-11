import unittest

import numpy as np
import tensorflow as tf

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.model = DeepContextChoiceModel(num_items=5)

    def test_unavailable_items_have_near_zero_probability(self):
        available = tf.constant([[1, 0, 1, 0, 0]], dtype=tf.float32)
        item_ids = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)
        out = self.model({"available": available, "item_ids": item_ids}, training=False)
        log_probs = out["log_probs"].numpy()[0]
        for idx in [1, 3, 4]:
            self.assertLess(log_probs[idx], -1e5)

    def test_available_items_probabilities_sum_to_one(self):
        available = tf.constant([[1, 0, 1, 0, 0]], dtype=tf.float32)
        item_ids = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)
        out = self.model({"available": available, "item_ids": item_ids}, training=False)
        probs = tf.exp(out["log_probs"]).numpy()[0]
        self.assertTrue(np.allclose(probs[[0, 2]].sum(), 1.0, atol=1e-6))

    def test_all_available_sums_to_one(self):
        B, J = 4, 5
        available = tf.ones([B, J], dtype=tf.float32)
        item_ids = tf.broadcast_to(tf.range(J, dtype=tf.int32)[tf.newaxis], [B, J])
        out = self.model({"available": available, "item_ids": item_ids}, training=False)
        probs = tf.exp(out["log_probs"]).numpy()
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-6))

    def test_single_available_item_gets_probability_one(self):
        # Only item 2 available — it must receive probability ~1.
        available = tf.constant([[0, 0, 1, 0, 0]], dtype=tf.float32)
        item_ids = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)
        out = self.model({"available": available, "item_ids": item_ids}, training=False)
        probs = tf.exp(out["log_probs"]).numpy()[0]
        self.assertAlmostEqual(float(probs[2]), 1.0, places=5)

    def test_masked_item_ids_do_not_affect_available_probs(self):
        # Changing item_ids for unavailable slots must not change available-item probs.
        available = tf.constant([[1, 1, 1, 0, 0]], dtype=tf.float32)
        ids_a = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)
        ids_b = tf.constant([[0, 1, 2, 0, 1]], dtype=tf.int32)  # different padded ids
        p_a = tf.exp(self.model({"available": available, "item_ids": ids_a})["log_probs"]).numpy()[0]
        p_b = tf.exp(self.model({"available": available, "item_ids": ids_b})["log_probs"]).numpy()[0]
        self.assertTrue(np.allclose(p_a[:3], p_b[:3], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
