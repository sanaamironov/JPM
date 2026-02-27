import unittest
import numpy as np
import tensorflow as tf

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel


class TestPaddingInvariance(unittest.TestCase):
    def test_padding_does_not_change_probs_for_real_items(self):
        """
        Padding invariance should be tested WITHIN the same model.

        If items are masked out (available=0), changing their item_ids must not
        change the probabilities assigned to the real (available=1) items.
        """
        # Single model with J=5
        model = DeepContextChoiceModel(num_items=5, featureless=True, d_embed=8, n_blocks=1)

        # Only first 3 items are available; last 2 are padding/unavailable
        avail = tf.constant([[1, 1, 1, 0, 0]], dtype=tf.float32)

        # Two different assignments for padded item_ids (positions 3 and 4)
        ids_a = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)
        ids_b = tf.constant([[0, 1, 2, 0, 1]], dtype=tf.int32)

        out_a = model({"available": avail, "item_ids": ids_a}, training=False)
        p_a = tf.exp(out_a["log_probs"]).numpy()[0]

        out_b = model({"available": avail, "item_ids": ids_b}, training=False)
        p_b = tf.exp(out_b["log_probs"]).numpy()[0]

        # Probabilities for unavailable items should be ~0
        self.assertTrue(np.allclose(p_a[3:], 0.0, atol=1e-7))
        self.assertTrue(np.allclose(p_b[3:], 0.0, atol=1e-7))

        # Real items should sum to 1 (because masked softmax renormalizes)
        self.assertTrue(np.allclose(p_a[:3].sum(), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(p_b[:3].sum(), 1.0, atol=1e-6))

        # Key property: real item probabilities do not change
        self.assertTrue(np.allclose(p_a[:3], p_b[:3], atol=1e-5))