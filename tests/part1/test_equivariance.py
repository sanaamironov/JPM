import unittest

import numpy as np
import tensorflow as tf

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel


class TestPermutationEquivariance(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    def _model(self, num_items: int) -> DeepContextChoiceModel:
        return DeepContextChoiceModel(num_items=num_items)

    def test_full_availability_permutation(self):
        J = 6
        model = self._model(J)
        available = tf.ones((1, J), dtype=tf.float32)
        item_ids = tf.range(J, dtype=tf.int32)[tf.newaxis, :]

        perm = np.random.permutation(J)
        inv_perm = np.argsort(perm)
        perm_tf = tf.constant(perm, dtype=tf.int32)

        out_orig = model({"available": available, "item_ids": item_ids}, training=False)
        out_perm = model(
            {
                "available": tf.gather(available, perm_tf, axis=1),
                "item_ids": tf.gather(item_ids, perm_tf, axis=1),
            },
            training=False,
        )
        lp_orig = out_orig["log_probs"].numpy()
        lp_back = out_perm["log_probs"].numpy()[:, inv_perm]
        self.assertTrue(np.allclose(lp_orig, lp_back, atol=1e-5))

    def test_partial_availability_permutation(self):
        # Only first 4 of 6 items available; permute only within the available set.
        J = 6
        model = self._model(J)
        available = tf.constant([[1, 1, 1, 1, 0, 0]], dtype=tf.float32)
        item_ids = tf.range(J, dtype=tf.int32)[tf.newaxis, :]

        perm = np.array([3, 0, 2, 1, 4, 5])  # swap first four, keep last two
        inv_perm = np.argsort(perm)
        perm_tf = tf.constant(perm, dtype=tf.int32)

        out_orig = model({"available": available, "item_ids": item_ids}, training=False)
        out_perm = model(
            {
                "available": tf.gather(available, perm_tf, axis=1),
                "item_ids": tf.gather(item_ids, perm_tf, axis=1),
            },
            training=False,
        )
        lp_orig = out_orig["log_probs"].numpy()
        lp_back = out_perm["log_probs"].numpy()[:, inv_perm]
        # Available items should be equivariant; unavailable items will both be -inf.
        self.assertTrue(np.allclose(lp_orig[:, :4], lp_back[:, :4], atol=1e-5))

    def test_equivariance_batch(self):
        # Same permutation applied to every row of a batch.
        J, B = 5, 8
        model = self._model(J)
        available = tf.ones((B, J), dtype=tf.float32)
        item_ids = tf.broadcast_to(tf.range(J, dtype=tf.int32)[tf.newaxis], [B, J])

        perm = np.random.permutation(J)
        inv_perm = np.argsort(perm)
        perm_tf = tf.constant(perm, dtype=tf.int32)

        out_orig = model({"available": available, "item_ids": item_ids}, training=False)
        out_perm = model(
            {
                "available": tf.gather(available, perm_tf, axis=1),
                "item_ids": tf.gather(item_ids, perm_tf, axis=1),
            },
            training=False,
        )
        lp_back = out_perm["log_probs"].numpy()[:, inv_perm]
        self.assertTrue(np.allclose(out_orig["log_probs"].numpy(), lp_back, atol=1e-5))

    def test_equivariance_holds_across_seeds(self):
        """Property must hold for several independent random permutations, not just one.

        This guards against a test that passes by luck when the drawn permutation
        happens to be close to the identity.
        """
        J = 6
        model = self._model(J)
        available = tf.ones((1, J), dtype=tf.float32)
        item_ids = tf.range(J, dtype=tf.int32)[tf.newaxis, :]

        for seed in range(8):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(J)
            inv_perm = np.argsort(perm)
            perm_tf = tf.constant(perm, dtype=tf.int32)

            out_orig = model({"available": available, "item_ids": item_ids},
                             training=False)
            out_perm = model(
                {
                    "available": tf.gather(available, perm_tf, axis=1),
                    "item_ids": tf.gather(item_ids, perm_tf, axis=1),
                },
                training=False,
            )
            lp_orig = out_orig["log_probs"].numpy()
            lp_back = out_perm["log_probs"].numpy()[:, inv_perm]
            self.assertTrue(
                np.allclose(lp_orig, lp_back, atol=1e-5),
                msg=f"Equivariance failed for permutation seed={seed}, perm={perm}",
            )


if __name__ == "__main__":
    unittest.main()
