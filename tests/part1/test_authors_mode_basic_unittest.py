
import unittest
import numpy as np

try:
    import tensorflow as tf
except Exception as e:
    tf = None

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel

@unittest.skipIf(tf is None, "TensorFlow not available")
class TestAuthorsModeBasic(unittest.TestCase):
    def test_authors_mode_masking_and_normalization(self):
        # Simple batch with J=4; item 2 unavailable
        J = 4
        avail = np.array([[1, 1, 0, 1]], dtype=np.float32)
        item_ids = np.arange(J, dtype=np.int32)[None, :]

        model = DeepContextChoiceModel(
            num_items=J,
            featureless=True,
            n_blocks=2,
            n_heads=2,
            d_embed=8,
            authors_mode=True,
            authors_resnet_width=64,
            authors_block_types="exa,qua",
        )

        out = model(
            {"available": tf.constant(avail, dtype=tf.float32),
            "item_ids": tf.constant(item_ids, dtype=tf.int32)},
            training=False,
        )
        p = tf.exp(out["log_probs"]).numpy()[0]
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(p[2]), 0.0, places=8)  # unavailable
        self.assertTrue(np.all(p >= 0.0))


if __name__ == "__main__":
    unittest.main()
