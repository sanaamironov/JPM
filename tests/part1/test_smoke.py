import unittest

import numpy as np
import tensorflow as tf

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.training import make_dataset


class TestSmoke(unittest.TestCase):
    def test_compile_and_fit_one_epoch(self):
        model = DeepContextChoiceModel(num_items=5)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-2))

        available = tf.constant([[1, 1, 1, 0, 0], [1, 0, 1, 0, 0]], dtype=tf.float32)
        item_ids = tf.constant([[0, 1, 2, 3, 4], [0, 2, 1, 3, 4]], dtype=tf.int32)
        choices = tf.constant([1, 0], dtype=tf.int32)

        ds = make_dataset(
            {"available": available, "item_ids": item_ids, "choice": choices},
            batch_size=2,
            shuffle=False,
        )
        history = model.fit(ds, epochs=1, verbose=0)
        loss_val = history.history["loss"][0]
        self.assertGreaterEqual(loss_val, 0.0)
        self.assertTrue(np.isfinite(loss_val))


if __name__ == "__main__":
    unittest.main()
