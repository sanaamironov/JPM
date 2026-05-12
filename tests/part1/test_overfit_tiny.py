import unittest

import tensorflow as tf

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.training import make_dataset


class TestOverfitTiny(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(0)

    def test_overfit_single_choice_pattern(self):
        """Model should strongly favour item 1 after training on a dataset
        where item 1 is always chosen from a 3-item set."""
        model = DeepContextChoiceModel(num_items=3)
        model.compile(optimizer=tf.keras.optimizers.Adam(5e-2))

        N = 40
        available = tf.constant([[1, 1, 1]] * N, dtype=tf.float32)
        item_ids = tf.constant([[0, 1, 2]] * N, dtype=tf.int32)
        choices = tf.constant([1] * N, dtype=tf.int32)

        # batch_size == N → one gradient step per epoch; 150 epochs ≈ 150 steps
        ds = make_dataset(
            {"available": available, "item_ids": item_ids, "choice": choices},
            batch_size=N,
            shuffle=False,
        )
        history = model.fit(ds, epochs=150, verbose=0)

        out = model({"available": available, "item_ids": item_ids}, training=False)
        probs = tf.exp(out["log_probs"]).numpy()
        mean_p1 = probs[:, 1].mean()

        self.assertGreater(mean_p1, 0.9)
        self.assertLess(history.history["loss"][-1], 0.5)


if __name__ == "__main__":
    unittest.main()
