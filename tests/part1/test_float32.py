import unittest

import numpy as np
import tensorflow as tf

from choice_learn_ext.models.deep_context.data_io import validate_arrays
from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel


def _batch(B=2, J=5):
    return {
        "available": tf.ones([B, J], dtype=tf.float32),
        "item_ids": tf.broadcast_to(tf.range(J, dtype=tf.int32)[tf.newaxis], [B, J]),
    }


class TestFloat32(unittest.TestCase):
    def test_model_outputs_are_float32(self):
        model = DeepContextChoiceModel(num_items=5)
        out = model(_batch(), training=False)
        self.assertEqual(out["log_probs"].dtype, tf.float32)
        self.assertEqual(out["utilities"].dtype, tf.float32)

    def test_nll_is_float32(self):
        model = DeepContextChoiceModel(num_items=5)
        batch = {**_batch(), "choice": tf.zeros([2], dtype=tf.int32)}
        nll = model.nll(batch, training=False)
        self.assertEqual(nll.dtype, tf.float32)

    def test_float64_available_upcast_by_model(self):
        model = DeepContextChoiceModel(num_items=5)
        batch = {
            "available": tf.cast(tf.ones([2, 5]), tf.float64),
            "item_ids": tf.broadcast_to(tf.range(5, dtype=tf.int32)[tf.newaxis], [2, 5]),
        }
        out = model(batch, training=False)
        self.assertEqual(out["log_probs"].dtype, tf.float32)

    def test_validate_arrays_casts_float64_available(self):
        available64 = np.ones((3, 4), dtype=np.float64)
        batch = validate_arrays(available=available64)
        self.assertEqual(batch.available.dtype.name, "float32")

    def test_validate_arrays_casts_float64_x(self):
        available = np.ones((2, 3), dtype=np.float32)
        X64 = np.ones((2, 3, 4), dtype=np.float64)
        batch = validate_arrays(available=available, X=X64)
        self.assertEqual(batch.X.dtype.name, "float32")

    def test_all_trainable_weights_are_float32(self):
        model = DeepContextChoiceModel(num_items=5)
        model(_batch(), training=False)  # build
        for var in model.trainable_variables:
            self.assertEqual(var.dtype, tf.float32, msg=f"{var.name} is {var.dtype}")


if __name__ == "__main__":
    unittest.main()
