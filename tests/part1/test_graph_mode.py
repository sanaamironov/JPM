import unittest

import tensorflow as tf

from choice_learn_ext.models.deep_context.deep_halo_core import DeepContextChoiceModel
from choice_learn_ext.models.deep_context.utils import masked_mean


def _batch(B: int = 4, J: int = 5) -> dict:
    available = tf.ones([B, J], dtype=tf.float32)
    item_ids = tf.broadcast_to(tf.range(J, dtype=tf.int32)[tf.newaxis], [B, J])
    choices = tf.zeros([B], dtype=tf.int32)
    return {"available": available, "item_ids": item_ids, "choice": choices}


class TestGraphMode(unittest.TestCase):
    def test_forward_compiles_under_tf_function(self):
        model = DeepContextChoiceModel(num_items=5)

        @tf.function
        def forward(inputs):
            return model(inputs, training=False)

        out = forward(_batch())
        self.assertEqual(out["log_probs"].dtype, tf.float32)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["log_probs"])))

    def test_nll_compiles_under_tf_function(self):
        model = DeepContextChoiceModel(num_items=5)

        @tf.function
        def compute_nll(inputs):
            return model.nll(inputs, training=False)

        nll = compute_nll(_batch())
        self.assertEqual(nll.dtype, tf.float32)
        self.assertGreaterEqual(float(nll.numpy()), 0.0)
        self.assertTrue(bool(tf.math.is_finite(nll).numpy()))

    def test_no_retrace_on_same_shape(self):
        model = DeepContextChoiceModel(num_items=5)
        b = _batch(B=4)

        # Pre-build the model with one eager call so that all Keras variables
        # exist before the @tf.function closure is created.  Without this,
        # TF legitimately traces a second time after build() adds new variables
        # to the closure — that first re-trace is expected TF behaviour, not a bug.
        model(b, training=False)

        trace_count = [0]

        @tf.function
        def forward(inputs):
            # Python side-effects only execute at trace time.
            trace_count[0] += 1
            return model(inputs, training=False)

        forward(b)  # first @tf.function call — traces once (model already built)
        forward(b)  # second call — must reuse the cached trace
        self.assertEqual(trace_count[0], 1,
                         "Model retraced on second call with identical inputs")

    def test_masked_mean_compiles_under_tf_function(self):
        @tf.function
        def compute(x, mask):
            return masked_mean(x, mask)

        x = tf.ones([2, 4, 8], dtype=tf.float32)
        mask = tf.constant([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=tf.float32)
        out = compute(x, mask)
        self.assertEqual(out.dtype, tf.float32)
        self.assertEqual(out.shape, (2, 8))

    def test_masked_mean_rank_guard_fires(self):
        # Passing a rank-3 mask should trigger the assert_rank guard.
        @tf.function
        def compute(x, mask):
            return masked_mean(x, mask)

        x = tf.ones([2, 4, 8], dtype=tf.float32)
        bad_mask = tf.ones([2, 4, 8], dtype=tf.float32)  # rank 3 — wrong
        with self.assertRaises((tf.errors.InvalidArgumentError, Exception)):
            compute(x, bad_mask)

    def test_train_step_compiles_under_tf_function(self):
        model = DeepContextChoiceModel(num_items=5)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

        b = _batch(B=4)
        model(b, training=False)  # pre-build before @tf.function boundary

        @tf.function
        def one_step(data):
            return model.train_step(data)

        result1 = one_step(b)
        result2 = one_step(b)
        self.assertTrue(bool(tf.math.is_finite(result1["loss"]).numpy()))
        self.assertTrue(bool(tf.math.is_finite(result2["loss"]).numpy()))


if __name__ == "__main__":
    unittest.main()
