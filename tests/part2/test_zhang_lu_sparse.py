import unittest
import numpy as np
import tensorflow as tf

from jpm_q3.hybrid.zhang_lu_sparse import (
    ZhangSparseConfig,
    ZhangSparseDeepHalo,
    simulate_context_plus_sparse,
    choice_dataset_to_tensors,
)


class _ZeroHalo(tf.keras.Model):
    """Deterministic halo used for unit tests: utilities are identically 0."""
    def __init__(self, n_items: int):
        super().__init__()
        self._n_items = int(n_items)

    def call(self, inputs, training=False):
        # inputs["available"] has shape (B, n_items)
        B = tf.shape(inputs["available"])[0]
        u = tf.zeros((B, self._n_items), dtype=tf.float32)
        return {"utilities": u}


class TestZhangLuSparse(unittest.TestCase):
    def setUp(self):
        # Make tests deterministic
        np.random.seed(0)
        tf.random.set_seed(0)

    def test_forward_shapes_and_logprob_normalization(self):
        """
        Smoke test: forward pass returns correct shapes and log_probs normalize
        (softmax sums to 1 across available items).
        """
        cfg = ZhangSparseConfig(epochs=1, verbose=0)
        T = 3
        J_inside = 4
        n_items = J_inside + 1
        B = 5

        model = ZhangSparseDeepHalo(num_items=n_items, T=T, J_inside=J_inside, cfg=cfg)

        batch = {
            "item_ids": tf.constant(np.tile(np.arange(n_items)[None, :], (B, 1)), dtype=tf.int32),
            "available": tf.constant(np.ones((B, n_items), dtype=np.float32)),
            "choice": tf.constant(np.zeros((B,), dtype=np.int32)),
            "market_id": tf.constant(np.arange(B) % T, dtype=tf.int32),
        }

        out = model(batch, training=False)
        self.assertIn("utilities", out)
        self.assertIn("log_probs", out)

        utilities = out["utilities"].numpy()
        log_probs = out["log_probs"].numpy()

        self.assertEqual(utilities.shape, (B, n_items))
        self.assertEqual(log_probs.shape, (B, n_items))

        probs = np.exp(log_probs)
        row_sums = probs.sum(axis=1)
        # softmax rows should sum to 1 (numerical tolerance)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))

    def test_center_d_within_market_constraint(self):
        """
        Key Lu-style identification constraint:
        when center_d_within_market=True, for each market the mean of inside-good
        d_{t, j} applied in the utility should be ~0 (after centering).

        We enforce determinism by replacing halo with a ZeroHalo so utilities are
        driven only by (mu, d).
        """
        cfg = ZhangSparseConfig(center_d_within_market=True, verbose=0)
        T = 2
        J_inside = 3
        n_items = J_inside + 1
        B = 4

        model = ZhangSparseDeepHalo(num_items=n_items, T=T, J_inside=J_inside, cfg=cfg)
        # Replace halo with deterministic zero-utilities model
        model.halo = _ZeroHalo(n_items)

        # Set mu = 0 so only d matters
        model.mu.assign(tf.zeros_like(model.mu))

        # Make d nonzero and asymmetric to ensure centering is actually doing work
        d_init = np.array([[1.0, -2.0, 3.0],
                           [-4.0, 0.5, 1.5]], dtype=np.float32)
        model.d.assign(d_init)

        # Batch: alternating markets 0,1,0,1
        market_id = np.array([0, 1, 0, 1], dtype=np.int32)
        batch = {
            "item_ids": tf.constant(np.tile(np.arange(n_items)[None, :], (B, 1)), dtype=tf.int32),
            "available": tf.constant(np.ones((B, n_items), dtype=np.float32)),
            "choice": tf.constant(np.zeros((B,), dtype=np.int32)),
            "market_id": tf.constant(market_id, dtype=tf.int32),
        }

        out = model(batch, training=False)
        u_aug = out["utilities"].numpy()  # (B, n_items)

        # Extract inside goods utilities excluding outside option at index 0
        inside = u_aug[:, 1:]  # (B, J_inside)

        # With halo=0 and mu=0, inside utilities equal centered d for that market.
        # Check per-row mean ~ 0.
        inside_means = inside.mean(axis=1)
        self.assertTrue(np.allclose(inside_means, 0.0, atol=1e-6))

    def test_simulation_outputs_and_gamma_true(self):
        """
        Simulation test: simulate_context_plus_sparse returns:
        - ChoiceDataset with expected lengths/shapes
        - meta containing mu_true, d_true, gamma_true with consistent shapes
        - gamma_true has exactly k nonzeros per market (k = max(1, floor(frac * J_inside)))
        """
        T = 5
        J_inside = 10
        n_items = J_inside + 1
        N_t = 20
        seed = 123

        frac = 0.4
        dataset, meta = simulate_context_plus_sparse(
            T=T,
            J_inside=J_inside,
            N_t=N_t,
            seed=seed,
            d_embed=8,
            n_blocks=1,
            n_heads=1,
            mu_sd=2.0,
            sparse_frac_nonzero=frac,
            d_sd=1.0,
        )

        self.assertIn("mu_true", meta)
        self.assertIn("d_true", meta)
        self.assertIn("gamma_true", meta)

        mu_true = meta["mu_true"]
        d_true = meta["d_true"]
        gamma_true = meta["gamma_true"]

        self.assertEqual(mu_true.shape, (T,))
        self.assertEqual(d_true.shape, (T, J_inside))
        self.assertEqual(gamma_true.shape, (T, J_inside))

        # gamma_true should have exactly k ones per market
        k = max(1, int(frac * J_inside))
        nz_per_market = gamma_true.sum(axis=1)
        self.assertTrue(np.all(nz_per_market == k))

        # Dataset choices should be in [0, n_items-1]
        choices = np.asarray(dataset.choices).reshape(-1)
        self.assertEqual(choices.shape[0], T * N_t)
        self.assertTrue(np.all(choices >= 0))
        self.assertTrue(np.all(choices < n_items))

        # Conversion to tensors should produce correct shapes
        data = choice_dataset_to_tensors(dataset, n_items=n_items)
        self.assertEqual(data["item_ids"].shape, (T * N_t, n_items))
        self.assertEqual(data["available"].shape, (T * N_t, n_items))
        self.assertEqual(data["choice"].shape, (T * N_t,))
        self.assertEqual(data["market_id"].shape, (T * N_t,))


if __name__ == "__main__":
    unittest.main()