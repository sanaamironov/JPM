import unittest

import numpy as np

from jpm_q3.lu25.estimators.blp import (
    invert_delta_contraction,
    iv_2sls_beta,
    compute_delta_vec,
    build_matrices,
)


class TestBLPSmoke(unittest.TestCase):
    def test_delta_inversion_returns_finite_values(self):
        # Tiny market: J=3, R=5, loose tolerance — verifies the contraction runs.
        J = 3
        s_obs = np.array([0.30, 0.20, 0.15], dtype=np.float64)
        p = np.array([1.0, 2.0, 1.5], dtype=np.float64)
        delta = invert_delta_contraction(s_obs, p, sigma=0.5, R=5, max_iter=50, tol=1e-4)
        self.assertEqual(delta.shape, (J,))
        self.assertTrue(np.all(np.isfinite(delta.numpy())))

    def test_delta_inversion_shape_varies_with_j(self):
        for J in [2, 5, 10]:
            s = np.ones(J, dtype=np.float64) * 0.05
            p = np.ones(J, dtype=np.float64)
            delta = invert_delta_contraction(s, p, sigma=0.1, R=5, max_iter=20, tol=1e-3)
            self.assertEqual(delta.shape, (J,))

    def test_iv_2sls_returns_correct_shape_and_finite(self):
        rng = np.random.default_rng(0)
        N, k, l = 30, 3, 5
        delta = rng.normal(0, 1, N)
        X = rng.normal(0, 1, (N, k))
        Z = rng.normal(0, 1, (N, l))
        beta = iv_2sls_beta(delta, X, Z).numpy()
        self.assertEqual(beta.shape, (k,))
        self.assertTrue(np.all(np.isfinite(beta)))

    def test_iv_2sls_recovers_known_beta(self):
        # OLS case: Z = X, should recover true beta up to numerical noise.
        rng = np.random.default_rng(7)
        N, k = 200, 2
        true_beta = np.array([2.0, -1.0])
        X = rng.normal(0, 1, (N, k))
        delta = X @ true_beta + rng.normal(0, 0.01, N)
        beta = iv_2sls_beta(delta, X, Z=X).numpy()
        self.assertTrue(np.allclose(beta, true_beta, atol=0.1),
                        msg=f"beta={beta}, expected≈{true_beta}")

    def test_compute_delta_vec_stacks_markets(self):
        T, J = 3, 3
        rng = np.random.default_rng(1)
        markets = [
            {"s": np.ones(J) * 0.1, "p": rng.uniform(1, 3, J), "w": rng.normal(0, 1, J)}
            for _ in range(T)
        ]
        delta_vec = compute_delta_vec(markets, sigma=0.2, R=5)
        self.assertEqual(delta_vec.shape, (T * J,))
        self.assertTrue(np.all(np.isfinite(delta_vec)))

    def test_build_matrices_shapes_cost_iv(self):
        T, J = 2, 4
        rng = np.random.default_rng(2)
        markets = [
            {
                "s": np.ones(J) * 0.08,
                "p": rng.uniform(1, 3, J),
                "w": rng.normal(0, 1, J),
                "u": rng.normal(0, 1, J),
            }
            for _ in range(T)
        ]
        X, Z = build_matrices(markets, iv_type="cost")
        self.assertEqual(X.shape, (T * J, 3))   # [1, p, w]
        self.assertEqual(Z.shape, (T * J, 5))   # [1, w, w^2, u, u^2]


if __name__ == "__main__":
    unittest.main()
