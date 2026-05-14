import unittest

import numpy as np

from jpm_q3.lu25.estimators.blp import (
    invert_delta_contraction,
    iv_2sls_beta,
    compute_delta_vec,
    build_matrices,
)
from jpm_q3.lu25.experiments.replicate_section4 import build_matrices_paper


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


class TestBLPSigmaEstimation(unittest.TestCase):
    """Tests for the full sigma grid search in estimate_blp_sigma."""

    def _make_markets(self, T=5, J=4, seed=0):
        rng = np.random.default_rng(seed)
        markets = []
        for _ in range(T):
            p = rng.uniform(1.0, 3.0, J)
            w = rng.normal(0.0, 1.0, J)
            u = rng.normal(0.0, 0.5, J)
            s = np.ones(J) * 0.05
            markets.append({"s": s, "p": p, "w": w, "u": u})
        return markets

    def test_sigma_hat_in_search_range(self):
        """sigma_hat must lie within the grid bounds [0.05, 4.0]."""
        from jpm_q3.lu25.estimators.blp import estimate_blp_sigma
        markets = self._make_markets()
        sigma_hat, _, _ = estimate_blp_sigma(markets, iv_type="cost", R=10)
        self.assertGreaterEqual(sigma_hat, 0.0)
        self.assertLessEqual(sigma_hat, 4.5,
                             msg=f"sigma_hat={sigma_hat} is outside the search range")

    def test_sigma_hat_is_finite(self):
        from jpm_q3.lu25.estimators.blp import estimate_blp_sigma
        markets = self._make_markets()
        sigma_hat, beta_hat, _ = estimate_blp_sigma(markets, iv_type="cost", R=10)
        self.assertTrue(np.isfinite(sigma_hat))
        self.assertTrue(np.all(np.isfinite(beta_hat)))

    def test_extras_contain_expected_keys(self):
        """estimate_blp_sigma must return extras with delta_hat, X, Z, xi_hat, obj_hat."""
        from jpm_q3.lu25.estimators.blp import estimate_blp_sigma
        markets = self._make_markets()
        sigma_hat, beta_hat, extras = estimate_blp_sigma(markets, iv_type="cost", R=10)
        for key in ("obj_hat", "delta_hat", "X", "Z", "xi_hat"):
            self.assertIn(key, extras, msg=f"Missing key '{key}' in extras")
        # xi_hat shape must equal T*J
        T, J = 5, 4
        self.assertEqual(extras["xi_hat"].shape, (T * J,))


class TestBuildMatricesPaper(unittest.TestCase):
    def _make_markets(self, T=3, J=4):
        rng = np.random.default_rng(0)
        markets = []
        for _ in range(T):
            markets.append({
                "p": rng.uniform(1, 3, J),
                "w": rng.uniform(0.5, 2, J),
                "u": rng.normal(0, 1, J),
            })
        return markets

    def test_x_shape(self):
        markets = self._make_markets(T=3, J=4)
        X, Z, wbar = build_matrices_paper(markets, iv_type="cost")
        self.assertEqual(X.shape, (3 * 4, 2))

    def test_z_shape_cost_iv(self):
        markets = self._make_markets(T=3, J=4)
        X, Z, wbar = build_matrices_paper(markets, iv_type="cost")
        self.assertEqual(Z.shape, (3 * 4, 4))

    def test_centering_identity(self):
        T, J = 3, 4
        markets = self._make_markets(T=T, J=J)
        X, Z, wbar = build_matrices_paper(markets, iv_type="cost")
        w_c = X[:, 1]
        for t in range(T):
            block = w_c[t * J : (t + 1) * J]
            self.assertAlmostEqual(float(block.mean()), 0.0, places=10)

    def test_xi_correction_roundtrip(self):
        T, J = 3, 4
        markets = self._make_markets(T=T, J=J)
        X, Z, wbar = build_matrices_paper(markets, iv_type="cost")
        w_c = X[:, 1]
        w_recovered = w_c + wbar
        w_true = np.concatenate([m["w"] for m in markets])
        np.testing.assert_allclose(w_recovered, w_true, atol=1e-10)

    def test_no_constant_column(self):
        markets = self._make_markets(T=3, J=4)
        X, Z, _ = build_matrices_paper(markets, iv_type="cost")
        for col in range(X.shape[1]):
            self.assertFalse(
                np.allclose(X[:, col], 1.0),
                f"Column {col} of X appears to be a constant column",
            )

    # ------------------------------------------------------------------
    # nocost IV variant
    # ------------------------------------------------------------------

    def test_z_shape_nocost_iv(self):
        """Nocost IV: Z must be (T*J, 4) — columns [w_c, w_c^2, w_c^3, w_c^4]."""
        markets = self._make_markets(T=3, J=4)
        X, Z, _ = build_matrices_paper(markets, iv_type="nocost")
        self.assertEqual(X.shape, (3 * 4, 2))
        self.assertEqual(Z.shape, (3 * 4, 4))

    def test_nocost_centering_identity(self):
        """Within-market centering of w must hold for nocost IV too."""
        T, J = 3, 4
        markets = self._make_markets(T=T, J=J)
        X, Z, wbar = build_matrices_paper(markets, iv_type="nocost")
        w_c = X[:, 1]
        for t in range(T):
            block = w_c[t * J : (t + 1) * J]
            self.assertAlmostEqual(float(block.mean()), 0.0, places=10)

    def test_nocost_z_differs_from_cost_z(self):
        """Nocost and cost IV must produce different Z matrices (different instruments)."""
        markets = self._make_markets(T=3, J=4)
        _, Z_cost, _ = build_matrices_paper(markets, iv_type="cost")
        _, Z_nocost, _ = build_matrices_paper(markets, iv_type="nocost")
        self.assertFalse(
            np.allclose(Z_cost, Z_nocost),
            "Cost and nocost IV produced identical Z matrices",
        )

    def test_nocost_xi_correction_roundtrip(self):
        """wbar_vec + w_c recovers the original w vector for nocost IV."""
        T, J = 3, 4
        markets = self._make_markets(T=T, J=J)
        X, Z, wbar = build_matrices_paper(markets, iv_type="nocost")
        w_c = X[:, 1]
        w_true = np.concatenate([m["w"] for m in markets])
        np.testing.assert_allclose(w_c + wbar, w_true, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
