import unittest

import numpy as np

from jpm_q3.lu25.estimators.shrinkage import shrinkage_fit_beta_given_sigma


class TestShrinkageMCMC(unittest.TestCase):
    def _tiny_problem(self, N=40, k=2, seed=0):
        rng = np.random.default_rng(seed)
        true_beta = np.array([1.5, -0.8])[:k]
        X = rng.normal(0, 1, (N, k))
        delta = X @ true_beta + rng.normal(0, 0.1, N)
        return delta, X, true_beta

    def test_returns_finite_posterior_mean(self):
        delta, X, _ = self._tiny_problem()
        beta_mean, gamma_prob, score, acc_rate = shrinkage_fit_beta_given_sigma(
            delta_vec=delta, X=X, n_iter=40, burn=20, seed=0
        )
        self.assertEqual(beta_mean.shape, (2,))
        self.assertTrue(np.all(np.isfinite(beta_mean)),
                        msg=f"beta_mean has non-finite values: {beta_mean}")

    def test_gamma_prob_in_unit_interval(self):
        delta, X, _ = self._tiny_problem()
        _, gamma_prob, _, _ = shrinkage_fit_beta_given_sigma(
            delta_vec=delta, X=X, n_iter=40, burn=20, seed=1
        )
        self.assertEqual(gamma_prob.shape, (40,))
        self.assertTrue(np.all(gamma_prob >= 0.0))
        self.assertTrue(np.all(gamma_prob <= 1.0))

    def test_acceptance_rate_in_unit_interval(self):
        delta, X, _ = self._tiny_problem()
        _, _, _, acc_rate = shrinkage_fit_beta_given_sigma(
            delta_vec=delta, X=X, n_iter=40, burn=20, seed=2
        )
        self.assertGreaterEqual(float(acc_rate), 0.0)
        self.assertLessEqual(float(acc_rate), 1.0)

    def test_posterior_mean_near_truth_on_clean_data(self):
        # With very little noise and many observations, posterior mean should
        # be close to the true beta within a generous tolerance.
        rng = np.random.default_rng(99)
        N, k = 200, 2
        true_beta = np.array([2.0, -1.0])
        X = rng.normal(0, 1, (N, k))
        delta = X @ true_beta + rng.normal(0, 0.05, N)

        beta_mean, _, _, _ = shrinkage_fit_beta_given_sigma(
            delta_vec=delta, X=X, n_iter=200, burn=100, seed=0
        )
        self.assertTrue(
            np.allclose(beta_mean, true_beta, atol=0.5),
            msg=f"posterior mean {beta_mean} far from true {true_beta}",
        )

    def test_burn_lt_n_iter_enforced(self):
        delta, X, _ = self._tiny_problem()
        with self.assertRaises(ValueError):
            shrinkage_fit_beta_given_sigma(
                delta_vec=delta, X=X, n_iter=10, burn=10
            )


if __name__ == "__main__":
    unittest.main()
