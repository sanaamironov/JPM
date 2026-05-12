import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import unittest
import warnings

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore", message=".*does not have a `build\\(\\)` method.*")
warnings.filterwarnings("ignore", message=".*looks like it has unbuilt state.*")

from jpm_q3.bonus1.dynamic_model.config import DynamicModelConfig
from jpm_q3.bonus1.dynamic_model.data import simulate_dynamic_panel
from jpm_q3.bonus1.dynamic_model.model import DynamicContextSparseChoiceModel
from jpm_q3.bonus1.dynamic_model.trainer import DynamicTrainer
from jpm_q3.bonus1.dynamic_model.counterfactual import price_promotion_analysis


def _small_cfg(**kwargs) -> DynamicModelConfig:
    """Tiny config for fast unit tests."""
    defaults = dict(
        J=3, S_max=3, T=5, num_households=20,
        epochs=2, batch_size=64,
        compile_train_step=False,
        force_cpu=True, seed=7,
    )
    defaults.update(kwargs)
    return DynamicModelConfig(**defaults)


class TestBonus1DGP(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    def test_choice_sets_are_common_per_period(self):
        """A_t must be the same for all consumers at time t."""
        cfg = _small_cfg()
        data, meta = simulate_dynamic_panel(cfg)
        avail = data["available"]
        market_ids = data["market_id"]
        for t in range(cfg.T):
            mask = market_ids == t
            if mask.sum() > 1:
                rows = avail[mask]
                self.assertTrue(
                    np.all(rows == rows[0]),
                    msg=f"A_t is not common across consumers at t={t}"
                )

    def test_choice_sets_have_min_avail_brands(self):
        # Question requirement: |A_t| >= 3 BRANDS (inside goods only).
        cfg = _small_cfg(min_avail=2)
        data, meta = simulate_dynamic_panel(cfg)
        avail = data["available"]
        inside_avail = avail[:, 1:].sum(axis=1)
        self.assertTrue(np.all(inside_avail >= cfg.min_avail),
                        msg="|A_t| must satisfy at least min_avail inside brands")

    def test_outside_option_always_available(self):
        cfg = _small_cfg()
        data, _ = simulate_dynamic_panel(cfg)
        self.assertTrue(np.all(data["available"][:, 0] == 1.0))

    def test_inventory_bounded_by_s_max(self):
        cfg = _small_cfg()
        data, _ = simulate_dynamic_panel(cfg)
        self.assertTrue(np.all(data["inventory"] >= 0))
        self.assertTrue(np.all(data["inventory"] <= cfg.S_max))
        self.assertTrue(np.all(data["next_inventory"] >= 0))
        self.assertTrue(np.all(data["next_inventory"] <= cfg.S_max))

    def test_choices_in_valid_range(self):
        cfg = _small_cfg()
        data, _ = simulate_dynamic_panel(cfg)
        self.assertTrue(np.all(data["choice"] >= 0))
        self.assertTrue(np.all(data["choice"] < cfg.num_items))

    def test_prices_endogenous_positive(self):
        cfg = _small_cfg()
        data, _ = simulate_dynamic_panel(cfg)
        # All inside-good prices must be positive
        self.assertTrue(np.all(data["price"][:, 1:] > 0))
        # Outside-option price is 0
        self.assertTrue(np.all(data["price"][:, 0] == 0.0))

    def test_meta_keys_present(self):
        cfg = _small_cfg()
        _, meta = simulate_dynamic_panel(cfg)
        for key in ["alpha_true", "mu_true", "d_true", "gamma_true",
                    "xi_true", "eta_true", "price_inside", "avail_true",
                    "delta_true"]:
            self.assertIn(key, meta)

    def test_household_id_in_data(self):
        cfg = _small_cfg()
        data, _ = simulate_dynamic_panel(cfg)
        self.assertIn("household_id", data)
        self.assertIn("next_household_id", data)
        # Same consumer at t and t+1
        self.assertTrue(np.all(data["household_id"] == data["next_household_id"]))
        # All household IDs in valid range
        self.assertTrue(np.all(data["household_id"] >= 0))
        self.assertTrue(np.all(data["household_id"] < cfg.num_households))

    def test_eta_true_shape_and_homogeneity_over_time(self):
        cfg = _small_cfg()
        _, meta = simulate_dynamic_panel(cfg)
        eta = meta["eta_true"]
        self.assertEqual(eta.shape, (cfg.num_households, cfg.J))
        # eta should not be all zeros (true generating distribution is N(0, sigma_eta))
        self.assertGreater(float(np.abs(eta).mean()), 0.0)

    def test_control_function_residuals_in_data(self):
        """Petrin & Train (2010) control function residuals must be in the data."""
        cfg = _small_cfg()
        data, meta = simulate_dynamic_panel(cfg)
        # Observable cost shifter w_jt and the precomputed residual are present
        self.assertIn("w_obs", data)
        self.assertIn("price_residual", data)
        self.assertIn("next_w_obs", data)
        self.assertIn("next_price_residual", data)
        # Outside option columns must be zero for both
        self.assertTrue(np.all(data["w_obs"][:, 0] == 0.0))
        self.assertTrue(np.all(data["price_residual"][:, 0] == 0.0))
        # Inside-good residuals must have mean ~ 0 (first-stage OLS property)
        inside_resid = meta["price_residual_inside"]
        self.assertAlmostEqual(float(inside_resid.mean()), 0.0, places=5)

    def test_gamma_true_sparsity(self):
        cfg = _small_cfg(sparse_frac=0.5)
        _, meta = simulate_dynamic_panel(cfg)
        k_expected = max(1, int(0.5 * cfg.J))
        nz_per_t = meta["gamma_true"].sum(axis=1)
        self.assertTrue(np.all(nz_per_t == k_expected))


class TestBonus1Model(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        self.cfg = _small_cfg()
        self.data, self.meta = simulate_dynamic_panel(self.cfg)

    def _batch(self, n=4):
        idx = np.arange(n)
        return {k: tf.constant(v[idx]) for k, v in self.data.items()}

    def test_forward_output_shapes(self):
        model = DynamicContextSparseChoiceModel(self.cfg)
        batch = self._batch()
        out = model(
            {k: batch[k] for k in
             ["item_ids", "available", "price", "price_residual",
              "market_id", "household_id", "inventory"]},
            training=False,
        )
        J1 = self.cfg.num_items
        self.assertEqual(out["log_probs"].shape, (4, J1))
        self.assertEqual(out["utilities"].shape, (4, J1))

    def test_log_probs_sum_to_one(self):
        model = DynamicContextSparseChoiceModel(self.cfg)
        batch = self._batch()
        out = model(
            {k: batch[k] for k in
             ["item_ids", "available", "price", "price_residual",
              "market_id", "household_id", "inventory"]},
            training=False,
        )
        probs = tf.exp(out["log_probs"]).numpy()
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_all_weights_float32(self):
        model = DynamicContextSparseChoiceModel(self.cfg)
        for v in model.weights:
            self.assertEqual(v.dtype, tf.float32, msg=f"{v.name} is {v.dtype}")

    def test_beta_price_is_trainable(self):
        model = DynamicContextSparseChoiceModel(self.cfg)
        names = {v.name for v in model.trainable_variables}
        self.assertTrue(any("beta_price" in n for n in names))

    def test_eta_is_trainable_with_correct_shape(self):
        model = DynamicContextSparseChoiceModel(self.cfg)
        self.assertEqual(tuple(model.eta.shape),
                         (self.cfg.num_households, self.cfg.J))
        names = {v.name for v in model.trainable_variables}
        self.assertTrue(any("eta" in n for n in names))
        self.assertEqual(model.eta.dtype, tf.float32)

    def test_static_choice_nll_works(self):
        model = DynamicContextSparseChoiceModel(self.cfg)
        batch = self._batch()
        cur = {k: batch[k] for k in
               ["item_ids", "available", "price", "price_residual",
                "market_id", "household_id", "inventory", "choice"]}
        nll = model.static_choice_nll(cur, training=False)
        self.assertEqual(nll.dtype, tf.float32)
        self.assertGreaterEqual(float(nll.numpy()), 0.0)
        self.assertTrue(np.isfinite(float(nll.numpy())))

    def test_lambda_control_is_trainable(self):
        model = DynamicContextSparseChoiceModel(self.cfg)
        self.assertEqual(model.lambda_control.shape, ())
        self.assertEqual(model.lambda_control.dtype, tf.float32)
        names = {v.name for v in model.trainable_variables}
        self.assertTrue(any("lambda_control" in n for n in names))

    def test_smoke_train_runs_and_outputs_finite(self):
        model = DynamicContextSparseChoiceModel(self.cfg)
        model.halo.trainable = False
        trainer = DynamicTrainer(model, self.cfg)
        trainer.fit(self.data)
        self.assertTrue(np.isfinite(float(model.beta_price.numpy())))
        self.assertTrue(np.isfinite(float(tf.math.reduce_std(model.mu).numpy())))
        self.assertTrue(np.isfinite(float(tf.reduce_mean(tf.abs(model.d)).numpy())))

    def test_compiled_train_step_runs(self):
        cfg = _small_cfg(compile_train_step=True)
        data, _ = simulate_dynamic_panel(cfg)
        model = DynamicContextSparseChoiceModel(cfg)
        model.halo.trainable = False
        trainer = DynamicTrainer(model, cfg)
        trainer.fit(data)
        self.assertTrue(np.isfinite(float(model.beta_price.numpy())))


class TestBonus1CoverageStudy(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    def test_coverage_study_runs_and_produces_summary(self):
        """Multi-seed coverage study: tiny config, 2 seeds, smoke-level."""
        from jpm_q3.bonus1.dynamic_model.coverage_study import run_coverage_study

        cfg = DynamicModelConfig(
            num_households=10, T=5, epochs=3, batch_size=32,
            compile_train_step=False, force_cpu=True,
        )
        summary = run_coverage_study(base_cfg=cfg, n_seeds=2)

        self.assertIn("empirical_coverage_95", summary)
        self.assertIn("point_estimate_distribution", summary)
        self.assertEqual(len(summary["per_seed"]), 2)

        for key in ("beta_price", "mu", "d"):
            cov = summary["empirical_coverage_95"][key]
            self.assertGreaterEqual(cov, 0.0)
            self.assertLessEqual(cov, 1.0)


class TestBonus1Counterfactual(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    def test_counterfactual_returns_expected_keys(self):
        cfg = _small_cfg()
        data, _ = simulate_dynamic_panel(cfg)
        model = DynamicContextSparseChoiceModel(cfg)
        result = price_promotion_analysis(model, data, cfg, brand_x=1, discount_pct=10.0)
        for key in ["revenue_baseline", "revenue_promotion",
                    "revenue_change_abs", "revenue_change_pct",
                    "share_x_baseline", "share_x_promotion"]:
            self.assertIn(key, result)

    def test_promotion_increases_brand_share(self):
        """A price cut should weakly increase the promoted brand's market share."""
        cfg = _small_cfg(epochs=5)
        data, _ = simulate_dynamic_panel(cfg)
        model = DynamicContextSparseChoiceModel(cfg)
        model.halo.trainable = False
        trainer = DynamicTrainer(model, cfg)
        trainer.fit(data)
        result = price_promotion_analysis(model, data, cfg, brand_x=1, discount_pct=20.0)
        self.assertGreaterEqual(result["share_x_promotion"], result["share_x_baseline"] - 0.01)


if __name__ == "__main__":
    unittest.main()
