import os

# Must be set BEFORE importing tensorflow to reduce TF startup spam.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide INFO + WARNING
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import unittest
import warnings

import numpy as np
import tensorflow as tf
from jpm_q3.bonus1.dynamic_model.config import DynamicModelConfig
from jpm_q3.bonus1.dynamic_model.data import simulate_dynamic_panel
from jpm_q3.bonus1.dynamic_model.model import DynamicContextSparseChoiceModel
from jpm_q3.bonus1.dynamic_model.trainer import DynamicTrainer

warnings.filterwarnings(
    "ignore", message=".*does not have a `build\\(\\)` method implemented.*"
)
warnings.filterwarnings("ignore", message=".*looks like it has unbuilt state.*")


class TestBonus1DynamicSmoke(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        tf.random.set_seed(0)

        # Make test logs quieter and more stable across machines.
        tf.get_logger().setLevel("ERROR")

        # Force CPU for deterministic + quiet unit tests (best-effort).
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    def test_availability_has_outside_and_at_least_two(self):
        cfg = DynamicModelConfig(households=10, periods=5, num_items=8, seed=7)
        data, _ = simulate_dynamic_panel(cfg)
        avail = data["available"]
        self.assertTrue(np.all(avail[:, 0] == 1.0))  # outside always available
        self.assertTrue(np.all(avail.sum(axis=1) >= 2))  # at least one inside item

    def test_smoke_train_runs_and_outputs_finite(self):
        cfg = DynamicModelConfig(
            households=40,
            periods=10,
            epochs=2,
            batch_size=128,
            num_items=6,
            num_markets=5,
            availability_prob=0.8,
            force_cpu=True,
            compile_train_step=False,
            seed=123,
        )

        data, meta = simulate_dynamic_panel(cfg)
        self.assertIn("gamma_true", meta)

        model = DynamicContextSparseChoiceModel(cfg)

        # For the bonus demo we freeze Halo to force shocks to explain residuals.
        model.halo.trainable = False

        trainer = DynamicTrainer(model, cfg)
        trainer.fit(data)

        pi_hat = float(tf.math.sigmoid(model.logit_pi).numpy())
        self.assertTrue(np.isfinite(pi_hat))

        mu_sd = float(tf.math.reduce_std(model.mu).numpy())
        self.assertTrue(np.isfinite(mu_sd))

        mean_abs_d = float(tf.reduce_mean(tf.abs(model.d)).numpy())
        self.assertTrue(np.isfinite(mean_abs_d))

        # Predicted nonzero rate should drop as tau increases.
        d_hat = model.d.numpy()
        pred_010 = float((np.abs(d_hat) > 0.10).mean())
        pred_015 = float((np.abs(d_hat) > 0.15).mean())
        self.assertLessEqual(pred_015, pred_010)


if __name__ == "__main__":
    unittest.main()
