import unittest
import warnings
from pathlib import Path

# Cleaner test logs (filter known benign Keras/TensorFlow warnings)
warnings.filterwarnings(
    "ignore", message=".*does not have a `build\\(\\)` method implemented.*"
)
warnings.filterwarnings("ignore", message=".*looks like it has unbuilt state.*")
warnings.filterwarnings(
    "ignore", message=".*__zero_halo.*not a valid root scope name.*"
)

from jpm_q3.hybrid.zhang_lu_sparse import (
    ZhangSparseConfig,
    run_experiment,
    save_results,
)


class TestZhangLuSparseSmoke(unittest.TestCase):
    def test_smoke_sparse20_saves_and_full_is_competitive(self):
        """
        Smoke test (small run):
        - Saves artifacts to results/
        - Full model is competitive with best baseline (optimization can be noisy at tiny scale)
        - Full model uses the sparse layer (mean_abs_d > 0)
        """
        cfg = ZhangSparseConfig(
            epochs=3,
            batch_size=256,
            verbose=0,
            seed=123,
            l1_strength=0.25,
            tau_detect=0.5,
        )

        payload = run_experiment(
            cfg=cfg,
            T=8,
            J_inside=6,
            N_t=200,
            sparse_frac_nonzero=0.2,
            taus=[0.25, 0.5],
        )

        out_dir = Path("results/hybrid/zhang_lu_sparse/test_smoke_sparse20")
        save_results(out_dir, payload)

        self.assertTrue((out_dir / "results.json").exists())
        self.assertTrue((out_dir / "summary.csv").exists())
        self.assertTrue((out_dir / "support.csv").exists())

        rows = {r["model"]: r for r in payload["summary_rows"]}
        nll_full = float(rows["Full"]["nll"])
        best_baseline = min(
            float(rows["DeepHalo-only"]["nll"]), float(rows["Halo+mu"]["nll"])
        )

        # Full should be close to (or better than) the best baseline on tiny runs.
        # This avoids flaky tests due to optimizer noise at small epochs/data.
        self.assertLessEqual(nll_full, best_baseline + 0.02)

        # Ensure d is actually being used (otherwise "Full" wiring is broken).
        self.assertGreater(float(rows["Full"]["mean_abs_d"]), 0.0)

    def test_smoke_sparse40_threshold_tradeoff(self):
        """
        Smoke test (small run):
        - Saves artifacts to results/
        - Thresholding behaves correctly: increasing tau should not increase predicted nonzeros,
          and should not decrease specificity.
        """
        cfg = ZhangSparseConfig(
            epochs=3,
            batch_size=256,
            verbose=0,
            seed=123,
            l1_strength=0.25,
            tau_detect=0.5,
        )

        payload = run_experiment(
            cfg=cfg,
            T=8,
            J_inside=6,
            N_t=200,
            sparse_frac_nonzero=0.4,
            taus=[0.25, 0.5],
        )

        out_dir = Path("results/hybrid/zhang_lu_sparse/test_smoke_sparse40")
        save_results(out_dir, payload)
        self.assertTrue((out_dir / "results.json").exists())

        support = {float(r["tau"]): r for r in payload["support_rows"]}
        r025 = support[0.25]
        r050 = support[0.5]

        # As tau increases: predicted nonzero rate should weakly decrease.
        self.assertGreaterEqual(float(r025["pred_nz"]), float(r050["pred_nz"]))

        # Specificity should weakly increase with tau.
        self.assertGreaterEqual(float(r050["specificity"]), float(r025["specificity"]))


if __name__ == "__main__":
    unittest.main()
