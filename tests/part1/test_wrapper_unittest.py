import unittest
import numpy as np
import pandas as pd


from choice_learn_ext.models.deep_context.deep_halo_estimator import DeepHaloChoiceModel


class TestWrapper(unittest.TestCase):
    def test_fit_df_with_colmap(self):
        df = pd.DataFrame({
            "avail_col": [[1, 1, 1]] * 5,
            "ids_col": [[0, 1, 2]] * 5,
            "y_col": [0, 1, 2, 0, 1],
        })

        model = DeepHaloChoiceModel(num_items=3, featureless=True, epochs=1, batch_size=2)

        model.fit_df(
            df,
            colmap={"available": "avail_col", "item_ids": "ids_col", "choice": "y_col"},
            shuffle=False,
            seed=0,
        )

        probs = model.predict_proba_df(
            df,
            colmap={"available": "avail_col", "item_ids": "ids_col"},
        )
        self.assertEqual(probs.shape, (5, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-6))

    def test_missing_column_error_is_clear(self):
        df = pd.DataFrame({
            "available": [[1, 1, 1]],
            # missing item_ids
            "choice": [0],
        })

        model = DeepHaloChoiceModel(num_items=3, featureless=True, epochs=1)

        with self.assertRaises(Exception) as ctx:
            model.fit_df(df, colmap={"available": "available", "item_ids": "item_ids", "choice": "choice"})

        msg = str(ctx.exception).lower()
        # we want an error that hints about missing item_ids
        self.assertTrue(("item_ids" in msg) or ("ids" in msg) or ("column" in msg))

    def test_feature_based_requires_x(self):
        available = np.ones((4, 3), dtype=np.float32)
        choice = np.array([0, 1, 2, 0], dtype=np.int32)

        model = DeepHaloChoiceModel(num_items=3, featureless=False, epochs=1)

        with self.assertRaises(Exception) as ctx:
            model.fit(available=available, choices=choice, X=None)

        self.assertIn("x", str(ctx.exception).lower())