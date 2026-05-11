import unittest

import numpy as np
import pandas as pd

from choice_learn_ext.models.deep_context.deep_halo_estimator import DeepHaloChoiceModel


class TestWrapperBasic(unittest.TestCase):
    def _df(self):
        return pd.DataFrame(
            {
                "available": [[1, 1, 1]] * 3,
                "item_ids": [[0, 1, 2]] * 3,
                "choice": [0, 1, 2],
            }
        )

    def test_fit_df_and_predict_proba_df(self):
        model = DeepHaloChoiceModel(num_items=3, epochs=2, batch_size=2, lr=1e-2)
        model.fit_df(self._df())
        probs = model.predict_proba_df(self._df())
        self.assertEqual(probs.shape, (3, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_fit_arrays_and_predict_proba(self):
        df = self._df()
        available = np.array(df["available"].tolist(), dtype=np.float32)
        item_ids = np.array(df["item_ids"].tolist(), dtype=np.int32)
        choices = np.array(df["choice"].tolist(), dtype=np.int32)

        model = DeepHaloChoiceModel(num_items=3, epochs=2, batch_size=2, lr=1e-2)
        model.fit(available=available, choices=choices, item_ids=item_ids)
        probs = model.predict_proba(available=available, item_ids=item_ids)
        self.assertEqual(probs.shape, (3, 3))
        self.assertEqual(probs.argmax(axis=1).shape, (3,))

    def test_predict_proba_rows_sum_to_one(self):
        model = DeepHaloChoiceModel(num_items=3, epochs=1, batch_size=3, lr=1e-2)
        model.fit_df(self._df())
        probs = model.predict_proba_df(self._df())
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
