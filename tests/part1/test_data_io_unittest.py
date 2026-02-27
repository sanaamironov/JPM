import unittest
import numpy as np
import pandas as pd

from choice_learn_ext.models.deep_context.data_io import dataframe_to_arrays, validate_arrays


class TestDataIO(unittest.TestCase):
    def test_dataframe_colmap(self):
        df = pd.DataFrame(
            {
                "avail": [[1, 1, 0], [1, 0, 1]],
                "ids": [[0, 1, 2], [0, 1, 2]],
                "y": [1, 2],
            }
        )
        batch = dataframe_to_arrays(
            df,
            colmap={"available": "avail", "item_ids": "ids", "choice": "y"},
            require_choice=True,
        )
        self.assertEqual(tuple(batch.available.shape), (2, 3))
        self.assertEqual(tuple(batch.item_ids.shape), (2, 3))
        self.assertEqual(tuple(batch.choice.shape), (2,))

    def test_validate_arrays_padding_invariance_shapes(self):
        available = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
        item_ids = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
        choice = np.array([0, 2], dtype=np.int32)
        batch = validate_arrays(available=available, choice=choice, item_ids=item_ids)
        self.assertEqual(tuple(batch.available.shape), (2, 3))
        self.assertEqual(tuple(batch.item_ids.shape), (2, 3))
        self.assertEqual(tuple(batch.choice.shape), (2,))

    def test_missing_available_column_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with self.assertRaises(KeyError):
            dataframe_to_arrays(df, colmap={"available": "available"})


if __name__ == "__main__":
    unittest.main()
