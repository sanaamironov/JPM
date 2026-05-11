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

    def test_wrong_rank_available_raises(self):
        with self.assertRaises(ValueError):
            validate_arrays(available=np.ones(5, dtype=np.float32))

    def test_choice_wrong_rank_raises(self):
        available = np.ones((3, 4), dtype=np.float32)
        bad_choice = np.ones((3, 4), dtype=np.int32)  # should be rank 1
        with self.assertRaises(ValueError):
            validate_arrays(available=available, choice=bad_choice)

    def test_choice_n_mismatch_raises(self):
        available = np.ones((3, 4), dtype=np.float32)
        wrong_n_choice = np.zeros(5, dtype=np.int32)  # N=5, not 3
        with self.assertRaises(ValueError):
            validate_arrays(available=available, choice=wrong_n_choice)

    def test_float64_available_cast_to_float32(self):
        available64 = np.ones((2, 3), dtype=np.float64)
        batch = validate_arrays(available=available64)
        self.assertEqual(batch.available.dtype.name, "float32")

    def test_float64_x_cast_to_float32(self):
        available = np.ones((2, 3), dtype=np.float32)
        X64 = np.ones((2, 3, 4), dtype=np.float64)
        batch = validate_arrays(available=available, X=X64)
        self.assertEqual(batch.X.dtype.name, "float32")

    def test_x_wrong_rank_raises(self):
        available = np.ones((2, 3), dtype=np.float32)
        bad_X = np.ones((2, 3), dtype=np.float32)  # rank 2, needs rank 3
        with self.assertRaises(ValueError):
            validate_arrays(available=available, X=bad_X)


if __name__ == "__main__":
    unittest.main()
