import numpy as np
import pandas as pd

from choice_learn_ext.models.deep_context.deep_halo_estimator import DeepHaloChoiceModel


def test_wrapper_fit_predict_df_and_arrays():
    df = pd.DataFrame(
        {
            "available": [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            "item_ids": [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ],
            "choice": [0, 1, 2],
        }
    )

    model = DeepHaloChoiceModel(num_items=3, epochs=3, batch_size=2, lr=1e-2)

    # DataFrame convenience API
    model.fit_df(df)
    probs_df = model.predict_proba_df(df)
    assert probs_df.shape == (3, 3)

    # Array-first API
    available = np.array(df["available"].tolist(), dtype=np.float32)
    item_ids = np.array(df["item_ids"].tolist(), dtype=np.int32)
    choices = np.array(df["choice"].tolist(), dtype=np.int32)

    model2 = DeepHaloChoiceModel(num_items=3, epochs=3, batch_size=2, lr=1e-2)
    model2.fit(available=available, choices=choices, item_ids=item_ids)
    probs = model2.predict_proba(available=available, item_ids=item_ids)
    assert probs.shape == (3, 3)

    preds = probs.argmax(axis=1)
    assert preds.shape == (3,)
