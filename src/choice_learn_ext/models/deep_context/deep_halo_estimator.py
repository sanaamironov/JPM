from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .data_io import dataframe_to_arrays, validate_arrays
from .deep_halo_core import DeepContextChoiceModel
from .training import make_dataset, predict_proba


class DeepHaloChoiceModel:
    """Public-facing estimator that follows a choice-learn style API.

    Training is delegated to Keras via model.compile() + model.fit().
    Array/tensor inputs are the primary interface; DataFrames are
    supported through fit_df/predict_proba_df with a configurable column map.
    """

    def __init__(
        self,
        num_items: int,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128,
        d_embed: int = 16,
        n_blocks: int = 2,
        featureless: bool = True,
        verbose: int = 1,
        seed: Optional[int] = 0,
        width_multiplier: int = 1,
    ):
        self.num_items = int(num_items)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.d_embed = int(d_embed)
        self.n_blocks = int(n_blocks)
        self.featureless = bool(featureless)
        self.verbose = int(verbose)
        self.seed = seed
        self.width_multiplier = int(width_multiplier)

        # Feature-based models are built lazily in fit() once d_x is known.
        self.model: Optional[DeepContextChoiceModel] = None
        if self.featureless:
            self.model = DeepContextChoiceModel(
                num_items=self.num_items,
                d_embed=self.d_embed,
                n_blocks=self.n_blocks,
                featureless=True,
            )

    # ------------------------------------------------------------------
    # Primary API (arrays / tensors)
    # ------------------------------------------------------------------

    def fit(
        self,
        available: Any,
        choices: Any,
        *,
        item_ids: Optional[Any] = None,
        X: Optional[Any] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> "DeepHaloChoiceModel":
        if not self.featureless and X is None:
            raise ValueError("X must be provided when featureless=False.")

        if not self.featureless and self.model is None:
            d_x = int(np.asarray(X).shape[-1])
            self.model = DeepContextChoiceModel(
                num_items=self.num_items,
                d_embed=self.d_embed,
                n_blocks=self.n_blocks,
                featureless=False,
                d_x=d_x,
            )

        batch = validate_arrays(available=available, choice=choices, item_ids=item_ids, X=X)

        if self.model.cfg.featureless and batch.item_ids is None:
            raise ValueError("item_ids is required when featureless=True.")

        ds_inputs: dict = {"available": batch.available, "choice": batch.choice}
        if self.model.cfg.featureless:
            ds_inputs["item_ids"] = batch.item_ids
        else:
            ds_inputs["X"] = batch.X

        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.lr))
        ds = make_dataset(
            ds_inputs,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=seed if seed is not None else self.seed,
        )
        self.model.fit(ds, epochs=self.epochs, verbose=self.verbose)
        return self

    def predict_proba(
        self,
        available: Any,
        *,
        item_ids: Optional[Any] = None,
        X: Optional[Any] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        batch = validate_arrays(available=available, choice=None, item_ids=item_ids, X=X)
        ds_inputs: dict = {"available": batch.available}
        if self.model.cfg.featureless:
            ds_inputs["item_ids"] = batch.item_ids
        else:
            ds_inputs["X"] = batch.X
        bs = batch_size or max(256, self.batch_size)
        ds = make_dataset(ds_inputs, batch_size=bs, shuffle=False)
        return predict_proba(self.model, ds).numpy()

    def negative_log_likelihood(
        self,
        available: Any,
        choices: Any,
        *,
        item_ids: Optional[Any] = None,
        X: Optional[Any] = None,
    ) -> float:
        batch = validate_arrays(available=available, choice=choices, item_ids=item_ids, X=X)
        return float(
            self.model.nll(batch.as_dict(require_choice=True), training=False).numpy()
        )

    # ------------------------------------------------------------------
    # DataFrame convenience API
    # ------------------------------------------------------------------

    def fit_df(
        self,
        df: pd.DataFrame,
        *,
        colmap: Optional[Mapping[str, str]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> "DeepHaloChoiceModel":
        batch = dataframe_to_arrays(df, colmap=colmap, require_choice=True)
        return self.fit(
            available=batch.available,
            choices=batch.choice,
            item_ids=batch.item_ids,
            X=batch.X,
            shuffle=shuffle,
            seed=seed,
        )

    def predict_proba_df(
        self,
        df: pd.DataFrame,
        *,
        colmap: Optional[Mapping[str, str]] = None,
    ) -> np.ndarray:
        batch = dataframe_to_arrays(df, colmap=colmap, require_choice=False)
        return self.predict_proba(
            available=batch.available,
            item_ids=batch.item_ids,
            X=batch.X,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({
            "num_items": self.num_items,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "d_embed": self.d_embed,
            "n_blocks": self.n_blocks,
            "featureless": self.featureless,
            "width_multiplier": self.width_multiplier,
        })

    @staticmethod
    def from_json(s: str) -> "DeepHaloChoiceModel":
        return DeepHaloChoiceModel(**json.loads(s))
