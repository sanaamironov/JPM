from __future__ import annotations

import json
from typing import Mapping, Optional, Any

import numpy as np
import pandas as pd
from .data_io import dataframe_to_arrays, validate_arrays
from .deep_halo_core import DeepContextChoiceModel
from .trainer import Trainer


class DeepHaloChoiceModel:
    """Public-facing wrapper that follows a choice-learn style estimator API.

    Design notes (review feedback)
    ------------------------------
    - Array/tensor inputs are the primary interface.
    - Pandas DataFrames are supported via `fit_df`/`predict_proba_df` with a configurable column mapping.
    - This avoids brittle dependencies on exact column names and improves testability.
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

        # Lazy-build: in feature-based mode we don't know d_x until we see X in fit().
        self.model: Optional[DeepContextChoiceModel] = None
        self.trainer: Optional[Trainer] = None

        if self.featureless:
            self.model = DeepContextChoiceModel(
                num_items=self.num_items,
                d_embed=self.d_embed,
                n_blocks=self.n_blocks,
                featureless=True,
            )
            self.trainer = Trainer(self.model, lr=self.lr)

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
        # Fail fast with a clear message (needed for tests + usability)
        if not self.featureless and X is None:
            raise ValueError("X must be provided when featureless=False.")

        # Lazy-build for feature-based mode once X is available
        if not self.featureless and (self.model is None or self.trainer is None):
            d_x = int(np.asarray(X).shape[-1])
            self.model = DeepContextChoiceModel(
                num_items=self.num_items,
                d_embed=self.d_embed,
                n_blocks=self.n_blocks,
                featureless=False,
                d_x=d_x,
            )
            self.trainer = Trainer(self.model, lr=self.lr)

        batch = validate_arrays(available=available, choice=choices, item_ids=item_ids, X=X)

        if self.trainer is None:
            raise RuntimeError("Trainer not initialized.")

        self.trainer.fit_arrays(
            available=batch.available,
            choices=batch.choice,
            item_ids=batch.item_ids,
            X=batch.X,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            shuffle=shuffle,
            seed=seed,
        )
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
        probs = self.trainer.predict_probs(
            available=batch.available,
            item_ids=batch.item_ids,
            X=batch.X,
            batch_size=batch_size or max(256, self.batch_size),
        )
        return probs.numpy()

    def negative_log_likelihood(
        self,
        available: Any,
        choices: Any,
        *,
        item_ids: Optional[Any] = None,
        X: Optional[Any] = None,
    ) -> float:
        batch = validate_arrays(available=available, choice=choices, item_ids=item_ids, X=X)
        return float(self.model.nll(batch.as_dict(require_choice=True), training=False).numpy())

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
    # Backwards-compatible aliases (old signature used DataFrames)
    # ------------------------------------------------------------------

    def fit_legacy(self, df: pd.DataFrame) -> "DeepHaloChoiceModel":
        return self.fit_df(df)

    def predict_proba_legacy(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba_df(df)

    def to_json(self) -> str:
        payload = {
            "num_items": self.num_items,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "d_embed": self.d_embed,
            "n_blocks": self.n_blocks,
            "featureless": self.featureless,
            "width_multiplier": self.width_multiplier,
        }
        return json.dumps(payload)

    @staticmethod
    def from_json(s: str) -> "DeepHaloChoiceModel":
        d = json.loads(s)
        return DeepHaloChoiceModel(**d)
