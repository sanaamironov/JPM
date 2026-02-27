from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass(frozen=True)
class BatchSpec:
    """Validated batch spec for a discrete choice problem."""

    available: tf.Tensor  # (N, J) bool/float
    choice: Optional[tf.Tensor] = None  # (N,) int
    item_ids: Optional[tf.Tensor] = None  # (N, J) int (featureless)
    X: Optional[tf.Tensor] = None  # (N, J, d) float (feature-based)

    def as_dict(self, require_choice: bool = False) -> Dict[str, tf.Tensor]:
        out: Dict[str, tf.Tensor] = {"available": self.available}
        if require_choice:
            if self.choice is None:
                raise ValueError("choice is required but missing.")
            out["choice"] = self.choice
        elif self.choice is not None:
            out["choice"] = self.choice
        if self.item_ids is not None:
            out["item_ids"] = self.item_ids
        if self.X is not None:
            out["X"] = self.X
        return out


def _to_tensor(x: Any, dtype: tf.dtypes.DType) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype)
    return tf.convert_to_tensor(x, dtype=dtype)


def validate_arrays(
    *,
    available: Any,
    choice: Optional[Any] = None,
    item_ids: Optional[Any] = None,
    X: Optional[Any] = None,
) -> BatchSpec:
    """Validate and convert array-like inputs into tensors with consistent dtypes."""
    avail_t = _to_tensor(available, tf.float32)
    if avail_t.shape.rank != 2:
        raise ValueError(f"available must have shape (N,J). Got rank={avail_t.shape.rank}.")
    n, j = avail_t.shape[0], avail_t.shape[1]

    choice_t = None
    if choice is not None:
        choice_t = _to_tensor(choice, tf.int32)
        if choice_t.shape.rank != 1:
            raise ValueError(f"choice must have shape (N,). Got rank={choice_t.shape.rank}.")
        if n is not None and choice_t.shape[0] != n:
            raise ValueError("choice and available must have the same N dimension.")

    item_ids_t = None
    if item_ids is not None:
        item_ids_t = _to_tensor(item_ids, tf.int32)
        if item_ids_t.shape.rank != 2:
            raise ValueError("item_ids must have shape (N,J).")
        if item_ids_t.shape[0] != n or item_ids_t.shape[1] != j:
            raise ValueError("item_ids must match available shape (N,J).")

    X_t = None
    if X is not None:
        X_t = _to_tensor(X, tf.float32)
        if X_t.shape.rank != 3:
            raise ValueError("X must have shape (N,J,d).")
        if X_t.shape[0] != n or X_t.shape[1] != j:
            raise ValueError("X must match available shape (N,J,*) in first two dims.")

    return BatchSpec(available=avail_t, choice=choice_t, item_ids=item_ids_t, X=X_t)


DEFAULT_COLMAP: Mapping[str, str] = {
    "available": "available",
    "choice": "choice",
    "item_ids": "item_ids",
    "X": "X",
}


def dataframe_to_arrays(
    df: pd.DataFrame,
    *,
    colmap: Optional[Mapping[str, str]] = None,
    require_choice: bool = True,
) -> BatchSpec:
    """Convert a DataFrame to tensors using a configurable column mapping.

    Parameters
    ----------
    df:
        DataFrame containing array-like columns.
    colmap:
        Mapping from logical names {available, choice, item_ids, X} to DataFrame columns.
    require_choice:
        If True, raises when the mapped choice column is missing.

    Notes
    -----
    This function intentionally validates shapes and raises informative errors.
    """
    cm = dict(DEFAULT_COLMAP)
    if colmap:
        cm.update(colmap)

    def _get(name: str) -> Optional[Any]:
        col = cm.get(name)
        if col is None:
            return None
        if col not in df.columns:
            return None
        return df[col].to_numpy()

    available = _get("available")
    if available is None:
        raise KeyError(f"Missing required column for 'available'. Expected one of: {cm.get('available')}")
    choice = _get("choice")
    if require_choice and choice is None:
        raise KeyError(f"Missing required column for 'choice'. Expected: {cm.get('choice')}")
    item_ids = _get("item_ids")
    X = _get("X")

    # Most DataFrame columns here are object arrays containing per-row arrays/lists.
    # We stack them into dense arrays.
    def _stack_if_object(arr: Any) -> Any:
        if arr is None:
            return None
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            return np.stack(arr, axis=0)
        return arr

    available = _stack_if_object(available)
    choice = _stack_if_object(choice)
    item_ids = _stack_if_object(item_ids)
    X = _stack_if_object(X)

    return validate_arrays(available=available, choice=choice, item_ids=item_ids, X=X)
