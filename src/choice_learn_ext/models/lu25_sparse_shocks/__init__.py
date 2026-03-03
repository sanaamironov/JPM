"""
Lu & Shimizu (2025) sparse market-product shocks estimator (choice-learn integration).

This package exposes a choice-learn compatible estimator operating on aggregated market shares.
"""

from .data_io import MarketShareDataset, Market
from .estimator import Lu25SparseShocksEstimator

__all__ = ["MarketShareDataset", "Market", "Lu25SparseShocksEstimator"]