"""
Utility functions for financial analysis.

This module collects helper functions that support financial models and analysis.
Utilities are not models themselves, but provide functionality used by models.
"""

from .return_utils import (
    to_returns,
    historical_mean,
    ewma_return,
    capm_return,
)
from .basic import Finance

__all__ = [
    "to_returns",
    "historical_mean",
    "ewma_return",
    "capm_return",
    "Finance",
]
