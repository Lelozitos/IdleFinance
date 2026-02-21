"""
IdleFinance core module.

This module re-exports the accessor for backwards compatibility.
The actual accessor implementation is in accessor.py

Example
-------
    >>> import pandas as pd
    >>> import IdleFinance as idf
    >>> df_prices = pd.read_csv('prices.csv', index_col=0, parse_dates=True)
    >>> returns = df_prices.finance.returns()
    >>> cov = df_prices.finance.covariance()
"""

from .accessor import DataFrameFinanceAccessor, SeriesFinanceAccessor


__all__ = [
    "DataFrameFinanceAccessor",
    "SeriesFinanceAccessor",
]
