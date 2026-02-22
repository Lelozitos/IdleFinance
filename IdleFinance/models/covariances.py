"""
Covariance matrix estimation methods.

This module provides functions for calculating different types of covariance matrices
from return or price data, such as sample covariance and exponentially-weighted 
covariance (EMA).
"""

import numpy as np
import pandas as pd
from ..utils.return_utils import to_returns
from ..core._types import Union, Any, CovarianceMethod, PriceData, CovarianceMatrix

def sample_covariance(
    prices: PriceData, 
    returns_data: bool = False, 
    log_returns: bool = False,
    annualized: bool = True,
    frequency: int = 252
) -> CovarianceMatrix:
    """
    Calculate the sample covariance matrix of returns.

    Parameters
    ----------
    prices : pd.Series, pd.DataFrame, or np.ndarray
        Price series or DataFrame.
    returns_data : bool, default False
        If True, the input is treated as returns instead of prices.
    log_returns : bool, default False
        If True and returns_data is False, computes log returns.
    annualized : bool, default True
        If True, annualizes the covariance matrix.
    frequency : int, default 252
        Trading periods per year.

    Returns
    -------
    pd.DataFrame
        Sample covariance matrix.
    """
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = to_returns(prices, log_returns=log_returns)

    cov = returns.cov()
    if annualized:
        cov = cov * frequency
        
    return cov


def exponential_covariance(
    prices: PriceData, 
    span: int = 180,
    returns_data: bool = False, 
    log_returns: bool = False,
    annualized: bool = True,
    frequency: int = 252
) -> CovarianceMatrix:
    """
    Calculate the exponentially-weighted moving average (EWMA) covariance matrix.

    EWMA places more weight on recent observations, making the volatility estimate 
    more responsive to new market conditions.

    Parameters
    ----------
    prices : pd.Series, pd.DataFrame, or np.ndarray
        Price series or DataFrame.
    span : int, default 180
        The decay span (e.g., number of days) for the exponential weighting.
    returns_data : bool, default False
        If True, the input is treated as returns instead of prices.
    log_returns : bool, default False
        If True and returns_data is False, computes log returns.
    annualized : bool, default True
        If True, annualizes the covariance matrix.
    frequency : int, default 252
        Trading periods per year.

    Returns
    -------
    pd.DataFrame
        EWMA covariance matrix.
    """
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = to_returns(prices, log_returns=log_returns)

    cov = returns.ewm(span=span).cov().xs(returns.index[-1], level=0)
    
    if annualized:
        cov = cov * frequency
        
    return cov


def covariance(
    prices: PriceData, 
    method: CovarianceMethod = "sample",
    returns_data: bool = False, 
    **kwargs: Any
) -> CovarianceMatrix:
    """
    Calculate a covariance matrix using the specified method.

    Parameters
    ----------
    prices : pd.Series, pd.DataFrame, or np.ndarray
        Price series or DataFrame.
    method : str, default 'sample'
        The covariance method to use:
        - "sample": Standard sample covariance.
        - "ema" or "ewma": Exponentially-weighted covariance.
    returns_data : bool, default False
        If True, input is treated as returns.
    **kwargs
        Additional arguments passed to the specific covariance function 
        (e.g., span=180 for "ema", or frequency=252).

    Returns
    -------
    pd.DataFrame
        Covariance matrix.
    """
    kwargs["returns_data"] = returns_data
    
    if method == "sample":
        return sample_covariance(prices, **kwargs)
    elif method in ["ema", "ewma", "exponential"]:
        return exponential_covariance(prices, **kwargs)
    else:
        raise ValueError(f"Unknown covariance method: {method}")
