"""
Return estimation utilities.

Provides utility functions for calculating period-on-period returns from price data,
and for estimating expected returns using various methods: historical mean (CAGR),
exponentially-weighted mean, and CAPM.

These are helper functions intended for use with the pandas accessor (df.finance.*) 
or called directly. By convention, all return outputs are annualized unless otherwise specified.

Functions
---------
- time_series_returns: Convert prices to simple or log returns
- historical_mean: Annualized mean return (geometric CAGR or arithmetic)
- ewma_return: Annualized exponentially-weighted moving average return
- capm_return: CAPM-based expected returns estimate
"""

import warnings
import numpy as np
import pandas as pd


def to_returns(prices, log_returns=False):
    """
    Compute period-on-period returns from prices (DataFrame or Series).

    Accepts a single price Series (one asset) or a DataFrame (multiple assets).
    Same formula in both cases.

    Formula (simple): r_t = (P_t - P_{t-1}) / P_{t-1}
    Formula (log):    r_t = ln(P_t / P_{t-1}) = ln(1 + simple_r_t)

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price series (one asset) or DataFrame with dates as index and tickers as columns.
    log_returns : bool, default False
        If True, compute log returns instead of simple returns.

    Returns
    -------
    pd.Series or pd.DataFrame
        Period returns (same type and index as input, first row dropped).
    """
    if isinstance(prices, pd.DataFrame):
        if log_returns:
            out = np.log(1 + prices.pct_change()).dropna(how="all")
        else:
            out = prices.pct_change().dropna(how="all")
        return out
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    if log_returns:
        return np.log(1 + prices.pct_change()).dropna()
    return prices.pct_change().dropna()


def historical_mean(
    prices, returns_data=False, compounding=True, frequency=252, log_returns=False
):
    """
    Calculate annualized mean historical return.

    Formula (geometric, CAGR): (1 + R_geo)^frequency - 1  where (1+R_geo)^T = Π(1+r_t)
    Formula (arithmetic):      μ * frequency  where μ = mean of period returns

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices or returns data.
    returns_data : bool, default False
        If True, prices is actually returns data.
    compounding : bool, default True
        If True, use geometric mean (CAGR). If False, use arithmetic mean.
    frequency : int, default 252
        Trading periods per year.
    log_returns : bool, default False
        Whether data uses log returns.

    Returns
    -------
    pd.Series
        Annualized expected returns for each asset.
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices not in a dataframe, converting", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = to_returns(prices, log_returns)

    if compounding:
        return (1 + returns).prod() ** (frequency / returns.count()) - 1
    else:
        return returns.mean() * frequency


def ewma_return(
    prices,
    returns_data=False,
    compounding=True,
    span=500,
    frequency=252,
    log_returns=False,
):
    """
    Calculate exponentially-weighted moving average of historical returns.

    Formula: apply EWM (exponential weights) to returns, then annualize:
    geometric (1 + EWM_r)^frequency - 1  or  arithmetic EWM_r * frequency.
    Gives higher weight to recent data.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices or returns data.
    returns_data : bool, default False
        If True, prices is actually returns data.
    compounding : bool, default True
        If True, use geometric mean. If False, use arithmetic mean.
    span : int, default 500
        Span for exponential weighting (days).
    frequency : int, default 252
        Trading periods per year.
    log_returns : bool, default False
        Whether data uses log returns.

    Returns
    -------
    pd.Series
        Annualized exponentially-weighted expected returns.
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices not in a dataframe, converting", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices
    else:
        returns = to_returns(prices, log_returns)

    if compounding:
        return (1 + returns.ewm(span=span).mean().iloc[-1]) ** frequency - 1
    else:
        return returns.ewm(span=span).mean().iloc[-1] * frequency


def capm_return(
    prices,
    market_prices=None,
    returns_data=False,
    risk_free_rate=0.0,
    compounding=True,
    frequency=252,
    log_returns=False,
):
    r"""
    Estimate returns using CAPM.

    Formula: E[R_i] = R_f + β_i (E[R_m] - R_f),  with β_i = Cov(R_i, R_m) / Var(R_m).

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices or returns data.
    market_prices : pd.Series or pd.DataFrame, optional
        Market benchmark prices. If None, uses equal-weighted market proxy.
    returns_data : bool, default False
        If True, prices is actually returns data.
    risk_free_rate : float, default 0.0
        Risk-free rate (should match frequency).
    compounding : bool, default True
        If True, use geometric mean.
    frequency : int, default 252
        Trading periods per year.
    log_returns : bool, default False
        Whether data uses log returns.

    Returns
    -------
    pd.Series
        Annualized CAPM return estimates for each asset.
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices not in a dataframe, converting", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if returns_data:
        returns = prices.copy()
        market_returns = None
        if market_prices is not None:
            market_returns = market_prices
    else:
        returns = to_returns(prices, log_returns)
        market_returns = None
        if market_prices is not None:
            if not isinstance(market_prices, pd.DataFrame):
                market_prices = pd.DataFrame(market_prices)
            market_returns = to_returns(market_prices, log_returns)

    # Use equal-weighted proxy if no market provided
    if market_returns is None:
        returns["mkt"] = returns.mean(axis=1)
    else:
        market_returns = market_returns.squeeze()
        returns["mkt"] = market_returns

    # Calculate covariance
    cov = returns.cov()
    betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    betas = betas.drop("mkt")

    # Market return
    if compounding:
        mkt_mean_ret = (1 + returns["mkt"]).prod() ** (
            frequency / returns["mkt"].count()
        ) - 1
    else:
        mkt_mean_ret = returns["mkt"].mean() * frequency

    # CAPM formula
    return risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)
