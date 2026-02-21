"""
Risk metrics and performance analysis.

This module contains risk and performance metrics for pandas Series/DataFrames,
using explicit mathematical naming conventions.
"""

import numpy as np
import pandas as pd


def annualized_return(series, frequency=252):
    """
    Calculate annualized return from daily returns.

    Formula: R_annual = (1 + μ)^frequency - 1  where μ = mean of period returns.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Return series, or DataFrame of returns.
    frequency : int, default 252
        Trading periods per year.

    Returns
    -------
    float or pd.Series
        Annualized return(s).
    """
    return (1 + series.mean()) ** frequency - 1


def annualized_volatility(series, annualized=True, frequency=252):
    """
    Calculate volatility (standard deviation of returns).

    Formula: σ_period = std(r);  σ_annual = σ_period * √frequency.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Return series or DataFrame of returns.
    annualized : bool, default True
        If True, annualize the volatility.
    frequency : int, default 252
        Trading periods per year.

    Returns
    -------
    float or pd.Series
        Volatility.
    """
    vol = series.std()
    if annualized:
        vol = vol * np.sqrt(frequency)
    return vol


def sharpe_ratio(series, risk_free_rate=0.03, frequency=252):
    """
    Calculate Sharpe Ratio (excess return per unit of risk).

    Formula: SR = (E[R] - R_f) / σ  (using annualized return and volatility).

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Return series or DataFrame of returns.
    risk_free_rate : float, default 0.03
        Annual risk-free rate.
    frequency : int, default 252
        Trading periods per year.

    Returns
    -------
    float or pd.Series
        Sharpe ratio(s).
    """
    excess = series.mean() * frequency - risk_free_rate
    vol = series.std() * np.sqrt(frequency)
    if isinstance(series, pd.DataFrame):
        return (excess / vol).replace([np.inf, -np.inf], 0).fillna(0)
    return excess / vol if vol > 0 else 0


def sortino_ratio(series, risk_free_rate=0.03, target_return=0, frequency=252):
    """
    Calculate Sortino Ratio (excess return per unit of downside risk).

    Formula: Sortino = (E[R] - R_f) / σ_down  (using annualized metrics).
    σ_down = std(min(0, r - target)) * √frequency.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Return series or DataFrame of returns.
    risk_free_rate : float, default 0.03
        Annual risk-free rate.
    target_return : float, default 0
        Minimum acceptable return threshold per period for downside deviation.
    frequency : int, default 252
        Trading periods per year.

    Returns
    -------
    float or pd.Series
        Sortino ratio(s).
    """
    excess = series.mean() * frequency - risk_free_rate
    
    # Calculate downside deviation
    downside_returns = series.copy()
    if isinstance(series, pd.DataFrame):
        downside_returns[downside_returns > target_return] = 0
    else:
        downside_returns = downside_returns.clip(upper=target_return)
        
    downside_vol = downside_returns.std() * np.sqrt(frequency)
    
    if isinstance(series, pd.DataFrame):
        return (excess / downside_vol).replace([np.inf, -np.inf], 0).fillna(0)
    return excess / downside_vol if downside_vol > 0 else 0


def cumulative_returns(obj, log_returns=False):
    """
    Cumulative (compounded) returns from a return series or DataFrame.

    Formula (simple): Cum_t = Π_{s≤t} (1 + r_s).  Formula (log): Cum_t = exp(Σ_{s≤t} r_s).

    Parameters
    ----------
    obj : pd.Series or pd.DataFrame
        Return series or DataFrame of returns.
    log_returns : bool, default False
        If True, data is in log-return form.

    Returns
    -------
    pd.Series or pd.DataFrame
        Cumulative return index (1 = start).
    """
    if isinstance(obj, pd.DataFrame):
        if log_returns:
            return np.exp(obj.cumsum())
        return (1 + obj).cumprod()
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj)
    if log_returns:
        return np.exp(obj.cumsum())
    return (1 + obj).cumprod()


def drawdown(obj, from_returns=False):
    """
    Drawdown series from prices or cumulative returns (Series or DataFrame).

    Formula: DD_t = (Cum_t - Peak_t) / Peak_t  where Peak_t = max_{s≤t} Cum_s.

    Parameters
    ----------
    obj : pd.Series or pd.DataFrame
        Price series/DataFrame or cumulative return index (if from_returns=True).
    from_returns : bool, default False
        If True, obj is a cumulative return index.

    Returns
    -------
    pd.Series or pd.DataFrame
        Drawdown at each date (0 at peak, negative below peak).
    """
    if isinstance(obj, pd.DataFrame):
        if from_returns:
            level = obj
        else:
            level = obj / obj.iloc[0]
        running_max = level.cummax()
        return (level - running_max) / running_max
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj)
    if from_returns:
        level = obj
    else:
        level = obj / obj.iloc[0]
    running_max = level.cummax()
    return (level - running_max) / running_max


def max_drawdown(obj, from_returns=False):
    """
    Maximum drawdown from prices or cumulative returns (Series or DataFrame).

    Formula: MDD = min_t DD_t  (most negative drawdown over time).

    Parameters
    ----------
    obj : pd.Series or pd.DataFrame
        Price series/DataFrame or cumulative return index (if from_returns=True).
    from_returns : bool, default False
        If True, obj is a cumulative return index.

    Returns
    -------
    float or pd.Series
        Maximum drawdown (negative). Per-column for DataFrame.
    """
    dd = drawdown(obj, from_returns=from_returns)
    return dd.min()


def rolling_volatility(obj, window=20, annualized=True, frequency=252):
    """
    Rolling volatility of returns (Series or DataFrame).

    Formula: σ_window = std(r over window);  σ_annual = σ_window × √frequency.

    Parameters
    ----------
    obj : pd.Series or pd.DataFrame
        Return series or DataFrame.
    window : int, default 20
        Rolling window size.
    annualized : bool, default True
        If True, annualize volatility.
    frequency : int, default 252
        Periods per year.

    Returns
    -------
    pd.Series or pd.DataFrame
        Rolling volatility.
    """
    vol = obj.rolling(window).std()
    if annualized:
        vol = vol * np.sqrt(frequency)
    return vol