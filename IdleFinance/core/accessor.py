"""
Pandas accessor for IdleFinance.

Enables calling financial methods directly on pandas Series and DataFrame objects.

Examples
--------
>>> df_prices = pd.DataFrame(...)  # price data
>>> returns = df_prices.finance.returns()
>>> cov = df_prices.finance.covariance()
>>> post_ret, post_cov = df_prices.finance.black_litterman(prior_returns=..., views=...)
>>> weights = df_prices.finance.black_litterman_weights(post_ret, post_cov)
"""

import numpy as np
import pandas as pd
from ._types import (
    Any, Optional, Tuple, List, Dict, Union,
    OmegaMethod, TauMethod, CovarianceMethod, ReturnsMethod, ObjectiveType, 
    ArrayLike, MatrixLike, PriceData, BLFullOutput, Weights,
    NumericOutput, FinancialOutput, PortfolioBounds, PortfolioConstraints,
    ViewData, BLResult, BLSingleResult
)

from ..utils import (
    to_returns,
    historical_mean,
    ewma_return,
    capm_return,
)
from ..models import black_litterman, risk_metrics, covariances


@pd.api.extensions.register_dataframe_accessor("finance")
class DataFrameFinanceAccessor:
    """
    Financial analysis accessor for pandas DataFrames.

    Accessed via `df.finance.*` after importing IdleFinance.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def returns(self, log_returns: bool = False, returns_data: bool = False) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Formula (simple): r_t = (P_t - P_{t-1}) / P_{t-1};  (log): r_t = ln(P_t / P_{t-1}).

        Parameters
        ----------
        log_returns : bool, default False
            If True, compute log returns.
        returns_data : bool, default False
            If True, assume input is already returns and return as-is (no conversion).

        Returns
        -------
        pd.DataFrame
            Daily returns.
        """
        if returns_data:
            return self._obj
        return to_returns(self._obj, log_returns=log_returns)

    def correlation(self, returns_data: bool = False, **kwargs: Any) -> pd.DataFrame:
        """
        Correlation matrix of returns.

        Formula: ρ_ij = Cov(R_i, R_j) / (σ_i σ_j).

        Parameters
        ----------
        returns_data : bool, default False
            If True, assume input is returns; otherwise convert from prices.
        **kwargs
            Passed to pd.DataFrame.corr().

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        ret = self.returns(returns_data=returns_data)
        return ret.corr(**kwargs)

    def cumulative_returns(self, log_returns: bool = False, returns_data: bool = False) -> pd.DataFrame:
        """
        Cumulative (compounded) return index for each column.

        Formula (simple): Cum_t = Π_{s≤t}(1 + r_s);  (log): Cum_t = exp(Σ_{s≤t} r_s).

        Parameters
        ----------
        log_returns : bool, default False
            If True, treat/compute log returns.
        returns_data : bool, default False
            If True, assume input is returns; otherwise convert from prices.

        Returns
        -------
        pd.DataFrame
            Cumulative return index (1 = start).
        """
        ret = self.returns(log_returns=log_returns, returns_data=returns_data)
        return risk_metrics.cumulative_returns(ret, log_returns=log_returns)

    def drawdown(self, returns_data: bool = False) -> pd.DataFrame:
        """
        Drawdown series for each column (from prices or cumulative returns).

        Formula: DD_t = (Cum_t - Peak_t) / Peak_t,  Peak_t = max_{s≤t} Cum_s.

        Parameters
        ----------
        returns_data : bool, default False
            If True, assume input is cumulative return index; otherwise prices.

        Returns
        -------
        pd.DataFrame
            Drawdown at each date (0 at peak, negative below).
        """
        return risk_metrics.drawdown(self._obj, from_returns=returns_data)

    def max_drawdown(self, returns_data: bool = False) -> pd.Series:
        """
        Maximum drawdown per column.

        Formula: MDD = min_t DD_t  (most negative drawdown over time).

        Parameters
        ----------
        returns_data : bool, default False
            If True, assume input is cumulative return index; otherwise prices.

        Returns
        -------
        pd.Series
            Maximum drawdown (negative) per asset.
        """
        return risk_metrics.max_drawdown(self._obj, from_returns=returns_data)

    def rolling_volatility(self, window: int = 20, annualized: bool = True, frequency: int = 252, returns_data: bool = False) -> pd.DataFrame:
        """
        Rolling volatility for each column.

        Formula: σ_window = std(r over window);  σ_annual = σ_window × √frequency.

        Parameters
        ----------
        window : int, default 20
            Rolling window size.
        annualized : bool, default True
            If True, annualize volatility.
        frequency : int, default 252
            Periods per year for annualization.
        returns_data : bool, default False
            If True, assume input is returns; otherwise convert from prices.

        Returns
        -------
        pd.DataFrame
            Rolling volatility.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.rolling_volatility(ret, window=window, annualized=annualized, frequency=frequency)

    def covariance(self, returns_data: bool = False, method: CovarianceMethod = "sample", **kwargs: Any) -> pd.DataFrame:
        """
        Calculate covariance matrix from prices or returns using specific method.

        Formula: Σ_ij = Cov(R_i, R_j)  (sample covariance of period returns).
        """
        return covariances.covariance(self._obj, method=method, returns_data=returns_data, **kwargs)

    def expected_returns(self, method: ReturnsMethod = "mean_historical", returns_data: bool = False, **kwargs: Any) -> pd.Series:
        """
        Estimate expected returns using various methods.

        Parameters
        ----------
        method : str, default "mean_historical"
            Method to use: "mean_historical", "ema_historical", or "capm"
        returns_data : bool, default False
            If True, treat input as returns; otherwise as prices.
        **kwargs
            Passed to the underlying method.

        Returns
        -------
        pd.Series
            Expected returns for each asset.
        """
        kwargs.setdefault("returns_data", returns_data)
        if method == "mean_historical":
            return historical_mean(self._obj, **kwargs)
        elif method == "ema_historical":
            return ewma_return(self._obj, **kwargs)
        elif method == "capm":
            return capm_return(self._obj, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def annualized_return(self, frequency: int = 252, returns_data: bool = False) -> pd.Series:
        """
        Annualized return per column.

        Formula: R_annual = (1 + μ)^frequency - 1. If returns_data=False, converts prices to returns first.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.annualized_return(ret, frequency=frequency)

    def volatility(self, annualized: bool = True, frequency: int = 252, returns_data: bool = False) -> pd.Series:
        """
        Volatility per column.

        Formula: σ_annual = σ_period × √frequency. If returns_data=False, converts prices to returns first.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.annualized_volatility(ret, annualized=annualized, frequency=frequency)

    def sharpe_ratio(self, risk_free_rate: float = 0.03, frequency: int = 252, returns_data: bool = False) -> pd.Series:
        """
        Sharpe ratio per column.

        Formula: SR = (E[R] - R_f) / σ  (annualized). If returns_data=False, converts prices to returns first.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.sharpe_ratio(ret, risk_free_rate=risk_free_rate, frequency=frequency)

    def sortino_ratio(self, risk_free_rate: float = 0.03, target_return: float = 0, frequency: int = 252, returns_data: bool = False) -> Weights:
        """
        Sortino ratio per column.

        Formula: Sortino = (E[R] - R_f) / σ_down (annualized). If returns_data=False, converts prices to returns first.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.sortino_ratio(
            ret, risk_free_rate=risk_free_rate, target_return=target_return, frequency=frequency
        )


@pd.api.extensions.register_series_accessor("finance")
class SeriesFinanceAccessor:
    """
    Financial analysis accessor for pandas Series.

    Accessed via `series.finance.*` after importing IdleFinance.
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def black_litterman(
        self, 
        prior_return: Optional[float] = None, 
        view: Optional[float] = None, 
        view_confidence: float = 0.5, 
        tau: float = 0.05, 
        risk_aversion: float = 1.0
    ) -> BLSingleResult:
        """
        Apply Black-Litterman optimization for a single asset with a single view.

        Returns
        -------
        (float, float, float)
            Posterior Expected Return, Posterior Variance, Optimal Weight.
        """
        return black_litterman.black_litterman_single_asset(
            self._obj, prior_return, view, view_confidence, tau, risk_aversion
        )

    def returns(self, log_returns: bool = False, returns_data: bool = False) -> pd.Series:
        """
        Calculate returns from price series.

        Formula (simple): r_t = (P_t - P_{t-1}) / P_{t-1};  (log): r_t = ln(P_t / P_{t-1}).

        Parameters
        ----------
        log_returns : bool, default False
            If True, compute log returns.
        returns_data : bool, default False
            If True, assume input is already returns.

        Returns
        -------
        pd.Series
            Returns series.
        """
        if returns_data:
            return self._obj
        return to_returns(self._obj, log_returns=log_returns)

    def cumulative_returns(self, log_returns: bool = False, returns_data: bool = False) -> pd.Series:
        """
        Cumulative (compounded) return index.

        Formula (simple): Cum_t = Π_{s≤t}(1 + r_s);  (log): Cum_t = exp(Σ_{s≤t} r_s).

        Parameters
        ----------
        log_returns : bool, default False
            If True, treat/compute log returns.
        returns_data : bool, default False
            If True, assume input is returns.

        Returns
        -------
        pd.Series
            Cumulative return index (1 = start).
        """
        ret = self.returns(log_returns=log_returns, returns_data=returns_data)
        return risk_metrics.cumulative_returns(ret, log_returns=log_returns)

    def drawdown(self, from_returns: bool = False) -> pd.Series:
        """
        Drawdown series.

        Formula: DD_t = (Cum_t - Peak_t) / Peak_t,  Peak_t = max_{s≤t} Cum_s.

        Parameters
        ----------
        from_returns : bool, default False
            If True, assume input is cumulative return index; otherwise prices.

        Returns
        -------
        pd.Series
            Drawdown (0 at peak, negative below).
        """
        return risk_metrics.drawdown(self._obj, from_returns=from_returns)

    def max_drawdown(self, from_returns: bool = False) -> float:
        """
        Maximum drawdown over the timeframe.

        Formula: MDD = min_t DD_t.

        Parameters
        ----------
        from_returns : bool, default False
            If True, assume input is cumulative return index; otherwise prices.

        Returns
        -------
        float
            Maximum drawdown (negative).
        """
        return risk_metrics.max_drawdown(self._obj, from_returns=from_returns)

    def annualized_return(self, frequency: int = 252) -> float:
        """
        Annualized return.

        Formula: R_annual = (1 + μ)^frequency - 1.

        Parameters
        ----------
        frequency : int, default 252
            Trading periods per year.

        Returns
        -------
        float
            Annualized return.
        """
        return risk_metrics.annualized_return(self._obj, frequency)

    def volatility(self, annualized: bool = True, frequency: int = 252) -> float:
        """
        Volatility (standard deviation of returns).

        Formula: σ_annual = σ_period × √frequency.

        Parameters
        ----------
        annualized : bool, default True
            If True, annualize the result.
        frequency : int, default 252
            Trading periods per year.

        Returns
        -------
        float
            Volatility.
        """
        return risk_metrics.annualized_volatility(self._obj, annualized, frequency)

    def sharpe_ratio(self, risk_free_rate: float = 0.03, frequency: int = 252) -> float:
        """
        Sharpe ratio (excess return per unit of risk).

        Formula: SR = (E[R] - R_f) / σ (annualized).

        Parameters
        ----------
        risk_free_rate : float, default 0.03
            Annual risk-free rate.
        frequency : int, default 252
            Trading periods per year.

        Returns
        -------
        float
            Sharpe ratio.
        """
        return risk_metrics.sharpe_ratio(self._obj, risk_free_rate, frequency)

    def sortino_ratio(self, risk_free_rate: float = 0.03, target_return: float = 0, frequency: int = 252) -> float:
        """
        Sortino ratio (excess return per unit of downside risk).

        Formula: Sortino = (E[R] - R_f) / σ_down (annualized).

        Parameters
        ----------
        risk_free_rate : float, default 0.03
            Annual risk-free rate.
        target_return : float, default 0
            Minimum acceptable return threshold.
        frequency : int, default 252
            Trading periods per year.

        Returns
        -------
        float
            Sortino ratio.
        """
        return risk_metrics.sortino_ratio(self._obj, risk_free_rate, target_return, frequency)
