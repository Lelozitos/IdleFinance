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

from functools import wraps

import numpy as np
import pandas as pd
from ._types import (
    Any, Optional, Tuple, List, Dict, Union,
    OmegaMethod, TauMethod, CovarianceMethod, ReturnsMethod, ObjectiveType,
    ArrayLike, MatrixLike, PriceData, BLFullOutput, Weights,
    NumericOutput, FinancialOutput, PortfolioBounds, PortfolioConstraints,
    ViewData, BLResult, BLSingleResult, VectorOutput
)

from ..utils import (
    to_returns,
    historical_mean,
    ewma_return,
    capm_return,
)
from ..models import (
    black_litterman, risk_metrics, covariances, efficient_frontier,
    fixed_income, options, stochastic, equities,
)


def _passthrough(cls):
    """
    Attach every function listed in `cls._passthrough_map` to `cls` as a
    static, doc-preserving pass-through method (same name, same signature).

    Lets modules that don't operate on the DataFrame/Series itself (options,
    fixed_income, stochastic, equities, standalone efficient-frontier/
    covariance helpers) be called straight off `.finance`, e.g.
    `df.finance.black_scholes_call(S=100, K=105, T=1, r=0.03, sigma=0.2)`.
    """
    for module, names in cls._passthrough_map.items():
        for name in names:
            func = getattr(module, name)

            @wraps(func)
            def _wrapper(*args, _f=func, **kwargs):
                return _f(*args, **kwargs)

            setattr(cls, name, staticmethod(_wrapper))
    return cls


_PASSTHROUGH_MAP = {
    fixed_income: [
        "bond_price", "bond_ytm", "bond_duration", "bond_convexity",
        "credit_spread", "forward_rate", "bond_macaulay_duration",
        "bond_modified_duration", "macaulay_duration_from_cashflows",
    ],
    options: [
        "black_scholes_call", "black_scholes_put",
        "calculate_option_price", "delta", "gamma", "vega", "theta", "rho",
        "vanna", "volga", "charm", "calculate_greeks", "implied_volatility",
        "batch_option_pricing", "batch_greeks", "batch_implied_volatility",
        "scenario_pnl_grid", "stress_test", "aggregate_portfolio_greeks",
        "run_benchmark",
    ],
    stochastic: [
        "geometric_brownian_motion", "mean_reverting_process",
        "jump_diffusion_process", "monte_carlo",
    ],
    equities: [
        "cost_of_equity_capm", "cost_of_equity_build_up", "cost_of_debt", "wacc",
        "gordon_growth_model", "h_model", "two_stage_ddm", "three_stage_ddm",
        "dcf_valuation", "dcf_sensitivity", "reverse_dcf", "earnings_power_value",
        "residual_income_model", "abnormal_earnings_growth", "pe_implied_price",
        "ev_ebitda_implied_price", "price_to_book_implied",
    ],
    covariances: ["denoise_covariance", "sample_covariance", "exponential_covariance"],
}


class _LibraryMixin:
    """
    Gives `.finance` full, flat access to every standalone IdleFinance model
    (fixed income, options, stochastic processes, equity valuation) — not
    just the DataFrame/Series-native methods below. Same names, same
    signatures, same docstrings as the underlying `IdleFinance.models.*`
    functions; nothing here reads `self._obj`.
    """

    _passthrough_map = _PASSTHROUGH_MAP


_passthrough(_LibraryMixin)


@pd.api.extensions.register_dataframe_accessor("finance")
class DataFrameFinanceAccessor(_LibraryMixin):
    """
    Financial analysis accessor for pandas DataFrames.

    Accessed via `df.finance.*` after importing IdleFinance. Covers the
    full library: returns/risk/covariance/optimization on the DataFrame's
    own price data, plus flat pass-through access to fixed income, options,
    stochastic simulation, and equity valuation (e.g.
    `df.finance.black_scholes_call(...)`, `df.finance.bond_price(...)`).
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

    def risk_aversion(self, risk_free_rate: float = 0.02, frequency: int = 252) -> float:
        """
        Market-implied risk aversion parameter (lambda).
        
        Formula: λ = (E[R_m] - R_f) / σ_m²
        """
        return risk_metrics.market_risk_aversion(self._obj, risk_free_rate=risk_free_rate, frequency=frequency)

    def market_prior_returns(self, market_weights: ArrayLike, risk_aversion: float, cov_method: str = "sample", **kwargs) -> VectorOutput:
        """
        Calculate the implied equilibrium returns (Π) given market weights.
        
        Formula: Π = λ * Σ * w_mkt
        """
        cov_matrix = self.covariance(method=cov_method, **kwargs)
        return black_litterman.market_prior_returns(market_weights, risk_aversion, cov_matrix)

    def sortino_ratio(self, risk_free_rate: float = 0.03, target_return: float = 0, frequency: int = 252, returns_data: bool = False) -> Weights:
        """
        Sortino ratio per column.

        Formula: Sortino = (E[R] - R_f) / σ_down (annualized). If returns_data=False, converts prices to returns first.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.sortino_ratio(
            ret, risk_free_rate=risk_free_rate, target_return=target_return, frequency=frequency
        )

    def black_litterman(
        self,
        prior_returns: Optional[ArrayLike] = None,
        market_weights: Optional[ArrayLike] = None,
        views: Optional[ViewData] = None,
        view_confidences: Optional[ArrayLike] = None,
        relative_views: bool = False,
        tau: Optional[float] = None,
        tau_method: TauMethod = "default",
        omega_method: OmegaMethod = "idzorek",
        risk_aversion: float = 1.0,
        cov_matrix: Optional[pd.DataFrame] = None,
        cov_method: CovarianceMethod = "sample",
        bounds: PortfolioBounds = None,
        objective: ObjectiveType = "utility",
        benchmark_weights: Optional[ArrayLike] = None,
        custom_constraints: PortfolioConstraints = None,
        target_sum: Optional[float] = 1.0,
        risk_free_rate: float = 0.0,
    ) -> BLFullOutput:
        """
        Perform Black-Litterman optimization to obtain posterior returns, covariance, and weights.

        Parameters
        ----------
        prior_returns : ArrayLike, optional
            Explicit prior equilibrium returns (Π). Mutually exclusive with
            *market_weights* (market_weights takes precedence).
        market_weights : ArrayLike, optional
            Market-cap weights used to auto-compute the prior via reverse
            optimisation (Π = λ · Σ · w_mkt).
        views : ViewData, optional
            Investor views. Absolute: ``{'AAPL': 0.15}``.
            Relative (with ``relative_views=True``): ``{('AAPL', 'MSFT'): 0.03}``.
        view_confidences : ArrayLike, optional
            Confidence levels [0, 1] for each view.
        relative_views : bool, default False
            When True, view keys are (long, short) ticker pairs.
        tau : float, optional
            Uncertainty scaling factor τ. If None, derived via *tau_method*.
        tau_method : TauMethod, default 'default'
            How to compute τ when *tau* is None.
        omega_method : OmegaMethod, default 'idzorek'
            Method to build Ω: ``'idzorek'`` or ``'proportional'``.
        risk_aversion : float, default 1.0
            Risk-aversion coefficient λ.
        cov_matrix : pd.DataFrame, optional
            Covariance matrix. If None, computed from the DataFrame.
        cov_method : CovarianceMethod, default 'sample'
            Method used to compute the covariance when *cov_matrix* is None.
        bounds : PortfolioBounds, optional
            Weight bounds per asset.
        objective : ObjectiveType, default 'utility'
            ``'utility'`` or ``'tracking_error'``.
        benchmark_weights : ArrayLike, optional
            Benchmark weights for tracking-error objective.
        custom_constraints : PortfolioConstraints, optional
            Extra scipy constraint dicts.
        target_sum : float or None, default 1.0
            Target risky-weight sum (``None`` = unconstrained).
        risk_free_rate : float, default 0.0
            Risk-free rate; cash balance earns this rate.

        Returns
        -------
        BLFullOutput
            ``(posterior_returns, posterior_covariance, optimal_weights)``.
        """
        if cov_matrix is None:
            cov_matrix = self.covariance(method=cov_method)

        post_ret, post_cov = black_litterman.bl_posterior_distribution(
            cov_matrix,
            prior_returns=prior_returns,
            market_weights=market_weights,
            views=views,
            view_confidences=view_confidences,
            relative_views=relative_views,
            tau_val=tau,
            tau_method=tau_method,
            omega_method=omega_method,
            risk_aversion=risk_aversion,
            prices=self._obj,
        )

        weights = black_litterman.compute_bl_weights(
            post_ret,
            post_cov,
            risk_aversion=risk_aversion,
            bounds=bounds,
            objective=objective,
            benchmark_weights=benchmark_weights,
            custom_constraints=custom_constraints,
            target_sum=target_sum,
            risk_free_rate=risk_free_rate,
        )

        return post_ret, post_cov, weights

    def black_litterman_weights(
        self, 
        posterior_returns: ArrayLike, 
        posterior_cov: MatrixLike, 
        risk_aversion: float = 1.0,
        bounds: PortfolioBounds = None,
        objective: ObjectiveType = "utility",
        benchmark_weights: Optional[ArrayLike] = None,
        custom_constraints: PortfolioConstraints = None,
        target_sum: Optional[float] = 1.0,
        risk_free_rate: float = 0.0,
    ) -> Weights:
        """
        Compute optimal portfolio weights using the posterior distribution from Black-Litterman.
        """
        return black_litterman.compute_bl_weights(
            posterior_returns,
            posterior_cov,
            risk_aversion=risk_aversion,
            bounds=bounds,
            objective=objective,
            benchmark_weights=benchmark_weights,
            custom_constraints=custom_constraints,
            target_sum=target_sum,
            risk_free_rate=risk_free_rate,
        )

    def value_at_risk(
        self,
        confidence: float = 0.95,
        method: str = "historical",
        weights: Optional[ArrayLike] = None,
        frequency: int = 1,
        returns_data: bool = False,
    ) -> NumericOutput:
        """
        Value at Risk (VaR) per column, or portfolio VaR if `weights` given.
        See `risk_metrics.value_at_risk` for method details.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.value_at_risk(ret, confidence=confidence, method=method, weights=weights, frequency=frequency)

    def expected_shortfall(
        self,
        confidence: float = 0.95,
        method: str = "historical",
        weights: Optional[ArrayLike] = None,
        frequency: int = 1,
        returns_data: bool = False,
    ) -> NumericOutput:
        """
        Expected Shortfall / CVaR per column, or portfolio ES if `weights` given.
        See `risk_metrics.expected_shortfall` for method details.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.expected_shortfall(ret, confidence=confidence, method=method, weights=weights, frequency=frequency)

    def probabilistic_sharpe_ratio(
        self,
        benchmark_sr: float = 0.0,
        risk_free_rate: float = 0.0,
        frequency: int = 252,
        returns_data: bool = False,
    ) -> NumericOutput:
        """Probability the true Sharpe Ratio exceeds `benchmark_sr`. See `risk_metrics.probabilistic_sharpe_ratio`."""
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.probabilistic_sharpe_ratio(ret, benchmark_sr=benchmark_sr, risk_free_rate=risk_free_rate, frequency=frequency)

    def marginal_contribution_to_risk(
        self,
        weights: ArrayLike,
        cov_matrix: Optional[MatrixLike] = None,
        cov_method: CovarianceMethod = "sample",
        **kwargs: Any,
    ) -> pd.Series:
        """MCTR per asset. Computes the covariance from `self._obj` unless `cov_matrix` is given."""
        if cov_matrix is None:
            cov_matrix = self.covariance(method=cov_method, **kwargs)
        return risk_metrics.marginal_contribution_to_risk(weights, cov_matrix)

    def component_var(
        self,
        weights: ArrayLike,
        confidence: float = 0.95,
        method: str = "historical",
        returns_data: bool = False,
    ) -> pd.DataFrame:
        """Per-asset decomposition of portfolio VaR. See `risk_metrics.component_var`."""
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.component_var(weights, ret, confidence=confidence, method=method)

    def min_variance(
        self,
        cov_matrix: Optional[MatrixLike] = None,
        cov_method: CovarianceMethod = "sample",
        bounds: PortfolioBounds = None,
        target_sum: float = 1.0,
    ) -> np.ndarray:
        """Global Minimum Variance Portfolio weights. Computes covariance from `self._obj` unless `cov_matrix` is given."""
        if cov_matrix is None:
            cov_matrix = self.covariance(method=cov_method)
        return efficient_frontier.min_variance(cov_matrix, bounds=bounds, target_sum=target_sum)

    def max_sharpe(
        self,
        expected_returns: Optional[ArrayLike] = None,
        cov_matrix: Optional[MatrixLike] = None,
        returns_method: ReturnsMethod = "mean_historical",
        cov_method: CovarianceMethod = "sample",
        risk_free_rate: float = 0.0,
        bounds: PortfolioBounds = None,
        target_sum: float = 1.0,
    ) -> np.ndarray:
        """Maximum Sharpe (tangency) portfolio weights, auto-deriving inputs from `self._obj` if not given."""
        if cov_matrix is None:
            cov_matrix = self.covariance(method=cov_method)
        if expected_returns is None:
            expected_returns = self.expected_returns(method=returns_method)
        return efficient_frontier.max_sharpe(expected_returns, cov_matrix, risk_free_rate=risk_free_rate, bounds=bounds, target_sum=target_sum)

    def efficient_return(
        self,
        target_return: float,
        expected_returns: Optional[ArrayLike] = None,
        cov_matrix: Optional[MatrixLike] = None,
        returns_method: ReturnsMethod = "mean_historical",
        cov_method: CovarianceMethod = "sample",
        bounds: PortfolioBounds = None,
        target_sum: float = 1.0,
    ) -> np.ndarray:
        """Minimum-variance portfolio weights for a given `target_return`, auto-deriving inputs from `self._obj` if not given."""
        if cov_matrix is None:
            cov_matrix = self.covariance(method=cov_method)
        if expected_returns is None:
            expected_returns = self.expected_returns(method=returns_method)
        return efficient_frontier.efficient_return(expected_returns, cov_matrix, target_return, bounds=bounds, target_sum=target_sum)

    def efficient_frontier(
        self,
        expected_returns: Optional[ArrayLike] = None,
        cov_matrix: Optional[MatrixLike] = None,
        returns_method: ReturnsMethod = "mean_historical",
        cov_method: CovarianceMethod = "sample",
        n_points: int = 50,
        bounds: PortfolioBounds = None,
        target_sum: float = 1.0,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """Full efficient frontier (`['Return', 'Volatility', 'Sharpe']` per point), auto-deriving inputs from `self._obj` if not given."""
        if cov_matrix is None:
            cov_matrix = self.covariance(method=cov_method)
        if expected_returns is None:
            expected_returns = self.expected_returns(method=returns_method)
        return efficient_frontier.efficient_frontier(
            expected_returns, cov_matrix, n_points=n_points, bounds=bounds,
            target_sum=target_sum, risk_free_rate=risk_free_rate,
        )

    def portfolio_performance(
        self,
        weights: ArrayLike,
        expected_returns: Optional[ArrayLike] = None,
        cov_matrix: Optional[MatrixLike] = None,
        returns_method: ReturnsMethod = "mean_historical",
        cov_method: CovarianceMethod = "sample",
        risk_free_rate: float = 0.0,
    ) -> tuple:
        """`(return, volatility, sharpe)` for given `weights`, auto-deriving inputs from `self._obj` if not given."""
        if cov_matrix is None:
            cov_matrix = self.covariance(method=cov_method)
        if expected_returns is None:
            expected_returns = self.expected_returns(method=returns_method)
        return efficient_frontier.portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate=risk_free_rate)


@pd.api.extensions.register_series_accessor("finance")
class SeriesFinanceAccessor(_LibraryMixin):
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
        tau_val: float = 0.05,
        risk_aversion: float = 1.0,
    ) -> BLSingleResult:
        """
        Apply Black-Litterman optimization for a single asset with a single view.

        Parameters
        ----------
        prior_return : float, optional
            Prior equilibrium return (Π). If None, uses the historical mean.
        view : float, optional
            Investor's view on expected return (Q). If None, returns prior
            with inflated uncertainty.
        view_confidence : float, default 0.5
            Confidence in the view [0, 1].
        tau_val : float, default 0.05
            Uncertainty scaling factor (τ).
        risk_aversion : float, default 1.0
            Risk-aversion coefficient (λ).

        Returns
        -------
        BLSingleResult
            ``(posterior_return, posterior_variance, optimal_weight)``.
        """
        return black_litterman.black_litterman_single_asset(
            self._obj,
            prior_return=prior_return,
            view=view,
            view_confidence=view_confidence,
            tau_val=tau_val,
            risk_aversion=risk_aversion,
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

    def rolling_volatility(self, window: int = 20, annualized: bool = True, frequency: int = 252, returns_data: bool = False) -> pd.Series:
        """
        Rolling volatility.

        Formula: σ_window = std(r over window);  σ_annual = σ_window × √frequency.
        """
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.rolling_volatility(ret, window=window, annualized=annualized, frequency=frequency)

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

    def risk_aversion(self, risk_free_rate: float = 0.02, frequency: int = 252) -> float:
        """
        Market-implied risk aversion parameter (lambda).
        
        Formula: λ = (E[R_m] - R_f) / σ_m²
        """
        return risk_metrics.market_risk_aversion(self._obj, risk_free_rate=risk_free_rate, frequency=frequency)

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

    def value_at_risk(
        self,
        confidence: float = 0.95,
        method: str = "historical",
        frequency: int = 1,
        returns_data: bool = False,
    ) -> float:
        """Value at Risk (VaR). See `risk_metrics.value_at_risk` for method details."""
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.value_at_risk(ret, confidence=confidence, method=method, frequency=frequency)

    def expected_shortfall(
        self,
        confidence: float = 0.95,
        method: str = "historical",
        frequency: int = 1,
        returns_data: bool = False,
    ) -> float:
        """Expected Shortfall / CVaR. See `risk_metrics.expected_shortfall` for method details."""
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.expected_shortfall(ret, confidence=confidence, method=method, frequency=frequency)

    def probabilistic_sharpe_ratio(
        self,
        benchmark_sr: float = 0.0,
        risk_free_rate: float = 0.0,
        frequency: int = 252,
        returns_data: bool = False,
    ) -> float:
        """Probability the true Sharpe Ratio exceeds `benchmark_sr`. See `risk_metrics.probabilistic_sharpe_ratio`."""
        ret = self.returns(returns_data=returns_data)
        return risk_metrics.probabilistic_sharpe_ratio(ret, benchmark_sr=benchmark_sr, risk_free_rate=risk_free_rate, frequency=frequency)