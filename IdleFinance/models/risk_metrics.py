"""
Risk metrics and performance analysis.

This module contains risk and performance metrics for pandas Series/DataFrames,
using explicit mathematical naming conventions.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from ..core._types import NumericOutput, FinancialOutput, PriceData, ArrayLike, Optional, Union, MatrixLike


def market_risk_aversion(
    market_prices: PriceData, 
    risk_free_rate: float = 0.02, 
    frequency: int = 252
) -> float:
    """
    Calculate the market-implied risk aversion parameter (lambda).
    
    This parameter represents the "Market Price of Risk" and is used to 
    reverse-engineer expected returns from market weights in Mean-Variance 
    frameworks like Black-Litterman.
    
    Formula: λ = (E[R_m] - R_f) / σ_m²
    
    Parameters
    ----------
    market_prices : pd.Series, pd.DataFrame or np.ndarray
        Historical prices for a broad market index (e.g., S&P 500).
    risk_free_rate : float, default 0.02
        Annualized risk-free rate.
    frequency : int, default 252
        Trading periods per year.
        
    Returns
    -------
    float
        The market-implied risk aversion coefficient.

    Examples
    --------
    >>> market_risk_aversion(market_index_prices)
    15.792453
    """
    from ..utils.return_utils import to_returns
    returns = to_returns(market_prices, log_returns=True)
    mkt_ret = returns.mean() * frequency
    # mkt_ret = (1 + returns.mean())**frequency - 1
    mkt_var = returns.var() * frequency
    return (mkt_ret - risk_free_rate) / mkt_var


def annualized_return(series: PriceData, frequency: int = 252) -> NumericOutput:
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

    Examples
    --------
    >>> annualized_return(aapl_returns)
    0.155124
    """
    return (1 + series.mean()) ** frequency - 1


def annualized_volatility(
    series: PriceData, 
    annualized: bool = True, 
    frequency: int = 252
) -> NumericOutput:
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

    Examples
    --------
    >>> annualized_volatility(aapl_returns)
    0.308745
    """
    vol = series.std()
    if annualized:
        vol = vol * np.sqrt(frequency)
    return vol


def sharpe_ratio(
    series: PriceData, 
    risk_free_rate: float = 0.03, 
    frequency: int = 252
) -> NumericOutput:
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

    Examples
    --------
    >>> sharpe_ratio(aapl_returns)
    0.370012
    """
    excess = series.mean() * frequency - risk_free_rate
    vol = series.std() * np.sqrt(frequency)
    if isinstance(series, pd.DataFrame):
        return (excess / vol).replace([np.inf, -np.inf], 0).fillna(0)
    return excess / vol if vol > 0 else 0


def sortino_ratio(
    series: PriceData, 
    risk_free_rate: float = 0.03, 
    target_return: float = 0.0, 
    frequency: int = 252
) -> NumericOutput:
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

    Examples
    --------
    >>> sortino_ratio(aapl_returns)
    0.450123
    """
    excess = series.mean() * frequency - risk_free_rate
    downside = (series - target_return).clip(upper=0.0)
    downside_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(frequency)

    if isinstance(series, pd.DataFrame):
        return (excess / downside_vol).replace([np.inf, -np.inf], 0).fillna(0)
    return float(excess / downside_vol) if float(downside_vol) > 0 else 0


def cumulative_returns(
    obj: PriceData, 
    log_returns: bool = False
) -> FinancialOutput:
    """
    Cumulative (compounded) returns from a return series or DataFrame.

    Formula (simple): Cum_t = Π (1 + r_s).  Formula (log): Cum_t = exp(Σ r_s).

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


def drawdown(
    obj: PriceData, 
    from_returns: bool = False
) -> FinancialOutput:
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


def max_drawdown(
    obj: PriceData, 
    from_returns: bool = False
) -> NumericOutput:
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

    Examples
    --------
    >>> max_drawdown(aapl_prices)
    -0.268541
    """
    dd = drawdown(obj, from_returns=from_returns)
    return dd.min()


def rolling_volatility(
    obj: PriceData, 
    window: int = 20, 
    annualized: bool = True, 
    frequency: int = 252
) -> FinancialOutput:
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


def value_at_risk(
    returns: PriceData,
    confidence: float = 0.95,
    method: str = "historical",
    weights: Optional[ArrayLike] = None,
    frequency: int = 1,
) -> NumericOutput:
    r"""
    Compute Value at Risk (VaR) using historical, parametric, or Cornish-Fisher method.

    VaR is the maximum loss not exceeded with probability ``confidence``
    over the given horizon.

    Methods
    -------
    **historical**
        Non-parametric: VaR = -quantile(returns, 1-confidence).

    **parametric** (Gaussian)
        VaR = -(μ - z_α · σ)  where z_α = Φ⁻¹(1-confidence).

    **cornish_fisher** (modified parametric)
        Adjusts the Gaussian quantile for skewness (S) and excess kurtosis (K):

        z_cf = z + (z²-1)·S/6 + (z³-3z)·K/24 - (2z³-5z)·S²/36

        VaR = -(μ - z_cf · σ)

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period return series. Negative values represent losses.
    confidence : float, default 0.95
        Confidence level (e.g. 0.95 for 95% VaR).
    method : str, default 'historical'
        One of ``'historical'``, ``'parametric'``, ``'cornish_fisher'``.
    weights : ArrayLike, optional
        Portfolio weights. When provided with a DataFrame, computes
        *portfolio* VaR from weighted returns.
    frequency : int, default 1
        Annualisation scale (e.g. 252 to annualise daily VaR).
        Scale factor applied as ``VaR * sqrt(frequency)``.

    Returns
    -------
    float or pd.Series
        VaR (positive number; represents the loss).

    Raises
    ------
    ValueError
        If confidence is not in (0, 1) or method is unknown.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    valid_methods = ("historical", "parametric", "cornish_fisher")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'.")

    r = _to_returns_array(returns, weights)
    scale = np.sqrt(frequency) if frequency > 1 else 1.0

    if method == "historical":
        if isinstance(r, pd.DataFrame):
            return -r.quantile(1 - confidence) * scale
        return float(-np.quantile(r.dropna(), 1 - confidence)) * scale

    mu = r.mean() if isinstance(r, pd.Series) else r.mean(axis=0)
    sigma = r.std() if isinstance(r, pd.Series) else r.std(axis=0)
    alpha = 1 - confidence
    z = scipy_stats.norm.ppf(alpha)

    if method == "parametric":
        if isinstance(r, pd.Series):
            return float(-(mu + z * sigma)) * scale
        return -(mu + z * sigma) * scale

    skew = r.skew() if isinstance(r, pd.Series) else r.skew(axis=0)
    kurt = r.kurtosis() if isinstance(r, pd.Series) else r.kurtosis(axis=0)
    z_cf = (z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3*z) * kurt / 24
            - (2*z**3 - 5*z) * skew**2 / 36)
    if isinstance(r, pd.Series):
        return float(-(mu + z_cf * sigma)) * scale
    return -(mu + z_cf * sigma) * scale


def expected_shortfall(
    returns: PriceData,
    confidence: float = 0.95,
    method: str = "historical",
    weights: Optional[ArrayLike] = None,
    frequency: int = 1,
) -> NumericOutput:
    r"""
    Compute Expected Shortfall (ES / CVaR) — the *expected* loss given that
    the loss exceeds VaR.

    Also known as Conditional Value at Risk (CVaR), Average Tail Loss, or
    Tail VaR.

    Methods
    -------
    **historical**
        ES = -mean(r | r ≤ VaR_quantile).

    **parametric** (Gaussian)
        ES = -(μ - σ · φ(z_α) / (1-confidence))
        where φ is the normal PDF and z_α = Φ⁻¹(1-confidence).

    **cornish_fisher**
        Uses the Cornish-Fisher quantile adjustment (same as VaR) and
        then computes the conditional mean numerically over the tail.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period return series.
    confidence : float, default 0.95
        Confidence level.
    method : str, default 'historical'
        One of ``'historical'``, ``'parametric'``, ``'cornish_fisher'``.
    weights : ArrayLike, optional
        Portfolio weights for DataFrame input.
    frequency : int, default 1
        Annualisation scale (VaR * sqrt(frequency)).

    Returns
    -------
    float or pd.Series
        Expected Shortfall (positive number).

    Raises
    ------
    ValueError
        If confidence is outside (0, 1) or method is unknown.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    valid_methods = ("historical", "parametric", "cornish_fisher")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'.")

    r = _to_returns_array(returns, weights)
    scale = np.sqrt(frequency) if frequency > 1 else 1.0
    alpha = 1 - confidence

    if method == "historical":
        def _hist_es(s: pd.Series) -> float:
            cutoff = s.quantile(alpha)
            tail = s[s <= cutoff]
            return float(-tail.mean()) * scale if len(tail) > 0 else 0.0

        if isinstance(r, pd.DataFrame):
            return r.apply(_hist_es)
        return _hist_es(r.dropna())

    mu = r.mean() if isinstance(r, pd.Series) else r.mean(axis=0)
    sigma = r.std() if isinstance(r, pd.Series) else r.std(axis=0)
    z = scipy_stats.norm.ppf(alpha)

    if method == "parametric":
        # Closed-form Gaussian ES
        es = -(mu - sigma * scipy_stats.norm.pdf(z) / alpha) * scale
        return float(es) if isinstance(r, pd.Series) else es

    # Cornish-Fisher: adjust quantile then take tail conditional mean
    skew = r.skew() if isinstance(r, pd.Series) else r.skew(axis=0)
    kurt = r.kurtosis() if isinstance(r, pd.Series) else r.kurtosis(axis=0)
    z_cf = (z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3*z) * kurt / 24
            - (2*z**3 - 5*z) * skew**2 / 36)
    var_cf_level = mu + z_cf * sigma  # threshold

    if isinstance(r, pd.Series):
        tail = r[r <= var_cf_level]
        return float(-tail.mean()) * scale if len(tail) > 0 else float(-var_cf_level) * scale

    def _cf_col(col: pd.Series, threshold: float) -> float:
        tail = col[col <= threshold]
        return float(-tail.mean()) * scale if len(tail) > 0 else float(-threshold) * scale

    return pd.Series(
        {c: _cf_col(r[c], var_cf_level[c]) for c in r.columns},
        name="ES_CF",
    )


def probabilistic_sharpe_ratio(
    returns: PriceData,
    benchmark_sr: float = 0.0,
    risk_free_rate: float = 0.0,
    frequency: int = 252,
) -> NumericOutput:
    r"""
    Estimate the Probabilistic Sharpe Ratio (PSR).

    The PSR is the probability that the *true* Sharpe Ratio of a strategy
    exceeds a benchmark SR, accounting for estimation noise, skewness, and
    kurtosis of the return series.

    Formula (López de Prado, 2012):

    .. math::
        \widehat{PSR}(SR^*) = \Phi\!\left(
            \frac{(\widehat{SR} - SR^*)\sqrt{T-1}}
            {\sqrt{1 - \hat{\gamma}_3\widehat{SR}
                  + \tfrac{\hat{\gamma}_4-1}{4}\widehat{SR}^2}}
        \right)

    where γ₃ is the skewness and γ₄ is the (excess) kurtosis of returns,
    T is the sample size, and SR̂ is the annualised sample Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period return series.
    benchmark_sr : float, default 0.0
        Minimum SR to beat (annualised). Often set to SR of a passive benchmark.
    risk_free_rate : float, default 0.0
        Annualised risk-free rate.
    frequency : int, default 252
        Periods per year. Used to annualise SR and de-annualise benchmark_sr
        for within-period volatility scaling.

    Returns
    -------
    float or pd.Series
        Probability that the true SR exceeds ``benchmark_sr`` (in [0, 1]).

    Raises
    ------
    ValueError
        If the series has fewer than 4 observations.
    """
    def _psr_series(s: pd.Series) -> float:
        s = s.dropna()
        T = len(s)
        if T < 4:
            raise ValueError("PSR requires at least 4 observations.")

        sr_hat = (s.mean() * frequency - risk_free_rate) / (s.std() * np.sqrt(frequency))
        sr_star = benchmark_sr
        skew = float(s.skew())
        kurt = float(s.kurtosis())  # excess

        # Variance of SR estimator
        denom = np.sqrt(1 - skew * sr_hat + ((kurt + 2) / 4) * sr_hat**2)
        if denom <= 0:
            return float("nan")

        z = (sr_hat - sr_star) * np.sqrt(T - 1) / denom
        return float(scipy_stats.norm.cdf(z))

    if isinstance(returns, pd.DataFrame):
        return returns.apply(_psr_series)
    return _psr_series(returns if isinstance(returns, pd.Series) else pd.Series(returns))


def marginal_contribution_to_risk(
    weights: ArrayLike,
    cov_matrix: MatrixLike,
) -> pd.Series:
    r"""
    Marginal Contribution to Risk (MCTR) for each asset.

    MCTR_i measures how much portfolio volatility changes when asset i's
    weight is infinitesimally increased:

    .. math::
        MCTR_i = \frac{(\Sigma w)_i}{\sigma_p}

    where σ_p = √(w^T Σ w) is portfolio volatility.

    Parameters
    ----------
    weights : ArrayLike
        Portfolio weights (must sum to 1 for meaningful results).
    cov_matrix : MatrixLike
        Annualized covariance matrix (N × N).

    Returns
    -------
    pd.Series
        MCTR per asset (units: same as volatility).

    Raises
    ------
    ValueError
        If weights and cov_matrix dimensions are incompatible.
    """
    w = np.asarray(weights, dtype=float).flatten()
    cov = np.asarray(cov_matrix, dtype=float)
    if len(w) != cov.shape[0]:
        raise ValueError(
            f"weights length ({len(w)}) != cov_matrix size ({cov.shape[0]})."
        )

    port_vol = np.sqrt(w @ cov @ w)
    if port_vol <= 0:
        raise ValueError("Portfolio volatility is zero — check inputs.")

    mctr = (cov @ w) / port_vol

    index = (
        cov_matrix.columns.tolist()
        if hasattr(cov_matrix, "columns")
        else list(range(len(w)))
    )
    return pd.Series(mctr, index=index, name="MCTR")


def component_var(
    weights: ArrayLike,
    returns: PriceData,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.DataFrame:
    r"""
    Component VaR — decomposition of portfolio VaR into per-asset contributions.

    Each asset's Component VaR sums to the total portfolio VaR:

    .. math::
        CVaR_i = w_i \cdot \beta_i \cdot VaR_p

    where β_i = Cov(R_i, R_p) / Var(R_p) is the asset's beta to the
    portfolio and VaR_p is computed using the selected method.

    Parameters
    ----------
    weights : ArrayLike
        Portfolio weights.
    returns : pd.DataFrame
        Asset return series.
    confidence : float, default 0.95
        VaR confidence level.
    method : str, default 'historical'
        VaR method: ``'historical'``, ``'parametric'``, or ``'cornish_fisher'``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``['Weight', 'Beta_to_Port', 'Component_VaR', 'Pct_of_Total_VaR']``.
    """
    w = np.asarray(weights, dtype=float).flatten()
    if not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(returns)

    port_ret = returns @ w
    port_var = float(value_at_risk(port_ret, confidence=confidence, method=method))

    # Beta of each asset to the portfolio
    cov_w_port = returns.apply(lambda col: col.cov(port_ret))
    var_port = port_ret.var()
    beta = cov_w_port / var_port if var_port > 0 else pd.Series(np.zeros(len(w)), index=returns.columns)

    component = w * beta * port_var
    pct = component / component.sum() * 100 if component.sum() != 0 else component * 0

    result = pd.DataFrame({
        "Weight": w,
        "Beta_to_Port": beta.values,
        "Component_VaR": component.values,
        "Pct_of_Total_VaR": pct.values,
    }, index=returns.columns)
    return result


def _to_returns_array(
    returns: PriceData,
    weights: Optional[ArrayLike],
) -> Union[pd.Series, pd.DataFrame]:
    """Collapse a return DataFrame to a portfolio Series if weights are given."""
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        returns = pd.Series(returns)
    if weights is not None and isinstance(returns, pd.DataFrame):
        w = np.asarray(weights, dtype=float).flatten()
        return returns @ w
    return returns