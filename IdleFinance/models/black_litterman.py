"""
Black-Litterman methods for estimating posterior returns and covariance.

This module implements the canonical Black-Litterman models based on Jay Walters
and Thomas Idzorek, calculating full posterior distributions.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize
from ..utils.return_utils import to_returns
from ..core._types import (
    Union, Optional, BLResult, BLWeightsResult, BLSingleResult, 
    ViewData, PortfolioBounds, PortfolioConstraints,
    OmegaMethod, TauMethod, ArrayLike, MatrixLike, PriceData, VectorOutput
)

def market_prior_returns(
    market_weights: ArrayLike, 
    risk_aversion: float, 
    cov_matrix: MatrixLike
) -> VectorOutput:
    r"""
    Calculate the implied equilibrium returns (Π) given market weights.
    
    This process, also known as "Reverse Optimization," finds the returns 
    that would make the current market weights optimal in a Markowitz 
    Mean-Variance framework.
    
    Formula: Π = λ * Σ * w_mkt
    
    Parameters
    ----------
    market_weights : pd.Series or array-like
        The market capitalization weights (w_mkt).
    risk_aversion : float
        The market-implied risk aversion coefficient (λ).
    cov_matrix : pd.DataFrame or np.ndarray
        The prior covariance matrix (Σ).
        
    Returns
    -------
    pd.Series or np.ndarray
        The implied equilibrium returns (Π).
    """
    weights = np.asarray(market_weights).flatten()
    cov = np.asarray(cov_matrix)
    pi = risk_aversion * cov @ weights
    
    if isinstance(market_weights, pd.Series):
        return pd.Series(pi, index=market_weights.index)
    return pi

def tau(
    prices: Optional[pd.DataFrame] = None, 
    method: TauMethod = "default", 
    constant_value: float = 0.05
) -> float:
    """
    Calculate the uncertainty scaling factor (tau).
    
    Methods:
    - 'default': use a constant value (e.g., 0.05).
    - 'variance': 1 / number of historical observations (T).
    """
    if method == "default":
        return constant_value
    elif method == "variance":
        if prices is None:
            raise ValueError("Prices must be provided to calculate tau based on variance length.")
        return 1.0 / len(prices)
    else:
        raise ValueError(f"Unknown tau calculation method: {method}")

def omega(
    cov_matrix: MatrixLike, 
    P: np.ndarray, 
    tau: float = 0.05, 
    confidences: Optional[ArrayLike] = None,
    method: OmegaMethod = "idzorek"
) -> np.ndarray:
    """
    Calculate the uncertainty matrix (Omega) for views.
    
    Methods:
    - 'idzorek': Based on user confidence intervals [0, 1].
    - 'proportional': Proportional to the variance of the prior (tau * P * Sigma * P.T).
    """
    if method == "idzorek":
        if confidences is None:
            raise ValueError("Confidences must be provided for Idzorek's method.")
        return idzorek_confidence_to_omega(confidences, cov_matrix, P, tau=tau)
    elif method == "proportional":
        return np.diag(np.diag(tau * P @ np.asarray(cov_matrix) @ P.T))
    else:
        raise ValueError(f"Unknown omega method: {method}")

def idzorek_confidence_to_omega(
    confidences: ArrayLike, 
    cov_matrix: MatrixLike, 
    P: np.ndarray, 
    tau: float = 0.05
) -> np.ndarray:
    """
    Calculate the uncertainty matrix (Omega) from confidence levels using Idzorek's method.

    Formula: ω_k = τ * α * P_k * Σ * P_k^T  where α = (1 - C) / C.

    Parameters
    ----------
    confidences : array-like
        Confidence levels in [0, 1] for each view.
    cov_matrix : np.ndarray or pd.DataFrame
        Covariance matrix of asset returns.
    P : np.ndarray
        Picking matrix.
    tau : float, default 0.05
        Uncertainty scaling factor.

    Returns
    -------
    np.ndarray
        Diagonal Omega matrix.
    """
    confidences = np.asarray(confidences).flatten()
    n_views = len(confidences)
    if P.shape[0] != n_views:
        raise ValueError("Number of confidences must match number of views (rows in P).")

    cov = np.asarray(cov_matrix)
    omega = np.zeros((n_views, n_views))

    for i, conf in enumerate(confidences):
        if conf < 0 or conf > 1:
            raise ValueError(f"Confidence must be in [0,1], got {conf}")
        if conf == 0:
            omega[i, i] = 1e6  # High uncertainty
        else:
            alpha = (1 - conf) / conf
            P_view = P[i].reshape(1, -1)
            omega[i, i] = (tau * alpha * P_view @ cov @ P_view.T).item()
            
    return omega


def compute_bl_weights(
    posterior_returns: ArrayLike, 
    posterior_cov: MatrixLike, 
    risk_aversion: float = 1.0, 
    bounds: PortfolioBounds = None,
    objective: str = "utility",
    benchmark_weights: Optional[ArrayLike] = None,
    custom_constraints: PortfolioConstraints = None
) -> BLWeightsResult:
    r"""
    Calculate optimal portfolio weights using Markowitz Mean-Variance Utility.

    This function finds the weights that maximize the expected utility:
    U = w^T * E[R] - (λ / 2) * w^T * Σ * w
    
    Or minimizes tracking error variance:
    TE = (w - w_b)^T * Σ * (w - w_b)

    Parameters
    ----------
    posterior_returns : pd.Series or array-like
        The posterior expected returns (E[R]) from the Black-Litterman model.
    posterior_cov : pd.DataFrame or np.ndarray
        The posterior covariance matrix (Σ).
    risk_aversion : float, default 1.0
        The risk aversion coefficient (λ). Higher values lead to 
        lower risk portfolios.
    bounds : tuple or list of tuples, optional
        Weight constraints for each asset. e.g., (0, 0.20) restricts 
        allocations between 0% and 20%.
    objective : str, default 'utility'
        The optimization objective:
        - 'utility': Maximize Markowitz Mean-Variance utility.
        - 'tracking_error': Minimize variance relative to a benchmark.
    benchmark_weights : pd.Series or array-like, optional
        The weights of the benchmark portfolio (w_b). Required if 
        objective='tracking_error'.
    custom_constraints: list of dict, optional
        Additional constraints for `scipy.optimize.minimize` (e.g., sector caps).

    Returns
    -------
    pd.Series
        The optimized portfolio weights.
    """
    returns = np.asarray(posterior_returns).flatten()
    cov = np.asarray(posterior_cov)
    n_assets = len(returns)

    # Fast path: Unconstrained algebraic optimization
    if bounds is None and objective == "utility":
        A = risk_aversion * cov
        b = returns
        try:
            weights = solve(A, b)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(A, b, rcond=None)[0]
        
        weights = weights / weights.sum()
        if isinstance(posterior_returns, pd.Series):
            return pd.Series(weights, index=posterior_returns.index)
        return weights

    # Numerical optimization path
    def utility_obj(w):
        port_ret = w @ returns
        port_var = w @ cov @ w.T
        return -(port_ret - 0.5 * risk_aversion * port_var)
        
    def tracking_error_obj(w):
        if benchmark_weights is None:
            raise ValueError("benchmark_weights must be provided for tracking error optimization.")
        bench_w = np.asarray(benchmark_weights).flatten()
        diff = w - bench_w
        return diff @ cov @ diff.T
        
    target_obj = utility_obj if objective == "utility" else tracking_error_obj

    # Full investment constraint
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    if custom_constraints is not None:
        constraints.extend(custom_constraints)
    
    # Process bounds
    processed_bounds = None
    if bounds is not None:
        if isinstance(bounds, tuple) and len(bounds) == 2 and isinstance(bounds[0], (int, float)):
            processed_bounds = [bounds] * n_assets
        elif isinstance(bounds, (list, tuple)):
            if len(bounds) != n_assets:
                raise ValueError("List of bounds must match number of assets.")
            processed_bounds = bounds
        else:
            raise ValueError("Invalid bounds format.")

    # Initial guess
    x0 = np.asarray(benchmark_weights).flatten() if benchmark_weights is not None else np.ones(n_assets) / n_assets

    res = minimize(
        target_obj, 
        x0, 
        method='SLSQP', 
        bounds=processed_bounds, 
        constraints=constraints,
        tol=1e-8
    )
    
    if not res.success:
        warnings.warn(f"Optimization failed: {res.message}", UserWarning)
        
    res_x = res.x
    
    if isinstance(posterior_returns, pd.Series):
        return pd.Series(res_x, index=posterior_returns.index)
    return res_x


def bl_posterior_distribution(
    cov_matrix: MatrixLike,
    prior_returns: Optional[ArrayLike] = None,
    views: Optional[ViewData] = None,
    view_confidences: Optional[ArrayLike] = None,
    tau_val: Optional[float] = None,
    tau_method: TauMethod = "default",
    omega_method: OmegaMethod = "idzorek",
    prices: Optional[PriceData] = None,
) -> BLResult:
    r"""
    Calculate Black-Litterman posterior returns and covariance using the Walters form.

    The Black-Litterman model combines a prior equilibrium distribution with 
    subjective investor views to produce a refined posterior distribution.

    Mathematical Formulas:
    - Posterior Returns: E[R] = [(τ * Σ)^-1 + P^T * Ω^-1 * P]^-1 * [(τ * Σ)^-1 * Π + P^T * Ω^-1 * Q]
    - Posterior Covariance: Σ_post = Σ + [(τ * Σ)^-1 + P^T * Ω^-1 * P]^-1

    Parameters
    ----------
    cov_matrix : pd.DataFrame or np.ndarray
        The prior covariance matrix of asset returns (Σ).
    prior_returns : pd.Series or np.ndarray, optional
        The prior equilibrium returns (Π). Often sourced from 
        `market_implied_prior_returns`.
    views : dict or pd.Series, optional
        Subjective views (Q). Absolute return expectations for specific assets.
    view_confidences : array-like, optional
        Investor confidence levels for each view ([0, 1]).
    tau_val : float, optional
        The uncertainty scaling factor (τ). If None, calculated via `tau_method`.
    tau_method : str, default "default"
        Method to estimate τ if not provided. 'default' (0.05) or 'variance'.
    omega_method : str, default "idzorek"
        Method to calculate the uncertainty matrix (Ω): 'idzorek' 
        (confidence-based) or 'proportional'.
    prices : pd.DataFrame, optional
        Historical price data, required if `tau_method='variance'`.

    Returns
    -------
    BLResult
        A tuple containing (Posterior Returns, Posterior Covariance Matrix).
    """
    cov = np.asarray(cov_matrix)
    tickers = list(cov_matrix.columns) if hasattr(cov_matrix, "columns") else range(len(cov))
    n_assets = len(cov)
    
    if tau_val is None:
        tau_val = tau(prices, method=tau_method)

    if prior_returns is None:
        warnings.warn(
            "No prior returns provided, using equal-weighted (1/N)",
            UserWarning,
        )
        prior_ret = np.ones(n_assets) / n_assets
    elif isinstance(prior_returns, pd.Series):
        prior_ret = prior_returns.values
    else:
        prior_ret = np.asarray(prior_returns)
        
    prior_ret = prior_ret.reshape(-1, 1)

    if views is None:
        posterior_ret = prior_ret.flatten()
        posterior_cov = cov * (1 + tau)
        return (
            pd.Series(posterior_ret, index=tickers),
            pd.DataFrame(posterior_cov, index=tickers, columns=tickers)
        )

    if isinstance(views, dict):
        views = pd.Series(views)
    
    n_views = len(views)
    Q = np.zeros((n_views, 1))
    P = np.zeros((n_views, n_assets))
    
    for i, (ticker, ret) in enumerate(views.items()):
        if ticker not in tickers:
            raise ValueError(f"View ticker '{ticker}' not in asset universe")
        Q[i] = ret
        idx = tickers.index(ticker)
        P[i, idx] = 1

    if view_confidences is not None or omega_method == "proportional":
        omega_matrix = omega(cov, P, tau=tau_val, confidences=view_confidences, method=omega_method)
    else:
        omega_matrix = np.diag(np.diag(tau_val * P @ cov @ P.T))

    # Walters canonical inversions
    tau_cov = tau_val * cov
    try:
        tau_cov_inv = np.linalg.inv(tau_cov)
    except np.linalg.LinAlgError:
        tau_cov_inv = np.linalg.pinv(tau_cov)

    try:
        omega_inv = np.linalg.inv(omega_matrix)
    except np.linalg.LinAlgError:
        omega_inv = np.linalg.pinv(omega_matrix)

    # M = [(tau * cov)^-1 + P' * omega^-1 * P]^-1
    M_inverse = tau_cov_inv + P.T @ omega_inv @ P
    try:
        M = np.linalg.inv(M_inverse)
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(M_inverse)

    # E[R] = M * [(tau * cov)^-1 * pi + P' * omega^-1 * Q]
    posterior_ret = M @ (tau_cov_inv @ prior_ret + P.T @ omega_inv @ Q)
    
    # cov_post = cov + M
    posterior_cov = cov + M

    return (
        pd.Series(posterior_ret.flatten(), index=tickers),
        pd.DataFrame(posterior_cov, index=tickers, columns=tickers)
    )


def black_litterman_single_asset(
    price_series: PriceData, 
    prior_return: Optional[float] = None, 
    view: Optional[float] = None, 
    view_confidence: float = 0.5, 
    tau_val: float = 0.05, 
    risk_aversion: float = 1.0
) -> BLSingleResult:
    """
    Apply Black-Litterman optimization for a single asset with a view.

    Returns
    -------
    (float, float, float)
        Posterior Expected Return, Posterior Variance, Optimal Weight
    """
    if price_series.empty:
        return 0.0, 0.0, 0.0

    if (price_series > 1).any():
        returns = price_series.pct_change().dropna()
    else:
        returns = price_series

    variance = returns.var()
    if prior_return is None:
        prior_return = returns.mean()

    if view is None:
        posterior_return, posterior_variance = prior_return, variance * (1 + tau_val)
        weight = posterior_return / (risk_aversion * variance) if variance > 0 else 0.5
        return posterior_return, posterior_variance, min(max(weight, 0), 1)

    omega_val = ((1 - view_confidence) / view_confidence) * tau_val * variance if view_confidence > 0 else 1e6
    
    tau_var_inv = 1 / (tau_val * variance)
    omega_inv = 1 / omega_val
    
    M = 1 / (tau_var_inv + omega_inv)
    posterior_return = M * (tau_var_inv * prior_return + omega_inv * view)
    posterior_variance = variance + M
    
    weight = posterior_return / (risk_aversion * variance) if variance > 0 else 0.5

    return posterior_return, posterior_variance, min(max(weight, 0), 1)