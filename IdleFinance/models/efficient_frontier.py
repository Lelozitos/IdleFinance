"""
Efficient Frontier portfolio optimization.

Provides Mean-Variance optimization tools: minimum variance portfolios,
maximum Sharpe (tangency) portfolios, target-return portfolios,
and full frontier generation.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..core._types import Optional, ArrayLike, MatrixLike, PortfolioBounds


def portfolio_performance(
    weights: ArrayLike,
    expected_returns: ArrayLike,
    cov_matrix: MatrixLike,
    risk_free_rate: float = 0.0,
) -> tuple:
    """
    Calculate return, volatility, and Sharpe ratio for a given portfolio.

    Parameters
    ----------
    weights : ArrayLike
        Portfolio weights.
    expected_returns : ArrayLike
        Expected annualized returns per asset.
    cov_matrix : MatrixLike
        Annualized covariance matrix.
    risk_free_rate : float, default 0.0
        Risk-free rate for Sharpe ratio.

    Returns
    -------
    tuple
        (portfolio_return, portfolio_volatility, sharpe_ratio).
    """
    w = np.asarray(weights, dtype=float).flatten()
    mu = np.asarray(expected_returns, dtype=float).flatten()
    cov = np.asarray(cov_matrix, dtype=float)

    port_ret = w @ mu
    port_vol = np.sqrt(w @ cov @ w)
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return float(port_ret), float(port_vol), float(sharpe)


def _process_bounds(bounds, n_assets):
    """Normalize bounds into a list of (lo, hi) tuples."""
    if bounds is None:
        return [(0.0, 1.0)] * n_assets
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and isinstance(bounds[0], (int, float))
    ):
        return [bounds] * n_assets
    if isinstance(bounds, (list, tuple)):
        if len(bounds) != n_assets:
            raise ValueError(
                f"len(bounds)={len(bounds)} must equal n_assets={n_assets}."
            )
        return list(bounds)
    raise ValueError("bounds must be a (lo, hi) tuple or a list of (lo, hi) per asset.")


def min_variance(
    cov_matrix: MatrixLike,
    bounds: PortfolioBounds = None,
    target_sum: float = 1.0,
) -> np.ndarray:
    """
    Find the Global Minimum Variance Portfolio.

    Minimizes: **w^T Σ w**  subject to Σw_i = target_sum.

    Parameters
    ----------
    cov_matrix : MatrixLike
        Annualized covariance matrix (N × N).
    bounds : PortfolioBounds, optional
        Per-asset weight bounds. Default (0, 1) per asset.
    target_sum : float, default 1.0
        Sum constraint for weights.

    Returns
    -------
    np.ndarray
        Optimal weights.

    Raises
    ------
    ValueError
        If optimization fails.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]
    processed_bounds = _process_bounds(bounds, n)

    x0 = np.full(n, target_sum / n)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - target_sum}]

    res = minimize(
        lambda w: w @ cov @ w,
        x0,
        method="SLSQP",
        bounds=processed_bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not res.success:
        warnings.warn(f"Min-variance optimization: {res.message}", UserWarning, stacklevel=2)

    return res.x


def max_sharpe(
    expected_returns: ArrayLike,
    cov_matrix: MatrixLike,
    risk_free_rate: float = 0.0,
    bounds: PortfolioBounds = None,
    target_sum: float = 1.0,
) -> np.ndarray:
    """
    Find the Maximum Sharpe Ratio (tangency) portfolio.

    Maximizes: **(w^T μ - R_f) / √(w^T Σ w)**

    Parameters
    ----------
    expected_returns : ArrayLike
        Expected annualized returns per asset.
    cov_matrix : MatrixLike
        Annualized covariance matrix.
    risk_free_rate : float, default 0.0
        Risk-free rate.
    bounds : PortfolioBounds, optional
        Per-asset weight bounds. Default (0, 1).
    target_sum : float, default 1.0
        Sum constraint for weights.

    Returns
    -------
    np.ndarray
        Optimal weights.
    """
    mu = np.asarray(expected_returns, dtype=float).flatten()
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(mu)
    processed_bounds = _process_bounds(bounds, n)

    x0 = np.full(n, target_sum / n)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - target_sum}]

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 1e6
        return -(port_ret - risk_free_rate) / port_vol

    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=processed_bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not res.success:
        warnings.warn(f"Max-Sharpe optimization: {res.message}", UserWarning, stacklevel=2)

    return res.x


def efficient_return(
    expected_returns: ArrayLike,
    cov_matrix: MatrixLike,
    target_return: float,
    bounds: PortfolioBounds = None,
    target_sum: float = 1.0,
) -> np.ndarray:
    """
    Find the minimum-variance portfolio for a given target return.

    Minimizes: **w^T Σ w**  subject to w^T μ = target_return.

    Parameters
    ----------
    expected_returns : ArrayLike
        Expected annualized returns.
    cov_matrix : MatrixLike
        Annualized covariance matrix.
    target_return : float
        Target portfolio return.
    bounds : PortfolioBounds, optional
        Per-asset weight bounds.
    target_sum : float, default 1.0
        Sum constraint.

    Returns
    -------
    np.ndarray
        Optimal weights.
    """
    mu = np.asarray(expected_returns, dtype=float).flatten()
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(mu)
    processed_bounds = _process_bounds(bounds, n)

    x0 = np.full(n, target_sum / n)
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - target_sum},
        {"type": "eq", "fun": lambda w: w @ mu - target_return},
    ]

    res = minimize(
        lambda w: w @ cov @ w,
        x0,
        method="SLSQP",
        bounds=processed_bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not res.success:
        warnings.warn(f"Efficient-return optimization: {res.message}", UserWarning, stacklevel=2)

    return res.x


def efficient_frontier(
    expected_returns: ArrayLike,
    cov_matrix: MatrixLike,
    n_points: int = 50,
    bounds: PortfolioBounds = None,
    target_sum: float = 1.0,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Generate the full Efficient Frontier.

    Computes a set of portfolios from the minimum-variance to the
    maximum-return feasible portfolio.

    Parameters
    ----------
    expected_returns : ArrayLike
        Expected annualized returns.
    cov_matrix : MatrixLike
        Annualized covariance matrix.
    n_points : int, default 50
        Number of frontier points.
    bounds : PortfolioBounds, optional
        Per-asset weight bounds.
    target_sum : float, default 1.0
        Sum constraint.
    risk_free_rate : float, default 0.0
        Risk-free rate for Sharpe ratios.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Return', 'Volatility', 'Sharpe'] and
        one row per frontier point.
    """
    mu = np.asarray(expected_returns, dtype=float).flatten()
    cov = np.asarray(cov_matrix, dtype=float)

    # Find the min-variance portfolio to anchor the lower bound
    w_min = min_variance(cov, bounds=bounds, target_sum=target_sum)
    ret_min = w_min @ mu

    # Upper bound: max feasible return
    # With long-only bounds, this is the return of the single best asset
    processed_bounds = _process_bounds(bounds, len(mu))
    hi_weights = np.array([b[1] for b in processed_bounds])
    ret_max = mu.max()  # theoretical max with full concentration

    # Ensure ret_max is achievable: try optimizing for it
    try:
        w_max = efficient_return(mu, cov, ret_max, bounds=bounds, target_sum=target_sum)
        ret_max = w_max @ mu
    except Exception:
        ret_max = ret_min + (ret_max - ret_min) * 0.95

    target_returns = np.linspace(ret_min, ret_max, n_points)

    results = []
    for target_ret in target_returns:
        try:
            w = efficient_return(mu, cov, target_ret, bounds=bounds, target_sum=target_sum)
            p_ret, p_vol, p_sharpe = portfolio_performance(w, mu, cov, risk_free_rate)
            results.append({"Return": p_ret, "Volatility": p_vol, "Sharpe": p_sharpe})
        except Exception:
            continue

    return pd.DataFrame(results)
