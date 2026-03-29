"""
Covariance matrix estimation methods.

This module provides functions for calculating different types of covariance matrices
from return or price data, such as sample covariance and exponentially-weighted 
covariance (EMA).
"""

import numpy as np
import pandas as pd
from ..utils.return_utils import to_returns
from ..core._types import Union, Any, Optional, CovarianceMethod, DenoiseMethod, PriceData, CovarianceMatrix

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
    denoise: bool = False,
    n_obs: Optional[int] = None,
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
    denoise : bool, default False
        If True, apply Marchenko-Pastur eigenvalue clipping to remove
        noise from the covariance matrix.
    n_obs : int, optional
        Number of observations (T). If None, inferred from the data.
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
        cov = sample_covariance(prices, **kwargs)
    elif method in ["ema", "ewma", "exponential"]:
        cov = exponential_covariance(prices, **kwargs)
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    
    if denoise:
        if n_obs is None:
            n_obs = len(prices) if not returns_data else len(prices) + 1
        cov = denoise_covariance(cov, n_obs)
    
    return cov


def denoise_covariance(
    cov: CovarianceMatrix,
    n_obs: int,
    method: DenoiseMethod = "marchenko_pastur",
) -> CovarianceMatrix:
    """
    Remove noise from a covariance matrix using Random Matrix Theory.

    Applies the Marchenko-Pastur theorem to identify eigenvalues that are
    consistent with pure noise and replaces them with their average,
    preserving the total variance (trace).

    Formula:
        λ+ = σ² (1 + √(N/T))²     (upper noise threshold)
        λ- = σ² (1 - √(N/T))²     (lower noise threshold)

    Eigenvalues below λ+ are considered noise and are replaced by their
    mean value, then the matrix is reconstructed.

    Parameters
    ----------
    cov : pd.DataFrame or np.ndarray
        Input covariance matrix (N × N).
    n_obs : int
        Number of observations (T) used to estimate the matrix.
    method : DenoiseMethod, default 'marchenko_pastur'
        Denoising method.

    Returns
    -------
    pd.DataFrame
        Denoised covariance matrix of same shape.

    Raises
    ------
    ValueError
        If n_obs is non-positive or method is unknown.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if method != "marchenko_pastur":
        raise ValueError(f"Unknown denoise method: {method}")

    is_df = isinstance(cov, pd.DataFrame)
    idx = cov.index if is_df else None
    cols = cov.columns if is_df else None
    cov_arr = np.array(cov, dtype=float)
    n_assets = cov_arr.shape[0]

    # Extract volatilities and convert to correlation matrix
    vols = np.sqrt(np.diag(cov_arr))

    vols_safe = np.where(vols == 0, 1e-10, vols)
    corr = cov_arr / np.outer(vols_safe, vols_safe)
    np.fill_diagonal(corr, 1.0)

    # Eigendecomposition of correlation matrix
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Marchenko-Pastur bounds
    q = n_assets / n_obs
    sigma2 = 1.0
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2

    # Identify noise eigenvalues
    noise_mask = eigenvalues <= lambda_plus
    n_noise = noise_mask.sum()

    if n_noise > 0 and n_noise < n_assets:
        noise_avg = eigenvalues[noise_mask].mean()
        eigenvalues[noise_mask] = noise_avg

    # Reconstruct the correlation matrix
    denoised_corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    d = np.sqrt(np.diag(denoised_corr))
    d_safe = np.where(d == 0, 1e-10, d)
    denoised_corr = denoised_corr / np.outer(d_safe, d_safe)
    np.fill_diagonal(denoised_corr, 1.0)

    # Convert back to covariance
    denoised_cov = denoised_corr * np.outer(vols, vols)
    
    # Enforce symmetry
    denoised_cov = 0.5 * (denoised_cov + denoised_cov.T)

    if is_df:
        return pd.DataFrame(denoised_cov, index=idx, columns=cols)
    return denoised_cov
