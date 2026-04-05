"""
Option valuation models (Black-Scholes).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _bs_d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> tuple[float, float]:
    """Internal: compute d1 and d2 for the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_call(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    """
    Black-Scholes European call option price.

    Formula: **C = S·N(d₁) − K·e^(−rT)·N(d₂)**

    Parameters
    ----------
    spot : float
        Current asset price (S).
    strike : float
        Option strike price (K).
    time_to_expiry : float
        Time to expiration in years (T).
    risk_free_rate : float
        Continuously compounded risk-free rate (r).
    volatility : float
        Annualised asset volatility (σ).

    Returns
    -------
    float
        Call option price.

    Raises
    ------
    ValueError
        If *spot*, *strike*, or *volatility* is negative.

    Examples
    --------
    >>> black_scholes_call(spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.2)
    10.450583572185565
    """
    if spot < 0 or strike < 0 or volatility < 0:
        raise ValueError("spot, strike, and volatility must be non-negative.")
    if time_to_expiry <= 0:
        return max(spot - strike, 0.0)
    d1, d2 = _bs_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)


def black_scholes_put(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    """
    Black-Scholes European put option price (via put-call parity).

    Formula: **P = K·e^(−rT)·N(−d₂) − S·N(−d₁)**

    Parameters
    ----------
    spot : float
        Current asset price (S).
    strike : float
        Option strike price (K).
    time_to_expiry : float
        Time to expiration in years (T).
    risk_free_rate : float
        Continuously compounded risk-free rate (r).
    volatility : float
        Annualised asset volatility (σ).

    Returns
    -------
    float
        Put option price.

    Raises
    ------
    ValueError
        If *spot*, *strike*, or *volatility* is negative.

    Examples
    --------
    >>> black_scholes_put(spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.2)
    5.573526022256971
    """
    if spot < 0 or strike < 0 or volatility < 0:
        raise ValueError("spot, strike, and volatility must be non-negative.")
    if time_to_expiry <= 0:
        return max(strike - spot, 0.0)
    d1, d2 = _bs_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return (
        strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        - spot * norm.cdf(-d1)
    )


def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    bracket: tuple = (1e-6, 10.0),
) -> float:
    """
    Implied volatility via Brent's method.

    Parameters
    ----------
    option_price : float
        Observed market option price.
    spot, strike, time_to_expiry, risk_free_rate : float
        Standard Black-Scholes inputs.
    option_type : str, default 'call'
        ``'call'`` or ``'put'``.
    bracket : tuple, default (1e-6, 10.0)
        Search bounds for volatility.

    Returns
    -------
    float
        Implied annualised volatility.

    Raises
    ------
    ValueError
        If *option_price*, *spot*, or *strike* is negative, if *time_to_expiry*
        is non-positive, or if Brent's method fails to find a root.

    Examples
    --------
    >>> implied_volatility(option_price=10.45, spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free_rate=0.05, option_type="call")
    0.19998444801102064
    """
    if option_price < 0 or spot < 0 or strike < 0:
        raise ValueError("Prices and strikes must be non-negative.")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be strictly positive.")
        
    pricer = black_scholes_call if option_type == "call" else black_scholes_put

    def objective(sigma):
        return pricer(spot, strike, time_to_expiry, risk_free_rate, sigma) - option_price

    try:
        return brentq(objective, bracket[0], bracket[1])
    except ValueError:
        raise ValueError(
            f"Could not find implied volatility in bracket {bracket}. "
            "Check inputs or widen the bracket."
        )
