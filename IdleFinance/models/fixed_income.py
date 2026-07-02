"""
Fixed Income and Credit Valuation models.
"""

import numpy as np
from scipy.optimize import brentq
from ..core._types import Union, ArrayLike
from ..utils.basic import Finance


def bond_price(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 2,
) -> float:
    """
    Calculate the fair price of a fixed-coupon bond.

    Formula: **P = Σ C/(1+y)^t + F/(1+y)^T**

    Parameters
    ----------
    face_value : float
        Par / face value of the bond.
    coupon_rate : float
        Annual coupon rate (e.g. 0.05 for 5 %).
    ytm : float
        Annualised yield-to-maturity.
    years_to_maturity : float
        Remaining life of the bond in years.
    frequency : int, default 2
        Coupon payments per year

    Returns
    -------
    float
        Fair bond price.

    Raises
    ------
    ValueError
        If *face_value* or *years_to_maturity* is negative, or if *frequency* is non-positive.

    Examples
    --------
    >>> bond_price(face_value=1000.0, coupon_rate=0.05, ytm=0.06, years_to_maturity=10.0, frequency=2)
    925.6126256977221
    """
    if face_value < 0 or years_to_maturity < 0:
        raise ValueError("face_value and years_to_maturity must be non-negative.")
    if frequency <= 0:
        raise ValueError("frequency must be strictly positive.")
        
    if coupon_rate == 0:
        return Finance.present_value(face_value, ytm, years_to_maturity, frequency)

    periods = int(years_to_maturity * frequency)
    periodic_coupon = face_value * coupon_rate / frequency
    periodic_ytm = ytm / frequency

    if periodic_ytm == 0:
        return periodic_coupon * periods + face_value

    # Closed-form: PV of annuity + PV of par
    t = np.arange(1, periods + 1)
    discount = (1.0 + periodic_ytm) ** t
    return float(np.sum(periodic_coupon / discount) + face_value / discount[-1])


def bond_ytm(
    price: float,
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    frequency: int = 2,
    threshold: tuple = (1e-6, 0.99),
) -> float:
    """
    Solve for the Yield-to-Maturity (YTM) of a bond given its price.

    Parameters
    ----------
    price : float
        Current market price of the bond.
    face_value : float
        Par value.
    coupon_rate : float
        Annual coupon rate.
    years_to_maturity : float
        Remaining life in years.
    frequency : int, default 2
        Coupon payments per year.
    threshold : tuple, default (1e-6, 0.99)
        Search bounds for YTM.

    Returns
    -------
    float
        Annualised yield-to-maturity.

    Raises
    ------
    ValueError
        If *price*, *face_value*, or *years_to_maturity* is negative, if *frequency*
        is non-positive, or if Brent's method fails to find a root.

    Examples
    --------
    >>> bond_ytm(price=925.61, face_value=1000.0, coupon_rate=0.05, years_to_maturity=10.0, frequency=2)
    0.060000370085423534
    """
    if price <= 0 or years_to_maturity <= 0:
        raise ValueError("price and years_to_maturity must be strictly positive.")
    if frequency <= 0:
        raise ValueError("frequency must be strictly positive.")

    if coupon_rate == 0:
        periods = years_to_maturity * frequency
        return frequency * ((face_value / price) ** (1 / periods) - 1)

    def objective(y):
        return bond_price(face_value, coupon_rate, y, years_to_maturity, frequency) - price

    return brentq(objective, threshold[0], threshold[1])


def bond_duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 2,
    modified: bool = True,
) -> float:
    """
    Macaulay or Modified Duration of a fixed-coupon bond.

    Parameters
    ----------
    face_value : float
        Par value.
    coupon_rate : float
        Annual coupon rate.
    ytm : float
        Annual yield-to-maturity.
    years_to_maturity : float
        Remaining life in years.
    frequency : int, default 2
        Coupon payments per year.
    modified : bool, default True
        If True, returns Modified Duration; otherwise Macaulay Duration.

    Returns
    -------
    float
        Duration in years (modified or Macaulay).

    Examples
    --------
    >>> bond_duration(face_value=1000.0, coupon_rate=0.04, ytm=0.05, years_to_maturity=5.0, frequency=2)
    4.458056324352843
    """
    if modified:
        return bond_modified_duration(face_value, coupon_rate, ytm, years_to_maturity, frequency)
    return bond_macaulay_duration(face_value, coupon_rate, ytm, years_to_maturity, frequency)


def bond_convexity(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 2,
) -> float:
    """
    Convexity of a fixed-coupon bond.

    Parameters
    ----------
    face_value : float
        Par value.
    coupon_rate : float
        Annual coupon rate.
    ytm : float
        Annual yield-to-maturity.
    years_to_maturity : float
        Remaining life in years.
    frequency : int, default 2
        Coupon frequency per year.

    Returns
    -------
    float
        Convexity (in years²).

    Raises
    ------
    ValueError
        If *face_value* or *years_to_maturity* is negative, or if *frequency* is non-positive.

    Examples
    --------
    >>> bond_convexity(face_value=1000.0, coupon_rate=0.04, ytm=0.05, years_to_maturity=5.0, frequency=2)
    23.194409907616695
    """
    if face_value < 0 or years_to_maturity < 0:
        raise ValueError("face_value and years_to_maturity must be non-negative.")
    if frequency <= 0:
        raise ValueError("frequency must be strictly positive.")

    periods = int(years_to_maturity * frequency)
    periodic_coupon = face_value * coupon_rate / frequency
    periodic_ytm = ytm / frequency
    t = np.arange(1, periods + 1)

    cash_flows = np.full(periods, periodic_coupon, dtype=float)
    cash_flows[-1] += face_value

    discount_factors = (1 + periodic_ytm) ** t
    pv_cf = cash_flows / discount_factors

    price = bond_price(face_value, coupon_rate, ytm, years_to_maturity, frequency)

    convexity_sum = np.sum(t * (t + 1) * pv_cf / (1 + periodic_ytm) ** 2)
    return convexity_sum / (price * frequency ** 2)


def credit_spread(
    risky_ytm: float,
    risk_free_ytm: float,
) -> float:
    """
    Credit spread (option-adjusted spread in basis points not assumed).

    Formula: **Spread = risky_YTM − risk_free_YTM**

    Parameters
    ----------
    risky_ytm : float
        YTM of the risky (corporate) bond.
    risk_free_ytm : float
        YTM of the risk-free benchmark bond.

    Returns
    -------
    float
        Credit spread.

    Examples
    --------
    >>> credit_spread(risky_ytm=0.065, risk_free_ytm=0.04)
    0.025
    """
    return risky_ytm - risk_free_ytm


def forward_rate(
    spot_rate_t1: float,
    t1: float,
    spot_rate_t2: float,
    t2: float,
) -> float:
    """
    Calculate the implied forward rate between two periods.

    Formula (Annual compounding): **(1 + r_2)^t2 = (1 + r_1)^t1 * (1 + f)^{t2 - t1}**

    Parameters
    ----------
    spot_rate_t1 : float
        Spot rate for the shorter maturity period (t1).
    t1 : float
        Time to the shorter maturity in years.
    spot_rate_t2 : float
        Spot rate for the longer maturity period (t2).
    t2 : float
        Time to the longer maturity in years (must be > t1).

    Returns
    -------
    float
        Implied forward rate between t1 and t2.

    Raises
    ------
    ValueError
        If *t2* is not strictly greater than *t1*, or if times are negative.
        
    Examples
    --------
    >>> forward_rate(spot_rate_t1=0.03, t1=1.0, spot_rate_t2=0.04, t2=2.0)
    0.050097087378640824
    """
    if t1 < 0 or t2 < 0:
        raise ValueError("Time periods must be non-negative.")
    if t2 <= t1:
        raise ValueError("t2 must be strictly greater than t1.")
    return ((1 + spot_rate_t2) ** t2 / (1 + spot_rate_t1) ** t1) ** (1 / (t2 - t1)) - 1


def bond_macaulay_duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 2,
) -> float:
    """
    Calculate the Macaulay Duration of a fixed-coupon bond.

    Formula: **D_mac = [Σ (t * C / (1+y)^t) + T * F / (1+y)^T] / Price**

    Parameters
    ----------
    face_value : float
        Par value.
    coupon_rate : float
        Annual coupon rate.
    ytm : float
        Annual yield-to-maturity.
    years_to_maturity : float
        Remaining life in years.
    frequency : int, default 2
        Coupon payments per year.

    Returns
    -------
    float
        Macaulay duration in years.

    Raises
    ------
    ValueError
        If *face_value* or *years_to_maturity* is negative, or if *frequency* is non-positive.

    Examples
    --------
    >>> bond_macaulay_duration(1000, 0.04, 0.05, 5, 2)
    4.5695
    """
    if face_value < 0 or years_to_maturity < 0:
        raise ValueError("face_value and years_to_maturity must be non-negative.")
    if frequency <= 0:
        raise ValueError("frequency must be strictly positive.")

    if coupon_rate == 0:
        return float(years_to_maturity)

    periods = int(years_to_maturity * frequency)
    periodic_coupon = face_value * coupon_rate / frequency
    
    cash_flows = [periodic_coupon] * periods
    cash_flows[-1] += face_value
    
    return macaulay_duration_from_cashflows(cash_flows, ytm, frequency)


def bond_modified_duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 2,
) -> float:
    """
    Calculate the Modified Duration of a fixed-coupon bond.

    Formula: **D_mod = D_mac / (1 + y/k)**

    Parameters
    ----------
    face_value : float
        Par value.
    coupon_rate : float
        Annual coupon rate.
    ytm : float
        Annual yield-to-maturity.
    years_to_maturity : float
        Remaining life in years.
    frequency : int, default 2
        Coupon payments per year.

    Returns
    -------
    float
        Modified duration in years.

    Raises
    ------
    ValueError
        If *face_value* or *years_to_maturity* is negative, or if *frequency* is non-positive.

    Examples
    --------
    >>> bond_modified_duration(1000, 0.04, 0.05, 5, 2)
    4.4581
    """
    if face_value < 0 or years_to_maturity < 0:
        raise ValueError("face_value and years_to_maturity must be non-negative.")
    if frequency <= 0:
        raise ValueError("frequency must be strictly positive.")

    macaulay = bond_macaulay_duration(
        face_value, coupon_rate, ytm, years_to_maturity, frequency
    )
    periodic_ytm = ytm / frequency
    return macaulay / (1 + periodic_ytm)


def macaulay_duration_from_cashflows(
    cashflows: ArrayLike,
    ytm: float,
    frequency: int = 1,
) -> float:
    """
    Macaulay Duration for an arbitrary stream of cashflows.

    Parameters
    ----------
    cashflows : ArrayLike
        Array of cashflows occurring at periods 1, 2, ..., N.
    ytm : float
        Annual yield-to-maturity to discount the cashflows.
    frequency : int, default 1
        Number of periods per year.

    Returns
    -------
    float
        Macaulay duration in years.
        
    Raises
    ------
    ValueError
        If *frequency* is non-positive.
        
    Examples
    --------
    >>> macaulay_duration_from_cashflows([5, 5, 105], ytm=0.05, frequency=1)
    2.859410431525997
    """
    if frequency <= 0:
        raise ValueError("frequency must be strictly positive.")
    
    cf = Finance._as_cashflows(cashflows)
    n = len(cf)
    periodic_ytm = ytm / frequency
    t = np.arange(1, n + 1)
    
    discount_factors = (1 + periodic_ytm) ** t
    pv_cf = cf / discount_factors
    price = pv_cf.sum()
    
    if price == 0:
        return 0.0
        
    weighted_t = t * pv_cf
    macaulay_periods = weighted_t.sum() / price
    
    return macaulay_periods / frequency
