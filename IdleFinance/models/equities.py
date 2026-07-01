"""
Comprehensive valuation models for fundamental analysis.

Covers dividend discount models, DCF, cost-of-capital, equity multiples,
earnings-based models, and fixed-income / credit.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from ..core._types import Optional, List, Union, ArrayLike, NumericOutput


# ═══════════════════════════════════════════════════════════════════════════
# Cost of Capital
# ═══════════════════════════════════════════════════════════════════════════

def cost_of_equity_capm(
    risk_free_rate: float,
    beta: float,
    market_return: float,
) -> float:
    """
    Estimate cost of equity via the Capital Asset Pricing Model (CAPM).

    Formula: **E[R_i] = R_f + β_i · (E[R_m] − R_f)**

    Parameters
    ----------
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.04 for 4 %).
    beta : float
        Equity beta relative to the market portfolio.
    market_return : float
        Expected annual market return.

    Returns
    -------
    float
        Required return on equity.

    Raises
    ------
    ValueError
        If *beta* is negative.

    Examples
    --------
    >>> cost_of_equity_capm(risk_free_rate=0.03, beta=1.2, market_return=0.10)
    0.114
    """
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    return risk_free_rate + beta * (market_return - risk_free_rate)


def cost_of_equity_build_up(
    risk_free_rate: float,
    equity_risk_premium: float,
    size_premium: float = 0.0,
    company_specific_risk_premium: float = 0.0,
    industry_risk_premium: float = 0.0,
    country_risk_premium: float = 0.0,
) -> float:
    """
    Estimate cost of equity using the Build-Up method.

    Formula: **E[k_e] = r_f + ERP + size_premium + company_specific_risk_premium + industry_risk_premium + country_risk_premium**

    Parameters
    ----------
    risk_free_rate : float
        Base risk-free rate.
    equity_risk_premium : float
        Market equity risk premium.
    size_premium : float, default 0
        Small-cap illiquidity premium.
    company_specific_risk_premium : float, default 0
        Idiosyncratic risk adjustment.
    industry_risk_premium : float, default 0
        Industry-specific risk adjustment.
    country_risk_premium : float, default 0
        Country-specific risk adjustment.

    Returns
    -------
    float
        Required return on equity.

    Examples
    --------
    >>> cost_of_equity_build_up(risk_free_rate=0.04, equity_risk_premium=0.06, size_premium=0.02)
    0.12
    """
    return (
        risk_free_rate
        + equity_risk_premium
        + size_premium
        + company_specific_risk_premium
        + industry_risk_premium
        + country_risk_premium
    )


def cost_of_debt(
    coupon_rate: float,
    tax_rate: float,
) -> float:
    """
    After-tax cost of debt.

    Formula: **r_d = coupon_rate · (1 − tax_rate)**

    Parameters
    ----------
    coupon_rate : float
        Pre-tax cost of debt (coupon / face value).
    tax_rate : float
        Marginal corporate tax rate.

    Returns
    -------
    float
        After-tax cost of debt.

    Raises
    ------
    ValueError
        If *tax_rate* is not between 0 and 1.

    Examples
    --------
    >>> cost_of_debt(coupon_rate=0.06, tax_rate=0.25)
    0.045
    """
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax_rate must be between 0 and 1.")
    return coupon_rate * (1 - tax_rate)


def wacc(
    market_equity: float,
    market_debt: float,
    cost_of_equity: float,
    cost_of_debt_pretax: float,
    tax_rate: float = 0.25,
    preferred_equity: float = 0.0,
    cost_of_preferred: float = 0.0,
) -> float:
    """
    Weighted Average Cost of Capital (WACC).

    Formula: **WACC = (E/V)·r_e + (D/V)·r_d·(1−t) + (P/V)·r_p**

    Parameters
    ----------
    market_equity : float
        Market capitalisation.
    market_debt : float
        Market value of interest-bearing debt.
    cost_of_equity : float
        Required return on equity (r_e).
    cost_of_debt_pretax : float
        Pre-tax cost of debt (r_d).
    tax_rate : float, default 0.25
        Corporate tax rate.
    preferred_equity : float, default 0
        Market value of preferred equity.
    cost_of_preferred : float, default 0
        Required return on preferred equity.

    Returns
    -------
    float
        Weighted Average Cost of Capital.

    Raises
    ------
    ValueError
        If *tax_rate* is not between 0 and 1.

    Examples
    --------
    >>> wacc(market_equity=1000, market_debt=500, cost_of_equity=0.10, cost_of_debt_pretax=0.06, tax_rate=0.25)
    0.08166666666666668
    """
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax_rate must be between 0 and 1.")

    V = market_equity + market_debt + preferred_equity
    if V <= 0:
        raise ValueError("Total firm value must be positive.")
    return (
        (market_equity / V) * cost_of_equity
        + (market_debt / V) * cost_of_debt_pretax * (1 - tax_rate)
        + (preferred_equity / V) * cost_of_preferred
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dividend Discount Models
# ═══════════════════════════════════════════════════════════════════════════

def gordon_growth_model(
    dividend: float,
    cost_of_equity: float,
    growth_rate: float,
) -> float:
    """
    Intrinsic value via the Gordon Growth Model (constant-growth DDM).

    Formula: **P = D₁ / (r − g)** where **D₁ = D₀ · (1 + g)**

    Parameters
    ----------
    dividend : float
        Current dividend per share (D₀).
    cost_of_equity : float
        Required return on equity (r). Must exceed *growth_rate*.
    growth_rate : float
        Constant perpetual dividend growth rate (g).

    Returns
    -------
    float
        Intrinsic value per share.

    Raises
    ------
    ValueError
        If *dividend* or *cost_of_equity* is negative, or if *cost_of_equity*
        does not strictly exceed *growth_rate*.

    Examples
    --------
    >>> gordon_growth_model(dividend=2.0, cost_of_equity=0.10, growth_rate=0.05)
    42.0
    """
    if dividend < 0:
        raise ValueError("dividend must be non-negative.")
    if cost_of_equity < 0:
        raise ValueError("cost_of_equity must be non-negative.")
    if cost_of_equity <= growth_rate:
        raise ValueError("cost_of_equity must exceed growth_rate to yield a finite value.")
    return dividend * (1 + growth_rate) / (cost_of_equity - growth_rate)


def h_model(
    dividend: float,
    cost_of_equity: float,
    long_term_growth: float,
    short_term_growth: float,
    high_growth_period: int,
) -> float:
    """
    Intrinsic value via the H-Model (linearly declining two-stage DDM).

    Formula: **V = D₀·(1+g_L)/(r−g_L) + D₀·H·(g_S−g_L)/(r−g_L)**
    where **H = high_growth_period / 2**.

    Parameters
    ----------
    dividend : float
        Current dividend per share (D₀).
    cost_of_equity : float
        Required return on equity (r).
    long_term_growth : float
        Terminal stable growth rate (g_L).
    short_term_growth : float
        Initial high growth rate (g_S).
    high_growth_period : int
        Total length of high-growth phase (years).

    Returns
    -------
    float
        Intrinsic value per share.

    Raises
    ------
    ValueError
        If *dividend* or *high_growth_period* is negative, or if *cost_of_equity*
        does not strictly exceed *long_term_growth*.

    Examples
    --------
    >>> h_model(dividend=2.0, cost_of_equity=0.10, long_term_growth=0.03, short_term_growth=0.08, high_growth_period=10)
    36.57142857142857
    """
    if dividend < 0:
        raise ValueError("dividend must be non-negative.")
    if high_growth_period < 0:
        raise ValueError("high_growth_period must be non-negative.")
    h = high_growth_period / 2
    spread = cost_of_equity - long_term_growth
    if spread <= 0:
        raise ValueError("cost_of_equity must exceed long_term_growth to yield a finite value.")
    return (
        dividend * (1 + long_term_growth) / spread
        + dividend * h * (short_term_growth - long_term_growth) / spread
    )


def two_stage_ddm(
    dividend: float,
    cost_of_equity: float,
    high_growth_rate: float,
    stable_growth_rate: float,
    high_growth_years: int,
) -> float:
    """
    Intrinsic value via a two-stage Dividend Discount Model.

    Stage 1: Explicit dividends grow at *high_growth_rate* for *high_growth_years*.
    Stage 2: GGM terminal value at *stable_growth_rate*.

    Parameters
    ----------
    dividend : float
        Current dividend per share (D₀).
    cost_of_equity : float
        Required return on equity (r).
    high_growth_rate : float
        Dividend growth rate during Stage 1.
    stable_growth_rate : float
        Terminal dividend growth rate (Stage 2).
    high_growth_years : int
        Duration of Stage 1 (years).

    Returns
    -------
    float
        Intrinsic value per share.

    Raises
    ------
    ValueError
        If *dividend* or *high_growth_years* is negative, or if *cost_of_equity*
        does not strictly exceed *stable_growth_rate*.

    Examples
    --------
    >>> two_stage_ddm(dividend=2.0, cost_of_equity=0.10, high_growth_rate=0.08, stable_growth_rate=0.03, high_growth_years=5)
    36.31599388145781
    """
    if dividend < 0:
        raise ValueError("dividend must be non-negative.")
    if high_growth_years < 0:
        raise ValueError("high_growth_years must be non-negative.")
    if cost_of_equity <= stable_growth_rate:
        raise ValueError("cost_of_equity must exceed stable_growth_rate.")

    n = int(high_growth_years)
    t = np.arange(1, n + 1)
    # Vectorised dividend PV: D0 * (1+g)^t / (1+r)^t
    growth_factors = (1.0 + high_growth_rate) ** t
    discount_factors = (1.0 + cost_of_equity) ** t
    pv_dividends = float(np.sum(dividend * growth_factors / discount_factors))

    # Terminal value at end of Stage 1
    d_last = dividend * (1.0 + high_growth_rate) ** n
    d_terminal = d_last * (1.0 + stable_growth_rate)
    tv = d_terminal / (cost_of_equity - stable_growth_rate)
    pv_tv = tv / (1.0 + cost_of_equity) ** n

    return pv_dividends + pv_tv


def three_stage_ddm(
    dividend: float,
    cost_of_equity: float,
    high_growth_rate: float,
    stable_growth_rate: float,
    high_growth_years: int,
    transition_years: int,
) -> float:
    """
    Intrinsic value via a three-stage Dividend Discount Model.

    Stage 1: High growth for *high_growth_years* years.
    Stage 2: Linear transition from high to stable over *transition_years* years.
    Stage 3: GGM terminal value at *stable_growth_rate*.

    Parameters
    ----------
    dividend : float
        Current dividend per share (D₀).
    cost_of_equity : float
        Required return on equity.
    high_growth_rate : float
        Stage 1 growth rate.
    stable_growth_rate : float
        Stage 3 (terminal) growth rate.
    high_growth_years : int
        Length of Stage 1.
    transition_years : int
        Length of Stage 2 (linear decline).

    Returns
    -------
    float
        Intrinsic value per share.

    Examples
    --------
    >>> three_stage_ddm(dividend=2.0, cost_of_equity=0.10, high_growth_rate=0.12, stable_growth_rate=0.03, high_growth_years=5, transition_years=5)
    48.34960320473752
    """
    if dividend < 0:
        raise ValueError("dividend must be non-negative.")
    if high_growth_years < 0 or transition_years < 0:
        raise ValueError("years parameters must be non-negative.")
    if cost_of_equity <= stable_growth_rate:
        raise ValueError("cost_of_equity must exceed stable_growth_rate.")

    n1 = int(high_growth_years)
    n2 = int(transition_years)

    # Stage 1: constant high growth
    growth_1 = np.full(n1, 1 + high_growth_rate)

    # Stage 2: linearly declining growth rate
    if n2 > 0:
        steps = np.arange(1, n2 + 1)
        g2 = high_growth_rate - (high_growth_rate - stable_growth_rate) * (steps / n2)
        growth_2 = 1 + g2
    else:
        growth_2 = np.array([])

    growth_factors = np.concatenate([growth_1, growth_2])
    d_levels = dividend * np.cumprod(growth_factors)
    t = np.arange(1, n1 + n2 + 1)
    pv = float(np.sum(d_levels / (1 + cost_of_equity) ** t))

    # Stage 3 terminal value
    d_last = d_levels[-1] if len(d_levels) else dividend
    d_terminal = d_last * (1 + stable_growth_rate)
    tv = d_terminal / (cost_of_equity - stable_growth_rate)
    pv += tv / (1 + cost_of_equity) ** (n1 + n2)

    return pv


# ═══════════════════════════════════════════════════════════════════════════
# Discounted Cash Flow (DCF)
# ═══════════════════════════════════════════════════════════════════════════

def dcf_valuation(
    free_cash_flows: ArrayLike,
    discount_rate: float,
    terminal_growth: float,
    shares_outstanding: Optional[float] = None,
    net_debt: float = 0.0,
) -> Union[float, dict]:
    """
    Multi-stage Discounted Cash Flow valuation.

    Stage 1: PV of explicit FCF projections.
    Stage 2: Gordon-Growth terminal value on the last projected FCF.

    Parameters
    ----------
    free_cash_flows : ArrayLike
        Projected FCFs for years 1 … N (in the same currency units).
    discount_rate : float
        WACC or required return.
    terminal_growth : float
        Perpetual growth rate applied to the last FCF.
    shares_outstanding : float, optional
        If provided, returns ``{'enterprise_value', 'equity_value', 'price_per_share'}``.
    net_debt : float, default 0
        Net debt (debt − cash). Subtracted from EV to get equity value.

    Returns
    -------
    float or dict
        Enterprise value (float) or full breakdown dict when *shares_outstanding* given.

    Raises
    ------
    ValueError
        If *discount_rate* ≤ *terminal_growth*.

    Examples
    --------
    >>> dcf_valuation(free_cash_flows=[100, 110, 120, 130, 140], discount_rate=0.10, terminal_growth=0.02)
    1556.0350346383616
    """
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must exceed terminal_growth.")

    fcfs = np.asarray(free_cash_flows, dtype=float)
    years = np.arange(1, len(fcfs) + 1)

    pv_explicit = float(np.sum(fcfs / (1 + discount_rate) ** years))

    last_fcf = fcfs[-1]
    n = len(fcfs)
    terminal_value = last_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** n

    enterprise_value = pv_explicit + pv_terminal

    if shares_outstanding is None:
        return enterprise_value

    equity_value = enterprise_value - net_debt
    price_per_share = equity_value / shares_outstanding
    return {
        "enterprise_value": enterprise_value,
        "pv_explicit_fcf": pv_explicit,
        "pv_terminal_value": pv_terminal,
        "terminal_value": terminal_value,
        "equity_value": equity_value,
        "price_per_share": price_per_share,
    }


def dcf_sensitivity(
    free_cash_flows: ArrayLike,
    discount_rates: ArrayLike,
    terminal_growths: ArrayLike,
    net_debt: float = 0.0,
    shares_outstanding: Optional[float] = None,
) -> pd.DataFrame:
    """
    Two-dimensional DCF sensitivity table (WACC × terminal growth).

    Parameters
    ----------
    free_cash_flows : ArrayLike
        Projected FCFs.
    discount_rates : ArrayLike
        Range of discount rates (rows).
    terminal_growths : ArrayLike
        Range of terminal growth rates (columns).
    net_debt : float, default 0
        Subtracted from EV when computing per-share price.
    shares_outstanding : float, optional
        When provided, cells show implied share price; otherwise EV.

    Returns
    -------
    pd.DataFrame
        Rows = discount rates, columns = terminal growth rates.

    Examples
    --------
    >>> dcf_sensitivity([100, 110], [0.08, 0.10], [0.02, 0.03])
    """
    waccs = np.asarray(discount_rates, dtype=float)
    tgs = np.asarray(terminal_growths, dtype=float)
    table = np.zeros((len(waccs), len(tgs)))

    for i, w in enumerate(waccs):
        for j, g in enumerate(tgs):
            if w <= g:
                table[i, j] = np.nan
                continue
            result = dcf_valuation(
                free_cash_flows, w, g,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
            )
            if isinstance(result, dict):
                table[i, j] = result["price_per_share"]
            else:
                table[i, j] = result

    return pd.DataFrame(
        table,
        index=pd.Index([f"{w:.1%}" for w in waccs], name="WACC"),
        columns=pd.Index([f"{g:.1%}" for g in tgs], name="Terminal Growth"),
    )


def reverse_dcf(
    current_price: float,
    free_cash_flows: ArrayLike,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
    bracket: tuple = (0.01, 0.50),
) -> float:
    """
    Solve for the implied WACC that equates the DCF value to the current price.

    Parameters
    ----------
    current_price : float
        Observed market price per share.
    free_cash_flows : ArrayLike
        Projected FCFs.
    terminal_growth : float
        Terminal growth rate assumption.
    shares_outstanding : float
        Number of shares.
    net_debt : float, default 0
        Net debt subtracted to get equity value.
    bracket : tuple, default (0.01, 0.50)
        Search interval for WACC.

    Returns
    -------
    float
        Implied WACC (the market's embedded discount rate).

    Raises
    ------
    ValueError
        If *current_price* is negative, *shares_outstanding* is non-positive,
        or if Brent's method fails to find a root within the bracket.

    Examples
    --------
    >>> reverse_dcf(current_price=16.065, free_cash_flows=[100, 110, 120, 130, 140], terminal_growth=0.02, shares_outstanding=100)
    0.09756680055194629
    """
    if current_price < 0 or shares_outstanding <= 0:
        raise ValueError("current_price must be non-negative and shares_outstanding must be positive.")
    target_ev = current_price * shares_outstanding + net_debt

    def objective(wacc_guess):
        try:
            ev = dcf_valuation(free_cash_flows, wacc_guess, terminal_growth)
        except ValueError:
            return np.inf
        return float(ev) - target_ev

    try:
        return brentq(objective, bracket[0], bracket[1])
    except ValueError:
        raise ValueError(
            f"Could not solve for WACC in bracket {bracket}. "
            "Try widening the bracket or checking inputs."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Earnings-Based Models
# ═══════════════════════════════════════════════════════════════════════════

def earnings_power_value(
    earnings: float,
    cost_of_capital: float,
    tax_adjustment: float = 0.0,
) -> float:
    """
    Earnings Power Value (EPV) — zero-growth perpetuity.

    Formula: **EPV = Normalised Earnings / Cost of Capital**

    Parameters
    ----------
    earnings : float
        Sustainable, normalised after-tax earnings (EBIT × (1−tax) or net income).
    cost_of_capital : float
        WACC or cost of equity.
    tax_adjustment : float, default 0
        Additional tax adjustment on reported earnings.

    Returns
    -------
    float
        Intrinsic value under a no-growth assumption.

    Raises
    ------
    ValueError
        If *cost_of_capital* is not strictly positive, or if *tax_adjustment*
        is not between 0 and 1.

    Examples
    --------
    >>> earnings_power_value(earnings=5.0, cost_of_capital=0.10)
    50.0
    """
    if cost_of_capital <= 0:
        raise ValueError("cost_of_capital must be strictly positive.")
    if not 0 <= tax_adjustment <= 1:
        raise ValueError("tax_adjustment must be between 0 and 1.")
    adjusted = earnings * (1 - tax_adjustment)
    return adjusted / cost_of_capital


def residual_income_model(
    book_value: float,
    earnings_per_share: ArrayLike,
    cost_of_equity: float,
    terminal_growth: float = 0.0,
) -> float:
    """
    Intrinsic value via the Residual Income Model (Edwards-Bell-Ohlson).

    Formula: **V = BV₀ + Σ [RI_t / (1+r)^t] + TV**
    where **RI_t = EPS_t − r · BV_{t-1}**

    Parameters
    ----------
    book_value : float
        Current book value per share (BV₀).
    earnings_per_share : ArrayLike
        Projected EPS for the explicit forecast period.
    cost_of_equity : float
        Required return on equity (r).
    terminal_growth : float, default 0
        Perpetual growth in residual income after the forecast period.

    Returns
    -------
    float
        Intrinsic value per share.

    Examples
    --------
    >>> residual_income_model(book_value=10.0, earnings_per_share=[2.0, 2.5], cost_of_equity=0.10)
    22.479338842975203
    """
    if cost_of_equity <= 0:
        raise ValueError("cost_of_equity must be strictly positive.")
    if book_value < 0:
        raise ValueError("book_value must be non-negative.")
    eps = np.asarray(earnings_per_share, dtype=float)
    bv = float(book_value)
    pv_ri = 0.0
    current_bv = bv

    for t, e in enumerate(eps, start=1):
        ri = e - cost_of_equity * current_bv
        pv_ri += ri / (1 + cost_of_equity) ** t
        current_bv = current_bv + e - (cost_of_equity * current_bv)  # clean-surplus

    # Terminal residual income (perpetuity)
    last_ri = eps[-1] - cost_of_equity * current_bv
    if cost_of_equity > terminal_growth and last_ri != 0:
        tv = last_ri * (1 + terminal_growth) / (cost_of_equity - terminal_growth)
        pv_tv = tv / (1 + cost_of_equity) ** len(eps)
    else:
        pv_tv = 0.0

    return bv + pv_ri + pv_tv


def abnormal_earnings_growth(
    eps_base: float,
    eps_projections: ArrayLike,
    dps_projections: ArrayLike,
    cost_of_equity: float,
) -> float:
    """
    Intrinsic value via the Abnormal Earnings Growth (AEG / Ohlson-Juettner) model.

    Capitalises the core EPS (assuming no growth) plus the PV of abnormal
    earnings growth (AEG = next-year EPS − (1+r)·dividends reinvested at r).

    Parameters
    ----------
    eps_base : float
        Base-year EPS (year 1 'normal' earnings).
    eps_projections : ArrayLike
        Projected EPS for years 2 … N.
    dps_projections : ArrayLike
        Projected dividends per share for years 1 … N−1 (reinvested at r).
    cost_of_equity : float
        Required return on equity.

    Returns
    -------
    float
        Intrinsic value per share.

    Examples
    --------
    >>> abnormal_earnings_growth(eps_base=2.0, eps_projections=[2.2, 2.42], dps_projections=[1.0, 1.1], cost_of_equity=0.10)
    18.181818181818176
    """
    if cost_of_equity <= 0:
        raise ValueError("cost_of_equity must be strictly positive.")

    eps = np.asarray(eps_projections, dtype=float)
    dps = np.asarray(dps_projections, dtype=float)

    n = min(len(eps), len(dps))
    eps_n, dps_n = eps[:n], dps[:n]
    prev_eps = np.concatenate(([eps_base], eps_n[:-1]))
    cum_div = dps_n * cost_of_equity           # dividends reinvested
    aeg = eps_n - prev_eps * (1 + cost_of_equity) - cum_div
    t = np.arange(1, n + 1)
    pv_aeg = float(np.sum(aeg / (1 + cost_of_equity) ** t))

    return (eps_base + pv_aeg) / cost_of_equity


# ═══════════════════════════════════════════════════════════════════════════
# Equity Multiples
# ═══════════════════════════════════════════════════════════════════════════

def pe_implied_price(
    earnings_per_share: float,
    pe_ratio: float,
) -> float:
    """
    Implied share price from a P/E multiple.

    Formula: **Price = EPS × P/E**

    Parameters
    ----------
    earnings_per_share : float
        Trailing or forward EPS.
    pe_ratio : float
        Applied P/E multiple.

    Returns
    -------
    float
        Implied price per share.

    Examples
    --------
    >>> pe_implied_price(earnings_per_share=2.5, pe_ratio=15.0)
    37.5
    """
    if pe_ratio < 0:
        raise ValueError("pe_ratio must be non-negative.")
    return earnings_per_share * pe_ratio


def ev_ebitda_implied_price(
    ebitda: float,
    ev_ebitda_multiple: float,
    net_debt: float,
    shares_outstanding: float,
) -> float:
    """
    Implied share price from an EV/EBITDA multiple.

    Formula: **EV = EBITDA × multiple → Price = (EV − Net Debt) / Shares**

    Parameters
    ----------
    ebitda : float
        Earnings Before Interest, Taxes, Depreciation & Amortisation.
    ev_ebitda_multiple : float
        Applied EV/EBITDA multiple.
    net_debt : float
        Debt minus cash (subtracted to convert EV to equity value).
    shares_outstanding : float
        Number of diluted shares.

    Returns
    -------
    float
        Implied price per share.

    Examples
    --------
    >>> ev_ebitda_implied_price(ebitda=100.0, ev_ebitda_multiple=10.0, net_debt=200.0, shares_outstanding=50.0)
    16.0
    """
    if ev_ebitda_multiple < 0:
        raise ValueError("ev_ebitda_multiple must be non-negative.")
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be strictly positive.")
    ev = ebitda * ev_ebitda_multiple
    equity_value = ev - net_debt
    return equity_value / shares_outstanding


def price_to_book_implied(
    book_value_per_share: float,
    pb_ratio: float,
) -> float:
    """
    Implied share price from a Price-to-Book multiple.

    Formula: **Price = BV/share × P/B**

    Parameters
    ----------
    book_value_per_share : float
        Book value per share.
    pb_ratio : float
        Applied P/B multiple.

    Returns
    -------
    float
        Implied price per share.

    Examples
    --------
    >>> price_to_book_implied(book_value_per_share=12.0, pb_ratio=1.5)
    18.0
    """
    if pb_ratio < 0:
        raise ValueError("pb_ratio must be non-negative.")
    return book_value_per_share * pb_ratio
