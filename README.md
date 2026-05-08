# IdleFinance

**IdleFinance** is a Python library for quantitative finance that embeds portfolio theory, risk modeling, valuation, and asset pricing tools directly into the pandas ecosystem via a seamless `.finance` accessor.

```python
pip install IdleFinance
```

With optional Excel export support:

```python
pip install "IdleFinance[excel]"
```

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Financial Math](#core-financial-math)
4. [Pandas Accessor](#pandas-accessor)
5. [Expected Returns](#expected-returns)
6. [Covariance Estimation](#covariance-estimation)
7. [Portfolio Optimization](#portfolio-optimization)
8. [Risk Metrics](#risk-metrics)
9. [Fixed Income](#fixed-income)
10. [Options Pricing](#options-pricing)
11. [Equity Valuation](#equity-valuation)
12. [Stochastic Simulations](#stochastic-simulations)
13. [DCF Excel Export](#dcf-excel-export)
14. [API Reference](#api-reference)
15. [Type Aliases](#type-aliases)

---

## Overview

IdleFinance is organized into three layers:

| Layer | Entry point | What it covers |
|-------|-------------|----------------|
| **Accessor** | `df.finance.*` / `series.finance.*` | Returns, risk metrics, BL, covariance — all from raw price DataFrames |
| **Models** | `idf.risk_metrics`, `idf.fixed_income`, etc. | Standalone functions for every domain |
| **Utils** | `idf.fv`, `idf.npv`, `idf.irr`, … | Top-level time-value-of-money shortcuts |

---

## Quick Start

```python
import pandas as pd
import IdleFinance as idf

# Any price DataFrame — columns are assets, index is dates
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

# Single-line portfolio analytics
ret, cov, weights = prices.finance.black_litterman(
    views={"AAPL": 0.15, "MSFT": 0.10},
    view_confidences=[0.8, 0.6],
    bounds=(0.0, 0.30),
)
print(weights)
```

---

## Core Financial Math

Top-level shortcuts for time value of money, NPV/IRR, and loan math.

```python
import IdleFinance as idf

# Future / Present Value
idf.fv(1000, rate=0.05, time=3)          # → 1157.63
idf.pv(1157.63, rate=0.05, time=3)       # → 1000.00

# NPV & IRR
cashflows = [-1000, 300, 400, 500, 200]
idf.npv(cashflows, rate=0.10)            # → 154.89
idf.irr(cashflows)                       # → 0.2094  (20.94%)

# Payback period
idf.payback_period([-1000, 400, 400, 400])  # → 2.5 years

# Loan payment (PMT)
idf.loan_payment(10_000, rate=0.08, periods=10)  # → 1490.29 / period

# Profitability Index
idf.profitability_index(cashflows, rate=0.10)    # → 1.155

# Effective annual rate
idf.effective_annual_rate(period_rate=0.01, periods_per_year=12)  # → 12.68%

# Compound / simple interest
idf.compound_interest(1000, rate=0.05, time=3)   # → 1157.63
idf.simple_interest(1000, rate=0.05, time=3)     # → 1150.00
```

### Reference

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `fv(principal, rate, time, n=1)` | principal: float, rate: float, time: float, n: int | float | Future value with compound interest |
| `pv(future_value, rate, time, n=1)` | future_value: float, rate: float, time: float, n: int | float | Present value with compound interest |
| `npv(cashflows, rate)` | cashflows: list, rate: float | float | Net present value of cashflow stream |
| `irr(cashflows, guess=0.1)` | cashflows: list, guess: float | float | Internal rate of return (Newton-Raphson) |
| `payback_period(cashflows)` | cashflows: list | float | Payback period with linear interpolation |
| `profitability_index(cashflows, rate)` | cashflows: list, rate: float | float | PV of future flows / initial outlay |
| `effective_annual_rate(period_rate, periods_per_year)` | period_rate: float, periods_per_year: int | float | EAR from periodic rate |
| `compound_interest(principal, rate, time, n=1)` | | float | Ending balance with compound interest |
| `simple_interest(principal, rate, time)` | | float | Ending balance with simple interest |
| `annuity_payment(present_value, rate, periods)` | | float | Fixed payment for ordinary annuity |
| `loan_payment(principal, rate, periods)` | | float | Fixed amortizing loan payment (PMT) |

---

## Pandas Accessor

Both `pd.DataFrame` and `pd.Series` gain a `.finance` accessor after importing IdleFinance.

```python
import IdleFinance as idf  # registers .finance on DataFrame and Series
```

### DataFrame accessor

Expects a price DataFrame with assets as columns and dates as index.

```python
# Returns
returns = prices.finance.returns()                   # simple returns
log_ret = prices.finance.returns(log_returns=True)   # log returns

# Risk measures
cov   = prices.finance.covariance()                  # sample covariance
corr  = prices.finance.correlation()
vol   = prices.finance.volatility()                  # annualized vol per asset
ann_r = prices.finance.annualized_return()
sr    = prices.finance.sharpe_ratio(risk_free_rate=0.03)
so    = prices.finance.sortino_ratio()
mdd   = prices.finance.max_drawdown()
dd    = prices.finance.drawdown()                    # full drawdown series
rvol  = prices.finance.rolling_volatility(window=20)

# Expected returns (see Expected Returns section)
mu_hist = prices.finance.expected_returns(method="mean_historical")
mu_ema  = prices.finance.expected_returns(method="ema_historical")
mu_capm = prices.finance.expected_returns(method="capm")

# Market-implied risk aversion
lam = prices.finance.risk_aversion(risk_free_rate=0.02)

# Market prior returns (reverse optimization)
market_weights = pd.Series([0.3, 0.2, 0.2, 0.2, 0.1], index=prices.columns)
pi = prices.finance.market_prior_returns(market_weights, risk_aversion=2.5)

# Black-Litterman (see Portfolio Optimization section)
post_ret, post_cov, weights = prices.finance.black_litterman(
    views={"AAPL": 0.15},
    bounds=(0.0, 0.30),
)
```

**DataFrame accessor method reference**

| Method | Key Parameters | Returns | Description |
|--------|---------------|---------|-------------|
| `returns(log_returns, returns_data)` | log_returns: bool=False | DataFrame | Simple or log returns |
| `correlation(returns_data)` | | DataFrame | Correlation matrix |
| `cumulative_returns(log_returns, returns_data)` | | DataFrame | Compounded return index |
| `drawdown(returns_data)` | | DataFrame | Drawdown series |
| `max_drawdown(returns_data)` | | Series | Max drawdown per asset |
| `rolling_volatility(window, annualized, frequency)` | window: int=20 | DataFrame | Rolling volatility |
| `covariance(returns_data, method, **kwargs)` | method: "sample"\|"ema" | DataFrame | Covariance matrix |
| `expected_returns(method, returns_data, **kwargs)` | method: str | Series | Expected returns |
| `annualized_return(frequency, returns_data)` | frequency: int=252 | Series | Annualized return |
| `volatility(annualized, frequency, returns_data)` | | Series | Volatility |
| `sharpe_ratio(risk_free_rate, frequency)` | rf: float=0.03 | Series | Sharpe ratio |
| `sortino_ratio(risk_free_rate, target_return, frequency)` | rf: float=0.03 | Series | Sortino ratio |
| `risk_aversion(risk_free_rate, frequency)` | rf: float=0.02 | float | Market-implied λ |
| `market_prior_returns(market_weights, risk_aversion, cov_method)` | | Series/array | Implied equilibrium returns |
| `black_litterman(...)` | see Portfolio section | (Series, DataFrame, Series) | Full BL optimization |
| `black_litterman_weights(posterior_returns, posterior_cov, ...)` | | Series | Weights from pre-computed posterior |

### Series accessor

Expects a price or return Series.

```python
series = prices["AAPL"]

ann   = series.finance.annualized_return()
vol   = series.finance.volatility()
sr    = series.finance.sharpe_ratio()
so    = series.finance.sortino_ratio()
mdd   = series.finance.max_drawdown()
dd    = series.finance.drawdown()
cumr  = series.finance.cumulative_returns()
ret   = series.finance.returns()

# Single-asset Black-Litterman
post_ret, post_var, weight = series.finance.black_litterman(
    view=0.12, view_confidence=0.8
)
```

**Series accessor method reference**

| Method | Key Parameters | Returns | Description |
|--------|---------------|---------|-------------|
| `returns(log_returns, returns_data)` | | Series | Simple or log returns |
| `cumulative_returns(log_returns, returns_data)` | | Series | Compounded return index |
| `drawdown(from_returns)` | | Series | Drawdown series |
| `max_drawdown(from_returns)` | | float | Maximum drawdown |
| `annualized_return(frequency)` | frequency: int=252 | float | Annualized return |
| `volatility(annualized, frequency)` | | float | Volatility |
| `sharpe_ratio(risk_free_rate, frequency)` | rf: float=0.03 | float | Sharpe ratio |
| `sortino_ratio(risk_free_rate, target_return, frequency)` | | float | Sortino ratio |
| `risk_aversion(risk_free_rate, frequency)` | | float | Market-implied λ |
| `black_litterman(prior_return, view, view_confidence, tau_val, risk_aversion)` | | (float, float, float) | Single-asset BL |

---

## Expected Returns

Three estimation methods are available via the accessor or standalone functions.

```python
import IdleFinance as idf

# Via accessor
mu = prices.finance.expected_returns(method="mean_historical")
mu = prices.finance.expected_returns(method="ema_historical", span=500)
mu = prices.finance.expected_returns(method="capm", risk_free_rate=0.03)

# Standalone functions
mu = idf.historical_mean(prices)
mu = idf.ewma_return(prices, span=500)
mu = idf.capm_return(prices, risk_free_rate=0.03)

# Convert prices → returns
ret = idf.to_returns(prices)
ret = idf.to_returns(prices, log_returns=True)
```

| Function | Key Parameters | Description |
|----------|---------------|-------------|
| `to_returns(prices, log_returns=False)` | log_returns: bool | Period returns from price data |
| `historical_mean(prices, compounding=True, frequency=252)` | | Annualized historical mean (CAGR) |
| `ewma_return(prices, span=500, frequency=252)` | span: int | EW mean from exponentially-decayed returns |
| `capm_return(prices, market_prices=None, risk_free_rate=0.0, frequency=252)` | | CAPM expected returns (β × MRP + rf) |

---

## Covariance Estimation

```python
import IdleFinance as idf

# Via accessor
cov = prices.finance.covariance()                      # sample
cov = prices.finance.covariance(method="ema", span=90) # EWMA

# Standalone
cov = idf.covariances.sample_covariance(prices)
cov = idf.covariances.exponential_covariance(prices, span=90)

# Denoised (Marchenko-Pastur eigenvalue clipping)
cov = idf.covariances.covariance(prices, denoise=True, n_obs=252)

# Or in two steps
raw_cov = idf.covariances.sample_covariance(prices)
den_cov = idf.covariances.denoise_covariance(raw_cov, n_obs=252)
```

| Function | Key Parameters | Description |
|----------|---------------|-------------|
| `sample_covariance(prices, annualized=True, frequency=252)` | | Standard sample covariance |
| `exponential_covariance(prices, span=180, annualized=True, frequency=252)` | span: int | EWMA covariance with exponential decay |
| `covariance(prices, method="sample", denoise=False, n_obs=None)` | method: "sample"\|"ema" | Dispatcher — optionally denoises via MP |
| `denoise_covariance(cov, n_obs, method="marchenko_pastur")` | | Clip noise eigenvalues using MP theorem |

---

## Portfolio Optimization

### Black-Litterman

The full Walters/Idzorek model blends a market equilibrium prior with subjective views.

```python
# Minimal call — market equilibrium prior via reverse optimization
post_ret, post_cov, weights = prices.finance.black_litterman(
    views={"AAPL": 0.15},
)

# With investor confidence and per-asset bounds
post_ret, post_cov, weights = prices.finance.black_litterman(
    views={"AAPL": 0.15, "MSFT": 0.10},
    view_confidences=[0.8, 0.6],
    bounds=(0.0, 0.30),        # long-only, max 30% per asset
)

# Relative views (long AAPL vs short GOOGL by 5%)
post_ret, post_cov, weights = prices.finance.black_litterman(
    views={("AAPL", "GOOGL"): 0.05},
    relative_views=True,
)

# Partial allocation — 80% in risky assets, 20% in cash earning risk_free_rate
post_ret, post_cov, weights = prices.finance.black_litterman(
    views={"AAPL": 0.15},
    bounds=(0.0, 1.0),
    target_sum=0.80,
    risk_free_rate=0.04,
)

# Auto-determine optimal risky allocation (no sum constraint)
post_ret, post_cov, weights = prices.finance.black_litterman(
    views={"AAPL": 0.15},
    target_sum=None,
    risk_free_rate=0.04,
)

# Allow short positions
post_ret, post_cov, weights = prices.finance.black_litterman(
    views={"AAPL": 0.15},
    bounds=(-0.20, 0.50),      # up to 20% short, up to 50% long
    target_sum=1.0,
)

# Tracking-error minimization vs benchmark
bench = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=prices.columns)
post_ret, post_cov, weights = prices.finance.black_litterman(
    views={"AAPL": 0.15},
    objective="tracking_error",
    benchmark_weights=bench,
)
```

**`black_litterman()` parameter reference**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `views` | dict / Series | None | Absolute: `{ticker: return}`. Relative: `{(long, short): spread}` |
| `prior_returns` | ArrayLike | None | Prior Π. Auto-computed via reverse optimization if None |
| `market_weights` | ArrayLike | None | Market-cap weights for reverse optimization (overrides `prior_returns`) |
| `view_confidences` | ArrayLike | None | Confidence [0, 1] per view (Idzorek method) |
| `relative_views` | bool | False | Interpret view keys as (long, short) pairs |
| `tau` | float | None | Override τ directly |
| `tau_method` | "default" \| "variance" | "default" | τ derivation method |
| `omega_method` | "idzorek" \| "proportional" | "idzorek" | View uncertainty method |
| `risk_aversion` | float | 1.0 | Risk-aversion λ for optimization |
| `cov_matrix` | MatrixLike | None | Override covariance matrix |
| `cov_method` | "sample" \| "ema" | "sample" | Covariance estimation method |
| `bounds` | (lo, hi) or list | None | Per-asset weight bounds. Single tuple applied to all assets |
| `objective` | "utility" \| "tracking_error" | "utility" | Optimization objective |
| `benchmark_weights` | ArrayLike | None | Required for tracking_error objective |
| `custom_constraints` | list[dict] | None | Extra scipy constraint dicts |
| `target_sum` | float \| None | 1.0 | Risky weight sum. `None` = unconstrained (cash earns rf) |
| `risk_free_rate` | float | 0.0 | Cash/risk-free rate for utility and partial allocation |

**Allocation modes summary**

| Goal | `target_sum` | `bounds` |
|------|-------------|---------|
| Fully invested, long-only | `1.0` (default) | `(0, max)` |
| Partial allocation (X% risky, rest in cash) | `0.8` | `(0, 1.0)` |
| Auto-determine optimal risky exposure | `None` | None or your bounds |
| Long/short (130/30 style) | `1.0` | `(-0.30, 1.30)` |
| Leveraged | `1.5` | None or per-asset |

### Efficient Frontier

```python
import IdleFinance as idf

mu  = prices.finance.expected_returns()
cov = prices.finance.covariance()

# Global Minimum Variance portfolio
w_gmv = idf.efficient_frontier.min_variance(cov)

# Maximum Sharpe (tangency) portfolio
w_msr = idf.efficient_frontier.max_sharpe(mu, cov, risk_free_rate=0.03)

# Minimum-variance portfolio for a specific target return
w_eff = idf.efficient_frontier.efficient_return(mu, cov, target_return=0.12)

# Full frontier curve (returns DataFrame with columns: Return, Volatility, Sharpe)
frontier = idf.efficient_frontier.efficient_frontier(mu, cov, n_points=100, risk_free_rate=0.03)

# Portfolio performance given weights
ret, vol, sharpe = idf.efficient_frontier.portfolio_performance(w_msr, mu, cov, risk_free_rate=0.03)
```

| Function | Key Parameters | Returns | Description |
|----------|---------------|---------|-------------|
| `portfolio_performance(weights, expected_returns, cov, risk_free_rate=0.0)` | | (return, vol, sharpe) | Return, volatility, Sharpe for given weights |
| `min_variance(cov, bounds=None, target_sum=1.0)` | | ndarray | Global minimum variance weights |
| `max_sharpe(mu, cov, risk_free_rate=0.0, bounds=None, target_sum=1.0)` | | ndarray | Tangency portfolio weights |
| `efficient_return(mu, cov, target_return, bounds=None, target_sum=1.0)` | target_return: float | ndarray | Min-variance at target return |
| `efficient_frontier(mu, cov, n_points=50, bounds=None, target_sum=1.0, risk_free_rate=0.0)` | n_points: int | DataFrame | Full frontier curve |

---

## Risk Metrics

```python
import IdleFinance as idf
rm = idf.risk_metrics

# Returns and volatility
ann_ret = rm.annualized_return(returns)
ann_vol = rm.annualized_volatility(returns)

# Ratios
sharpe  = rm.sharpe_ratio(returns, risk_free_rate=0.03)
sortino = rm.sortino_ratio(returns, risk_free_rate=0.03)

# Drawdown
dd  = rm.drawdown(prices)                # full drawdown series (from prices)
mdd = rm.max_drawdown(prices)

# Rolling volatility
rv = rm.rolling_volatility(returns, window=20)

# VaR and Expected Shortfall
var95 = rm.value_at_risk(returns, confidence=0.95, method="historical")
var95 = rm.value_at_risk(returns, confidence=0.95, method="parametric")
var95 = rm.value_at_risk(returns, confidence=0.95, method="cornish_fisher")
es95  = rm.expected_shortfall(returns, confidence=0.95, method="historical")

# Portfolio-level VaR with weights
weights = [0.2, 0.2, 0.2, 0.2, 0.2]
port_var = rm.value_at_risk(returns, weights=weights, confidence=0.99)

# Risk decomposition
mctr   = rm.marginal_contribution_to_risk(weights, cov)   # Series
cvar_d = rm.component_var(weights, returns)               # DataFrame

# Probabilistic Sharpe Ratio (PSR)
psr = rm.probabilistic_sharpe_ratio(returns, benchmark_sr=0.5)

# Market-implied risk aversion
lam = rm.market_risk_aversion(market_prices, risk_free_rate=0.02)
```

| Function | Key Parameters | Returns | Description |
|----------|---------------|---------|-------------|
| `annualized_return(series, frequency=252)` | | float / Series | Annualized mean return |
| `annualized_volatility(series, annualized=True, frequency=252)` | | float / Series | Annualized volatility |
| `sharpe_ratio(series, risk_free_rate=0.03, frequency=252)` | | float / Series | Sharpe ratio |
| `sortino_ratio(series, risk_free_rate=0.03, target_return=0.0, frequency=252)` | | float / Series | Sortino ratio |
| `cumulative_returns(obj, log_returns=False)` | | Series / DataFrame | Compounded return index |
| `drawdown(obj, from_returns=False)` | | Series / DataFrame | Drawdown series from prices |
| `max_drawdown(obj, from_returns=False)` | | float / Series | Maximum drawdown |
| `rolling_volatility(obj, window=20, annualized=True, frequency=252)` | | Series / DataFrame | Rolling vol over time |
| `value_at_risk(returns, confidence=0.95, method="historical", weights=None)` | method: str | float / Series | Value at Risk |
| `expected_shortfall(returns, confidence=0.95, method="historical", weights=None)` | | float / Series | Expected Shortfall (CVaR) |
| `probabilistic_sharpe_ratio(returns, benchmark_sr=0.0, risk_free_rate=0.0, frequency=252)` | | float / Series | P(true SR > benchmark SR) |
| `marginal_contribution_to_risk(weights, cov_matrix)` | | Series | MCTR per asset |
| `component_var(weights, returns, confidence=0.95, method="historical")` | | DataFrame | Component VaR per asset |
| `market_risk_aversion(market_prices, risk_free_rate=0.02, frequency=252)` | | float | Market-implied λ |

**VaR / ES methods**

| Method | Description |
|--------|-------------|
| `"historical"` | Empirical quantile of past returns |
| `"parametric"` | Gaussian assumption: `z × σ` |
| `"cornish_fisher"` | Adjusts for skewness and kurtosis (Cornish-Fisher expansion) |

---

## Fixed Income

```python
import IdleFinance as idf
fi = idf.fixed_income

# Bond pricing
price = fi.bond_price(face_value=1000, coupon_rate=0.05, ytm=0.06,
                      years_to_maturity=10, frequency=2)   # → 925.61

# Yield to Maturity
ytm = fi.bond_ytm(price=925.61, face_value=1000, coupon_rate=0.05,
                  years_to_maturity=10, frequency=2)       # → 6.00%

# Duration
mac = fi.bond_macaulay_duration(1000, 0.05, 0.06, 10, 2)
mod = fi.bond_modified_duration(1000, 0.05, 0.06, 10, 2)
d   = fi.bond_duration(1000, 0.05, 0.06, 10, 2)           # modified by default

# Convexity
conv = fi.bond_convexity(1000, 0.05, 0.06, 10, 2)

# Forward rate: 1Y spot 3%, 2Y spot 4% → implied 1Y-into-1Y forward
fr = fi.forward_rate(spot_rate_t1=0.03, t1=1.0,
                     spot_rate_t2=0.04, t2=2.0)            # → ~5.01%

# Credit spread
spread = fi.credit_spread(risky_ytm=0.065, risk_free_ytm=0.04)  # → 2.5%

# Duration from arbitrary cashflows
cfs = [50, 50, 50, 1050]
mac_cf = fi.macaulay_duration_from_cashflows(cfs, ytm=0.05)
```

| Function | Key Parameters | Returns | Description |
|----------|---------------|---------|-------------|
| `bond_price(face_value, coupon_rate, ytm, years_to_maturity, frequency=2)` | | float | Fair price of coupon bond |
| `bond_ytm(price, face_value, coupon_rate, years_to_maturity, frequency=2)` | | float | YTM via Brent root-finding |
| `bond_duration(face_value, coupon_rate, ytm, years_to_maturity, frequency=2, modified=True)` | modified: bool | float | Macaulay or Modified Duration |
| `bond_macaulay_duration(face_value, coupon_rate, ytm, years_to_maturity, frequency=2)` | | float | Macaulay Duration |
| `bond_modified_duration(face_value, coupon_rate, ytm, years_to_maturity, frequency=2)` | | float | Modified Duration |
| `bond_convexity(face_value, coupon_rate, ytm, years_to_maturity, frequency=2)` | | float | Bond convexity |
| `forward_rate(spot_rate_t1, t1, spot_rate_t2, t2)` | | float | Implied forward rate |
| `credit_spread(risky_ytm, risk_free_ytm)` | | float | Credit spread |
| `macaulay_duration_from_cashflows(cashflows, ytm, frequency=1)` | | float | Macaulay Duration from arbitrary CFs |

---

## Options Pricing

```python
import IdleFinance as idf
from IdleFinance.models.options import (
    black_scholes_call,
    black_scholes_put,
    implied_volatility,
)

# European call / put (Black-Scholes)
call = black_scholes_call(spot=100, strike=100, time_to_expiry=1.0,
                          risk_free_rate=0.05, volatility=0.20)  # → 10.45

put  = black_scholes_put(spot=100, strike=100, time_to_expiry=1.0,
                         risk_free_rate=0.05, volatility=0.20)   # → 5.57

# Put-call parity: call - put = spot - K·e^(-rT)

# Implied volatility (Brent root-finding)
iv = implied_volatility(option_price=10.45, spot=100, strike=100,
                        time_to_expiry=1.0, risk_free_rate=0.05,
                        option_type="call")                      # → 0.20
```

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, volatility)` | All float | float | European call price |
| `black_scholes_put(spot, strike, time_to_expiry, risk_free_rate, volatility)` | All float | float | European put price |
| `implied_volatility(option_price, spot, strike, time_to_expiry, risk_free_rate, option_type="call")` | | float | IV from market price |

---

## Equity Valuation

```python
import IdleFinance as idf
eq = idf.models.equities  # or: from IdleFinance.models import equities as eq
```

### Cost of Capital

```python
# CAPM cost of equity
ke = eq.cost_of_equity_capm(risk_free_rate=0.04, beta=1.2, market_return=0.10)

# Build-up method
ke = eq.cost_of_equity_build_up(
    risk_free_rate=0.04,
    equity_risk_premium=0.05,
    size_premium=0.02,
    company_specific_risk_premium=0.01,
)

# After-tax cost of debt
kd = eq.cost_of_debt(coupon_rate=0.06, tax_rate=0.30)

# WACC
wacc = eq.wacc(
    market_equity=800_000,
    market_debt=200_000,
    cost_of_equity=0.10,
    cost_of_debt_pretax=0.06,
    tax_rate=0.30,
)
```

### DCF Valuation

```python
# Full DCF with terminal value
fcfs = [50, 60, 70, 80, 90]   # explicit free cash flows
result = eq.dcf_valuation(
    free_cash_flows=fcfs,
    discount_rate=0.10,
    terminal_growth=0.03,
    shares_outstanding=100,
    net_debt=50,
)
# result["equity_value"], result["price_per_share"], result["terminal_value"]

# Sensitivity table (WACC × terminal growth)
table = eq.dcf_sensitivity(
    free_cash_flows=fcfs,
    discount_rates=[0.08, 0.09, 0.10, 0.11, 0.12],
    terminal_growths=[0.02, 0.025, 0.03, 0.035],
)

# Reverse DCF — implied WACC from market price
implied_wacc = eq.reverse_dcf(
    current_price=45.0,
    free_cash_flows=fcfs,
    terminal_growth=0.03,
    shares_outstanding=100,
    net_debt=50,
)
```

### Dividend Discount Models

```python
# Gordon Growth Model (constant-growth DDM)
price = eq.gordon_growth_model(dividend=2.0, cost_of_equity=0.10, growth_rate=0.04)

# H-Model (linearly declining growth)
price = eq.h_model(
    dividend=2.0, cost_of_equity=0.10,
    long_term_growth=0.03, short_term_growth=0.12, high_growth_period=10,
)

# Two-stage DDM
price = eq.two_stage_ddm(
    dividend=2.0, cost_of_equity=0.10,
    high_growth_rate=0.12, stable_growth_rate=0.03, high_growth_years=5,
)

# Three-stage DDM
price = eq.three_stage_ddm(
    dividend=2.0, cost_of_equity=0.10,
    high_growth_rate=0.15, stable_growth_rate=0.03,
    high_growth_years=5, transition_years=5,
)
```

### Residual Income and Other Models

```python
# Earnings Power Value (zero-growth perpetuity)
epv = eq.earnings_power_value(earnings=100, cost_of_capital=0.10)

# Residual Income Model (Edwards-Bell-Ohlson)
price = eq.residual_income_model(
    book_value=50,
    earnings_per_share=[6, 7, 8, 9, 10],
    cost_of_equity=0.10,
    terminal_growth=0.03,
)

# Abnormal Earnings Growth
price = eq.abnormal_earnings_growth(
    eps_base=5.0,
    eps_projections=[5.5, 6.0, 6.5],
    dps_projections=[2.0, 2.2, 2.4],
    cost_of_equity=0.10,
)
```

### Multiples Valuation

```python
price = eq.pe_implied_price(earnings_per_share=5.0, pe_ratio=20)
price = eq.ev_ebitda_implied_price(ebitda=100, ev_ebitda_multiple=10,
                                    net_debt=50, shares_outstanding=20)
price = eq.price_to_book_implied(book_value_per_share=25, pb_ratio=2.5)
```

**Equities function reference**

| Function | Description |
|----------|-------------|
| `cost_of_equity_capm(risk_free_rate, beta, market_return)` | CAPM Ke |
| `cost_of_equity_build_up(risk_free_rate, equity_risk_premium, ...)` | Build-up Ke |
| `cost_of_debt(coupon_rate, tax_rate)` | After-tax Kd |
| `wacc(market_equity, market_debt, cost_of_equity, cost_of_debt_pretax, tax_rate)` | WACC |
| `dcf_valuation(free_cash_flows, discount_rate, terminal_growth, ...)` | DCF intrinsic value |
| `dcf_sensitivity(free_cash_flows, discount_rates, terminal_growths, ...)` | 2D sensitivity table |
| `reverse_dcf(current_price, free_cash_flows, terminal_growth, ...)` | Implied WACC |
| `earnings_power_value(earnings, cost_of_capital)` | EPV (zero-growth) |
| `gordon_growth_model(dividend, cost_of_equity, growth_rate)` | Constant-growth DDM |
| `h_model(dividend, cost_of_equity, long_term_growth, short_term_growth, high_growth_period)` | H-Model DDM |
| `two_stage_ddm(dividend, cost_of_equity, high_growth_rate, stable_growth_rate, high_growth_years)` | Two-stage DDM |
| `three_stage_ddm(dividend, cost_of_equity, high_growth_rate, stable_growth_rate, ...)` | Three-stage DDM |
| `residual_income_model(book_value, earnings_per_share, cost_of_equity, terminal_growth)` | Edwards-Bell-Ohlson RIM |
| `abnormal_earnings_growth(eps_base, eps_projections, dps_projections, cost_of_equity)` | AEG model |
| `pe_implied_price(earnings_per_share, pe_ratio)` | P/E implied price |
| `ev_ebitda_implied_price(ebitda, ev_ebitda_multiple, net_debt, shares_outstanding)` | EV/EBITDA implied price |
| `price_to_book_implied(book_value_per_share, pb_ratio)` | P/B implied price |

---

## Stochastic Simulations

```python
import IdleFinance as idf
from IdleFinance.models.stochastic import (
    geometric_brownian_motion,
    mean_reverting_process,
    jump_diffusion_process,
    monte_carlo,
)

# Geometric Brownian Motion — shape: (n_steps + 1, n_paths)
paths = geometric_brownian_motion(
    s0=100, mu=0.08, sigma=0.20,
    dt=1/252, n_steps=252, n_paths=1000, random_seed=42,
)

# Ornstein-Uhlenbeck (mean-reverting) — useful for interest rates
paths = mean_reverting_process(
    s0=0.05, theta=0.3, mu=0.04, sigma=0.01,
    dt=1/252, n_steps=252, n_paths=500, random_seed=42,
)

# Merton Jump Diffusion
paths = jump_diffusion_process(
    s0=100, mu=0.08, sigma=0.20,
    lamb=2.0,      # jump intensity (jumps per year)
    mu_j=-0.05,    # mean jump size (log)
    sigma_j=0.10,  # jump size std (log)
    dt=1/252, n_steps=252, n_paths=500, random_seed=42,
)

# Dispatcher — same API for all methods
paths = monte_carlo("gbm",          s0=100, mu=0.08, sigma=0.20, dt=1/252, n_steps=252)
paths = monte_carlo("mean_reversion", s0=0.05, theta=0.3, mu=0.04, sigma=0.01, dt=1/252, n_steps=252)
paths = monte_carlo("jump_diffusion", s0=100, mu=0.08, sigma=0.20,
                    lamb=2.0, mu_j=-0.05, sigma_j=0.10, dt=1/252, n_steps=252)
```

All simulation functions return an `ndarray` of shape `(n_steps + 1, n_paths)` where row 0 is the initial value `s0`.

| Function | Key Parameters | Description |
|----------|---------------|-------------|
| `geometric_brownian_motion(s0, mu, sigma, dt, n_steps, n_paths=1, random_seed=None)` | | GBM price paths |
| `mean_reverting_process(s0, theta, mu, sigma, dt, n_steps, n_paths=1, random_seed=None)` | theta: mean-reversion speed | OU process paths |
| `jump_diffusion_process(s0, mu, sigma, lamb, mu_j, sigma_j, dt, n_steps, n_paths=1, random_seed=None)` | lamb: jump intensity | Merton jump diffusion |
| `monte_carlo(method, **kwargs)` | method: "gbm"\|"mean_reversion"\|"jump_diffusion" | Dispatcher for all methods |

---

## DCF Excel Export

Exports a fully-formatted, formula-driven DCF workbook. Requires `openpyxl`.

```python
pip install "IdleFinance[excel]"
```

```python
from IdleFinance.models.dcf_excel import export_dcf_to_excel, import_dcf_from_excel

# Export — creates a workbook with Dashboard, WACC, Valuation, Sensitivity sheets
path = export_dcf_to_excel(
    "my_company_dcf.xlsx",
    company_name="Acme Corp",
    current_price=45.0,
    base_revenue=5000.0,
    revenue_growth_rates=[0.10, 0.09, 0.08, 0.07, 0.06],
    ebit_margins=[0.20, 0.21, 0.22, 0.23, 0.24],
    risk_free_rate=0.04,          # or a list of yearly rates
    beta=1.1,
    market_risk_premium=0.06,
    cost_of_debt_pre_tax=0.05,
    tax_rate=0.25,
    terminal_growth=0.025,
    shares_outstanding=200,
    net_debt=500,
)

# Import — round-trip the inputs back from the workbook
params = import_dcf_from_excel("my_company_dcf.xlsx")
# params["company_name"], params["base_revenue"], params["beta"], ...
```

**`export_dcf_to_excel()` parameter reference**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str / Path | — | Output file path (.xlsx) |
| `company_name` | str | "Company" | Header label in workbook |
| `years` | int | 5 | Forecast horizon |
| `current_price` | float | 100.0 | Current share price for margin-of-safety |
| `base_revenue` | float | 1000.0 | Base-year revenue |
| `revenue_growth_rates` | list[float] | None | Per-year revenue growth (length = years) |
| `ebit_margins` | list[float] | None | Per-year EBIT margins (length = years) |
| `tax_rate` | float | 0.25 | Corporate tax rate |
| `capex_pct` | list[float] | None | CapEx as % of revenue per year |
| `da_pct` | list[float] | None | D&A as % of revenue per year |
| `nwc_change_pct` | list[float] | None | ΔNWC as % of revenue per year |
| `risk_free_rate` | float / list[float] | 0.04 | Risk-free rate. List → uses last value |
| `beta` | float | 1.0 | Equity beta |
| `market_risk_premium` | float | 0.06 | ERP |
| `cost_of_debt_pre_tax` | float | 0.06 | Pre-tax cost of debt |
| `terminal_growth` | float | 0.025 | Gordon terminal growth rate |
| `shares_outstanding` | float | 100.0 | Diluted shares |
| `net_debt` | float | 0.0 | Net debt (debt − cash) |
| `sensitivity_waccs` | list[float] | None | WACC range for sensitivity table |
| `sensitivity_tgs` | list[float] | None | Terminal growth range for sensitivity table |

---

## API Reference

### `IdleFinance` top-level namespace

```python
import IdleFinance as idf
```

| Name | Type | Description |
|------|------|-------------|
| `idf.fv` | function | Future value |
| `idf.pv` | function | Present value |
| `idf.npv` | function | Net present value |
| `idf.irr` | function | Internal rate of return |
| `idf.payback_period` | function | Payback period |
| `idf.profitability_index` | function | Profitability index |
| `idf.effective_annual_rate` | function | Effective annual rate |
| `idf.compound_interest` | function | Compound interest balance |
| `idf.simple_interest` | function | Simple interest balance |
| `idf.annuity_payment` / `idf.pmt` | function | Annuity payment |
| `idf.loan_payment` | function | Amortizing loan payment |
| `idf.to_returns` | function | Prices → returns |
| `idf.historical_mean` | function | Historical mean return |
| `idf.ewma_return` | function | EWMA expected return |
| `idf.capm_return` | function | CAPM expected return |
| `idf.risk_metrics` | module | Risk analytics |
| `idf.covariances` | module | Covariance estimation |
| `idf.fixed_income` | module | Bond analytics |
| `idf.efficient_frontier` | module | Portfolio optimization |
| `idf.options` | module | Options pricing |
| `idf.stochastic` | module | Stochastic simulations |
| `idf.Finance` | class | Static TVM / finance math class |
| `idf.DataFrameFinanceAccessor` | class | `.finance` accessor for DataFrame |
| `idf.SeriesFinanceAccessor` | class | `.finance` accessor for Series |

---

## Type Aliases

| Alias | Underlying Type | Description |
|-------|----------------|-------------|
| `ArrayLike` | ndarray / Series / list[float] | Any 1-D numeric input |
| `MatrixLike` | ndarray / DataFrame | Any 2-D numeric input |
| `PriceData` | DataFrame / Series / ndarray | Price or return data |
| `FinancialOutput` | Series / DataFrame | Generic financial output |
| `NumericOutput` | float / Series | Scalar or per-asset output |
| `VectorOutput` | Series / ndarray | 1-D output |
| `Weights` | Series | Portfolio weights |
| `CovarianceMatrix` | DataFrame | Square covariance matrix |
| `ReturnsMethod` | Literal | `"mean_historical"` \| `"ema_historical"` \| `"capm"` |
| `CovarianceMethod` | Literal | `"sample"` \| `"ema"` |
| `StochasticMethod` | Literal | `"gbm"` \| `"mean_reversion"` \| `"jump_diffusion"` |
| `OmegaMethod` | Literal | `"idzorek"` \| `"proportional"` |
| `TauMethod` | Literal | `"default"` \| `"variance"` |
| `ObjectiveType` | Literal | `"utility"` \| `"tracking_error"` |
| `DenoiseMethod` | Literal | `"marchenko_pastur"` |
