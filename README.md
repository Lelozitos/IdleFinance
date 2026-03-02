# IdleFinance

**IdleFinance** reimagines quantitative finance in Python by embedding portfolio theory, risk modeling, and asset pricing tools directly into the pandas ecosystem.

Rather than acting as a standalone optimizer, **IdleFinance** extends financial data structures with institutional-grade analytical capabilities through a seamless .finance accessor — transforming raw market data into a research-ready quantitative framework.

## 🚀 Key Pillars

**IdleFinance** is built on four core modules:

1.  **Portfolio Optimization (Black-Litterman)**:
    - Full implementation of the J. Walters & T. Idzorek models.
    - Flexible constraint handling (bounds, target sums) and objective functions (Utility vs. Tracking Error).
    - Automated derivation of market-implied risk aversion and prior equilibrium returns.

2.  **Risk Metrics**:
    - **Covariance Estimation**:
      - **Shrinkage Estimators**: Ledoit-Wolf and Oracle Approximating Shrinkage (OAS) for robust covariance matrix estimation.
      - **Factor Models**: Single-factor (Market) and Multi-factor (Fama-French 3/5 Factor) models for risk decomposition.
      - **Dynamic Models**: Constant Conditional Correlation (CCC) and Orthogonal GARCH (OGARCH) for time-varying volatility and correlation.
    - **Risk Decomposition**:
      - **Marginal Contribution to Risk (MCTR)**: Contribution of each asset to total portfolio risk.
      - **Component Value at Risk (CVaR)**: Risk contribution of each asset to the portfolio's Value at Risk.
      - **Risk Decomposition**: Decomposition of portfolio risk into systematic and idiosyncratic components.

3.  **Fixed Income**:
    - **Unified Bond Pricing**: Automatic detection of zero-coupon vs. coupon-bearing bonds.
    - **Risk Metrics**: Macaulay/Modified Duration, Convexity, and Effective Duration/Convexity (for bonds with embedded options).
    - **Yield Analysis**: Brent-method optimized YTM, Yield to Call (YTC), and current yield.
    - **Arbitrary Cashflows**: Duration and NPV analysis for any stream of payments.

4.  **Core Utilities**:
    - Standard financial math: NPV, IRR, Payback Period, Loan Payments (PMT), and more.
    - Seamless conversion between price data and return series.

## ~~📦 Installation~~ (not yet available)

```bash
pip install IdleFinance
```

## 🛠️ Quick Start

### 1. Pandas Objects accessor

IdleFinance extends pandas DataFrames and Series with the `.finance` accessor.

```python
import pandas as pd
import IdleFinance as idf

prices = pd.Series([100, 102, 101, 105, 107])

print(f"Annualized Return: {prices.finance.annualized_return():.2%}")
print(f"Sharpe Ratio: {prices.finance.sharpe_ratio():.2f}")
print(f"Max Drawdown: {prices.finance.max_drawdown():.2%}")
```

### 1. Portfolio Optimization with Views

Combine market priors with investor views using the Black-Litterman model.

```python
df_prices = pd.read_csv("portfolio_prices.csv", index_col=0)

post_ret, post_cov, weights = df_prices.finance.black_litterman(
    views={'AAPL': 0.12, 'MSFT': 0.08},
    view_confidences=[0.8, 0.6],
    bounds=(0, 0.30)
)
```

### 2. Risk Decomposition

```python
prices = pd.Series([100, 102, 101, 105, 107])

print(f"Annualized Return: {prices.finance.annualized_return():.2%}")
print(f"Sharpe Ratio: {prices.finance.sharpe_ratio():.2f}")
print(f"Max Drawdown: {prices.finance.max_drawdown():.2%}")
```

### 3. Comprehensive Asset Valuation

```python
ytm = idf.fixed_income.bond_ytm(price=925.61, face_value=1000, coupon_rate=0.05, years_to_maturity=10)
```
