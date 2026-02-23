"""
Example usage of IdleFinance library.
"""

import numpy as np
import pandas as pd
import IdleFinance as idf

def get_demo_data():
    """Generate simulated price data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', periods=252, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    prices_data = {}
    for asset in assets:
        period_returns = np.random.normal(0.0005, 0.02, 252)
        prices_data[asset] = 100 * np.exp(np.cumsum(period_returns))

    return pd.DataFrame(prices_data, index=dates)

def run_returns_and_cov_demo(df_prices):
    print("--- 1. Returns & Covariance ---")
    ret_df = df_prices.finance.returns()
    cov_matrix = df_prices.finance.covariance()
    print(f"Returns shape: {ret_df.shape}")
    print(f"Covariance shape: {cov_matrix.shape}")
    print(f"First 5 correlations:\n{df_prices.pct_change().corr().iloc[:2, :2]}\n")

def run_expected_returns_demo(df_prices):
    print("--- 2. Expected Returns ---")
    mean_returns = df_prices.finance.expected_returns(method='mean_historical')
    print(f"Mean historical:\n{mean_returns}\n")
    return mean_returns

def run_basic_bl_demo(df_prices, mean_returns):
    print("--- 3. Standard Black-Litterman ---")
    assets = df_prices.columns
    # Derive risk aversion from the data (Equal-Weighted Benchmark)
    eq_w_mkt = pd.Series(1/len(assets), index=assets)
    mkt_index_eq = (df_prices * eq_w_mkt).sum(axis=1)
    risk_av_auto = mkt_index_eq.finance.risk_aversion()
    
    views = {'AAPL': 0.15, 'MSFT': 0.12}
    view_confidences = [0.8, 0.7]

    post_ret, post_cov, optimal_weights = df_prices.finance.black_litterman(
        prior_returns=mean_returns,
        views=views,
        view_confidences=view_confidences,
        tau=0.05,
        risk_aversion=risk_av_auto,
    )
    print(f"Optimal weights (Auto λ={risk_av_auto:.2f}):\n{optimal_weights}\n")

def run_asset_analysis_demo(df_prices):
    print("--- 4. Individual Asset Analysis (AAPL) ---")
    aapl_prices = df_prices['AAPL']
    aapl_returns = aapl_prices.finance.returns()
    
    print(f"Annualized return: {aapl_returns.finance.annualized_return():.4f}")
    print(f"Volatility: {aapl_returns.finance.volatility():.4f}")
    print(f"Sharpe ratio: {aapl_returns.finance.sharpe_ratio():.4f}")
    print(f"Max drawdown: {aapl_prices.finance.max_drawdown():.4f}\n")

def run_financial_math_demo():
    print("--- 5. Basic Financial Calculations ---")
    cashflows = [-1000, 300, 400, 500, 600]
    print(f"NPV (10%): ${idf.npv(cashflows, 0.1):.2f}")
    print(f"IRR: {idf.irr(cashflows)*100:.2f}%")
    print(f"Payback period: {idf.payback_period(cashflows):.2f} periods")
    print(f"Loan payment (PV=10k, 8%, 10p): ${idf.loan_payment(10000, 0.08, 10):.2f}\n")

def run_advanced_bl_demo(df_prices):
    print("--- 6. Advanced Black-Litterman (Accessor) ---")
    assets = df_prices.columns
    market_caps = pd.Series([2000, 1500, 1800, 1000, 800], index=assets)
    market_weights = market_caps / market_caps.sum()
    
    # Use Value-Weighted Market Index for Risk Aversion
    mkt_index_vw = (df_prices * market_weights).sum(axis=1)
    risk_av_vw = mkt_index_vw.finance.risk_aversion()
    
    # Use Accessor for Equilibrium Returns (Π)
    prior_ret = df_prices.finance.market_prior_returns(market_weights, risk_av_vw, cov_method="ema", span=100)

    _, _, w_mod = df_prices.finance.black_litterman(
        prior_returns=prior_ret,
        views={"AAPL": 0.15, "TSLA": 0.20},
        tau_method="variance",
        cov_method="ema",
        omega_method="proportional"
    )
    print(f"Advanced weights (VW Index λ={risk_av_vw:.2f}):\n{w_mod.round(6)}\n")

def run_constrained_opt_demo(df_prices, mean_returns):
    print("--- 7. Constrained Portfolio Optimization ---")
    _, _, w_constrained = df_prices.finance.black_litterman(
        prior_returns=mean_returns,
        views={'AAPL': 0.15},
        bounds=(0.0, 0.30) # Max 30% allocation per asset
    )
    print(f"Constrained Weights (Max 30% per asset):\n{w_constrained.round(4)}\n")
    
    bench_w = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    _, _, w_tracker = df_prices.finance.black_litterman(
        prior_returns=mean_returns,
        views={'AAPL': 0.15},
        objective="tracking_error",
        benchmark_weights=bench_w
    )
    print(f"Tracking Error Minimized Weights (vs Benchmark):\n{w_tracker.round(4)}\n")

def run_custom_constraints_demo(df_prices, mean_returns):
    print("--- 8. Arbitrary Custom Constraints ---")
    # For example, enforce that AAPL (index 0) weight must be exactly 15%
    # and GOOGL (index 1) >= MSFT (index 2)
    my_constraints = [
        {'type': 'eq', 'fun': lambda w: w[0] - 0.15},
        {'type': 'ineq', 'fun': lambda w: w[1] - w[2]}
    ]
    _, _, w_custom = df_prices.finance.black_litterman(
        prior_returns=mean_returns,
        views={'AAPL': 0.15},
        custom_constraints=my_constraints
    )
    print("Custom Constraints (AAPL=15%, GOOGL >= MSFT):")
    print(w_custom.round(4))

def main():
    df_prices = get_demo_data()
    
    run_returns_and_cov_demo(df_prices)
    mean_returns = run_expected_returns_demo(df_prices)
    run_basic_bl_demo(df_prices, mean_returns)
    run_asset_analysis_demo(df_prices)
    run_financial_math_demo()
    run_advanced_bl_demo(df_prices)
    run_constrained_opt_demo(df_prices, mean_returns)
    run_custom_constraints_demo(df_prices, mean_returns)

if __name__ == "__main__":
    main()
    print("\n--- ALL METHODS DEMONSTRATED SUCCESSFULLY ---")