from .utils import (
    to_returns,
    historical_mean,
    ewma_return,
    capm_return,
)
from .models import risk_metrics, covariances, fixed_income
from .utils.basic import Finance
from .core.accessor import DataFrameFinanceAccessor, SeriesFinanceAccessor

# # Expose individual functions for convenience
# gordon_growth_model = equities.gordon_growth_model
# h_model = equities.h_model
# dcf_valuation = equities.dcf_valuation
# earnings_power_value = equities.earnings_power_value
# wacc = equities.wacc
# black_scholes_call = equities.black_scholes_call
# black_scholes_put = equities.black_scholes_put
# implied_volatility = equities.implied_volatility

# bond_price = fixed_income.bond_price
# bond_ytm = fixed_income.bond_ytm
# bond_duration = fixed_income.bond_duration
# bond_convexity = fixed_income.bond_convexity
# credit_spread = fixed_income.credit_spread
# forward_rate = fixed_income.forward_rate
# macaulay_duration_from_cashflows = fixed_income.macaulay_duration_from_cashflows

# sharpe_ratio = risk_metrics.sharpe_ratio
# market_risk_aversion = risk_metrics.market_risk_aversion

# Basic Financial Calculations
future_value = Finance.future_value
fv = future_value
present_value = Finance.present_value
pv = present_value
net_present_value = Finance.net_present_value
npv = net_present_value
internal_rate_of_return = Finance.internal_rate_of_return
irr = internal_rate_of_return
payback_period = Finance.payback_period
profitability_index = Finance.profitability_index
effective_annual_rate = Finance.effective_annual_rate
compound_interest = Finance.compound_interest
simple_interest = Finance.simple_interest
annuity_payment = Finance.annuity_payment
pmt = annuity_payment
loan_payment = Finance.loan_payment

__all__ = [
    "to_returns",
    "historical_mean",
    "ewma_return",
    "capm_return",
    "sharpe_ratio",
    "market_risk_aversion",
    "Finance",
    "future_value",
    "fv",
    "present_value",
    "pv",
    "net_present_value",
    "npv",
    "internal_rate_of_return",
    "irr",
    "payback_period",
    "profitability_index",
    "effective_annual_rate",
    "compound_interest",
    "simple_interest",
    "annuity_payment",
    "pmt",
    "loan_payment",
    "DataFrameFinanceAccessor",
    "SeriesFinanceAccessor",
    "fixed_income",
    "risk_metrics",
    "covariances",
]