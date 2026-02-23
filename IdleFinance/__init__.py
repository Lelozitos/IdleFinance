from .utils import (
    to_returns,
    historical_mean,
    ewma_return,
    capm_return,
)
from .models.risk_metrics import sharpe_ratio, market_risk_aversion
from .utils.basic import Finance
from .core.accessor import DataFrameFinanceAccessor, SeriesFinanceAccessor

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
]