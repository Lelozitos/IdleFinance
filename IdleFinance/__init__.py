from .utils import (
    to_returns,
    historical_mean,
    ewma_return,
    capm_return,
)
from .utils.basic import Finance
from . import core

future_value = Finance.future_value
present_value = Finance.present_value
net_present_value = Finance.net_present_value
internal_rate_of_return = Finance.internal_rate_of_return
payback_period = Finance.payback_period
profitability_index = Finance.profitability_index
effective_annual_rate = Finance.effective_annual_rate
compound_interest = Finance.compound_interest
simple_interest = Finance.simple_interest
annuity_payment = Finance.annuity_payment
loan_payment = Finance.loan_payment

__all__ = [
    "to_returns",
    "historical_mean",
    "ewma_return",
    "capm_return",
    "sharpe_ratio",
    "Finance",
    "future_value",
    "present_value",
    "net_present_value",
    "internal_rate_of_return",
    "payback_period",
    "profitability_index",
    "effective_annual_rate",
    "compound_interest",
    "simple_interest",
    "annuity_payment",
    "loan_payment",
]