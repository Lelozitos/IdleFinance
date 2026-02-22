"""
IdleFinance.models
===================

Financial models and optimization algorithms. Each model is self-contained,
implements a specific financial algorithm, and can be extended independently.

Available models:
    - risk_metrics: annualized_return, volatility, sharpe_ratio, drawdown, max_drawdown, cumulative_returns
    - covariances: sample, ema, exponentially_weighted
"""

from . import risk_metrics
from . import covariances

__all__ = [
    "risk_metrics",
    "covariances",
]