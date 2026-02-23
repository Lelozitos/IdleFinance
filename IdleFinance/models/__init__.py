"""
IdleFinance.models
===================

Financial models and optimization algorithms. Each model is self-contained,
implements a specific financial algorithm, and can be extended independently.

Available models:
    - black_litterman: Black-Litterman portfolio optimization for Series
    - risk_metrics: annualized_return, volatility, sharpe_ratio, drawdown, max_drawdown, cumulative_returns
    - covariances: sample, ema, exponentially_weighted
"""

from .black_litterman import bl_posterior_distribution, black_litterman_single_asset
from . import risk_metrics
from . import covariances

__all__ = [
    "bl_posterior_distribution",
    "black_litterman_single_asset",
    "risk_metrics",
    "covariances",
]