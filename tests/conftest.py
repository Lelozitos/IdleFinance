import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def prices():
    """Reproducible price history for 5 assets over 252 business days."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    data = {a: 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252))) for a in assets}
    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="session")
def returns(prices):
    """Simple percentage returns derived from the prices fixture."""
    return prices.pct_change().dropna()
