"""
Tests for portfolio analytics: returns, covariance, Black-Litterman, expected returns.
"""

import numpy as np
import pandas as pd
import pytest
import IdleFinance as idf


class TestAccessorReturns:
    def test_returns_shape(self, prices):
        ret = prices.finance.returns()
        # pct_change drops first row
        assert ret.shape == (prices.shape[0] - 1, prices.shape[1])

    def test_returns_columns_match_prices(self, prices):
        ret = prices.finance.returns()
        assert list(ret.columns) == list(prices.columns)

    def test_log_returns_differ_from_simple(self, prices):
        simple = prices.finance.returns(log_returns=False)
        log    = prices.finance.returns(log_returns=True)
        assert not simple.equals(log)


class TestAccessorCovariance:
    def test_cov_is_square_symmetric(self, prices):
        cov = prices.finance.covariance()
        assert cov.shape[0] == cov.shape[1]
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)

    def test_cov_diagonal_is_positive(self, prices):
        cov = prices.finance.covariance()
        assert np.all(np.diag(cov.values) > 0)


class TestExpectedReturns:
    def test_mean_historical_returns_shape(self, prices):
        mu = prices.finance.expected_returns(method="mean_historical")
        assert len(mu) == prices.shape[1]

    def test_ema_historical_returns_shape(self, prices):
        mu = prices.finance.expected_returns(method="ema_historical")
        assert len(mu) == prices.shape[1]


class TestBlackLitterman:
    def test_weights_sum_to_one(self, prices, returns):
        mu = prices.finance.expected_returns(method="mean_historical")
        _, _, weights = prices.finance.black_litterman(
            prior_returns=mu,
            views={"AAPL": 0.15},
        )
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_weights_are_series_with_asset_index(self, prices, returns):
        mu = prices.finance.expected_returns(method="mean_historical")
        _, _, weights = prices.finance.black_litterman(
            prior_returns=mu,
            views={"AAPL": 0.15},
        )
        assert isinstance(weights, pd.Series)
        assert set(weights.index) == set(prices.columns)

    def test_constrained_weights_respect_bounds(self, prices):
        mu = prices.finance.expected_returns(method="mean_historical")
        max_alloc = 0.30
        _, _, weights = prices.finance.black_litterman(
            prior_returns=mu,
            views={"AAPL": 0.15},
            bounds=(0.0, max_alloc),
        )
        assert np.all(weights <= max_alloc + 1e-6)
        assert np.all(weights >= -1e-6)

    def test_posterior_cov_is_positive_definite(self, prices):
        mu = prices.finance.expected_returns(method="mean_historical")
        _, post_cov, _ = prices.finance.black_litterman(
            prior_returns=mu,
            views={"AAPL": 0.15},
        )
        eigenvalues = np.linalg.eigvalsh(post_cov.values)
        assert np.all(eigenvalues > -1e-10)


class TestSeriesAccessor:
    def test_annualized_return(self, prices):
        r = prices["AAPL"].finance.returns()
        ann = r.finance.annualized_return()
        assert isinstance(ann, float)

    def test_volatility(self, prices):
        r = prices["AAPL"].finance.returns()
        vol = r.finance.volatility()
        assert vol > 0

    def test_sharpe_ratio(self, prices):
        r = prices["AAPL"].finance.returns()
        sr = r.finance.sharpe_ratio()
        assert isinstance(sr, float)

    def test_max_drawdown_nonpositive(self, prices):
        mdd = prices["AAPL"].finance.max_drawdown()
        assert mdd <= 0
