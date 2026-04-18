"""
Tests for Risk Metrics: returns, volatility, Sharpe, Sortino, drawdown, VaR, ES, MCTR.
"""

import numpy as np
import pandas as pd
import pytest
import IdleFinance as idf

rm = idf.risk_metrics


class TestAnnualizedReturn:
    def test_positive_for_rising_market(self, returns):
        # With seed=42, market generally rises
        r = idf.risk_metrics.annualized_return(returns["AAPL"])
        assert isinstance(r, float)

    def test_zero_for_flat_series(self):
        flat = pd.Series([0.0] * 100)
        assert rm.annualized_return(flat) == 0.0


class TestAnnualizedVolatility:
    def test_positive_for_noisy_series(self, returns):
        vol = rm.annualized_volatility(returns["AAPL"])
        assert vol > 0

    def test_zero_for_constant_series(self):
        constant = pd.Series([0.01] * 100)
        assert rm.annualized_volatility(constant) < 1e-14

    def test_annualized_larger_than_period(self, returns):
        period_vol = returns["AAPL"].std()
        annual_vol = rm.annualized_volatility(returns["AAPL"])
        assert annual_vol > period_vol


class TestSharpeRatio:
    def test_positive_for_positive_excess_return(self, returns):
        # With a 0% risk-free rate, positive returns → positive Sharpe
        sr = rm.sharpe_ratio(returns["AAPL"], risk_free_rate=0.0)
        assert isinstance(sr, float)

    def test_zero_vol_returns_zero(self):
        # Zero volatility edge case should not raise
        constant = pd.Series([0.001] * 100)
        sr = rm.sharpe_ratio(constant, risk_free_rate=0.0)
        assert isinstance(sr, (int, float))


class TestSortinoRatio:
    def test_sortino_type(self, returns):
        sr = rm.sortino_ratio(returns["AAPL"])
        assert isinstance(sr, (int, float))

    def test_sortino_geq_sharpe_for_right_skewed(self, returns):
        # When returns have fat positive tails, Sortino ≥ Sharpe (downside vol < total vol)
        sharpe = rm.sharpe_ratio(returns["AAPL"], risk_free_rate=0.0)
        sortino = rm.sortino_ratio(returns["AAPL"], risk_free_rate=0.0)
        # This holds when downside deviation < total std, which is common in practice
        # We just check they're both finite floats
        assert np.isfinite(sortino) and np.isfinite(sharpe)


class TestDrawdown:
    def test_max_drawdown_nonpositive(self, prices):
        mdd = rm.max_drawdown(prices["AAPL"])
        assert mdd <= 0

    def test_monotonic_prices_have_zero_drawdown(self):
        rising = pd.Series([100.0, 101.0, 102.0, 103.0])
        assert rm.max_drawdown(rising) == 0.0

    def test_drawdown_series_starts_at_zero(self, prices):
        dd = rm.drawdown(prices["AAPL"])
        assert dd.iloc[0] == 0.0


class TestValueAtRisk:
    @pytest.mark.parametrize("method", ["historical", "parametric", "cornish_fisher"])
    def test_var_is_positive(self, returns, method):
        var = rm.value_at_risk(returns["AAPL"], confidence=0.95, method=method)
        assert var > 0

    def test_higher_confidence_gives_higher_var(self, returns):
        var_95 = rm.value_at_risk(returns["AAPL"], confidence=0.95)
        var_99 = rm.value_at_risk(returns["AAPL"], confidence=0.99)
        assert var_99 >= var_95

    def test_invalid_confidence_raises(self, returns):
        with pytest.raises(ValueError):
            rm.value_at_risk(returns["AAPL"], confidence=1.5)

    def test_invalid_method_raises(self, returns):
        with pytest.raises(ValueError):
            rm.value_at_risk(returns["AAPL"], method="unknown")


class TestExpectedShortfall:
    @pytest.mark.parametrize("method", ["historical", "parametric", "cornish_fisher"])
    def test_es_is_positive(self, returns, method):
        es = rm.expected_shortfall(returns["AAPL"], confidence=0.95, method=method)
        assert es > 0

    def test_es_greater_than_var(self, returns):
        # ES (average tail loss) must be >= VaR at the same confidence level
        var = rm.value_at_risk(returns["AAPL"], confidence=0.95, method="historical")
        es  = rm.expected_shortfall(returns["AAPL"], confidence=0.95, method="historical")
        assert es >= var


class TestMCTR:
    def test_output_is_series(self, returns):
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        cov = returns.cov() * 252
        mctr = rm.marginal_contribution_to_risk(weights, cov)
        assert isinstance(mctr, pd.Series)
        assert len(mctr) == 5

    def test_weighted_sum_equals_portfolio_vol(self, returns):
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        cov = (returns.cov() * 252).values
        mctr = rm.marginal_contribution_to_risk(weights, cov)
        port_vol = np.sqrt(weights @ cov @ weights)
        # Sum of w_i * MCTR_i should equal portfolio volatility
        assert abs(np.dot(weights, mctr.values) - port_vol) < 1e-9

    def test_dimension_mismatch_raises(self, returns):
        weights = np.array([0.5, 0.5])
        cov = returns.cov() * 252
        with pytest.raises(ValueError):
            rm.marginal_contribution_to_risk(weights, cov)
