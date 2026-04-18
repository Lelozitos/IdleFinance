"""
Tests for Fixed Income: bond pricing, YTM, duration, forward rates, credit spread.
"""

import pytest
import IdleFinance as idf

fi = idf.fixed_income


class TestBondPrice:
    def test_discount_bond(self):
        # 5% coupon bond priced at 6% YTM → below par (~$925.61)
        price = fi.bond_price(1000, 0.05, 0.06, 10, 2)
        assert abs(price - 925.61) < 0.01

    def test_par_bond(self):
        # When coupon rate == YTM, price equals face value
        price = fi.bond_price(1000, 0.05, 0.05, 10, 2)
        assert abs(price - 1000) < 0.01

    def test_premium_bond(self):
        # 6% coupon bond priced at 4% YTM → above par
        price = fi.bond_price(1000, 0.06, 0.04, 10, 2)
        assert price > 1000

    def test_zero_coupon_bond(self):
        # PV of $100 in 5 years at 5% annual → ~$78.35 (annual compounding, frequency=1)
        price = fi.bond_price(100, 0.0, 0.05, 5, frequency=1)
        assert abs(price - 78.35) < 0.01


class TestBondYTM:
    def test_ytm_roundtrip(self):
        # Price a bond at 6% YTM, then recover YTM from that price
        price = fi.bond_price(1000, 0.05, 0.06, 10, 2)
        ytm = fi.bond_ytm(price, 1000, 0.05, 10, 2)
        assert abs(ytm - 0.06) < 1e-4

    def test_ytm_at_par(self):
        # When priced at par, YTM equals coupon rate
        ytm = fi.bond_ytm(1000, 1000, 0.05, 10, 2)
        assert abs(ytm - 0.05) < 1e-4


class TestBondDuration:
    def test_duration_is_positive(self):
        d = fi.bond_duration(1000, 0.04, 0.05, 5, 2)
        assert d > 0

    def test_longer_maturity_has_longer_duration(self):
        short = fi.bond_duration(1000, 0.05, 0.05, 3, 2)
        long_ = fi.bond_duration(1000, 0.05, 0.05, 10, 2)
        assert long_ > short

    def test_modified_duration_less_than_macaulay(self):
        # Modified Duration = Macaulay Duration / (1 + y/m)
        mac = fi.bond_macaulay_duration(1000, 0.04, 0.05, 5, 2)
        mod = fi.bond_modified_duration(1000, 0.04, 0.05, 5, 2)
        assert mac > mod > 0


class TestForwardRate:
    def test_one_year_forward(self):
        # 1Y spot 3%, 2Y spot 4% → implied 1Y forward ~5.01%
        fr = fi.forward_rate(0.03, 1.0, 0.04, 2.0)
        assert abs(fr - 0.0501) < 0.001


class TestCreditSpread:
    def test_spread_equals_yield_minus_risk_free(self):
        spread = fi.credit_spread(0.065, 0.04)
        assert abs(spread - 0.025) < 1e-9
