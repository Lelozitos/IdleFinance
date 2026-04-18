"""
Tests for core financial math: Time Value of Money, NPV, IRR, Payback, Loan Payment.
"""

import pytest
import IdleFinance as idf


class TestTimeValue:
    def test_future_value(self):
        # $1,000 at 5% for 3 years → $1,157.63
        assert abs(idf.fv(1000, 0.05, 3) - 1157.625) < 1e-3

    def test_present_value(self):
        # Discounting $1,157.63 at 5% for 3 years → $1,000
        assert abs(idf.pv(1157.625, 0.05, 3) - 1000) < 1e-3

    def test_fv_pv_roundtrip(self):
        assert abs(idf.pv(idf.fv(500, 0.08, 5), 0.08, 5) - 500) < 1e-9


class TestNPV:
    CASHFLOWS = [-1000, 300, 400, 500, 600]

    def test_positive_npv_at_low_discount_rate(self):
        assert idf.npv(self.CASHFLOWS, 0.10) > 0

    def test_negative_npv_at_very_high_discount_rate(self):
        assert idf.npv(self.CASHFLOWS, 0.99) < 0

    def test_zero_rate_equals_undiscounted_sum(self):
        assert abs(idf.npv(self.CASHFLOWS, 0.0) - sum(self.CASHFLOWS)) < 1e-9


class TestIRR:
    def test_irr_zeroes_npv(self):
        cashflows = [-1000, 300, 400, 500, 600]
        rate = idf.irr(cashflows)
        assert abs(idf.npv(cashflows, rate)) < 1e-4


class TestPayback:
    def test_exact_payback(self):
        # Outlay $100, receive $50 per period → exactly 2 periods
        assert abs(idf.payback_period([-100, 50, 50]) - 2.0) < 1e-9

    def test_interpolated_payback(self):
        # Outlay $100, recover $60 in period 1, need $40 more from period 2 → 1 + 40/60
        assert abs(idf.payback_period([-100, 60, 60]) - (1 + 40 / 60)) < 1e-9


class TestLoanPayment:
    def test_payment_amount(self):
        # $10k loan, 8% annual rate, 10 periods → ~$1,490.29/period
        assert abs(idf.loan_payment(10000, 0.08, 10) - 1490.29) < 0.5
