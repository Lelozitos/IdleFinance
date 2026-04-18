"""
Tests for Options: Black-Scholes pricing, put-call parity, implied volatility.
"""

import numpy as np
import pytest
from IdleFinance.models.options import black_scholes_call, black_scholes_put, implied_volatility

# Standard ATM option inputs used across tests
SPOT   = 100.0
STRIKE = 100.0
T      = 1.0    # 1 year
R      = 0.05   # 5% risk-free rate
SIGMA  = 0.20   # 20% volatility


class TestBlackScholesCall:
    def test_atm_call_price(self):
        # Well-known ATM result: ~$10.45
        price = black_scholes_call(SPOT, STRIKE, T, R, SIGMA)
        assert abs(price - 10.4506) < 0.001

    def test_deep_itm_call(self):
        # S >> K → call price approaches S - K*e^(-rT)
        price = black_scholes_call(200, 100, T, R, SIGMA)
        intrinsic = 200 - 100 * np.exp(-R * T)
        assert price > intrinsic * 0.99

    def test_call_at_expiry_itm(self):
        assert black_scholes_call(110, 100, 0, R, SIGMA) == 10.0

    def test_call_at_expiry_otm(self):
        assert black_scholes_call(90, 100, 0, R, SIGMA) == 0.0

    def test_invalid_negative_spot(self):
        with pytest.raises(ValueError):
            black_scholes_call(-1, STRIKE, T, R, SIGMA)

    def test_invalid_negative_volatility(self):
        with pytest.raises(ValueError):
            black_scholes_call(SPOT, STRIKE, T, R, -0.1)


class TestBlackScholesPut:
    def test_atm_put_price(self):
        # Well-known ATM result: ~$5.57
        price = black_scholes_put(SPOT, STRIKE, T, R, SIGMA)
        assert abs(price - 5.5735) < 0.001

    def test_put_at_expiry_itm(self):
        assert black_scholes_put(90, 100, 0, R, SIGMA) == 10.0

    def test_put_at_expiry_otm(self):
        assert black_scholes_put(110, 100, 0, R, SIGMA) == 0.0


class TestPutCallParity:
    def test_parity_holds(self):
        # C - P = S - K * e^(-rT)
        call = black_scholes_call(SPOT, STRIKE, T, R, SIGMA)
        put  = black_scholes_put(SPOT, STRIKE, T, R, SIGMA)
        parity_rhs = SPOT - STRIKE * np.exp(-R * T)
        assert abs((call - put) - parity_rhs) < 1e-9


class TestImpliedVolatility:
    def test_roundtrip_call(self):
        price = black_scholes_call(SPOT, STRIKE, T, R, SIGMA)
        iv = implied_volatility(price, SPOT, STRIKE, T, R, option_type="call")
        assert abs(iv - SIGMA) < 1e-4

    def test_roundtrip_put(self):
        price = black_scholes_put(SPOT, STRIKE, T, R, SIGMA)
        iv = implied_volatility(price, SPOT, STRIKE, T, R, option_type="put")
        assert abs(iv - SIGMA) < 1e-4

    def test_invalid_zero_expiry(self):
        with pytest.raises(ValueError):
            implied_volatility(10.45, SPOT, STRIKE, 0, R)
