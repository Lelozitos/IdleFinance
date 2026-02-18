"""
Basic financial calculations.

Fundamental formulas for TVM, NPV/IRR, and loans. Accepts array-like
or pd.Series for cashflow arguments.
"""

from asyncio import futures
import numpy as np

class Finance:
    """
    All methods are static. Cashflow inputs accept list, array, or pd.Series.
    """

    @staticmethod
    def future_value(principal, rate, time, n=1):
        """
        Future value of a compound interest.

        Formula: FV = P * (1 + r/n)^(n*t)
        P is the principal, r is the rate, n is the number of compounding periods per year, and t is the time in years.
        """
        return Finance.compound_interest(principal, rate, time, n)

    @staticmethod
    def present_value(future_value, rate, time, n=1):
        """
        Present value of a compound interest.

        Formula: PV = FV / (1 + r/n)^(n*t)
        FV is the future value, r is the rate, n is the number of compounding periods per year, and t is the time in years.
        """
        return future_value / (1 + rate / n) ** (n * time)

    @staticmethod
    def net_present_value(cashflows, rate):
        """
        Net present value of a stream of cashflows.

        Formula: NPV = Σ_t CF_t / (1 + r)^t
        CF_t is the cashflow at time t, r is the discount rate, and t is the time in periods.
        """
        cf = Finance._as_cashflows(cashflows)
        return sum(cf[t] / (1 + rate) ** t for t in range(len(cf)))

    @staticmethod
    def internal_rate_of_return(cashflows, guess=0.1, max_iter=100, tol=1e-6):
        """
        Internal rate of return (periodic) via Newton-Raphson.

        Formula: solve for r such that NPV(r) = Σ_t CF_t / (1 + r)^t = 0.
        CF_t is the cashflow at time t, r is the discount rate, and t is the time in periods.

        Solves for the discount rate that sets NPV to zero. Use guess/max_iter
        if the solution does not converge.
        """
        cf = Finance._as_cashflows(cashflows)
        r = guess
        for _ in range(max_iter):
            npv = Finance.net_present_value(cf, r)
            dnpv_dr = sum(-i * cf[i] / (1 + r) ** (i + 1) for i in range(len(cf)))
            if abs(dnpv_dr) < tol:
                break
            r -= npv / dnpv_dr
            if abs(npv) < tol:
                break
        return r

    @staticmethod
    def payback_period(cashflows):
        """
        Payback period in number of periods (with linear interpolation).

        Formula: smallest t such that Σ_{s=0..t} CF_s ≥ 0 (with linear
        interpolation within the period where cumulative CF turns positive).
        CF_s is the cashflow at time s, and t is the time in periods.

        Returns np.nan if the project never pays back.
        """
        cf = Finance._as_cashflows(cashflows)
        cumulative_cf = np.cumsum(cf)
        if cumulative_cf[-1] < 0:
            return np.nan
        idx = np.searchsorted(cumulative_cf, 0, side="left")
        if idx == 0:
            return 0.0
        if idx >= len(cumulative_cf):
            return np.nan
        c0, c1 = cumulative_cf[idx - 1], cumulative_cf[idx]
        if c1 == c0:
            return float(idx)
        frac = -c0 / (c1 - c0)
        return (idx - 1) + frac

    @staticmethod
    def profitability_index(cashflows, rate):
        """
        Profitability index: PV of future cashflows / |initial investment|.

        Formula: PI = (Σ_{t≥1} CF_t / (1+r)^t) / |CF_0|
        CF_t is the cashflow at time t, r is the discount rate, t is the time in periods and CF_0 is the initial investment.

        PI > 1 indicates value creation.
        """
        cf = Finance._as_cashflows(cashflows)
        if cf[0] >= 0:
            return np.nan
        pv_future = sum(cf[t] / (1 + rate) ** t for t in range(1, len(cf)))
        return pv_future / abs(cf[0])

    @staticmethod
    def effective_annual_rate(period_rate, periods_per_year):
        """
        Effective annual rate from a periodic rate.

        Formula: EAR = (1 + r_period)^m - 1, where m = periods per year.
        r_period is the periodic rate, and m is the number of periods per year.
        """
        return (1 + period_rate) ** periods_per_year - 1

    @staticmethod
    def compound_interest(principal, rate, time, n=1):
        """
        Ending balance with compound interest.

        Formula: A = P * (1 + r/n)^(n*t)  (same as future_value).
        P is the principal, r is the rate, n is the number of compounding periods per year, and t is the time in years.
        """
        return principal * (1 + rate / n) ** (n * time)

    @staticmethod
    def simple_interest(principal, rate, time):
        """
        Ending balance with simple (non-compounding) interest.

        Formula: A = P * (1 + r*t);  interest = P * r * t.
        P is the principal, r is the rate, and t is the time in years.
        """
        return principal * (1 + rate * time)

    @staticmethod
    def annuity_payment(present_value, rate, periods):
        """
        Fixed payment per period for an ordinary annuity.

        Formula: PMT = PV * [r(1+r)^n] / [(1+r)^n - 1]
        PV is the present value, r is the rate, n is the number of periods, and t is the time in years.
        """
        if rate == 0:
            return present_value / periods
        return present_value * (rate * (1 + rate) ** periods) / ((1 + rate) ** periods - 1)

    @staticmethod
    def loan_payment(principal, rate, periods):
        """
        Fixed payment per period for an amortizing loan.

        Formula: PMT = P * [r(1+r)^n] / [(1+r)^n - 1]  (same as annuity_payment).
        P is the principal, r is the rate, n is the number of periods, and t is the time in years.
        """
        return Finance.annuity_payment(principal, rate, periods)

    @staticmethod
    def _as_cashflows(cashflows):
        """Normalize cashflows to a 1D numpy array (for NPV/IRR/payback)."""
        if hasattr(cashflows, "values"):
            return np.asarray(cashflows.values).ravel()
        return np.asarray(cashflows).ravel()