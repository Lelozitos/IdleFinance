"""Options pricing, Greeks, and volatility analytics (Black-Scholes)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm as _norm

_DAYS_PER_YEAR: float = 365.0
_MIN_T: float = 1e-8
_MIN_SIGMA: float = 1e-8
_MAX_SIGMA: float = 20.0
_SQRT_2PI: float = float(np.sqrt(2.0 * np.pi))

Flag = Literal["c", "p", "call", "put"]
ArrayLike = Union[np.ndarray, list, float, int]


def _flag_sign(flag: Flag) -> int:
    f = str(flag).lower().strip()
    if f in ("c", "call"):
        return 1
    if f in ("p", "put"):
        return -1
    raise ValueError(f"flag must be 'c'/'call' or 'p'/'put', got {flag!r}.")


def _validate(S, K, sigma) -> None:
    if S <= 0:
        raise ValueError(f"spot must be positive, got {S}.")
    if K <= 0:
        raise ValueError(f"strike must be positive, got {K}.")
    if sigma < 0:
        raise ValueError(f"volatility must be non-negative, got {sigma}.")


def _d1_d2(S, K, T, r, sigma) -> tuple[float, float]:
    """Black-Scholes d1/d2 — scalar or array."""
    sqrt_T = np.sqrt(np.maximum(T, _MIN_T))
    sig = np.maximum(sigma, _MIN_SIGMA)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sqrt_T)
    return d1, d1 - sig * sqrt_T


def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x ** 2) / _SQRT_2PI


def _Phi(x: np.ndarray) -> np.ndarray:
    return _norm.cdf(x)


# --- Scalar pricing ---------------------------------------------------------

def black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, volatility) -> float:
    """C = S*N(d1) - K*e^(-rT)*N(d2)."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return max(spot - strike, 0.0)
    if volatility == 0.0:
        return max(spot - strike * np.exp(-risk_free_rate * time_to_expiry), 0.0)
    d1, d2 = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return spot * _Phi(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * _Phi(d2)


def black_scholes_put(spot, strike, time_to_expiry, risk_free_rate, volatility) -> float:
    """P = K*e^(-rT)*N(-d2) - S*N(-d1)."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return max(strike - spot, 0.0)
    if volatility == 0.0:
        return max(strike * np.exp(-risk_free_rate * time_to_expiry) - spot, 0.0)
    d1, d2 = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return (
        strike * np.exp(-risk_free_rate * time_to_expiry) * _Phi(-d2)
        - spot * _Phi(-d1)
    )


def calculate_option_price(
    spot, strike, time_to_expiry, risk_free_rate, volatility, flag: Flag = "c",
) -> float:
    """Unified Black-Scholes pricer, call or put."""
    if _flag_sign(flag) == 1:
        return black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return black_scholes_put(spot, strike, time_to_expiry, risk_free_rate, volatility)


# --- Greeks ------------------------------------------------------------------

@dataclass
class GreeksBundle:
    """First- and second-order Greeks for a single option position."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta, "gamma": self.gamma, "vega": self.vega,
            "theta": self.theta, "rho": self.rho, "vanna": self.vanna,
            "volga": self.volga, "charm": self.charm,
        }


def delta(spot, strike, time_to_expiry, risk_free_rate, volatility, flag: Flag = "c") -> float:
    """delta = N(d1) for call, N(d1)-1 for put."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        s = _flag_sign(flag)
        return float(s > 0 and spot > strike) - float(s < 0 and spot < strike) * 1.0
    d1, _ = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    if _flag_sign(flag) == 1:
        return float(_Phi(d1))
    return float(_Phi(d1) - 1.0)


def gamma(spot, strike, time_to_expiry, risk_free_rate, volatility) -> float:
    """gamma = phi(d1) / (S*sigma*sqrt(T)), same for calls and puts."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    sig = max(volatility, _MIN_SIGMA)
    sqrt_T = np.sqrt(time_to_expiry)
    return float(_phi(d1) / (spot * sig * sqrt_T))


def vega(spot, strike, time_to_expiry, risk_free_rate, volatility) -> float:
    """vega per 1% vol move = S*phi(d1)*sqrt(T) / 100."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    return float(spot * _phi(d1) * np.sqrt(time_to_expiry) / 100.0)


def theta(spot, strike, time_to_expiry, risk_free_rate, volatility, flag: Flag = "c") -> float:
    """theta per calendar day."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return 0.0
    S, K, T, r = spot, strike, time_to_expiry, risk_free_rate
    sig = max(volatility, _MIN_SIGMA)
    d1, d2 = _d1_d2(S, K, T, r, sig)
    sqrt_T = np.sqrt(T)
    disc_r = np.exp(-r * T)
    decay = -S * sig * _phi(d1) / (2.0 * sqrt_T)

    if _flag_sign(flag) == 1:
        th = decay - r * K * disc_r * _Phi(d2)
    else:
        th = decay + r * K * disc_r * _Phi(-d2)
    return float(th / _DAYS_PER_YEAR)


def rho(spot, strike, time_to_expiry, risk_free_rate, volatility, flag: Flag = "c") -> float:
    """rho per 1% rate move = K*T*e^(-rT)*N(d2)/100 (call), negated N(-d2) for put."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return 0.0
    _, d2 = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    base = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry)
    if _flag_sign(flag) == 1:
        return float(base * _Phi(d2) / 100.0)
    return float(-base * _Phi(-d2) / 100.0)


def vanna(spot, strike, time_to_expiry, risk_free_rate, volatility) -> float:
    """vanna = d(delta)/d(sigma) = -phi(d1)*d2/sigma."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return 0.0
    sig = max(volatility, _MIN_SIGMA)
    d1, d2 = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, sig)
    return float(-_phi(d1) * d2 / sig)


def volga(spot, strike, time_to_expiry, risk_free_rate, volatility) -> float:
    """volga (vomma) = d^2V/d(sigma)^2 = vega_raw * d1 * d2 / sigma."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return 0.0
    sig = max(volatility, _MIN_SIGMA)
    d1, d2 = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, sig)
    sqrt_T = np.sqrt(time_to_expiry)
    vega_raw = spot * _phi(d1) * sqrt_T
    return float(vega_raw * d1 * d2 / sig)


def charm(spot, strike, time_to_expiry, risk_free_rate, volatility, flag: Flag = "c") -> float:
    """charm = d(delta)/dT per calendar day (delta bleed)."""
    _validate(spot, strike, volatility)
    if time_to_expiry <= 0.0:
        return 0.0
    r = risk_free_rate
    sig = max(volatility, _MIN_SIGMA)
    d1, d2 = _d1_d2(spot, strike, time_to_expiry, r, sig)
    sqrt_T = np.sqrt(time_to_expiry)
    inner = (2 * r * time_to_expiry - d2 * sig * sqrt_T) / (2 * time_to_expiry * sig * sqrt_T)
    return float(-_phi(d1) * inner / _DAYS_PER_YEAR)


def calculate_greeks(
    spot, strike, time_to_expiry, risk_free_rate, volatility,
    flag: Flag = "c", second_order: bool = False,
) -> GreeksBundle:
    """All Greeks for a single option in one call; second_order adds vanna/volga/charm."""
    d = delta(spot, strike, time_to_expiry, risk_free_rate, volatility, flag)
    g = gamma(spot, strike, time_to_expiry, risk_free_rate, volatility)
    v = vega(spot, strike, time_to_expiry, risk_free_rate, volatility)
    t = theta(spot, strike, time_to_expiry, risk_free_rate, volatility, flag)
    r = rho(spot, strike, time_to_expiry, risk_free_rate, volatility, flag)

    if second_order:
        va = vanna(spot, strike, time_to_expiry, risk_free_rate, volatility)
        vo = volga(spot, strike, time_to_expiry, risk_free_rate, volatility)
        ch = charm(spot, strike, time_to_expiry, risk_free_rate, volatility, flag)
        return GreeksBundle(delta=d, gamma=g, vega=v, theta=t, rho=r, vanna=va, volga=vo, charm=ch)
    return GreeksBundle(delta=d, gamma=g, vega=v, theta=t, rho=r)


def _iv_newton(target, spot, strike, T, r, flag, sigma0, max_iter, tol) -> float:
    """Newton-Raphson IV solver using vega as derivative."""
    sig = sigma0
    for _ in range(max_iter):
        price = calculate_option_price(spot, strike, T, r, sig, flag)
        err = price - target
        if abs(err) < tol:
            return sig
        d1, _ = _d1_d2(spot, strike, T, r, sig)
        vega_raw = spot * _phi(d1) * np.sqrt(max(T, _MIN_T))
        if abs(vega_raw) < 1e-12:
            break
        sig = sig - err / vega_raw
        sig = float(np.clip(sig, _MIN_SIGMA, _MAX_SIGMA))
    return sig


def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    flag: Flag = "c",
    method: Literal["brent", "newton", "hybrid"] = "hybrid",
    bracket: tuple[float, float] = (_MIN_SIGMA, _MAX_SIGMA),
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Implied volatility for a European Black-Scholes option.

    method: 'brent' (robust root-find), 'newton' (fast, may diverge), 'hybrid' (Newton then Brent fallback).
    Raises ValueError on invalid inputs or no solution within bracket.
    """
    if option_price < 0.0:
        raise ValueError(f"option_price must be non-negative, got {option_price}.")
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive.")
    if time_to_expiry <= 0.0:
        raise ValueError("time_to_expiry must be strictly positive.")

    T = time_to_expiry
    r = risk_free_rate

    if _flag_sign(flag) == 1:
        intrinsic = max(spot - strike * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(strike * np.exp(-r * T) - spot, 0.0)
    if option_price < intrinsic - tol:
        raise ValueError(
            f"option_price {option_price:.4f} below intrinsic {intrinsic:.4f}."
        )

    def _brent() -> float:
        def obj(sig: float) -> float:
            return calculate_option_price(spot, strike, T, r, sig, flag) - option_price
        try:
            return brentq(obj, bracket[0], bracket[1], xtol=tol, maxiter=200)
        except ValueError as exc:
            lo, hi = obj(bracket[0]), obj(bracket[1])
            raise ValueError(
                f"IV not bracketed in {bracket}. "
                f"f(lo)={lo:.4f}, f(hi)={hi:.4f}. "
                "Check inputs or widen bracket."
            ) from exc

    if method == "brent":
        return _brent()

    if method == "newton":
        return _iv_newton(option_price, spot, strike, T, r, flag, 0.20, max_iter, tol)

    try:
        iv_nr = _iv_newton(option_price, spot, strike, T, r, flag, 0.20, max_iter, tol)
        check = calculate_option_price(spot, strike, T, r, iv_nr, flag)
        if abs(check - option_price) < tol * 10.0:
            return iv_nr
    except Exception:
        pass
    return _brent()


def _vbroadcast(*arrays) -> list[np.ndarray]:
    return [np.atleast_1d(a).astype(float) for a in np.broadcast_arrays(*arrays)]


def _flag_signs(flags: Union[str, list[str]], n: int) -> np.ndarray:
    if isinstance(flags, str):
        return np.full(n, _flag_sign(flags), dtype=float)
    return np.array([_flag_sign(f) for f in flags], dtype=float)


def batch_option_pricing(
    spot: ArrayLike, strike: ArrayLike, time_to_expiry: ArrayLike,
    risk_free_rate: ArrayLike, volatility: ArrayLike,
    flags: Union[str, list[str]] = "c",
) -> np.ndarray:
    """Vectorised Black-Scholes pricing; all numeric inputs broadcast via NumPy."""
    S, K, T, r, sig = _vbroadcast(spot, strike, time_to_expiry, risk_free_rate, volatility)
    n = len(S)
    sign = _flag_signs(flags, n)

    T_s = np.maximum(T, _MIN_T)
    sig_s = np.maximum(sig, _MIN_SIGMA)
    sqrt_T = np.sqrt(T_s)

    d1 = (np.log(S / K) + (r + 0.5 * sig_s ** 2) * T_s) / (sig_s * sqrt_T)
    d2 = d1 - sig_s * sqrt_T
    disc_r = np.exp(-r * T_s)

    call_p = S * _Phi(d1) - K * disc_r * _Phi(d2)
    put_p = K * disc_r * _Phi(-d2) - S * _Phi(-d1)
    prices = np.where(sign > 0, call_p, put_p)

    intrinsic = np.where(sign > 0, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    return np.where(T <= 0, intrinsic, prices)


def batch_greeks(
    spot: ArrayLike, strike: ArrayLike, time_to_expiry: ArrayLike,
    risk_free_rate: ArrayLike, volatility: ArrayLike,
    flags: Union[str, list[str]] = "c",
) -> dict[str, np.ndarray]:
    """Vectorised delta/gamma/vega/theta/rho for arrays of inputs."""
    S, K, T, r, sig = _vbroadcast(spot, strike, time_to_expiry, risk_free_rate, volatility)
    n = len(S)
    sign = _flag_signs(flags, n)

    T_s = np.maximum(T, _MIN_T)
    sig_s = np.maximum(sig, _MIN_SIGMA)
    sqrt_T = np.sqrt(T_s)

    d1 = (np.log(S / K) + (r + 0.5 * sig_s ** 2) * T_s) / (sig_s * sqrt_T)
    d2 = d1 - sig_s * sqrt_T
    disc_r = np.exp(-r * T_s)
    phi_d1 = _phi(d1)

    delta_c = _Phi(d1)
    delta_p = _Phi(d1) - 1.0
    delta_arr = np.where(sign > 0, delta_c, delta_p)

    gamma_arr = phi_d1 / (S * sig_s * sqrt_T)
    vega_arr = S * phi_d1 * sqrt_T / 100.0

    decay = -S * sig_s * phi_d1 / (2.0 * sqrt_T)
    theta_c = (decay - r * K * disc_r * _Phi(d2)) / _DAYS_PER_YEAR
    theta_p = (decay + r * K * disc_r * _Phi(-d2)) / _DAYS_PER_YEAR
    theta_arr = np.where(sign > 0, theta_c, theta_p)

    base_rho = K * T_s * disc_r
    rho_c = base_rho * _Phi(d2) / 100.0
    rho_p = base_rho * -_Phi(-d2) / 100.0
    rho_arr = np.where(sign > 0, rho_c, rho_p)

    expired = T <= 0.0
    if expired.any():
        for arr in (delta_arr, gamma_arr, vega_arr, theta_arr, rho_arr):
            arr[expired] = 0.0

    return {
        "delta": delta_arr,
        "gamma": gamma_arr,
        "vega": vega_arr,
        "theta": theta_arr,
        "rho": rho_arr,
    }


def batch_implied_volatility(
    option_prices: ArrayLike, spot: ArrayLike, strike: ArrayLike,
    time_to_expiry: ArrayLike, risk_free_rate: ArrayLike,
    flags: Union[str, list[str]] = "c",
    tol: float = 1e-6,
) -> np.ndarray:
    """Vectorised IV inversion (Newton -> Brent per element); failed solves return NaN."""
    prices_a, S_a, K_a, T_a, r_a = _vbroadcast(
        option_prices, spot, strike, time_to_expiry, risk_free_rate
    )
    n = len(prices_a)
    if isinstance(flags, str):
        flag_list: list[str] = [flags] * n
    else:
        flag_list = list(flags)

    ivs = np.full(n, np.nan)
    for i in range(n):
        if T_a[i] <= 0.0 or np.isnan(prices_a[i]):
            continue
        try:
            ivs[i] = implied_volatility(
                prices_a[i], S_a[i], K_a[i], T_a[i], r_a[i],
                flag=flag_list[i], method="hybrid", tol=tol,
            )
        except Exception:
            pass
    return ivs


class VolatilitySurface:
    """Volatility surface builder/analyser over a (strike x expiry) grid."""

    def __init__(self, df: pd.DataFrame) -> None:
        required = {"strike", "expiry", "iv"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}.")
        self.data = df.copy().sort_values(["expiry", "strike"]).reset_index(drop=True)
        self._expiries: list[float] = sorted(self.data["expiry"].unique().tolist())
        self._strikes: list[float] = sorted(self.data["strike"].unique().tolist())

    @classmethod
    def from_market_quotes(
        cls, spot: float, strikes: ArrayLike, expiries: ArrayLike,
        option_prices: np.ndarray, risk_free_rate: float, flag: Flag = "c",
    ) -> "VolatilitySurface":
        """Build surface from a 2-D grid of market prices, shape (n_expiries, n_strikes)."""
        K = np.asarray(strikes, dtype=float)
        T = np.asarray(expiries, dtype=float)
        P = np.asarray(option_prices, dtype=float)

        records = []
        for i, t in enumerate(T):
            for j, k in enumerate(K):
                try:
                    iv_val = implied_volatility(P[i, j], spot, k, t, risk_free_rate, flag=flag)
                except Exception:
                    iv_val = np.nan
                records.append({"strike": k, "expiry": t, "iv": iv_val, "price": P[i, j]})
        return cls(pd.DataFrame(records))

    def smile(self, expiry: float) -> pd.DataFrame:
        """Vol smile for the nearest stored expiry."""
        closest = min(self._expiries, key=lambda t: abs(t - expiry))
        return (
            self.data[self.data["expiry"] == closest][["strike", "iv"]]
            .copy()
            .reset_index(drop=True)
        )

    def term_structure(self, strike: Optional[float] = None) -> pd.DataFrame:
        """IV term structure at the given strike (median strike if None)."""
        k = strike if strike is not None else self._strikes[len(self._strikes) // 2]
        closest_k = min(self._strikes, key=lambda x: abs(x - k))
        return (
            self.data[self.data["strike"] == closest_k][["expiry", "iv"]]
            .copy()
            .sort_values("expiry")
            .reset_index(drop=True)
        )

    def skew(self) -> pd.DataFrame:
        """Risk_reversal (OTM put IV - OTM call IV) and butterfly per expiry."""
        rows = []
        for t in self._expiries:
            sl = self.data[self.data["expiry"] == t].dropna(subset=["iv"])
            if len(sl) < 3:
                continue
            mid = len(sl) // 2
            rows.append({
                "expiry": t,
                "atm_iv": sl.iloc[mid]["iv"],
                "otm_put_iv": sl.iloc[0]["iv"],
                "otm_call_iv": sl.iloc[-1]["iv"],
                "risk_reversal": sl.iloc[0]["iv"] - sl.iloc[-1]["iv"],
                "butterfly": (sl.iloc[0]["iv"] + sl.iloc[-1]["iv"]) / 2.0 - sl.iloc[mid]["iv"],
            })
        return pd.DataFrame(rows)

    def interpolate(self, strike: float, expiry: float, method: Literal["linear", "nearest"] = "linear") -> float:
        """Bilinear (or nearest-neighbour) interpolation of IV at arbitrary (K, T)."""
        pivot = self.data.pivot_table(index="expiry", columns="strike", values="iv")

        if method == "nearest":
            K_n = min(self._strikes, key=lambda k: abs(k - strike))
            T_n = min(self._expiries, key=lambda t: abs(t - expiry))
            return float(pivot.loc[T_n, K_n])

        K_lo = max((k for k in self._strikes if k <= strike), default=self._strikes[0])
        K_hi = min((k for k in self._strikes if k >= strike), default=self._strikes[-1])
        T_lo = max((t for t in self._expiries if t <= expiry), default=self._expiries[0])
        T_hi = min((t for t in self._expiries if t >= expiry), default=self._expiries[-1])

        def _get(t: float, k: float) -> float:
            try:
                return float(pivot.loc[t, k])
            except KeyError:
                return np.nan

        v00, v01 = _get(T_lo, K_lo), _get(T_lo, K_hi)
        v10, v11 = _get(T_hi, K_lo), _get(T_hi, K_hi)

        kw = (strike - K_lo) / (K_hi - K_lo) if K_hi != K_lo else 0.5
        tw = (expiry - T_lo) / (T_hi - T_lo) if T_hi != T_lo else 0.5

        return float((v00 * (1 - kw) + v01 * kw) * (1 - tw)
                     + (v10 * (1 - kw) + v11 * kw) * tw)

    def to_dataframe(self) -> pd.DataFrame:
        """Pivot table: rows = expiry, columns = strike, values = IV."""
        return self.data.pivot_table(index="expiry", columns="strike", values="iv")

    def __repr__(self) -> str:
        return f"VolatilitySurface({len(self._strikes)} strikes x {len(self._expiries)} expiries)"


def scenario_pnl_grid(
    spot_range: ArrayLike, vol_range: ArrayLike, strike: float,
    time_to_expiry: float, risk_free_rate: float, flag: Flag = "c",
    current_spot: Optional[float] = None, current_vol: Optional[float] = None,
) -> pd.DataFrame:
    """
    P&L (or price) grid across (spot, vol) scenarios.

    If current_spot/current_vol given, values are P&L relative to that base case;
    otherwise absolute prices.
    """
    spots = np.asarray(spot_range, dtype=float)
    vols = np.asarray(vol_range, dtype=float)
    S_g, sig_g = np.meshgrid(spots, vols, indexing="ij")

    prices = batch_option_pricing(
        S_g.ravel(), strike, time_to_expiry, risk_free_rate, sig_g.ravel(), flag,
    ).reshape(S_g.shape)

    if current_spot is not None and current_vol is not None:
        base = calculate_option_price(current_spot, strike, time_to_expiry, risk_free_rate, current_vol, flag)
        prices = prices - base

    return pd.DataFrame(
        prices,
        index=pd.Index(spots, name="spot"),
        columns=pd.Index(np.round(vols * 100, 2), name="vol_%"),
    )


def stress_test(
    spot: float, strike: float, time_to_expiry: float, risk_free_rate: float,
    volatility: float, flag: Flag = "c",
    spot_shocks: Optional[list[float]] = None,
    vol_shocks: Optional[list[float]] = None,
    rate_shocks: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Discrete P&L shocks across spot/vol/rate dimensions."""
    if spot_shocks is None:
        spot_shocks = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
    if vol_shocks is None:
        vol_shocks = [-0.10, -0.05, 0.0, 0.05, 0.10]
    if rate_shocks is None:
        rate_shocks = [0.0]

    base_price = calculate_option_price(spot, strike, time_to_expiry, risk_free_rate, volatility, flag)
    records = []
    for ds in spot_shocks:
        for dv in vol_shocks:
            for dr in rate_shocks:
                S_ = spot * (1.0 + ds)
                sig_ = max(volatility + dv, _MIN_SIGMA)
                r_ = risk_free_rate + dr
                p_ = calculate_option_price(S_, strike, time_to_expiry, r_, sig_, flag)
                records.append({
                    "spot_shock_%": round(ds * 100, 1),
                    "vol_shock_%": round(dv * 100, 1),
                    "rate_shock_%": round(dr * 100, 2),
                    "spot": S_,
                    "vol": sig_,
                    "rate": r_,
                    "price": p_,
                    "pnl": p_ - base_price,
                })
    return pd.DataFrame(records)


def aggregate_portfolio_greeks(positions: list[dict]) -> pd.DataFrame:
    """
    Compute and aggregate Greeks across an option portfolio.

    Each position dict needs: spot, strike, time_to_expiry, risk_free_rate,
    volatility, flag, quantity. Returns per-position rows plus a TOTAL summary row.
    """
    rows = []
    for pos in positions:
        qty = pos.get("quantity", 1)
        g = calculate_greeks(
            spot=pos["spot"], strike=pos["strike"],
            time_to_expiry=pos["time_to_expiry"],
            risk_free_rate=pos["risk_free_rate"],
            volatility=pos["volatility"], flag=pos["flag"],
            second_order=True,
        )
        price = calculate_option_price(
            pos["spot"], pos["strike"], pos["time_to_expiry"],
            pos["risk_free_rate"], pos["volatility"], pos["flag"],
        )
        rows.append({
            "flag": pos["flag"], "strike": pos["strike"],
            "expiry": pos["time_to_expiry"], "quantity": qty,
            "price": price, "value": price * qty,
            "delta": g.delta * qty, "gamma": g.gamma * qty,
            "vega": g.vega * qty, "theta": g.theta * qty,
            "rho": g.rho * qty, "vanna": g.vanna * qty,
            "volga": g.volga * qty,
        })

    df = pd.DataFrame(rows)
    greek_cols = ["value", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]
    total = df[greek_cols].sum()
    total_row = total.to_frame().T
    total_row.index = ["TOTAL"]
    total_row.insert(0, "flag", "")
    total_row.insert(1, "strike", float("nan"))
    total_row.insert(2, "expiry", float("nan"))
    total_row.insert(3, "quantity", int(df["quantity"].sum()))
    total_row.insert(4, "price", float("nan"))
    return pd.concat([df, total_row])


def run_benchmark(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Throughput benchmark: batch pricing, Greeks, IV inversion (IV capped at n=5000)."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(50.0, 200.0, n)
    K = rng.uniform(50.0, 200.0, n)
    T = rng.uniform(0.05, 2.0, n)
    r = rng.uniform(0.01, 0.08, n)
    sig = rng.uniform(0.05, 0.80, n)
    flags = rng.choice(["c", "p"], n).tolist()

    records = []

    def _record(label: str, elapsed: float, count: int) -> None:
        records.append({
            "task": label,
            "elapsed_s": round(elapsed, 5),
            "throughput": f"{int(count / elapsed):,} /s",
        })

    t0 = time.perf_counter()
    prices = batch_option_pricing(S, K, T, r, sig, flags)
    _record(f"batch_option_pricing  (n={n:,})", time.perf_counter() - t0, n)

    t0 = time.perf_counter()
    batch_greeks(S, K, T, r, sig, flags)
    _record(f"batch_greeks          (n={n:,})", time.perf_counter() - t0, n)

    n_sc = min(n, 10_000)
    t0 = time.perf_counter()
    for i in range(n_sc):
        calculate_option_price(S[i], K[i], T[i], r[i], sig[i], flags[i])
    _record(f"scalar pricing loop   (n={n_sc:,})", time.perf_counter() - t0, n_sc)

    n_iv = min(n, 5_000)
    t0 = time.perf_counter()
    batch_implied_volatility(prices[:n_iv], S[:n_iv], K[:n_iv], T[:n_iv], r[:n_iv], flags[:n_iv])
    _record(f"batch_implied_vol     (n={n_iv:,})", time.perf_counter() - t0, n_iv)

    return pd.DataFrame(records).set_index("task")
