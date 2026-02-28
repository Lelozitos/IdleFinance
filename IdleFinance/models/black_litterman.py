"""
Black-Litterman methods for estimating posterior returns and covariance.

This module implements the canonical Black-Litterman models based on Jay Walters
and Thomas Idzorek, calculating full posterior distributions.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import solve, cho_factor, cho_solve
from scipy.optimize import minimize
from ..utils.return_utils import to_returns
from ..core._types import (
    Union, Optional, List, BLResult, BLWeightsResult, BLSingleResult,
    ViewData, PortfolioBounds, PortfolioConstraints,
    OmegaMethod, TauMethod, ArrayLike, MatrixLike, PriceData, VectorOutput
)


def market_prior_returns(
    market_weights: ArrayLike,
    risk_aversion: float,
    cov_matrix: MatrixLike,
) -> VectorOutput:
    r"""
    Calculate the implied equilibrium returns (Π) given market weights.

    This process — also known as *Reverse Optimization* — derives the returns
    that would render the current market-cap weights mean-variance optimal.

    Formula: **Π = λ · Σ · w_mkt**

    Parameters
    ----------
    market_weights : ArrayLike
        Market-capitalisation weights (w_mkt).
    risk_aversion : float
        Market-implied risk-aversion coefficient (λ). Must be positive.
    cov_matrix : MatrixLike
        Prior covariance matrix (Σ). Must be square and consistent with weights.

    Returns
    -------
    VectorOutput
        The implied equilibrium returns (Π), as a ``pd.Series`` when
        *market_weights* is a ``pd.Series``, otherwise as ``np.ndarray``.

    Raises
    ------
    ValueError
        If *risk_aversion* ≤ 0, or if shape mismatch is detected.

    Examples
    --------
    >>> market_prior_returns(weights=[0.6, 0.4], risk_aversion=2.5, cov_matrix=[[0.04, 0.01], [0.01, 0.05]])
    array([0.07 , 0.065])
    """
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}.")

    weights = np.asarray(market_weights, dtype=float).flatten()
    cov = np.asarray(cov_matrix, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square 2-D matrix.")
    if cov.shape[0] != len(weights):
        raise ValueError(
            f"cov_matrix has {cov.shape[0]} assets but market_weights has {len(weights)}."
        )

    pi = risk_aversion * cov @ weights

    if isinstance(market_weights, pd.Series):
        return pd.Series(pi, index=market_weights.index)
    if isinstance(cov_matrix, pd.DataFrame):
        return pd.Series(pi, index=cov_matrix.columns)
    return pi


def tau(
    prices: Optional[PriceData] = None,
    method: TauMethod = "default",
    constant_value: float = 0.05,
) -> float:
    """
    Calculate the uncertainty scaling factor (τ).

    This factor scales the uncertainty of the prior relative to the views.
    Typical values lie in [0.01, 0.10].

    Parameters
    ----------
    prices : PriceData, optional
        Historical price or return data. Required when ``method='variance'``.
    method : TauMethod, default 'default'
        How to derive τ:

        - ``'default'``: Returns the fixed ``constant_value``.
        - ``'variance'``: Uses ``1 / T``, where T is the number of observations.
    constant_value : float, default 0.05
        Fixed τ value used with ``method='default'``. Must be positive.

    Returns
    -------
    float
        The uncertainty scaling factor (τ).

    Raises
    ------
    ValueError
        If ``method='variance'`` but *prices* is None or empty,
        or if *constant_value* ≤ 0.

    Examples
    --------
    >>> tau(prices=df, method="variance")
    0.003968253968253968
    """
    if method == "default":
        if constant_value <= 0:
            raise ValueError(f"constant_value must be positive, got {constant_value}.")
        return float(constant_value)
    elif method == "variance":
        if prices is None:
            raise ValueError("prices must be provided when method='variance'.")
        T = len(prices)
        if T == 0:
            raise ValueError("prices has no observations; cannot compute tau.")
        return 1.0 / T
    else:
        raise ValueError(
            f"Unknown tau method '{method}'. Choose 'default' or 'variance'."
        )


def idzorek_confidence_to_omega(
    confidences: ArrayLike,
    cov_matrix: MatrixLike,
    P: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """
    Build the view-uncertainty matrix (Ω) from confidence levels via Idzorek's method.

    Idzorek's method maps investor confidence (0 → complete uncertainty,
    1 → absolute certainty) to diagonal Ω entries without requiring manual
    specification.

    Formula: **Ω_k = τ · α · P_k · Σ · P_k^T**,  where **α = (1 − C) / C**

    Parameters
    ----------
    confidences : ArrayLike
        Confidence levels in **[0, 1]** for each view. Shape (K × 1).
    cov_matrix : MatrixLike
        Covariance matrix of asset returns (Σ). Shape (N × N).
    P : np.ndarray
        Picking matrix (K × N), mapping K views to N assets.
    tau : float, default 0.05
        Uncertainty scaling factor (τ). Must be positive.

    Returns
    -------
    np.ndarray
        Diagonal Omega matrix (Ω) of shape (K × K).

    Raises
    ------
    ValueError
        If any confidence is outside [0, 1], *tau* ≤ 0, or shapes mismatch.

    Examples
    --------
    >>> idzorek_confidence_to_omega(confidences=[0.8], cov_matrix=[[0.04]], P=[[1]], tau=0.05)
    array([[0.0005]])
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")

    conf = np.asarray(confidences, dtype=float).flatten()
    cov = np.asarray(cov_matrix, dtype=float)
    n_views = len(conf)

    if np.any(conf < 0) or np.any(conf > 1):
        raise ValueError(
            f"All confidences must be in [0, 1]; invalid values: {conf[(conf < 0) | (conf > 1)]}."
        )
    if P.shape[0] != n_views:
        raise ValueError(
            f"P has {P.shape[0]} rows but confidences has {n_views} entries."
        )

    # alpha_k = (1 - c_k) / c_k  (c=0 → very high uncertainty via eps guard)
    eps = 1e-12
    alpha = (1.0 - conf) / np.maximum(conf, eps)           # shape (K,)

    # variance per view: diag(P Σ P^T) computed without forming the full matrix
    var_per_view = np.einsum("ki,ij,kj->k", P, cov, P)    # shape (K,)

    return np.diag(tau * alpha * var_per_view)


def omega(
    cov_matrix: MatrixLike,
    P: np.ndarray,
    tau: float = 0.05,
    confidences: Optional[ArrayLike] = None,
    method: OmegaMethod = "idzorek",
) -> np.ndarray:
    """
    Calculate the view-uncertainty matrix (Ω) for a set of investor views.

    Ω is a diagonal covariance matrix of view-error terms. Larger diagonal
    entries imply lower confidence in the corresponding view.

    Parameters
    ----------
    cov_matrix : MatrixLike
        Prior covariance matrix (Σ). Shape (N × N).
    P : np.ndarray
        Picking matrix (K × N), mapping K views to N assets.
    tau : float, default 0.05
        Uncertainty scaling factor (τ). Must be positive.
    confidences : ArrayLike, optional
        Confidence levels in [0, 1] per view. Required for ``'idzorek'``.
    method : OmegaMethod, default 'idzorek'
        How Ω is constructed:

        - ``'idzorek'``: Confidence-calibrated (requires *confidences*).
        - ``'proportional'``: Ω = diag(τ · P · Σ · P^T).

    Returns
    -------
    np.ndarray
        Diagonal uncertainty matrix (Ω) of shape (K × K).

    Raises
    ------
    ValueError
        If ``method='idzorek'`` and *confidences* is None, or unknown method.

    Examples
    --------
    >>> omega(cov_matrix=[[0.04]], P=[[1]], tau=0.05, method="proportional")
    array([[0.002]])
    """
    if method == "idzorek":
        if confidences is None:
            raise ValueError("confidences must be provided when method='idzorek'.")
        return idzorek_confidence_to_omega(confidences, cov_matrix, P, tau=tau)
    elif method == "proportional":
        cov = np.asarray(cov_matrix, dtype=float)
        var_per_view = np.einsum("ki,ij,kj->k", P, cov, P) # (P @ cov * P).sum(axis=1)
        return np.diag(tau * var_per_view)
    else:
        raise ValueError(
            f"Unknown omega method '{method}'. Choose 'idzorek' or 'proportional'."
        )


def compute_bl_weights(
    posterior_returns: ArrayLike,
    posterior_cov: MatrixLike,
    risk_aversion: float = 1.0,
    bounds: PortfolioBounds = None,
    objective: str = "utility",
    benchmark_weights: Optional[ArrayLike] = None,
    custom_constraints: PortfolioConstraints = None,
    target_sum: Optional[float] = None,
    risk_free_rate: float = 0.0,
    max_iter: int = 1000,
) -> BLWeightsResult:
    r"""
    Compute optimal portfolio weights from Black-Litterman posterior estimates.

    Maximises expected utility or minimises tracking error:

    - **Utility**: :math:`U = w^T (E[R] - R_f) + R_f - \tfrac{\lambda}{2} w^T \Sigma w`
    - **Tracking Error**: :math:`TE = (w - w_b)^T \Sigma (w - w_b)`

    When no bounds or custom constraints are imposed and ``objective='utility'``,
    the closed-form solution is used:

    .. math::
        w^* = \frac{1}{\lambda} \Sigma^{-1} (\mu - R_f)

    Parameters
    ----------
    posterior_returns : ArrayLike
        Posterior expected returns E[R]. Shape (N,).
    posterior_cov : MatrixLike
        Posterior covariance matrix Σ. Shape (N, N).
    risk_aversion : float, default 1.0
        Risk-aversion coefficient λ. Must be positive.
    bounds : PortfolioBounds, optional
        Per-asset weight bounds. Either a single ``(lo, hi)`` applied to all
        assets, or a list of ``(lo, hi)`` per asset.
    objective : str, default 'utility'
        ``'utility'`` or ``'tracking_error'``.
    benchmark_weights : ArrayLike, optional
        Benchmark weights w_b. Required when ``objective='tracking_error'``.
    custom_constraints : PortfolioConstraints, optional
        Extra ``scipy.optimize`` constraint dicts (e.g. sector caps).
    target_sum : float or None, default 1.0
        Target risky-weight sum.
        ``1.0`` → fully invested; ``None`` → unconstrained (remaining
        cash earns *risk_free_rate*); any positive float enables leverage.
    risk_free_rate : float, default 0.0
        Annualised risk-free rate (R_f). Cash balance earns this rate.
    max_iter : int, default 1000
        Maximum SLSQP iterations for the numerical path.

    Returns
    -------
    BLWeightsResult
        Optimal portfolio weights as a ``pd.Series`` (when inputs are Series)
        or ``np.ndarray``.

    Raises
    ------
    ValueError
        If *risk_aversion* ≤ 0, unknown *objective*, or mismatched shapes.

    Examples
    --------
    >>> compute_bl_weights(posterior_returns=[0.05, 0.08], posterior_cov=[[0.04, 0.01], [0.01, 0.05]], risk_aversion=3.0)
    array([0.29824561, 0.47368421])
    """
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}.")
    if objective not in ("utility", "tracking_error"):
        raise ValueError(
            f"Unknown objective '{objective}'. Choose 'utility' or 'tracking_error'."
        )

    returns = np.asarray(posterior_returns, dtype=float).flatten()
    cov = np.asarray(posterior_cov, dtype=float)
    n_assets = len(returns)

    if cov.shape != (n_assets, n_assets):
        raise ValueError(
            f"posterior_cov shape {cov.shape} does not match "
            f"posterior_returns length {n_assets}."
        )

    # ------------------------------------------------------------------
    # Fast path: closed-form unconstrained utility maximisation
    # ------------------------------------------------------------------
    no_constraints = not custom_constraints
    if bounds is None and objective == "utility" and no_constraints:
        A = risk_aversion * cov
        b = returns - risk_free_rate
        try:
            weights = solve(A, b, assume_a="pos")
        except Exception:
            try:
                weights = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(A) @ b

        if target_sum is not None:
            total = weights.sum()
            weights = (weights / total * target_sum) if total != 0 else weights

        if isinstance(posterior_returns, pd.Series):
            return pd.Series(weights, index=posterior_returns.index)
        return weights

    # ------------------------------------------------------------------
    # Numerical optimisation path (SLSQP)
    # ------------------------------------------------------------------
    def utility_obj(w: np.ndarray) -> float:
        risky_ret = w @ returns
        cash_ret = (1.0 - w.sum()) * risk_free_rate
        port_var = w @ cov @ w
        return -(risky_ret + cash_ret - 0.5 * risk_aversion * port_var)

    def tracking_error_obj(w: np.ndarray) -> float:
        if benchmark_weights is None:
            raise ValueError(
                "benchmark_weights must be provided for tracking_error objective."
            )
        bench_w = np.asarray(benchmark_weights, dtype=float).flatten()
        diff = w - bench_w
        return diff @ cov @ diff

    target_obj = utility_obj if objective == "utility" else tracking_error_obj

    # Constraints
    constraints: list = []
    if target_sum is not None:
        constraints.append({"type": "eq", "fun": lambda w: w.sum() - target_sum})
    if custom_constraints:
        constraints.extend(custom_constraints)

    # Bounds processing
    processed_bounds = None
    if bounds is not None:
        if (
            isinstance(bounds, tuple)
            and len(bounds) == 2
            and isinstance(bounds[0], (int, float))
        ):
            processed_bounds = [bounds] * n_assets
        elif isinstance(bounds, (list, tuple)):
            if len(bounds) != n_assets:
                raise ValueError(
                    f"len(bounds)={len(bounds)} must equal n_assets={n_assets}."
                )
            processed_bounds = list(bounds)
        else:
            raise ValueError(
                "bounds must be a (lo, hi) tuple or a list of (lo, hi) per asset."
            )

    # Smart initial guess: try the closed-form solution first
    try:
        x0 = np.linalg.solve(risk_aversion * cov, returns - risk_free_rate)
        total = x0.sum()
        if target_sum is not None and total != 0:
            x0 = x0 / total * target_sum
        if not np.all(np.isfinite(x0)):
            raise ValueError
    except Exception:
        if benchmark_weights is not None:
            x0 = np.asarray(benchmark_weights, dtype=float).flatten()
        else:
            t = target_sum if target_sum is not None else 1.0
            x0 = np.full(n_assets, t / n_assets)

    res = minimize(
        target_obj,
        x0,
        method="SLSQP",
        bounds=processed_bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-9},
    )

    if not res.success:
        warnings.warn(
            f"Portfolio optimisation did not fully converge: {res.message}",
            UserWarning,
            stacklevel=2,
        )

    if isinstance(posterior_returns, pd.Series):
        return pd.Series(res.x, index=posterior_returns.index)
    return res.x


def bl_posterior_distribution(
    cov_matrix: MatrixLike,
    prior_returns: Optional[ArrayLike] = None,
    views: Optional[ViewData] = None,
    view_confidences: Optional[ArrayLike] = None,
    tau_val: Optional[float] = None,
    tau_method: TauMethod = "default",
    omega_method: OmegaMethod = "idzorek",
    prices: Optional[PriceData] = None,
    market_weights: Optional[ArrayLike] = None,
    risk_aversion: float = 1.0,
    relative_views: bool = False,
) -> BLResult:
    r"""
    Compute Black-Litterman posterior returns and covariance (Walters form).

    The Black-Litterman model blends a prior equilibrium distribution with
    subjective investor views into a refined posterior:

    .. math::
        E[R] = \bigl[(\tau\Sigma)^{-1} + P^T\Omega^{-1}P\bigr]^{-1}
               \bigl[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q\bigr]

    .. math::
        \Sigma_{post} = \Sigma + \bigl[(\tau\Sigma)^{-1}
                         + P^T\Omega^{-1}P\bigr]^{-1}

    Parameters
    ----------
    cov_matrix : MatrixLike
        Prior covariance matrix Σ. Shape (N, N).
    prior_returns : ArrayLike, optional
        Prior equilibrium returns Π. If None and *market_weights* is also None,
        a uniform 1/N prior is used with a warning.
    views : ViewData, optional
        Subjective views Q.

        - **Absolute** (default): ``{ticker: expected_return}``
        - **Relative** (``relative_views=True``): ``{(long, short): spread}``
          — the long leg is expected to outperform the short by *spread*.
    view_confidences : ArrayLike, optional
        Confidence levels in [0, 1] per view. Required for ``'idzorek'``.
    tau_val : float, optional
        Uncertainty scaling factor τ. If None, derived via *tau_method*.
    tau_method : TauMethod, default 'default'
        Method to compute τ when *tau_val* is None.
    omega_method : OmegaMethod, default 'idzorek'
        Method to build Ω: ``'idzorek'`` or ``'proportional'``.
    prices : PriceData, optional
        Historical prices, used when ``tau_method='variance'``.
    market_weights : ArrayLike, optional
        Market-cap weights. When provided, the prior is auto-computed via
        :func:`market_prior_returns` (reverse optimisation).
    risk_aversion : float, default 1.0
        Risk-aversion coefficient λ used for reverse optimisation when
        *market_weights* is supplied.
    relative_views : bool, default False
        When True, view keys are ``(long_ticker, short_ticker)`` pairs.

    Returns
    -------
    BLResult
        ``(posterior_returns, posterior_covariance)`` — both indexed by the
        asset names taken from *cov_matrix*.

    Raises
    ------
    ValueError
        On shape mismatches, unknown tickers in views, or invalid view keys.

    Examples
    --------
    >>> post_ret, post_cov = bl_posterior_distribution(cov_matrix=cov, prior_returns=pi, views={'AAPL': 0.15}, tau_val=0.05, omega_method="proportional")
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n_assets = len(cov)
    tickers = (
        list(cov_matrix.columns)
        if hasattr(cov_matrix, "columns")
        else list(range(n_assets))
    )

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square 2-D matrix.")

    # ── τ ──────────────────────────────────────────────────────────────────
    if tau_val is None:
        tau_val = tau(prices, method=tau_method)

    # ── Prior returns Π ────────────────────────────────────────────────────
    if market_weights is not None:
        prior_ret = np.asarray(
            market_prior_returns(market_weights, risk_aversion, cov_matrix),
            dtype=float,
        ).flatten()
    elif prior_returns is not None:
        if isinstance(prior_returns, pd.Series):
            prior_ret = prior_returns.reindex(tickers).values.astype(float)
        else:
            prior_ret = np.asarray(prior_returns, dtype=float).flatten()
        if len(prior_ret) != n_assets:
            raise ValueError(
                f"prior_returns length {len(prior_ret)} != n_assets {n_assets}."
            )
    else:
        warnings.warn(
            "No prior returns or market_weights provided; using equal-weighted prior (1/N).",
            UserWarning,
            stacklevel=2,
        )
        prior_ret = np.ones(n_assets) / n_assets

    prior_ret = prior_ret.reshape(-1, 1)

    # ── No views → inflate prior uncertainty and return ────────────────────
    if views is None:
        posterior_cov = cov * (1.0 + tau_val)
        return (
            pd.Series(prior_ret.flatten(), index=tickers),
            pd.DataFrame(posterior_cov, index=tickers, columns=tickers),
        )

    # ── Build P (K×N) and Q (K×1) from views ─────────────────────────────
    if isinstance(views, dict):
        views = pd.Series(views)

    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n_views = len(views)
    P = np.zeros((n_views, n_assets))
    Q = np.zeros((n_views, 1))

    for i, (key, ret) in enumerate(views.items()):
        if relative_views:
            if not (isinstance(key, (tuple, list)) and len(key) == 2):
                raise ValueError(
                    "Relative views require keys as (long_ticker, short_ticker) pairs."
                )
            long_t, short_t = key
            if long_t not in ticker_idx:
                raise ValueError(f"Long-leg ticker '{long_t}' not in asset universe.")
            if short_t not in ticker_idx:
                raise ValueError(f"Short-leg ticker '{short_t}' not in asset universe.")
            P[i, ticker_idx[long_t]] = 1.0
            P[i, ticker_idx[short_t]] = -1.0
        else:
            if key not in ticker_idx:
                raise ValueError(f"View ticker '{key}' not in asset universe.")
            P[i, ticker_idx[key]] = 1.0
        Q[i] = ret

    # ── Ω ──────────────────────────────────────────────────────────────────
    if view_confidences is not None or omega_method == "proportional":
        omega_matrix = omega(
            cov_matrix, P, tau=tau_val,
            confidences=view_confidences, method=omega_method,
        )
    else:
        # Default: proportional when no confidences supplied
        omega_matrix = omega(cov_matrix, P, tau=tau_val, method="proportional")

    # ── Walters canonical form — stable matrix inversions ──────────────────
    tau_cov = tau_val * cov

    try:
        c, low = cho_factor(tau_cov)
        tau_cov_inv = cho_solve((c, low), np.eye(n_assets))
    except Exception:
        try:
            tau_cov_inv = np.linalg.inv(tau_cov)
        except np.linalg.LinAlgError:
            tau_cov_inv = np.linalg.pinv(tau_cov)

    try:
        c, low = cho_factor(omega_matrix)
        omega_inv = cho_solve((c, low), np.eye(n_views))
    except Exception:
        try:
            omega_inv = np.linalg.inv(omega_matrix)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(omega_matrix)

    # M_inv = (τΣ)⁻¹ + P^T Ω⁻¹ P  →  M = M_inv⁻¹
    M_inv = tau_cov_inv + P.T @ omega_inv @ P
    try:
        c, low = cho_factor(M_inv)
        M = cho_solve((c, low), np.eye(n_assets))
    except Exception:
        try:
            M = np.linalg.inv(M_inv)
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(M_inv)

    # E[R] = M [(τΣ)⁻¹ Π + P^T Ω⁻¹ Q]
    posterior_ret = M @ (tau_cov_inv @ prior_ret + P.T @ omega_inv @ Q)

    # Σ_post = Σ + M  (Walters 2008, eq. 17); enforce symmetry
    posterior_cov = cov + M
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)

    return (
        pd.Series(posterior_ret.flatten(), index=tickers),
        pd.DataFrame(posterior_cov, index=tickers, columns=tickers),
    )


def black_litterman_single_asset(
    price_series: PriceData,
    prior_return: Optional[float] = None,
    view: Optional[float] = None,
    view_confidence: float = 0.5,
    tau_val: float = 0.05,
    risk_aversion: float = 1.0,
) -> BLSingleResult:
    """
    Apply Black-Litterman to a single asset with an optional view.

    This is the scalar reduction of the full model:

    .. math::
        E[R] = \\left[\\frac{1}{\\tau\\sigma^2} + \\frac{1}{\\omega}\\right]^{-1}
               \\left[\\frac{\\Pi}{\\tau\\sigma^2} + \\frac{Q}{\\omega}\\right]

    .. math::
        \\sigma^2_{post} = \\sigma^2 +
               \\left[\\frac{1}{\\tau\\sigma^2} + \\frac{1}{\\omega}\\right]^{-1}

    Parameters
    ----------
    price_series : PriceData
        Price or return series for the single asset.
    prior_return : float, optional
        Prior equilibrium return (Π). If None, uses the historical mean.
    view : float, optional
        Investor's view on the expected return (Q). If None, returns the
        prior with inflated uncertainty.
    view_confidence : float, default 0.5
        Investor confidence in the view, in [0, 1].
    tau_val : float, default 0.05
        Uncertainty scaling factor (τ). Must be positive.
    risk_aversion : float, default 1.0
        Risk-aversion coefficient (λ). Must be positive.

    Returns
    -------
    BLSingleResult
        A tuple ``(posterior_return, posterior_variance, optimal_weight)``.

    Raises
    ------
    ValueError
        If *tau_val* or *risk_aversion* ≤ 0, or *view_confidence* ∉ [0, 1].

    Examples
    --------
    >>> black_litterman_single_asset(price_series=prices, view=0.10, view_confidence=0.8)
    (0.08, 0.04, 0.5)
    """
    if tau_val <= 0:
        raise ValueError(f"tau_val must be positive, got {tau_val}.")
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}.")
    if not 0 <= view_confidence <= 1:
        raise ValueError(f"view_confidence must be in [0, 1], got {view_confidence}.")

    series = (
        pd.Series(price_series)
        if not isinstance(price_series, pd.Series)
        else price_series
    )

    if series.empty:
        return 0.0, 0.0, 0.0

    # Convert prices → returns when values look like price levels (> 1)
    returns = series.pct_change().dropna() if (series > 1).all() else series.dropna()

    variance = float(returns.var())
    if variance <= 0:
        variance = 1e-10  # guard against degenerate series

    if prior_return is None:
        prior_return = float(returns.mean())

    # No view: return prior with inflated variance
    if view is None:
        posterior_return = prior_return
        posterior_variance = variance * (1.0 + tau_val)
        weight = posterior_return / (risk_aversion * variance)
        return posterior_return, posterior_variance, float(np.clip(weight, 0, 1))

    # Idzorek scalar ω
    eps = 1e-12
    alpha = (1.0 - view_confidence) / max(view_confidence, eps)
    omega_val = tau_val * alpha * variance

    tau_var_inv = 1.0 / (tau_val * variance)
    omega_inv = 1.0 / max(omega_val, eps)

    M = 1.0 / (tau_var_inv + omega_inv)
    posterior_return = float(M * (tau_var_inv * prior_return + omega_inv * view))
    posterior_variance = variance + M
    weight = posterior_return / (risk_aversion * variance)

    return posterior_return, posterior_variance, float(np.clip(weight, 0, 1))