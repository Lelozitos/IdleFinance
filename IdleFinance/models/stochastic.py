"""
Stochastic process simulations for financial modeling.
"""

import numpy as np
from ..core._types import StochasticMethod

def geometric_brownian_motion(
    s0: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int = 1,
    random_seed: int = None,
) -> np.ndarray:
    """
    Simulate price paths using Geometric Brownian Motion (GBM).

    Formula: **S_t = S_{t-1} * exp((μ - σ²/2)*dt + σ*sqrt(dt)*Z)**

    Parameters
    ----------
    s0 : float
        Initial asset price.
    mu : float
        Annualized expected return (drift).
    sigma : float
        Annualized volatility.
    dt : float
        Time step (e.g., 1/252 for daily steps).
    n_steps : int
        Number of steps to simulate.
    n_paths : int, default 1
        Number of independent trajectories to simulate.
    random_seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (n_steps + 1, n_paths) containing the simulated paths.

    Raises
    ------
    ValueError
        If s0, sigma, or dt is not positive, or if n_steps is not a positive integer.

    Examples
    --------
    >>> paths = geometric_brownian_motion(s0=100.0, mu=0.05, sigma=0.2, dt=1/252, n_steps=252)
    >>> paths.shape
    (253, 1)
    """
    if s0 <= 0 or sigma < 0 or dt <= 0 or n_steps <= 0:
        raise ValueError("s0, sigma, dt, and n_steps must be positive.")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Pre-calculate deterministic component
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Generate shock terms
    shocks = np.random.normal(0, 1, (n_steps, n_paths))

    # Calculate log returns
    log_returns = drift + diffusion * shocks

    # Accumulate log returns and convert to price paths
    price_paths = np.zeros((n_steps + 1, n_paths))
    price_paths[0] = s0
    price_paths[1:] = s0 * np.exp(np.cumsum(log_returns, axis=0))

    return price_paths

def mean_reverting_process(
    s0: float,
    theta: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int = 1,
    random_seed: int = None,
) -> np.ndarray:
    """
    Simulate a Mean-Reverting process (Ornstein-Uhlenbeck).

    Formula: **dS_t = θ * (μ - S_t) * dt + σ * dW_t**
    Discretized: **S_t = S_{t-1} + θ * (μ - S_{t-1}) * dt + σ * sqrt(dt) * Z**

    Parameters
    ----------
    s0 : float
        Initial value.
    theta : float
        Speed of mean reversion.
    mu : float
        Long-term mean level.
    sigma : float
        Volatility.
    dt : float
        Time step.
    n_steps : int
        Number of steps.
    n_paths : int, default 1
        Number of paths.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Simulated paths.

    Raises
    ------
    ValueError
        If theta, sigma, dt, or n_steps are not positive.
    """
    if theta <= 0 or sigma < 0 or dt <= 0 or n_steps <= 0:
        raise ValueError("theta, sigma, dt, and n_steps must be positive.")

    if random_seed is not None:
        np.random.seed(random_seed)

    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = s0

    for t in range(1, n_steps + 1):
        z = np.random.normal(0, 1, n_paths)
        paths[t] = paths[t-1] + theta * (mu - paths[t-1]) * dt + sigma * np.sqrt(dt) * z

    return paths


def jump_diffusion_process(
    s0: float,
    mu: float,
    sigma: float,
    lamb: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    n_steps: int,
    n_paths: int = 1,
    random_seed: int = None,
) -> np.ndarray:
    """
    Simulate price paths using Merton Jump Diffusion.

    Formula: **dS_t/S_t = μ dt + σ dW_t + J_t dN_t**
    Where N_t is a Poisson process and J_t is the jump size.

    Parameters
    ----------
    s0 : float
        Initial price.
    mu : float
        Drift.
    sigma : float
        Volatility.
    lamb : float
        Jump intensity (expected jumps per year).
    mu_j : float
        Mean jump size (logarithmic).
    sigma_j : float
        Standard deviation of jump size.
    dt : float
        Time step.
    n_steps : int
        Number of steps.
    n_paths : int, default 1
        Number of paths.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Simulated paths.

    Raises
    ------
    ValueError
        If s0, sigma, lamb, or dt are not positive, or n_steps is not a positive integer.
    """
    if s0 <= 0 or sigma < 0 or lamb < 0 or dt <= 0 or n_steps <= 0:
        raise ValueError("s0, sigma, lamb, dt, and n_steps must be positive.")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Deterministic drift and diffusion
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Generate Brownian shocks
    z = np.random.normal(0, 1, (n_steps, n_paths))

    # Generate Jump shocks (Poisson for count, Normal for size)
    n_jumps = np.random.poisson(lamb * dt, (n_steps, n_paths))
    jump_sizes = np.random.normal(mu_j, sigma_j, (n_steps, n_paths))
    
    # Net jump component
    jump_term = n_jumps * jump_sizes

    # Calculate log returns
    log_returns = drift + diffusion * z + jump_term

    # Accumulate
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = s0
    paths[1:] = s0 * np.exp(np.cumsum(log_returns, axis=0))

    return paths


def monte_carlo(
    method: StochasticMethod = "gbm",
    **kwargs,
) -> np.ndarray:
    """
    General Monte Carlo simulation dispatcher.

    Parameters
    ----------
    method : StochasticMethod, default 'gbm'
        Simulation method: 'gbm', 'mean_reversion', 'jump_diffusion'.
    **kwargs
        Arguments passed to the specific simulation function.

    Returns
    -------
    np.ndarray
        Simulated paths.

    Raises
    ------
    ValueError
        If method is not a member of StochasticMethod.
    """
    if method == "gbm":
        return geometric_brownian_motion(**kwargs)
    elif method == "mean_reversion":
        return mean_reverting_process(**kwargs)
    elif method == "jump_diffusion":
        return jump_diffusion_process(**kwargs)
    else:
        raise ValueError(f"Unknown simulation method: {method}")