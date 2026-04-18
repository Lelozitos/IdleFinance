"""
Tests for Stochastic Processes: GBM, Mean-Reversion (OU), Jump Diffusion, Monte Carlo.
"""

import numpy as np
import pytest
from IdleFinance.models.stochastic import (
    geometric_brownian_motion,
    mean_reverting_process,
    jump_diffusion_process,
    monte_carlo,
)

SEED = 42


class TestGeometricBrownianMotion:
    def test_output_shape_single_path(self):
        paths = geometric_brownian_motion(100, 0.05, 0.2, 1 / 252, 252, random_seed=SEED)
        assert paths.shape == (253, 1)  # n_steps + 1 rows, 1 path

    def test_output_shape_multiple_paths(self):
        paths = geometric_brownian_motion(100, 0.05, 0.2, 1 / 252, 252, n_paths=5, random_seed=SEED)
        assert paths.shape == (253, 5)

    def test_all_paths_start_at_s0(self):
        paths = geometric_brownian_motion(100, 0.05, 0.2, 1 / 252, 252, n_paths=3, random_seed=SEED)
        assert np.all(paths[0] == 100.0)

    def test_prices_stay_positive(self):
        # GBM cannot produce negative prices
        paths = geometric_brownian_motion(100, 0.05, 0.2, 1 / 252, 252, n_paths=50, random_seed=SEED)
        assert np.all(paths > 0)

    def test_reproducible_with_seed(self):
        a = geometric_brownian_motion(100, 0.05, 0.2, 1 / 252, 100, random_seed=SEED)
        b = geometric_brownian_motion(100, 0.05, 0.2, 1 / 252, 100, random_seed=SEED)
        np.testing.assert_array_equal(a, b)

    def test_invalid_nonpositive_s0(self):
        with pytest.raises(ValueError):
            geometric_brownian_motion(0, 0.05, 0.2, 1 / 252, 252)

    def test_invalid_nonpositive_dt(self):
        with pytest.raises(ValueError):
            geometric_brownian_motion(100, 0.05, 0.2, 0, 252)


class TestMeanRevertingProcess:
    def test_output_shape(self):
        paths = mean_reverting_process(0.05, 0.3, 0.04, 0.01, 1 / 252, 252, n_paths=3, random_seed=SEED)
        assert paths.shape == (253, 3)

    def test_all_paths_start_at_s0(self):
        paths = mean_reverting_process(0.05, 0.3, 0.04, 0.01, 1 / 252, 252, n_paths=3, random_seed=SEED)
        assert np.all(paths[0] == 0.05)

    def test_invalid_nonpositive_theta(self):
        with pytest.raises(ValueError):
            mean_reverting_process(0.05, 0, 0.04, 0.01, 1 / 252, 252)


class TestJumpDiffusionProcess:
    def test_output_shape(self):
        paths = jump_diffusion_process(100, 0.05, 0.2, 2.0, -0.05, 0.1, 1 / 252, 252, n_paths=4, random_seed=SEED)
        assert paths.shape == (253, 4)

    def test_all_paths_start_at_s0(self):
        paths = jump_diffusion_process(100, 0.05, 0.2, 2.0, -0.05, 0.1, 1 / 252, 252, random_seed=SEED)
        assert np.all(paths[0] == 100.0)


class TestMonteCarlo:
    def test_dispatches_to_gbm(self):
        paths = monte_carlo("gbm", s0=100, mu=0.05, sigma=0.2, dt=1 / 252, n_steps=252, random_seed=SEED)
        assert paths.shape == (253, 1)

    def test_dispatches_to_mean_reversion(self):
        paths = monte_carlo("mean_reversion", s0=0.05, theta=0.3, mu=0.04, sigma=0.01, dt=1 / 252, n_steps=100, random_seed=SEED)
        assert paths.shape == (101, 1)

    def test_dispatches_to_jump_diffusion(self):
        paths = monte_carlo("jump_diffusion", s0=100, mu=0.05, sigma=0.2, lamb=2.0, mu_j=-0.05, sigma_j=0.1, dt=1 / 252, n_steps=100, random_seed=SEED)
        assert paths.shape == (101, 1)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            monte_carlo("unknown_method", s0=100)
