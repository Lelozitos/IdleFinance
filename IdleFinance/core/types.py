"""
Centralized type definitions for IdleFinance configuration parameters.

This module provides Literal types and Union aliases to ensure consistent 
configuration and better IDE support across the library.
"""

from typing import Literal, Union, List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Configuration Settings
OmegaMethod = Literal["idzorek", "proportional"]
TauMethod = Literal["default", "variance"]
ObjectiveType = Literal["utility", "tracking_error"]
CovarianceMethod = Literal["sample", "ema"]
ReturnsMethod = Literal["mean_historical", "ema", "capm"]

# Common Type Aliases
ArrayLike = Union[np.ndarray, pd.Series, List[float]]
MatrixLike = Union[np.ndarray, pd.DataFrame]
PriceData = Union[pd.DataFrame, pd.Series, np.ndarray]
