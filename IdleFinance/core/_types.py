"""
Centralized type definitions for IdleFinance configuration parameters.

This module provides Literal types and Union aliases to ensure consistent 
configuration and better IDE support across the library.
"""

from typing import Literal, Union, List, Dict, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd

# Configuration Settings
OmegaMethod = Literal["idzorek", "proportional"]
TauMethod = Literal["default", "variance"]
ObjectiveType = Literal["utility", "tracking_error"]
CovarianceMethod = Literal["sample", "ema"]
ReturnsMethod = Literal["mean_historical", "ema", "capm"]
DenoiseMethod = Literal["marchenko_pastur"]

# Common Type Aliases
ArrayLike = Union[np.ndarray, pd.Series, List[float]]
MatrixLike = Union[np.ndarray, pd.DataFrame]
PriceData = Union[pd.DataFrame, pd.Series, np.ndarray]

# Domain-Specific Output Aliases
NumericOutput = Union[float, pd.Series]
FinancialOutput = Union[pd.Series, pd.DataFrame]
VectorOutput = Union[pd.Series, np.ndarray]
Weights = pd.Series
CovarianceMatrix = pd.DataFrame

# Black-Litterman specific
ViewData = Union[Dict[str, float], pd.Series]
PortfolioBounds = Optional[Union[Tuple, List[Tuple]]]
PortfolioConstraints = Optional[List[Dict]]
BLResult = Tuple[pd.Series, pd.DataFrame]
BLWeightsResult = pd.Series
BLSingleResult = Tuple[float, float, float]
BLFullOutput = Tuple[pd.Series, pd.DataFrame, pd.Series]
