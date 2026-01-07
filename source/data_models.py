"""
Shared Data Models for Portfolio Volatility Targeting
======================================================

Contains dataclasses used across the volatility targeting system.
Must be importable by both fixture extraction and test modules.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DataBundle:
    """Container for all portfolio data with strict validation.
    
    Holds returns, allocations, contract values, and prices in a validated bundle.
    All DataFrames must have identical DatetimeIndex and column names.
    """
    
    returns: pd.DataFrame
    alloc_lots: pd.DataFrame
    alloc_raw: pd.DataFrame
    alloc_usd: pd.DataFrame
    contract_values: pd.DataFrame
    prices: pd.DataFrame
    
    def __post_init__(self) -> None:
        """Validate all DataFrames have consistent index and columns."""
        frames = [
            self.returns, self.alloc_lots, self.alloc_raw,
            self.alloc_usd, self.contract_values, self.prices
        ]
        if not all(f.index.equals(self.returns.index) for f in frames):
            raise ValueError("All DataFrames must have identical DatetimeIndex")
        if not all(f.columns.equals(self.returns.columns) for f in frames):
            raise ValueError("All DataFrames must have identical column names")


@dataclass(slots=True)
class PortfolioConfig:
    """Configuration for portfolio volatility targeting.
    
    Defines target volatility, estimation parameters, and annualization.
    """
    
    target_vol: float
    lookback_days: int
    decay_factor: float
    annualization_factor: float = 252.0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.target_vol <= 0:
            raise ValueError(f"target_vol must be positive, got {self.target_vol}")
        if self.lookback_days < 2:
            raise ValueError(f"lookback_days must be >= 2, got {self.lookback_days}")
        if not 0 < self.decay_factor < 1:
            raise ValueError(f"decay_factor must be in (0, 1), got {self.decay_factor}")
        if self.annualization_factor <= 0:
            raise ValueError(f"annualization_factor must be positive, got {self.annualization_factor}")


@dataclass(slots=True)
class CovarianceResult:
    """Result of covariance matrix estimation.
    
    Contains the estimated covariance matrix with metadata about estimation.
    Validates positive semi-definiteness on construction.
    """
    
    matrix: np.ndarray
    estimation_date: pd.Timestamp
    method: str
    lookback_days: int
    
    def __post_init__(self) -> None:
        """Validate covariance matrix is square and PSD."""
        if self.matrix.ndim != 2:
            raise ValueError(f"Covariance must be 2D, got {self.matrix.ndim}D")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Covariance must be square, got {self.matrix.shape}")
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(f"Covariance not PSD, min eigenvalue: {eigenvalues.min()}")


@dataclass(slots=True)
class VolatilityResult:
    """Result of volatility targeting calculation.
    
    Contains scaling factor and ex-ante volatility with supporting data.
    """
    
    scale_factor: float
    ex_ante_vol: float
    target_vol: float
    weights: np.ndarray
    covariance: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate volatility result."""
        if self.scale_factor < 0:
            raise ValueError(f"scale_factor cannot be negative, got {self.scale_factor}")
        if self.ex_ante_vol < 0:
            raise ValueError(f"ex_ante_vol cannot be negative, got {self.ex_ante_vol}")


@dataclass(slots=True)
class PnLResult:
    """Result of PnL calculation.
    
    Contains daily and cumulative PnL with performance metrics.
    """
    
    daily_pnl: np.ndarray
    cumulative_pnl: np.ndarray
    realized_vol: float
    sharpe_ratio: float
    
    def __post_init__(self) -> None:
        """Validate PnL arrays have consistent length."""
        if len(self.daily_pnl) != len(self.cumulative_pnl):
            raise ValueError("daily_pnl and cumulative_pnl must have same length")


@dataclass(slots=True, frozen=True)
class TestResult:
    """Immutable result from a single test execution.
    
    Used for structured test reporting and logging.
    """
    
    test_id: str
    test_name: str
    passed: bool
    message: str
    metric_value: Union[float, None] = None
    metric_name: Union[str, None] = None
