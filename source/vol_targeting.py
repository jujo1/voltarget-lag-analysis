"""
Portfolio Volatility Targeting Core Module
==========================================

Implements exact single-step covariance-based volatility targeting:
    k = σ_target / √(w' @ Σ @ w)

Features:
- Dataclasses with __slots__ for memory efficiency
- EWMA and Ledoit-Wolf covariance estimation
- Walk-forward estimation preventing lookahead bias
- Zero fallbacks - explicit error handling only

References:
- Moreira & Muir (2017) "Volatility-Managed Portfolios"
- Ledoit & Wolf (2004) "Shrinkage Estimator"
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PortfolioConfig:
    """Immutable configuration for volatility targeting."""
    
    target_vol: float
    lookback_days: int
    decay_factor: float
    annualization_factor: float = np.sqrt(252)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.01 <= self.target_vol <= 1.0:
            raise ValueError(f"target_vol must be in [0.01, 1.0], got {self.target_vol}")
        if self.lookback_days < 10:
            raise ValueError(f"lookback_days must be >= 10, got {self.lookback_days}")
        if not 0.5 <= self.decay_factor <= 0.99:
            raise ValueError(f"decay_factor must be in [0.5, 0.99], got {self.decay_factor}")


@dataclass(slots=True)
class CovarianceResult:
    """Result of covariance matrix estimation."""
    
    matrix: NDArray[np.float64]
    estimation_date: pd.Timestamp
    method: Literal["ewma", "ledoit_wolf"]
    lookback_days: int
    
    def __post_init__(self) -> None:
        """Validate covariance matrix properties."""
        if self.matrix.ndim != 2:
            raise ValueError(f"Covariance matrix must be 2D, got {self.matrix.ndim}D")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Covariance matrix must be square, got {self.matrix.shape}")
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Covariance matrix is not positive semi-definite")


@dataclass(slots=True)
class VolatilityResult:
    """Result of volatility targeting computation."""
    
    scale_factor: float
    ex_ante_vol: float
    target_vol: float
    weights: NDArray[np.float64]
    covariance: CovarianceResult
    
    def __post_init__(self) -> None:
        """Validate volatility result."""
        if self.scale_factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {self.scale_factor}")
        if not np.isclose(self.ex_ante_vol, self.target_vol, rtol=1e-9):
            raise ValueError(
                f"Ex-ante vol {self.ex_ante_vol:.10f} does not equal target {self.target_vol:.10f}"
            )


@dataclass(slots=True)
class PnLResult:
    """Result of PnL calculation."""
    
    daily_pnl: pd.Series
    cumulative_pnl: pd.Series
    realized_vol: float
    sharpe_ratio: float
    
    def __post_init__(self) -> None:
        """Validate PnL result."""
        if not isinstance(self.daily_pnl, pd.Series):
            raise TypeError("daily_pnl must be pd.Series")
        if not isinstance(self.daily_pnl.index, pd.DatetimeIndex):
            raise TypeError("daily_pnl index must be DatetimeIndex")


def compute_ewma_covariance(
    returns: pd.DataFrame,
    decay_factor: float,
    estimation_date: pd.Timestamp
) -> CovarianceResult:
    """
    Compute EWMA covariance matrix using only past data.
    
    Uses exponentially weighted moving average with strict causality:
    - Only data before estimation_date is used
    - No forward-looking bias possible
    """
    logger.debug(f"Computing EWMA covariance for {estimation_date}")
    
    if returns.empty:
        raise ValueError("Returns DataFrame is empty")
    
    if estimation_date not in returns.index:
        past_data = returns.loc[returns.index < estimation_date]
    else:
        idx_pos = returns.index.get_loc(estimation_date)
        past_data = returns.iloc[:idx_pos]
    
    if len(past_data) < 10:
        raise ValueError(f"Insufficient data: {len(past_data)} rows, need >= 10")
    
    if past_data.isna().all().any():
        cols_all_nan = past_data.columns[past_data.isna().all()].tolist()
        raise ValueError(f"Columns with all NaN values: {cols_all_nan}")
    
    clean_data = past_data.dropna(axis=1, how="all").fillna(0.0)
    
    n_obs = len(clean_data)
    weights = np.array([(1 - decay_factor) * (decay_factor ** i) for i in range(n_obs)])
    weights = weights[::-1]
    weights = weights / weights.sum()
    
    centered = clean_data.values - np.average(clean_data.values, axis=0, weights=weights)
    cov_matrix = np.einsum("t,ti,tj->ij", weights, centered, centered)
    
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    
    return CovarianceResult(
        matrix=cov_matrix,
        estimation_date=estimation_date,
        method="ewma",
        lookback_days=n_obs
    )


def compute_ledoit_wolf_covariance(
    returns: pd.DataFrame,
    estimation_date: pd.Timestamp,
    lookback_days: Optional[int] = None
) -> CovarianceResult:
    """
    Compute Ledoit-Wolf shrinkage covariance matrix.
    
    Uses sklearn's implementation with strict causality.
    Shrinkage toward diagonal provides regularization.
    """
    logger.debug(f"Computing Ledoit-Wolf covariance for {estimation_date}")
    
    if returns.empty:
        raise ValueError("Returns DataFrame is empty")
    
    if estimation_date not in returns.index:
        past_data = returns.loc[returns.index < estimation_date]
    else:
        idx_pos = returns.index.get_loc(estimation_date)
        past_data = returns.iloc[:idx_pos]
    
    if lookback_days is not None:
        past_data = past_data.iloc[-lookback_days:]
    
    if len(past_data) < 10:
        raise ValueError(f"Insufficient data: {len(past_data)} rows, need >= 10")
    
    clean_data = past_data.dropna(axis=1, how="all").fillna(0.0)
    
    lw = LedoitWolf()
    lw.fit(clean_data.values)
    
    return CovarianceResult(
        matrix=lw.covariance_,
        estimation_date=estimation_date,
        method="ledoit_wolf",
        lookback_days=len(clean_data)
    )


def compute_scale_factor(
    weights: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
    target_vol: float,
    annualization: float = np.sqrt(252)
) -> float:
    """
    Compute volatility targeting scale factor using exact formula.
    
    Formula: k = σ_target / (√(w' @ Σ @ w) * annualization)
    
    This ensures ex-ante annualized volatility equals target exactly.
    """
    if weights.ndim != 1:
        raise ValueError(f"Weights must be 1D, got {weights.ndim}D")
    if cov_matrix.ndim != 2:
        raise ValueError(f"Covariance must be 2D, got {cov_matrix.ndim}D")
    if len(weights) != cov_matrix.shape[0]:
        raise ValueError(
            f"Dimension mismatch: weights {len(weights)}, cov {cov_matrix.shape[0]}"
        )
    
    portfolio_variance = weights @ cov_matrix @ weights
    
    if portfolio_variance <= 0:
        raise ValueError(f"Portfolio variance must be positive, got {portfolio_variance}")
    
    portfolio_daily_vol = np.sqrt(portfolio_variance)
    portfolio_annual_vol = portfolio_daily_vol * annualization
    
    scale_factor = target_vol / portfolio_annual_vol
    
    logger.debug(
        f"Scale factor: {scale_factor:.6f} "
        f"(daily_vol={portfolio_daily_vol:.6f}, annual_vol={portfolio_annual_vol:.6f})"
    )
    
    return scale_factor


def apply_volatility_target(
    allocations: pd.DataFrame,
    returns: pd.DataFrame,
    config: PortfolioConfig,
    cov_method: Literal["ewma", "ledoit_wolf"] = "ewma"
) -> pd.DataFrame:
    """
    Apply volatility targeting to allocation weights using walk-forward approach.
    
    For each date t:
    1. Estimate Σ using data up to t-1 (no lookahead)
    2. Compute scale factor k = σ_target / √(w' @ Σ @ w)
    3. Scale positions: w_scaled = k * w
    
    Returns scaled allocations with target volatility.
    """
    logger.info(f"Applying volatility target: {config.target_vol:.1%}")
    
    if allocations.empty:
        raise ValueError("Allocations DataFrame is empty")
    if returns.empty:
        raise ValueError("Returns DataFrame is empty")
    
    if not allocations.index.equals(returns.index):
        raise ValueError("Allocations and returns must have same index")
    if not allocations.columns.equals(returns.columns):
        raise ValueError("Allocations and returns must have same columns")
    
    scaled_allocations = pd.DataFrame(
        index=allocations.index,
        columns=allocations.columns,
        dtype=np.float64
    )
    
    start_idx = config.lookback_days
    
    for i in range(start_idx, len(allocations)):
        current_date = allocations.index[i]
        current_weights = allocations.iloc[i].fillna(0.0).values
        
        if np.allclose(current_weights, 0):
            scaled_allocations.iloc[i] = 0.0
            continue
        
        if cov_method == "ewma":
            cov_result = compute_ewma_covariance(
                returns=returns,
                decay_factor=config.decay_factor,
                estimation_date=current_date
            )
        else:
            cov_result = compute_ledoit_wolf_covariance(
                returns=returns,
                estimation_date=current_date,
                lookback_days=config.lookback_days
            )
        
        scale = compute_scale_factor(
            weights=current_weights,
            cov_matrix=cov_result.matrix,
            target_vol=config.target_vol,
            annualization=config.annualization_factor
        )
        
        scaled_allocations.iloc[i] = current_weights * scale
    
    scaled_allocations.iloc[:start_idx] = np.nan
    
    logger.info(f"Volatility targeting applied to {len(allocations) - start_idx} dates")
    
    return scaled_allocations


def compute_portfolio_returns(
    positions: pd.DataFrame,
    returns: pd.DataFrame
) -> pd.Series:
    """
    Compute portfolio returns from positions and instrument returns.
    
    Timing: positions[t] * returns[t+1] = pnl[t+1]
    This ensures no forward-looking bias.
    """
    if positions.empty:
        raise ValueError("Positions DataFrame is empty")
    if returns.empty:
        raise ValueError("Returns DataFrame is empty")
    
    positions_shifted = positions.shift(1)
    
    portfolio_returns = (positions_shifted * returns).sum(axis=1)
    
    portfolio_returns = portfolio_returns.iloc[1:]
    
    return portfolio_returns


def compute_realized_volatility(
    portfolio_returns: pd.Series,
    annualization: float = np.sqrt(252)
) -> float:
    """Compute annualized realized volatility from portfolio returns."""
    if len(portfolio_returns) < 2:
        raise ValueError("Need at least 2 returns to compute volatility")
    
    clean_returns = portfolio_returns.dropna()
    
    if len(clean_returns) < 2:
        raise ValueError("Not enough non-NaN returns to compute volatility")
    
    daily_std = clean_returns.std()
    annualized_vol = daily_std * annualization
    
    return annualized_vol


def calculate_portfolio_pnl(
    positions: pd.DataFrame,
    returns: pd.DataFrame
) -> PnLResult:
    """
    Calculate full PnL results for a portfolio.
    
    Returns PnLResult with daily/cumulative PnL, realized vol, and Sharpe.
    """
    logger.info("Calculating portfolio PnL")
    
    portfolio_rets = compute_portfolio_returns(positions, returns)
    
    daily_pnl = portfolio_rets.copy()
    cumulative_pnl = daily_pnl.cumsum()
    
    realized_vol = compute_realized_volatility(portfolio_rets)
    
    mean_return = portfolio_rets.mean()
    if realized_vol > 0:
        sharpe = (mean_return * 252) / realized_vol
    else:
        sharpe = 0.0
    
    logger.info(f"Realized vol: {realized_vol:.2%}, Sharpe: {sharpe:.2f}")
    
    return PnLResult(
        daily_pnl=daily_pnl,
        cumulative_pnl=cumulative_pnl,
        realized_vol=realized_vol,
        sharpe_ratio=sharpe
    )
