# Test Strategy: Real Data Portfolio Volatility Targeting

## Overview

This document defines the test strategy for using REAL allocation data from CSV files instead of synthetic `np.random()` data.

## User Request (Verbatim)
> "use PK data files *.csv to build 100% testing of all test cases, using real data"

## Current State (Problem)
- 9 tests use `np.random.randn()` for allocations
- 0 tests reference real allocation data
- 9,527 non-zero allocation values in fixtures are IGNORED

## Target State (Solution)
- ALL tests use `bundle.alloc_raw`, `bundle.alloc_usd`, or `bundle.alloc_lots`
- ZERO uses of `np.random` for allocation generation
- Real market dynamics captured in tests

## Instruments Selected for Testing

Based on analysis, these 20 instruments have:
- 100% date coverage in tail 300 days
- Both allocation AND return data available
- Actual trading activity (non-zero allocations)

```python
TEST_INSTRUMENTS = [
    'CC-1-Comdty', 'CC-2-Comdty',   # Cocoa futures
    'CF-1-Index',                     # CAC 40 Index
    'C_-1-Comdty', 'C_-2-Comdty',   # Corn futures
    'ES-1-Index',                     # S&P 500 E-mini
    'IB-1-Index',                     # IBEX 35 Index
    'SM-1-Index',                     # SMI Index
    'ST-1-Index',                     # STOXX 50 Index
    'VG-1-Index',                     # VIX Index
    'W_-1-Comdty', 'W_-2-Comdty',   # Wheat futures
    'Z_-1-Index',                     # FTSE 100 Index
    'LMNIDP-3-Comdty',               # Nickel
    'LMPBDP-3-Comdty',               # Lead
    'QC-1-Index',                     # NASDAQ 100
    'QS-1-Comdty', 'QS-2-Comdty',   # Gasoil
    'PA-1-Comdty', 'PA-2-Comdty',   # Palladium
]
```

## Test Categories and Changes

### Category A: Forward-Looking Bias Detection (6 tests)
**Current**: Uses synthetic data - OK (tests covariance causality)
**Change**: NO CHANGE NEEDED - These test algorithmic properties

### Category B: Portfolio Volatility Targeting (5 tests)
**Current**: Uses `np.random.randn()` for weights/allocations
**Change Required**:

| Test | Current | Change To |
|------|---------|-----------|
| B1: test_ex_ante_equals_target_exactly | `np.random.randn(n)` | `bundle.alloc_raw.iloc[100].values` |
| B2: test_ex_post_within_realistic_tolerance | `np.random.randn(n) * 0.01` | `bundle.alloc_raw[test_instruments]` |
| B3: test_scale_factor_formula_correctness | `np.random.randn(n)` | Keep synthetic (tests formula) |
| B4: test_covariance_positive_semidefinite | N/A | N/A |
| B5: test_different_target_levels | `np.random.randn(n) * 0.01` | `bundle.alloc_raw[test_instruments]` |

### Category C: Covariance Estimation (3 tests)
**Current**: Uses synthetic data - OK
**Change**: NO CHANGE NEEDED - Tests algorithmic properties

### Category D: PnL Calculation (3 tests)
**Current**: Uses synthetic allocations
**Change Required**:

| Test | Current | Change To |
|------|---------|-----------|
| D1: test_pnl_timing_correctness | `np.random.randn() * 0.01` | `bundle.alloc_raw` |
| D2: test_pnl_matches_returns_times_positions | Synthetic | `bundle.alloc_raw` |
| D3: test_pnl_aggregation | Synthetic | `bundle.alloc_raw` |

### Category E: Error Handling (3 tests)
**Current**: Tests error conditions - OK
**Change**: NO CHANGE NEEDED - Tests edge cases

### Category F: Integration Tests (2 tests)
**Current**: Uses synthetic allocations
**Change Required**:

| Test | Current | Change To |
|------|---------|-----------|
| F1: test_full_pipeline_real_data | `np.random.randn() * 0.01` | `bundle.alloc_raw[TEST_INSTRUMENTS]` |
| F2: test_stress_test_high_volatility_regime | `np.random.randn() * 0.01` | Real COVID period allocations |

## New Tests to Add

### G: Contract Value Tests (2 new tests)
- G1: test_contract_value_consistency
- G2: test_lot_allocation_rounding

### H: Real Data Validation (2 new tests)
- H1: test_real_allocation_statistics
- H2: test_cross_sectional_allocation_properties

## Implementation Pattern

```python
def test_with_real_allocations() -> Tuple[bool, str, str]:
    """Use real allocation data from fixtures."""
    bundle = load_fixture("fixture_tail_300")
    
    # Select instruments with actual allocations
    test_instruments = [col for col in bundle.alloc_raw.columns 
                        if bundle.alloc_raw[col].abs().sum() > 1e-10][:20]
    
    # Use REAL allocations
    allocations = bundle.alloc_raw[test_instruments].fillna(0)
    returns = bundle.returns[test_instruments].fillna(0)
    
    # Apply volatility targeting with REAL data
    config = PortfolioConfig(target_vol=0.15, lookback_days=60, decay_factor=0.94)
    scaled = apply_volatility_target(allocations, returns, config)
    
    # Compute results
    pnl_result = calculate_portfolio_pnl(scaled, returns)
    
    # Evidence
    evidence = f"Real allocations used: {len(test_instruments)} instruments"
    
    return passed, message, evidence
```

## Success Criteria

| ID | Criterion | Test Command | Evidence |
|----|-----------|--------------|----------|
| SC1 | Zero np.random for allocations | `grep -c 'np.random' tests.py == 0` (except B3) | grep output |
| SC2 | All tests pass | `pytest -v` | log file |
| SC3 | Real realized vol ~15% | Pipeline test | log metrics |
| SC4 | Third-party validation | OpenAI review | review text |

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Some instruments have sparse allocations | LOW | Filter to >= 50% coverage |
| Real allocations may be all same sign | MEDIUM | Verify cross-sectional distribution |
| COVID period has extreme vol | LOW | Expected - validates stress handling |

---
Generated: 2026-01-07T04:08:00Z
Stage: PLAN
