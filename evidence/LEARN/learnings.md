# Learnings from Volatility Targeting Real Data Remediation

**Date**: 2026-01-07T04:23:00Z
**Workflow**: voltarget_real_alloc_remediation

## Key Learnings

### 1. Fixture Design for Real Data Testing
**Learning**: Fixtures must be generated from instruments with ACTUAL non-zero allocations. 
**Impact**: Previous fixtures (fixture_subset_10) contained instruments with zero allocations, defeating the purpose of "real data" testing.
**Solution**: Create `find_active_instruments()` function that filters by allocation coverage percentage.

### 2. DataBundle Field Order Matters
**Learning**: When using `@dataclass(slots=True)`, field order in pickle files must match class definition exactly.
**Evidence**: DataBundle in data_models.py has specific order; generate_fixtures.py must match.
**Pattern**: Always verify dataclass field order consistency across modules.

### 3. Continuous Contract (CC) Returns ≠ Price Changes
**Learning**: For futures data, CC returns are roll-adjusted and do NOT equal simple price changes.
**Impact**: Test G3 initially failed because it expected high correlation between prices and returns.
**Solution**: Test price data consistency via positivity, magnitude, and coverage - NOT price-return correlation.

### 4. NaN Handling in Lookback Windows
**Learning**: Volatility targeting produces NaN for the first N days (lookback window).
**Impact**: Manual PnL calculations must use `.fillna(0)` to match API behavior.
**Evidence**: Test D2 failed with NaN correlation until manual calculation aligned with API.

### 5. Third-Party Validation Value
**Learning**: External review (OpenAI GPT) identified gaps that would have been missed.
**Impact**: Added tests for contract_value (G2) and prices (G3) per third-party recommendation.
**Pattern**: Always include third-party validation in workflow.

### 6. Allocation Coverage Analysis
**Finding**: Of 181 instruments in CSV, only 30 have >= 50% allocation coverage.
**Implication**: Tests should focus on actively traded instruments, not all columns.
**Data**: Top instruments: CC-1-Comdty (98.7%), CC-2-Comdty, C_-2-Comdty, VG-1-Index

## Workflow Compliance Checklist

| Rule | Requirement | Compliance |
|------|-------------|------------|
| M19 | Workflow chain | ✅ PLAN→REVIEW→DEBATE→IMPLEMENT→TEST→VALIDATE→LEARN |
| M20 | Learn step | ✅ This document |
| M25 | Plan first | ✅ Created plan before implementation |
| M26 | Test first | ✅ Designed tests during planning |
| M27 | Review | ✅ Review and Debate stages executed |
| M29 | Third-party | ✅ OpenAI validation |
| M35 | Always test | ✅ 25 tests executed, all passing |

## Artifacts Produced

1. **Source Code**:
   - `/source/generate_fixtures.py` - Fixture generator using real allocations
   - `/source/vol_targeting.py` - Core volatility targeting (unchanged)
   - `/source/data_models.py` - Dataclasses (unchanged)

2. **Tests**:
   - `/tests/test_comprehensive_real_data.py` - 25 tests using REAL allocation data
   - `/tests/fixtures/fixture_tail_300.pkl` - 30 instruments, 300 days
   - `/tests/fixtures/fixture_active_20.pkl` - 20 instruments, 300 days
   - `/tests/fixtures/fixture_quick_10.pkl` - 10 instruments, 100 days

3. **Documentation**:
   - `/docs/PLAN/test_strategy.md`
   - `/docs/REVIEW/review_v1.md`
   - `/docs/DEBATE/debates/debate_v1.md`
   - `/evidence/VALIDATE/validation_report.md`

## Memory Updates

- Real allocation testing requires instruments with actual positions
- CC returns are roll-adjusted for futures (not simple price changes)
- DataBundle field order must be consistent across modules
- Third-party validation adds significant value to review process
