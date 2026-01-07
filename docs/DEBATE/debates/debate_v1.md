# DEBATE: Challenging Plan Assumptions

**Stage**: DEBATE (3/10)
**Date**: 2026-01-07T04:09:30Z

---

## Position A: Plan is Sufficient

**Argument**: The plan addresses the user's core request:
1. Tests will use real allocation data from CSV files
2. Tail 300 days provides realistic market conditions
3. 20+ instruments with 100% coverage ensures robust testing

**Evidence**: 
- 32 instruments identified with valid allocations
- Real data produces 15.55% realized vol (close to 15% target)
- Sharpe 0.51 with real data vs 0.05 with synthetic

---

## Position B: Plan Has Critical Gaps

**Argument**: The plan does NOT achieve "100% testing" as requested:
1. Only 3 of 6 CSV files are actually tested (returns, alloc_raw, alloc_usd)
2. Contract values and prices are NOT tested
3. Lot allocations are mentioned but not implemented

**Evidence**:
- alloc_lots_*.csv: Not tested
- contract_value_lc_*.csv: Not tested  
- prices_settle_*.csv: Not tested
- User said "use PK data files *.csv" - implies ALL files

---

## Dialectical Resolution

### Challenge 1: Do ALL CSV files need tests?

**Counter-argument**: The user's primary request was about "portfolio volatility targeting" testing. Contract values and prices are input data for computing contract values but may not directly impact volatility targeting tests.

**Resolution**: 
- MUST: Test allocations (alloc_raw, alloc_usd) - core to request
- SHOULD: Test lot allocations - common in production
- COULD: Test contract values and prices - supplementary validation

### Challenge 2: Is fixture_subset_10 fixable?

**Counter-argument**: The fixture selected first 10 columns alphabetically, which happen to have zero allocations (AD-1-Curncy, AI-1-Index, etc.). This was a selection bug.

**Resolution**: 
- MUST: Regenerate fixture_subset_10 with instruments that have allocations
- Alternative: Create new fixture fixture_active_20 with 20 active instruments

### Challenge 3: Should tests use np.random at all?

**Counter-argument**: Some tests (B3: scale_factor_formula) test mathematical correctness with controlled inputs. Random data is appropriate for testing formula invariants.

**Resolution**:
- B3 (formula test): np.random OK - tests math property
- B1, B2, D1-D3, F1, F2: Must use real allocations - tests real behavior

---

## Debate Verdict

| Item | Decision | Rationale |
|------|----------|-----------|
| Update B1, B2, D1-D3, F1, F2 | ✅ MANDATORY | Core user request |
| Keep B3 with np.random | ✅ OK | Tests formula property |
| Add lot allocation test | ✅ ADD | Production relevance |
| Add contract value test | ⚠️ OPTIONAL | Nice to have |
| Add price test | ⚠️ OPTIONAL | Nice to have |
| Regenerate fixture | ✅ MANDATORY | Current fixture broken |

---

## Final Decision

**Proceed to IMPLEMENT with**:
1. Regenerate fixtures with active instruments
2. Update 7 tests to use real allocations (B1, B2, D1, D2, D3, F1, F2)
3. Add 1 new test for lot allocations (G1)
4. Total: 23 tests (22 existing + 1 new)

---
Generated: 2026-01-07T04:09:30Z
