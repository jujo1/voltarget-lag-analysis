# Volatility Targeting Lag Analysis

## Walk-Forward Validation Results

Rigorous validation of scale factor lag pattern in portfolio volatility targeting using blind k-fold testing.

### Dataset
- **Period**: 2012-01-02 to 2025-12-09 (3,637 trading days)
- **Instruments**: 181 futures contracts
- **Methodology**: Contiguous non-overlapping K-fold (K=4,5,6,8,10)
- **Total OOS tests**: 28 independent folds

### Key Findings

| Finding | Initial Analysis | Walk-Forward OOS | Verdict |
|---------|------------------|------------------|---------|
| Lag gap predicts vol | R²=14%, p<0.001 | Corr=0.03, p=0.70 | ❌ FAILS |
| Lag=1 > Lag=0 Sharpe | +0.50 Sharpe | +0.17 Sharpe, 79% win | ✅ VALIDATED |

### Finding 1: Lag Gap Predictive Signal ❌ FAILS

The hypothesis that the difference between lag=1 and lag=0 scale factors predicts next-period volatility **does not hold out-of-sample**.

- Significant folds (p<0.05): **0/28 (0%)**
- Mean OOS correlation: **0.033**
- Fisher's combined p-value: **0.70**

### Finding 2: Sharpe Paradox ✅ VALIDATED

The counter-intuitive finding that lag=1 scaling produces **higher risk-adjusted returns** than lag=0 despite higher volatility **is robust and statistically significant**.

- Folds where Lag=1 wins: **22/28 (79%)**
- Binomial test p-value: **0.0019**
- Mean Sharpe improvement: **+0.17**

### Implications

1. **DO NOT** use lag gap as a volatility timing signal - it was sample-specific
2. **CONSIDER** using lag=1 for position sizing - consistently better Sharpe
3. The improvement is smaller OOS (+0.17 vs +0.50 in-sample) but still meaningful

### Structure

```
├── source/
│   ├── vol_targeting.py      # Core volatility targeting implementation
│   ├── data_models.py        # Data structures
│   └── generate_fixtures.py  # Test data generation
├── tests/
│   └── test_comprehensive_real_data.py  # 26-test suite
├── docs/BRAINSTORM/
│   ├── walk_forward_validated_findings.md  # Final validation report
│   ├── vol_lag_pattern_analysis.md        # Initial analysis
│   └── vol_lag_complete_analysis.md       # Deep dive
└── evidence/
    ├── VALIDATE/validation_report.md
    └── LEARN/learnings.md
```

### Cautions

- Transaction costs not modeled
- Effect size reduced OOS (+0.17 vs +0.50)
- 21% of folds still showed lag=0 winning

### Further Research

1. Model transaction costs - Does lag=1 advantage survive?
2. Regime analysis - When does lag=0 outperform?
3. Optimal lag - Is lag=1 best, or is there an intermediate optimum?
