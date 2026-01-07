# Walk-Forward Validation Results: Scale Factor Lag Pattern

## Validation Methodology

**Dataset**: 3,637 days (2012-01-02 to 2025-12-09), 181 instruments

**K-Fold Structure**: Contiguous, non-overlapping, chronological splits
- K=4: 3 folds × 909 days
- K=5: 4 folds × 727 days  
- K=6: 5 folds × 606 days
- K=8: 7 folds × 454 days
- K=10: 9 folds × 363 days

**Total**: 28 out-of-sample test folds

---

## Finding 1: Lag Gap Predictive Signal

### Hypothesis
The difference between lag=1 and lag=0 scale factors predicts next-period volatility.

### Walk-Forward Results
```
Significant folds (p<0.05):  0 / 28 (0%)
Mean OOS correlation:        0.0328
Correlation range:           [-0.001, +0.082]
Fisher's combined p-value:   0.705
```

### VERDICT: ❌ FAILS VALIDATION

The predictive signal observed in the initial analysis was **sample-specific**. 
When tested on truly out-of-sample data across 28 folds, the correlation is 
near-zero and not statistically significant.

**Root Cause**: The original finding (R²=14%, p<0.001) likely benefited from:
- Overlapping estimation windows
- Implicit lookahead from using the full dataset
- Multiple hypothesis testing without correction

---

## Finding 2: Sharpe Paradox (Lag=1 > Lag=0)

### Hypothesis
Lag=1 scaling produces higher risk-adjusted returns than Lag=0 despite higher volatility.

### Walk-Forward Results
```
Folds where Lag=1 wins:      22 / 28 (79%)
Binomial test p-value:       0.0019 ✅ SIGNIFICANT
Mean Sharpe improvement:     +0.168
Sharpe diff range:           [-0.32, +0.70]
```

### By K Value
| K  | Folds | Lag=1 Win Rate |
|----|-------|----------------|
| 4  |   3   | 100%           |
| 5  |   4   | 75%            |
| 6  |   5   | 80%            |
| 8  |   7   | 86%            |
| 10 |   9   | 67%            |

### VERDICT: ✅ ROBUST AND VALIDATED

The Sharpe paradox is **real and statistically significant**:
- 79% win rate across 28 independent OOS tests
- p=0.0019 from binomial test (highly significant)
- Consistent across different fold sizes (K=4 to K=10)
- Mean improvement of +0.17 Sharpe ratio

---

## Summary Table

| Finding | Initial Analysis | Walk-Forward OOS | Verdict |
|---------|------------------|------------------|---------|
| Lag gap predicts vol | R²=14%, p<0.001 | Corr=0.03, p=0.70 | ❌ FAILS |
| Lag=1 > Lag=0 Sharpe | +0.50 Sharpe | +0.17 Sharpe, 79% win | ✅ VALIDATED |

---

## Implications

### What This Means

1. **DO NOT** use lag gap as a volatility timing signal - it doesn't work OOS
2. **CONSIDER** using lag=1 for position sizing - consistently better Sharpe
3. The Sharpe improvement is smaller OOS (+0.17 vs +0.50) but still meaningful

### Why Lag=1 Works (Hypotheses)

1. **Smoothing Effect**: Reduces overreaction to transient vol spikes
2. **Momentum Alignment**: Lagged scaling better captures trend continuation  
3. **Reduced Noise**: Less frequent scale factor changes
4. **Regime Buffering**: Provides natural delay during regime transitions

### Cautions

1. **Transaction costs not modeled** - higher vol with lag=1 may require more rebalancing
2. **Smaller effect OOS** - +0.17 Sharpe vs +0.50 in-sample suggests some overfitting
3. **Not universal** - 21% of folds still showed lag=0 winning

---

## Recommendations

### For Production

```python
# Current (keep for vol targeting accuracy):
scaled_weights = weights * scale_factor_lag0

# Consider testing (for alpha):
scaled_weights = weights * scale_factor_lag1  # +0.17 Sharpe OOS
```

### For Further Research

1. **Model transaction costs** - Does lag=1 advantage survive after costs?
2. **Regime analysis** - When does lag=0 outperform?
3. **Optimal lag** - Is lag=1 best, or is there an optimal intermediate?
4. **Combination** - Can lag=0 and lag=1 be blended?

---

## Evidence Files

- Walk-forward test: `/tmp/walk_forward_final.py`
- Full output: 28 folds across 5 K values
- Binomial test: p=0.0019 for Sharpe paradox

