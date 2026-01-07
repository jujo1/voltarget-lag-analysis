# Brainstorm: Volatility Increases with Scale Factor Lag

## The Observation

```
Lag (days) | Realized Vol | Delta from Target
    0      |    14.49%    |    -0.51%
    1      |    17.29%    |    +2.29%
    2      |    18.52%    |    +3.52%
    3      |    21.19%    |    +6.19%
    5      |    21.97%    |    +6.97%
```

**Pattern**: Realized volatility INCREASES monotonically with lag.

---

## Hypothesis Generation

### H1: Volatility Clustering (GARCH Effect)
**Theory**: Volatility is autocorrelated - high vol days cluster together.
- When vol spikes, yesterday's scale factor underestimates current vol
- Scale factor is "too high" → positions too large → realized vol increases
- This is the standard explanation for vol-of-vol

**Implication**: No alpha - this is well-known market microstructure

### H2: Mean Reversion in Volatility
**Theory**: Vol mean-reverts, so lagged estimates are biased.
- If vol is above average, lagged scale is too high
- If vol is below average, lagged scale is too low
- Net effect: systematic under-scaling during calm, over-scaling during stress

**Implication**: Could exploit by adjusting scale factor based on vol regime

### H3: Asymmetric Volatility Response (Leverage Effect)
**Theory**: Negative returns increase vol more than positive returns decrease it.
- After a down day, vol spikes but lagged scale doesn't reflect it
- Positions are too large on high-vol days following losses

**Implication**: Could improve by asymmetric vol estimation

### H4: Information Decay
**Theory**: Yesterday's covariance estimate contains "stale" information.
- Correlations change faster than covariance estimates adapt
- Lagged estimates miss correlation regime changes

**Implication**: Use faster-adapting correlation estimator

### H5: Systematic Under-Estimation of Tail Risk
**Theory**: EWMA covariance underestimates tail events.
- When tails hit, lagged estimates are way off
- More lag = more exposure to tail events

**Implication**: Use robust covariance estimators (shrinkage, etc.)

---

## Exploitable Patterns?

### Potential Alpha Sources

1. **Vol Timing Signal**
   - If lag=0 achieves 14.49% but lag=1 achieves 17.29%
   - The DIFFERENCE (2.8%) represents vol timing alpha
   - Current impl captures this by using fresher estimates

2. **Vol Regime Switching**
   - Adjust lag dynamically based on vol regime
   - Low vol regime: longer lag OK (more stable)
   - High vol regime: shorter lag needed (fast adaptation)

3. **Predictive Power of Lag Gap**
   - The gap between lag=0 and lag=1 vol might predict future vol
   - Large gap → vol is changing fast → expect higher vol ahead

### Counter-Arguments (Why This May NOT Be Alpha)

1. **Transaction Costs**
   - Fresher estimates = more frequent rebalancing
   - The 2.8% vol difference may be eaten by costs

2. **Implementation Complexity**
   - Dynamic lag adjustment adds model risk
   - Overfitting danger in regime detection

3. **Already Priced In**
   - Vol clustering is well-known (VIX term structure)
   - Market makers already account for this

---

## Deeper Questions

### Q1: Is the 2.8% gap stable over time?
- Need to test on different periods
- If gap varies, there may be predictive information

### Q2: Does the gap predict next-day returns?
- If lagged vol is "wrong", does the market correct?
- Could be vol risk premium signal

### Q3: Is the pattern symmetric for long vs short positions?
- Does lagging hurt longs more than shorts?
- Leverage effect suggests asymmetry

### Q4: Does the pattern hold across asset classes?
- Commodities vs equities vs FX
- Different vol dynamics per asset class

---

## Testable Hypotheses

1. **H_autocorr**: Corr(lag_gap_t, realized_vol_{t+1}) > 0
2. **H_regime**: Gap is larger during high-vol regimes
3. **H_asymmetry**: Gap is larger after negative returns
4. **H_decay**: Gap increases faster than linear with lag

---

## Preliminary Conclusion

The pattern is REAL and MEANINGFUL but likely NOT directly exploitable because:
1. It's a known effect (vol clustering)
2. Transaction costs to capture it
3. Current implementation ALREADY exploits it (lag=0)

The VALUE is in understanding WHY and ensuring implementation is optimal.

