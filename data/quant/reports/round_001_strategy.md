# Round 1 Strategy Plan

## 1. Round objective

Round 1 should **build on `lgbm_smart_money`**, not restart from naive multifactor. That direction is consistent with both the manager notes and the actual backtest reports already present in the repo.

### Official round targets (`orchestration/state.json`)

| Metric | Target |
|---|---:|
| Annual return | 15% |
| Sharpe | 1.0 |
| Max drawdown | -25% |

### Practical Round 1 success standard

Given the current baseline quality, Round 1 should be judged against the strongest existing strategy rather than the final end-state target only:

1. **Keep `lgbm_smart_money` as the return anchor**
2. Improve **Sharpe** and **max drawdown** materially
3. Do not collapse annual return toward the weaker cross-sectional / multifactor variants

A reasonable Round 1 pass condition is:

- Annual return **> 9.42%** or at least stays in the same range with much better risk
- Sharpe **> 0.33**
- Max drawdown **better than -35.68%**
- Turnover and crowding behavior become more controlled and explainable

---

## 2. Baseline comparison from existing reports

All numbers below are taken from existing markdown reports under `backtest/`.

| Strategy | Rebalance | Annual return | Sharpe | Max drawdown | Comment |
|---|---|---:|---:|---:|---|
| `lgbm_smart_money` | W | **+9.42%** | 0.33 | -35.68% | strongest return baseline |
| `lgbm_cross_sectional` | M | +6.34% | **0.35** | **-22.57%** | best risk profile among meaningful baselines |
| `lgbm_index_enhancement` | M | +0.22% | 0.01 | -32.00% | weak edge |
| `multifactor_vqm` | M | -1.64% | -0.06 | -51.19% | naive multifactor failed |
| `lgbm_multifactor` | BW | +0.00% | NaN | +0.00% | effectively broken / no useful trading |
| `lgbm_cross_sectional_v5` | BW | -3.91% | -0.19 | -52.02% | feature rework degraded badly |
| `lgbm_cross_sectional_v6` | BW | +0.17% | 0.01 | -28.34% | safer than some variants, but no return |
| `lgbm_cross_sectional_v7` | M | -4.66% | -0.26 | -42.36% | reranking variant failed |

### Baseline conclusion

- `lgbm_smart_money` is still the correct **anchor strategy** for Round 1 because it has the best realized annual return.
- `lgbm_cross_sectional` is the best **risk-control reference**, because its Sharpe and drawdown are better.
- Existing evidence does **not** support restarting from simple multifactor or pushing harder toward the V6/V7 “more fundamental / larger cap / reranking” direction.

---

## 3. Diagnosed weaknesses of the current strongest baseline

The strongest baseline is `strategies/lgbm_smart_money.py`.

### 3.1 Return is good, but risk-adjusted quality is still weak

From `backtest/lgbm_smart_money_report.md`:

- Annual return: **+9.42%**
- Annual volatility: **28.26%**
- Sharpe: **0.33**
- Max drawdown: **-35.68%**

This is not a stable high-quality alpha profile yet. It is a profitable strategy with incomplete risk control.

### 3.2 “Dynamic position sizing” is not translating into real de-risking often enough

From `backtest/lgbm_smart_money_monthly_returns.csv`:

- Rebalance periods: **408**
- `n_stocks` min: **26**
- `n_stocks` max: **32**
- `n_stocks` mean: **29.74**
- `n_stocks == 0`: **0** periods
- `n_stocks < 20`: **0** periods

Although the strategy description says it can hold fewer names or even go empty when signals are weak, the realized backtest behavior is close to **persistent near-full deployment**. In practice, the current “dynamic position sizing” is not delivering meaningful cash defense.

### 3.3 Tail-loss periods still happen at near-full exposure

Worst single-period outcomes from `lgbm_smart_money_monthly_returns.csv`:

| Date | Period return | Holdings |
|---|---:|---:|
| 2019-01-11 | -20.21% | 30 |
| 2024-02-02 | -16.28% | 30 |
| 2025-01-10 | -13.00% | 30 |
| 2018-02-02 | -11.59% | 31 |
| 2018-06-22 | -8.64% | 30 |

This strongly suggests that the current strategy does **not reduce gross exposure enough during bad regimes**.

### 3.4 Weekly cadence plus equal weighting still causes high trading intensity

From `backtest/lgbm_smart_money-trade.csv`:

- Trade days with fills: **409**
- Average filled trades per trade day: **38.91**
- Max filled trades per trade day: **58**

Weekly rebalancing is likely still appropriate for smart-money style signals, but the current portfolio construction is too close to “weekly reshuffle of ~30 equal-weight positions”. That behavior is expensive in Sharpe terms.

### 3.5 Short-horizon label is likely too noisy for this signal family

`lgbm_smart_money.py` trains on next-period forward return using weekly rebalance steps. That means the model is effectively learning **next-week return ranking**.

For a smart-money / accumulation thesis, one week is often too short. The accumulation-to-markup transition can take longer than a single week, so the current label likely injects avoidable noise.

### 3.6 Portfolio construction is too coarse

Current selection logic in `_select_stocks()`:

- `ml_score` above a quantile threshold
- minimum smart-money signal count
- industry cap
- max positions
- then **equal weight**

This means the best stock and the marginal last-admitted stock receive the same target weight. That wastes signal strength.

---

## 4. Round 1 strategy direction

## 4.1 Proposed strategy name

Create a new strategy file rather than modifying the current baseline in place:

- `strategies/lgbm_smart_money_r1.py`

Recommended strategy name inside the class:

- `lgbm_smart_money_r1`

This preserves a clean apples-to-apples baseline comparison against `lgbm_smart_money`.

### 4.2 Core Round 1 thesis

**Keep the smart-money feature family and model framework, but upgrade regime control, holding persistence, and portfolio construction.**

Round 1 should aim for:

1. Better **Sharpe** via lower unnecessary turnover and better gross-exposure control
2. Better **drawdown** via explicit market regime gating
3. Equal or better **annual return** via improved weighting and cleaner selection persistence

---

## 5. Proposed strategy changes with rationale

### 5.1 Add an explicit market regime filter

#### Change
Add a regime classifier with three states:

- `risk_on`
- `neutral`
- `risk_off`

The regime should be computed only from data already available in the current framework.

#### Inputs that can be implemented using the current project

1. **Index trend** using baseline CSVs already in `data/quant/baseline/`
   - e.g. `000300.SH.csv` or fallback `000001.SH.csv`
   - compute `close`, `MA20`, `MA60`
2. **Market breadth proxy** from current cross-section using `accessor.get_date()` or cached bulk data
   - fraction of stocks above MA20 or with positive 20-day return
3. **Market risk proxy** from cross-sectional `rvol_20`
   - median or upper-quantile realized volatility

#### Why
The backtest evidence shows that the current baseline does not materially reduce exposure during bad periods. Regime gating is the most direct way to improve drawdown and Sharpe without abandoning the signal family.

---

### 5.2 Replace “always near-full deployment” with regime-linked gross exposure

#### Change
Make total target portfolio weight dependent on regime:

| Regime | Target gross exposure | Target positions |
|---|---:|---:|
| `risk_on` | 0.90 - 1.00 | 18 - 24 |
| `neutral` | 0.60 - 0.80 | 12 - 18 |
| `risk_off` | 0.00 - 0.35 | 0 - 8 |

Return those weights directly from `generate_target_weights()`. The current broker and engine already support partial cash naturally, because weights do not need to sum to 1 before normalization if the strategy intentionally returns a lower gross set and explicitly scales them.

#### Why
The current strategy description promises dynamic sizing, but realized holdings show otherwise. This change makes de-risking explicit and testable.

---

### 5.3 Move from equal weight to tiered signal-strength weighting

#### Change
Within selected names, replace equal weighting with simple tiered weights based on ranking by combined selection strength.

Recommended implementation for Round 1:

- Top 30% selected names: multiplier **1.3**
- Middle 40%: multiplier **1.0**
- Bottom 30%: multiplier **0.7**
- Then normalize inside current target gross exposure
- Add a single-name cap such as **10%**

#### Why
This is easy to implement with the existing framework and should improve efficiency of capital allocation without requiring a full optimizer.

---

### 5.4 Strengthen hold-buffer and separate buy vs hold thresholds

#### Change
Current baseline uses:

- `score_quantile = 0.85`
- `min_signal_count = 3`
- `buffer_sigma = 0.3`

Round 1 should separate **new-entry** and **hold** conditions.

Recommended initial parameters:

| Parameter | Entry | Hold |
|---|---:|---:|
| Score quantile | 0.88 | 0.80 |
| Min smart-money signals | 4 | 3 |
| Buffer sigma | — | 0.6 |

Implementation idea:

- current holdings get the score boost first
- held names are judged against looser persistence rules
- new names must pass stricter thresholds

#### Why
This should reduce weekly churn while preserving exposure to names already working. It is the most direct turnover control available inside the current design.

---

### 5.5 Reduce max positions from 30 toward a controlled range

#### Change
Do not always allow up to 30 names. Use regime-linked max positions and a lower normal cap.

Recommended starting point:

- `risk_on`: max **24**
- `neutral`: max **16**
- `risk_off`: max **8**

#### Why
The current average realized holdings are ~30 names almost all the time. That dilutes strong signals and makes “dynamic conviction” mostly cosmetic.

---

### 5.6 Keep weekly rebalance, but make holdings sticky

#### Change
Stay on weekly rebalancing (`W`), but alter selection logic so the strategy behaves more like selective weekly maintenance than weekly full resorting.

Mechanics:

1. Keep currently held names that still pass hold rules
2. Fill remaining slots with new names passing entry rules
3. Only then rank and assign weights

#### Why
The manager direction explicitly prefers improving the smart-money family rather than restarting. Keeping weekly cadence preserves the signal’s timing edge; reducing unnecessary replacement should help Sharpe.

---

### 5.7 Do not make Round 1 a “fundamental reranking” project

#### Change
Do **not** follow the V6/V7 path as the main upgrade theme.

- Keep existing 22-feature smart-money-plus-context framework
- Do not re-anchor the strategy around value/quality reranking
- Do not shrink the universe aggressively to large caps only

#### Why
Actual backtests already showed V6/V7 underperformed badly.

---

## 6. Precise implementation plan using the current framework

This section is intentionally concrete so a programmer can implement it using only the existing `data/quant` framework.

### 6.1 Files to create or modify

#### Create
- `data/quant/strategies/lgbm_smart_money_r1.py`

#### Reuse without modification if possible
- `data/quant/strategies/lgbm_smart_money.py`
- `data/quant/strategies/utils.py`
- `data/quant/engine/strategy_base.py`
- `data/quant/engine/backtest.py`
- `data/quant/engine/broker.py`

#### Optional minor helper reuse only
- `data/quant/baseline/000300.SH.csv`
- `data/quant/baseline/000001.SH.csv`

### 6.2 Code structure to copy from the current baseline

`lgbm_smart_money_r1.py` should copy and reuse these baseline components from `lgbm_smart_money.py`:

- `FEATURE_COLUMNS`
- `FEATURE_NAMES`
- `_compute_momentum`
- `_get_bulk_date_index`
- `compute_features_from_memory`
- `compute_forward_return_from_memory`
- `rank_normalize`
- `train_lgbm_model`
- warmup and bulk-data cache structure in `LGBMSmartMoney`

That keeps Round 1 as an incremental version, not a rewrite.

### 6.3 New functions to add in `lgbm_smart_money_r1.py`

#### A. `compute_market_regime()`
Suggested signature:

```python
def compute_market_regime(
    date: pd.Timestamp,
    feat_df: pd.DataFrame,
    baseline_dir: str = "data/quant/baseline",
) -> dict:
```

Suggested outputs:

```python
{
    "regime": "risk_on" | "neutral" | "risk_off",
    "target_gross": 0.0 ~ 1.0,
    "target_positions": int,
}
```

Suggested internal logic:

1. Read baseline CSV once and cache in memory inside strategy instance
2. Truncate baseline history to `date`
3. Compute `MA20`, `MA60`
4. Compute breadth/risk proxies from `feat_df` or same-date ranked feature frame
5. Convert those into regime state

#### B. `_select_stocks_with_dual_thresholds()`
This should replace `_select_stocks()` from the baseline.

Responsibilities:

- apply stricter rules for new names
- apply looser rules for held names
- keep industry cap
- honor target positions from regime output
- preserve holdings before adding replacements

#### C. `_build_tiered_weights()`
Suggested signature:

```python
def _build_tiered_weights(
    self,
    selected_df: pd.DataFrame,
    target_gross: float,
) -> Dict[str, float]:
```

Responsibilities:

- assign 1.3 / 1.0 / 0.7 tier multipliers
- normalize within `target_gross`
- enforce per-name cap

#### D. Optional: `compute_multiperiod_label()`
Round 1 can remain on the current forward-return label for the first implementation pass. If time allows, a second variant can test a 2-week / multi-week label.

But the primary Round 1 document recommends **first implementing regime + weighting + turnover controls**, because those are lower-risk code changes with more direct drawdown benefit.

### 6.4 Strategy class design

Suggested class outline:

```python
class LGBMSmartMoneyR1(StrategyBase):
    def __init__(...):
        super().__init__("lgbm_smart_money_r1")
        ...

    def describe(self) -> str:
        ...

    def generate_target_weights(self, date, accessor, current_holdings):
        ...
```

Parameter suggestions for the constructor:

```python
train_window_years=3
entry_score_quantile=0.88
hold_score_quantile=0.80
entry_min_signal_count=4
hold_min_signal_count=3
max_per_industry=3
risk_on_max_positions=24
neutral_max_positions=16
risk_off_max_positions=8
buffer_sigma=0.6
mv_pct_upper=0.85
feature_lookback=260
```

### 6.5 No engine rewrite is required

Important implementation constraint: **Round 1 should be done entirely at the strategy layer.**

The current framework already supports what we need:

- `StrategyBase.generate_target_weights()` can return any weight dictionary
- `broker.rebalance()` already supports partial cash implicitly
- `backtest.py` already records cash, holdings count, and trades

So no framework rewrite is necessary.

---

## 7. Backtest command and output path assumptions

### 7.1 Expected command path

Assuming project root is `/projects/portal`, the expected run command is:

```bash
cd /projects/portal
python3 -m data.quant.strategies.lgbm_smart_money_r1
```

This mirrors the entry-point style already used in `lgbm_smart_money.py`.

### 7.2 Expected outputs

The new strategy should write to the existing output directory:

- `data/quant/backtest/lgbm_smart_money_r1_nav.csv`
- `data/quant/backtest/lgbm_smart_money_r1_monthly_returns.csv`
- `data/quant/backtest/lgbm_smart_money_r1-trade.csv`
- `data/quant/backtest/lgbm_smart_money_r1_report.md`

### 7.3 Comparison requirement

Round 1 should be compared directly against:

- `lgbm_smart_money`
- `lgbm_cross_sectional`

Those are the right return-risk reference points.

---

## 8. Risks and validation checks

### 8.1 Main risks

#### Risk 1: Over-defensiveness kills the alpha
If regime control is too aggressive, the strategy may improve drawdown but give up the main smart-money upside, ending with returns closer to `lgbm_cross_sectional` or worse.

#### Risk 2: Tiered weighting amplifies noise instead of conviction
If the ranking signal is unstable, increasing top-tier weights could worsen tail losses rather than help return.

#### Risk 3: Hold-buffer becomes stale-position bias
If hold thresholds are too loose, the strategy may cling to decayed setups and lose the benefit of weekly refresh.

#### Risk 4: Regime computation accidentally introduces hidden look-ahead bugs
Any direct baseline CSV usage must be truncated to current `date` only.

### 8.2 Validation checks

Programmer should validate at least the following:

1. **Look-ahead safety**
   - All regime calculations must use history truncated at `date`
   - No future rows from baseline CSV or bulk data

2. **Exposure behavior check**
   - Confirm `risk_off` periods actually produce lower gross exposure and fewer holdings
   - Confirm cash percentage rises meaningfully in weak regimes

3. **Turnover check**
   - Compare average trade count and overlap with prior holdings versus baseline `lgbm_smart_money`

4. **Selection stability check**
   - Measure average overlap of holdings period-to-period
   - Confirm hold-buffer meaningfully increases persistence

5. **Performance check**
   - Compare annual return, Sharpe, and max drawdown directly to baseline

6. **Stress-period check**
   - Inspect periods like 2019-01-11, 2024-02-02, 2025-01-10
   - Verify whether Round 1 is less exposed in similar conditions

---

## 9. What counts as success or failure for Round 1

### Success
Round 1 should be considered successful if it achieves most of the following:

- Annual return stays competitive with `lgbm_smart_money` and preferably exceeds **+9.42%**
- Sharpe improves above **0.33**, ideally to **0.45+**
- Max drawdown improves from **-35.68%** toward **-30%** or better
- Realized holdings and cash behavior clearly reflect regime-dependent risk control
- Weekly trade intensity declines versus baseline

### Partial success
A version can still be considered directionally useful if:

- annual return is slightly lower than baseline,
- but Sharpe and max drawdown improve materially,
- and the exposure/turnover behavior becomes much more robust.

### Failure
Round 1 should be considered a failure if any of these happen:

- Return falls toward the cross-sectional or multifactor variants without clear risk compensation
- Sharpe does not improve meaningfully
- Drawdown remains baseline-like despite added complexity
- “Dynamic exposure” still does not show up in realized holdings/cash
- The implementation depends on framework changes or external data not already in this repo

---

## 10. Final recommendation

Round 1 should **not** restart from a naive multifactor design. The file evidence strongly supports staying in the **smart-money + ML** family.

The correct first-round move is:

> **Take `lgbm_smart_money` as the anchor, preserve its feature and training framework, and upgrade it with explicit regime gating, true gross-exposure control, stronger hold-buffer logic, and tiered weighting.**

This is the most concrete path to improving annual return quality and Sharpe while controlling max drawdown, using only the existing `/projects/portal/data/quant` framework.
