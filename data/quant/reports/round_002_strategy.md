# Round 2 Strategy Plan

## Conclusion

Round 2 should **stay on the `lgbm_smart_money` mainline**. Existing evidence does not support abandoning the smart-money direction; it supports **removing the parts of Round 1 that over-throttled exposure and diluted realized portfolio quality**.

The strongest reference remains the baseline `lgbm_smart_money`, because it is still the best realized return engine in the repo:

- baseline `lgbm_smart_money`: **annual return +9.42%, Sharpe 0.33, max drawdown -35.68%**
- Round 1 smoke test `lgbm_smart_money_r1`: **annual return -31.69%, Sharpe -2.39, max drawdown -7.66%** over **2018-01-01 ~ 2018-03-31** only

Round 1 did one thing clearly: it reduced drawdown in a short window. But it likely did so by **cutting too much participation, applying too much friction, and converting a high-conviction weekly strategy into a partially stuck, partially diluted portfolio**. That is the wrong trade-off under the current priority order. The Round 2 objective should therefore be:

1. **Recover Sharpe first** by removing avoidable negative-return friction and improving realized portfolio efficiency
2. Then **lift annual return above 15%** by preserving more of the baseline alpha capture
3. Keep drawdown control, but with **selective / soft risk control**, not hard over-throttling

The practical Round 2 recommendation is:

> Keep baseline smart-money features and weekly ML framework, but replace hard regime throttling with softer exposure scaling, reduce unnecessary turnover blending, enforce tighter realized-position discipline, and move from coarse equal-weight / tiered heuristics toward conviction-aware portfolio construction that preserves alpha instead of suppressing it.

---

## Baseline vs Round 1: what actually failed

### Baseline reference: `lgbm_smart_money`

From `backtest/lgbm_smart_money_report.md`:

- Annual return: **+9.42%**
- Sharpe: **0.33**
- Max drawdown: **-35.68%**
- Annual volatility: **28.26%**
- Backtest range: **2018-01-01 ~ 2025-12-31**

This is not target-grade, but it is the only strategy in the current set with a meaningful return base. It is the correct anchor.

### Round 1 result: `lgbm_smart_money_r1`

From `backtest/lgbm_smart_money_r1_report.md` and `orchestration/round_001_summary.md`:

- Annual return: **-31.69%**
- Sharpe: **-2.39**
- Max drawdown: **-7.66%**
- Annual volatility: **13.28%**
- Backtest range: **2018-01-01 ~ 2018-03-31**

### Why this is not a valid improvement

Round 1 only improved the lowest-priority metric, and only in a tiny smoke-test interval.

Against current targets:

- **Sharpe > 1.0**: failed badly
- **Annual return > 15%**: failed badly
- **Max drawdown better than -25%**: passed in the short smoke test, but not decision-grade evidence

So the correct interpretation is not “Round 1 made the strategy safer.”
The correct interpretation is:

> Round 1 likely removed too much useful risk, left too much implementation friction, and prevented the smart-money alpha from expressing itself.

---

## Exact weaknesses inferred from the actual files and metrics

## 1. The baseline already had an alpha source; Round 1 damaged capture instead of refining it

The baseline produced **+9.42% annual return** over the full 2018-2025 window. That means the smart-money feature family plus weekly LightGBM ranking is not broken.

Round 1 flipped that into **-31.69% annual return** and **-2.39 Sharpe** in the smoke window. The magnitude of deterioration is too large to explain as normal sampling noise alone. It points to portfolio-construction damage.

Implication for Round 2:

- do **not** rebuild from scratch
- do **not** replace the core feature family
- focus on the layer that changed most: **selection persistence, exposure control, realized holdings, and trade implementation efficiency**

---

## 2. Baseline weakness was “too much exposure”; Round 1 weakness became “too little usable exposure”

From `backtest/lgbm_smart_money_monthly_returns.csv`:

- periods: **408**
- `n_stocks` min/max/mean: **26 / 32 / 29.74**
- worst 3 periods:
  - 2019-01-11: **-20.21%**, holdings **30**
  - 2024-02-02: **-16.28%**, holdings **30**
  - 2025-01-10: **-13.00%**, holdings **30**

This shows the baseline stayed close to full deployment even in bad periods. That was a real problem.

But from `backtest/lgbm_smart_money_r1_monthly_returns.csv`:

- periods: **12**
- `n_stocks` min/max/mean: **8 / 41 / 24.0**
- path of holdings: 30, 37, 41, 35, 26, 20, 14, 8, 16, 24, 22, 15

Round 1 did reduce average exposure, but it also created unstable participation and likely de-anchored the strategy from its alpha sweet spot. For a weekly smart-money strategy, cutting down to **8-16** names repeatedly while also tightening entry rules and capping turnover is a strong recipe for under-capture.

Implication for Round 2:

- keep exposure control
- but make it **soft and continuous**, not abrupt
- target gross exposure and holdings should move in narrower bands unless the market state is truly extreme

---

## 3. Round 1 likely over-throttled through stacked constraints, not one single bad rule

The Round 1 code in `strategies/lgbm_smart_money_r1.py` added all of the following at once:

- regime-based gross exposure via `_compute_market_regime()`
- stricter buy vs hold thresholds in `_select_stocks_with_hold_buffer()`
- additional candidate pruning in `_apply_candidate_filters()`
- tiered weights in `_build_tiered_weights()`
- turnover blending in `_blend_for_turnover()`
- smaller target position counts by regime

Any one of these can be reasonable. All of them together can easily become over-control.

That matches the observed outcome: drawdown improved, but annual return and Sharpe collapsed.

Implication for Round 2:

- remove the “many conservative knobs at once” approach
- reintroduce alpha step by step
- only keep controls that show clear incremental value

---

## 4. There is strong evidence of a target-to-realized portfolio mismatch in Round 1

This is the most important implementation clue.

In `lgbm_smart_money_r1.py`, `_compute_market_regime()` sets:

- `risk_on target_n = 22`
- `neutral target_n = 14`
- `risk_off target_n = 6`
- class `max_positions = 24`

Yet the realized holdings in `backtest/lgbm_smart_money_r1_monthly_returns.csv` reached:

- **30** on 2018-01-12
- **37** on 2018-01-19
- **41** on 2018-01-26
- **35** on 2018-02-02

That is far above both `target_n` and the configured position cap.

This strongly suggests that the intended de-risking did **not** translate cleanly into actual realized holdings. The likely causes inside the current framework are:

1. turnover blending preserved too many legacy names
2. board-lot / partial liquidation mechanics left residual positions behind
3. weight-level caps were enforced in target weights, but not in the realized count after execution friction

This is a direct Sharpe killer:

- you pay turnover and signal friction to de-risk,
- but still carry too many names,
- so you get **less alpha concentration without actually achieving clean exposure control**.

Implication for Round 2:

- realized-position discipline must become a first-class objective
- do not judge only target weights; verify actual post-trade holdings count and residual tails

---

## 5. Round 1 likely reduced turnover, but in a way that trapped stale holdings instead of improving portfolio quality

Trade-file evidence:

- baseline `lgbm_smart_money-trade.csv`:
  - filled trade days: **409**
  - average filled trades per trade day: **38.91**
  - max filled trades per day: **58**
- Round 1 `lgbm_smart_money_r1-trade.csv`:
  - filled trade days: **13**
  - average filled trades per trade day: **27.0**
  - max filled trades per day: **42**

And in the smoke period, Round 1 trade counts by day were:

- 22, 30, 37, 42, 40, 33, 23, 17, 14, 15, 25, 23, 30

So yes, trading intensity came down relative to the baseline. But the return path still deteriorated sharply. That means the turnover reduction was not “efficient turnover reduction.” It was more likely **alpha suppression plus stale-position retention**.

Implication for Round 2:

- reduce churn by improving entry/exit quality
- do **not** use heavy blending as the main tool
- prefer selective no-trade bands and smaller incremental resizing over portfolio-wide blending

---

## 6. Round 1’s regime control was probably too binary for the evidence available

The Round 1 regime logic in `_compute_market_regime()` combines:

- benchmark trend
- cross-sectional breadth
- short-vs-history volatility ratio

and then jumps to these gross exposures:

- `risk_on`: **0.98**
- `neutral`: **0.72**
- `risk_off`: **0.25**

For this strategy family, that is probably too coarse and too aggressive.

Why:

- the baseline weakness was not “always be 100% or 0%”; it was “stay too close to 100% when the signal is weak”
- a smart-money strategy often needs to keep some participation through messy transitions
- forcing `risk_off` to **25% gross** plus stricter entry thresholds plus turnover caps compounds underinvestment risk

Implication for Round 2:

- use **continuous exposure scaling** rather than 3 large buckets
- avoid dropping to extreme low gross exposure unless multiple conditions are severely negative

---

## 7. Round 1 still did not solve portfolio efficiency cleanly

The baseline’s clear portfolio inefficiency was equal weighting: the top-ranked and marginal-ranked names got the same weight.

Round 1 improved that with tiered weighting, but the weight scheme remained coarse:

- top 30%: 1.3x
- middle 40%: 1.0x
- bottom 30%: 0.7x

This is better than equal weight, but still rough. Combined with turnover blending and realized residual names, the effective portfolio may still have been too diffuse.

Implication for Round 2:

- keep the direction away from equal weight
- but move to a **smooth conviction-weighted scheme** tied to rank strength and risk penalties
- avoid letting weak residual names survive just because of previous holdings

---

## Concrete Round 2 modifications

## A. Replace hard regime buckets with soft exposure scaling

### What to change

Keep regime awareness, but replace discrete `risk_on / neutral / risk_off` jumps as the main sizing mechanism.

Instead compute a continuous exposure multiplier, for example:

- start from base gross exposure **0.90**
- subtract small penalties for:
  - weak benchmark trend
  - weak breadth
  - elevated market volatility
- clamp final gross exposure into a narrower default band, such as **0.55 ~ 0.95**
- only allow **< 0.50** if all three regime inputs are strongly negative at the same time

### Why

This preserves drawdown control while avoiding the Round 1 failure mode where the strategy repeatedly throttles itself out of its own edge.

### Practical guidance

Inside the current framework:

- keep `_compute_market_regime()` but change its output from coarse labels to a richer numeric structure
- return fields like:
  - `gross_target`
  - `position_target`
  - `risk_penalty`
  - `extreme_risk_off` boolean
- do **not** map directly to 0.98 / 0.72 / 0.25 unless `extreme_risk_off` is true

Recommended initial exposure schedule:

- normal zone: **0.80 ~ 0.95**
- soft defensive zone: **0.65 ~ 0.80**
- extreme defense: **0.35 ~ 0.50**, rare

---

## B. Narrow the position-count range; do not starve the book

### What to change

Keep dynamic position count, but do not let it swing so far.

Recommended starting range:

- default target positions: **18 ~ 26**
- only allow **12 ~ 16** in clearly poor conditions
- avoid going to **6 ~ 8** except in true crash-like states

### Why

The baseline’s alpha came from a broad weekly smart-money book. Round 1 likely went too concentrated while also tightening entry rules, which reduced breadth and increased selection noise.

### Practical guidance

Use a soft formula such as:

- base target positions = 22
- add 2-4 when breadth is strong and signal opportunity set is deep
- subtract 2-6 when both breadth and trend are weak
- clamp to **[14, 26]** for most dates

---

## C. Remove portfolio-wide turnover blending as the default control

### What to change

Round 1’s `_blend_for_turnover()` is likely a central source of target-to-realized mismatch and stale holdings.

Replace it with **name-level trade frictions**, not whole-book blending.

Recommended rules:

1. **No-trade band**: if target weight change for a name is smaller than a threshold, skip the resize
   - e.g. `abs(new_w - old_w) < 0.75%` or `< 1.00%`
2. **Stricter entry than resize**:
   - new buys must clear full entry threshold
   - existing holdings can resize gradually if still valid
3. **Forced exit rule**:
   - if a holding fails hold threshold by a clear margin, or signal count collapses, liquidate it fully rather than blending indefinitely
4. **Residual cleanup rule**:
   - if a position weight falls below a minimum practical threshold, force full exit to avoid lot-size dust

### Why

This keeps good persistence but avoids carrying a long tail of legacy names that dilute the portfolio.

### Practical guidance

Delete or bypass `_blend_for_turnover()` for Round 2 initial tests. Replace it with a helper that:

- applies a no-trade band
- drops tiny target weights
- hard-exits names that are clearly invalid

This is much more likely to improve Sharpe than whole-portfolio weight interpolation.

---

## D. Enforce realized holdings discipline, not just target holdings discipline

### What to change

Round 2 must explicitly handle the mismatch between target count and actual count.

Recommended additions:

1. define a **minimum effective target weight**, e.g. **2.5% ~ 3.0%** before lot-size rounding
2. if a selected name falls below that threshold after normalization, drop it from the portfolio and redistribute weight
3. when trimming, prioritize fully exiting low-conviction legacy names instead of leaving many tiny residual positions
4. add post-target validation before returning weights:
   - ensure non-cash names count does not exceed intended cap
   - ensure weight tail is not too thin

### Why

If the strategy wants 18-22 names but execution leaves 30-40 realized names, portfolio construction is failing even if the signal model is fine.

### Practical guidance

After weight generation, run a final pass:

- sort selected names by conviction
- keep only names whose target weight is above minimum practical size
- if too many remain, cut weakest names until count cap is respected
- renormalize to target gross

---

## E. Keep buy-vs-hold asymmetry, but relax Round 1’s combined defensiveness

### What to change

The dual-threshold idea is good. The Round 1 magnitude was probably too conservative once combined with regime cuts and candidate pruning.

Recommended Round 2 starting point:

- entry score quantile: **0.86 ~ 0.88**
- hold score quantile: **0.78 ~ 0.82**
- entry min signal count: **3 or 4** depending on breadth
- hold min signal count: **2 or 3**

Make these **state-aware but only mildly**. Do not sharply tighten everything in weak regimes.

### Why

You want holdings persistence, not holdings paralysis.

### Practical guidance

Use mild state adaptation, for example:

- strong market / strong opportunity set:
  - entry 0.86, hold 0.79, entry signals 3
- weak market / weak opportunity set:
  - entry 0.88, hold 0.81, entry signals 4

That is enough. Don’t stack large jumps on top of lower gross exposure.

---

## F. Replace coarse tiered weighting with smooth conviction weighting

### What to change

Instead of 1.3 / 1.0 / 0.7 tiers, use a smoother weight formula driven by:

- normalized ML rank
- signal count / signal density
- optional risk penalty from `rvol_20`

Example structure:

- `conviction = 0.75 * score_rank + 0.25 * signal_density`
- `risk_adjust = 1 / (1 + k * rvol_rank)` or a mild cap-based penalty
- `raw_weight = max(conviction, floor) ^ alpha * risk_adjust`

Then:

- normalize to target gross
- cap single-name exposure, e.g. **7% ~ 8%** initially
- drop sub-minimum names

### Why

This improves portfolio efficiency without needing a full optimizer.

### Practical guidance

Keep it simple:

- use percentile ranks already available in current code
- avoid complex covariance optimization for Round 2
- just move from discrete tiers to monotonic smooth scaling

---

## G. Make candidate pruning selective, not broad

### What to change

Round 1’s `_apply_candidate_filters()` removed:

- high `rvol_20` names with insufficient signal count
- names near 60-day highs with weak flow / poor compression combination

Keep the spirit, but weaken the filter burden.

Recommended change:

- convert these from hard filters into **score penalties** first
- only hard-exclude the worst tail cases

### Why

Hard filters plus conservative exposure plus turnover constraints can easily eliminate the names that carry upside in this strategy family.

### Practical guidance

For Round 2 initial pass:

- remove hard exclusion for the second rule entirely
- for high-vol names, require a stronger penalty instead of automatic removal unless volatility is truly extreme

---

## H. Preserve the weekly cadence and the current smart-money feature family

### What to keep unchanged in Round 2

- weekly rebalance frequency
- baseline `FEATURE_COLUMNS`
- baseline `FEATURE_NAMES`
- existing `compute_features_from_memory()` pipeline
- existing LightGBM training utility unless profiling shows a separate issue

### Why

There is no evidence in the current files that the core weekly smart-money feature engine is the main failure. The evidence points to portfolio-construction and exposure-throttling problems.

---

## Implementation notes mapped to likely files and functions

## Main file to create

Create a new strategy rather than editing Round 1 in place:

- `/projects/portal/data/quant/strategies/lgbm_smart_money_r2.py`

Keep the baseline and Round 1 intact for clean comparison.

---

## Reuse from baseline `lgbm_smart_money.py`

These functions are still the correct base layer:

- `compute_features_from_memory()`
- `rank_normalize()`
- `train_lgbm_model()`
- `_count_smart_money_signals()` logic can be reused or lightly adapted
- baseline `generate_target_weights()` training flow is still a good reference for the core ML path

Relevant baseline locations observed in file:

- feature computation area near `compute_features_from_memory()`
- ranking: `rank_normalize()`
- model training: `train_lgbm_model()`
- selection: `_select_stocks()`
- main orchestration: `generate_target_weights()`

Use the baseline as the structural anchor, not Round 1.

---

## Functions from Round 1 worth keeping but rewriting

From `lgbm_smart_money_r1.py`:

### 1. `_compute_market_regime()`
Keep the data inputs, but rewrite the output logic.

Round 2 change:

- return a continuous gross-exposure target and softer position target
- reserve hard risk-off only for extreme cases
- avoid direct large bucket jumps

### 2. `_select_stocks_with_hold_buffer()`
Keep the asymmetry concept, but simplify.

Round 2 change:

- use milder entry/hold threshold gaps
- do not combine with overly restrictive position counts
- separate “hold because still valid” from “hold because blending preserved it”

### 3. `_build_tiered_weights()`
Replace with smooth conviction weighting.

Round 2 change:

- continuous weighting from blended rank / signal density
- enforce minimum practical position size
- cap single names lower but use fewer useless tail names

### 4. `_blend_for_turnover()`
This should **not** remain the primary control.

Round 2 change:

- replace with no-trade bands + hard invalidation exits + residual cleanup

### 5. `_apply_candidate_filters()`
Reduce hard exclusions.

Round 2 change:

- convert most exclusions into score penalties
- only hard-filter extreme low-quality setups

---

## Likely file/function mapping for the programmer

### File: `strategies/lgbm_smart_money_r2.py`

Recommended structure:

1. import and reuse from baseline:
   - `FEATURE_COLUMNS`
   - `FEATURE_NAMES`
   - `compute_features_from_memory`
   - `rank_normalize`
   - `train_lgbm_model`
2. implement these Round 2 helpers:
   - `_compute_market_regime_soft()`
   - `_score_candidates()`
   - `_select_hold_and_add()`
   - `_build_conviction_weights()`
   - `_apply_trade_band_and_cleanup()`
   - `_finalize_realistic_targets()`
3. main method:
   - `generate_target_weights()`

### Suggested responsibilities

#### `_compute_market_regime_soft(date, window_df)`
Outputs:

- `gross_target`
- `position_target`
- `extreme_risk_off`
- diagnostic fields for logging

#### `_score_candidates(scores_df, feat_ranked, current_holdings)`
Build columns such as:

- `signal_count`
- `signal_density`
- `held`
- `score_rank`
- `conviction_score`
- optional risk penalty columns

#### `_select_hold_and_add(candidate_df, regime_info, current_holdings)`
Process:

1. keep currently held names that still pass hold rules
2. add new names that pass entry rules
3. enforce industry cap
4. enforce soft position target range

#### `_build_conviction_weights(selected_df, gross_target)`
Process:

1. compute smooth raw weights
2. apply single-name cap
3. drop too-small names
4. renormalize

#### `_apply_trade_band_and_cleanup(new_weights, old_weights)`
Process:

1. skip tiny weight changes
2. force full exits on clearly invalid names
3. remove sub-minimum tail positions
4. renormalize

#### `_finalize_realistic_targets(weights)`
Process:

1. ensure effective count and min weight are consistent
2. ensure no unrealistic dust names remain
3. add cash only after equity targets are finalized

---

## Quick validation logic to add in logs

Inside `generate_target_weights()`, print at least:

- target gross exposure
- realized non-cash target count
- min / max target weight
- count of names dropped due to minimum weight
- count of names preserved by hold rule
- count of names force-exited
- turnover before and after no-trade band

This matters because Round 1’s failure was partly invisible unless you compare intended and realized holdings.

---

## Evaluation plan and success criteria

## 1. Do not repeat the “tiny smoke test only” mistake

Round 1’s 2018-01-01 ~ 2018-03-31 window was useful only for checking that the code ran. It was **not** enough to decide whether Sharpe improved.

Round 2 evaluation should therefore use **two stages**:

### Stage A: fast-but-decision-relevant validation

Use a window that is still short enough to iterate quickly but long enough to include multiple regimes.

Recommended fast validation window:

- **2018-01-01 ~ 2020-12-31**

Why this works better:

- materially larger than a 3-month smoke test
- includes weak, rebound, and shock-like periods
- still much faster than a full 2018-2025 pass

This should become the standard inner-loop test for Round 2.

### Stage B: full evaluation

After one or two promising variants pass Stage A, run the full window:

- **2018-01-01 ~ 2025-12-31**

This is the only window that should decide whether Round 2 actually beats the baseline.

---

## 2. Evaluation order: isolate the effects instead of stacking changes blindly

Test in this order:

### Variant R2-A: execution / portfolio cleanup only

Changes:

- remove turnover blending
- add no-trade bands
- add min effective weight and residual cleanup
- keep baseline selection thresholds mostly unchanged

Purpose:

- verify whether the main Sharpe drag was implementation friction and stale holdings

### Variant R2-B: add soft regime scaling

Changes:

- R2-A plus soft exposure / position scaling

Purpose:

- check whether selective, gentler de-risking helps without suppressing return

### Variant R2-C: add smooth conviction weights

Changes:

- R2-B plus smooth conviction weighting

Purpose:

- improve capital efficiency after realized holdings discipline is fixed

### Variant R2-D: mild state-aware thresholds

Changes:

- R2-C plus mild threshold adaptation by opportunity set / regime

Purpose:

- final refinement, not first lever

This sequencing matters. If everything changes at once again, you will not know what helped and what hurt.

---

## 3. Metrics to compare at each stage

Use the same primary priorities from `orchestration/state.json`:

1. Sharpe
2. annual return
3. max drawdown

But for Round 2 development, also track these operational metrics from CSV outputs:

- average realized holdings count
- holdings count range
- average filled trades per trade day
- max filled trades per trade day
- fraction of periods with unusually low exposure
- overlap with prior holdings
- number of residual tiny positions after rebalance

Round 2 should not be judged only by top-line return. The whole point is to improve **risk-adjusted return via better construction efficiency**.

---

## 4. Minimum success criteria for Round 2

### Stage A pass criteria (2018-2020 quick window)

A variant should advance only if it shows most of the following relative to the baseline behavior and Round 1 failure mode:

- Sharpe clearly better than the Round 1 profile and directionally better than baseline quality
- annual return not collapsing from under-participation
- realized holdings count consistent with intended target range
- no evidence of 30-40 name drift when target is much lower
- turnover lower than baseline for good reasons, not because the book gets stuck

### Full-window success criteria (2018-2025)

Round 2 should be considered successful only if it moves materially toward the official targets:

- **Sharpe: priority target is > 1.0**
- **Annual return: next target is > 15%**
- **Max drawdown: preferably materially better than baseline and moving toward better than -25%**

Given the current baseline level, an acceptable directional improvement would be:

- Sharpe materially above **0.33** first
- annual return at least preserved or improved while Sharpe rises
- max drawdown materially improved from **-35.68%** without repeating Round 1’s return destruction

If a candidate improves drawdown but again damages Sharpe and annual return, reject it.

---

## 5. Explicit reject conditions for Round 2

Reject a Round 2 variant immediately if any of these appear in the quick window:

1. realized holdings count again exceeds the intended cap by a wide margin
2. average exposure repeatedly collapses without a corresponding crash regime
3. turnover decreases but returns deteriorate sharply again
4. annual return weakens materially while Sharpe does not improve convincingly
5. the result depends on another tiny smoke test instead of at least a multi-year quick window

---

## Final recommendation

The repo evidence points to a very specific conclusion:

- **baseline `lgbm_smart_money` has usable alpha but poor risk efficiency**
- **Round 1 improved drawdown by over-throttling and likely creating target-to-realized portfolio inefficiency**

So Round 2 should not be a philosophical redesign. It should be a surgical repair.

The best Round 2 path is:

> **Keep the baseline smart-money ML engine, remove portfolio-wide turnover blending, enforce realized-position discipline, soften regime control into continuous scaling, and upgrade weighting into smooth conviction-aware allocation.**

That is the highest-probability route to recovering alpha capture, improving Sharpe first, and then pushing annual return above 15% without letting drawdown return to baseline extremes.
