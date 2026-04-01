# Round 1 Manager Summary

## Phase
Result evaluation completed.

## Strategy / Outputs
- Strategy doc: `/projects/portal/data/quant/reports/round_001_strategy.md`
- Implemented strategy: `/projects/portal/data/quant/strategies/lgbm_smart_money_r1.py`
- Backtest report: `/projects/portal/data/quant/backtest/lgbm_smart_money_r1_report.md`
- Other outputs:
  - `/projects/portal/data/quant/backtest/lgbm_smart_money_r1_nav.csv`
  - `/projects/portal/data/quant/backtest/lgbm_smart_money_r1_monthly_returns.csv`
  - `/projects/portal/data/quant/backtest/lgbm_smart_money_r1-trade.csv`

## Actual metrics from generated report
- Backtest range: 2018-01-01 ~ 2018-03-31
- Annual return: **-31.69%**
- Sharpe: **-2.39**
- Max drawdown: **-7.66%**

## Target comparison
Targets from `state.json`:
- Sharpe > **1.0** (highest priority)
- Annual return > **15%**
- Max drawdown better than **-25%**

Assessment:
- Sharpe: **missed badly**
- Annual return: **missed badly**
- Max drawdown: **passed**, but only over a short smoke-test window

## Baseline comparison
Current strongest baseline remains `lgbm_smart_money`:
- Baseline annual return: **9.42%**
- Baseline Sharpe: **0.33**
- Baseline max drawdown: **-35.68%**

Round 1 implementation result does **not** beat the baseline on annual return or Sharpe.
It only shows lower drawdown on a much shorter test window, so it is not sufficient evidence of improvement.

## Manager diagnosis
Main weaknesses observed:
1. The implementation/backtest only completed a **short smoke-test interval**, not a full evaluation window, so research throughput/execution efficiency is still a problem.
2. The tighter regime filter / turnover control / stronger buffers likely became **too conservative or mistimed**, harming return capture.
3. Risk controls reduced drawdown, but **destroyed Sharpe and annual return**, which violates current target priority.
4. The strategy should continue along the **smart-money** direction, but with more selective changes to avoid over-throttling alpha.

## Decision
- Mark Round 1 as **below target**.
- Increment to **Round 2**.
- Keep the research direction anchored on `lgbm_smart_money`.
- Next strategist must propose improvements that preserve smart-money alpha while prioritizing Sharpe recovery and ensuring a scalable/full-window evaluation plan.
