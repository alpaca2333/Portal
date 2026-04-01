# Manager Notes

- 2026-03-31 11:53 Asia/Shanghai: Initialized quant research loop.
- Existing results reviewed:
  - `lgbm_multifactor`: broken / effectively no trades (annual return 0, Sharpe NaN)
  - `lgbm_index_enhancement`: weak edge (annual return 0.22%, Sharpe 0.01)
  - `multifactor_vqm`: poor (annual return -1.64%, Sharpe -0.06)
  - `lgbm_smart_money`: currently strongest baseline (annual return 9.42%, Sharpe 0.33), but still below target and with large drawdown.
- Initial direction for Round 1:
  - Build on `lgbm_smart_money` family rather than restarting from naive multifactor.
  - Focus on improving risk-adjusted return, not just cumulative return.
  - Require explicit drawdown control, market regime filter, and turnover/crowding controls.
- 2026-03-31 12:18 Asia/Shanghai: User tightened target priority to **Sharpe > 1 first**, then **annual return > 15%**, while keeping drawdown acceptable (target better than -25%).
- 2026-03-31 12:24 Asia/Shanghai: Round 1 implementation produced `lgbm_smart_money_r1` and generated real backtest outputs, but only over a short smoke-test window (2018-01-01 ~ 2018-03-31). Metrics from the generated report: annual return **-31.69%**, Sharpe **-2.39**, max drawdown **-7.66%**. Result is far below target on the highest-priority metrics (Sharpe, annual return). Drawdown improved, but likely via over-restrictive regime/risk throttling. Keep the smart-money direction; Round 2 should focus on recovering alpha while preserving only the genuinely useful risk controls, and it must include a practical plan for scalable/full-window evaluation rather than another tiny smoke test.
