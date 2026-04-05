# V7 简单风格开关研究（自动迭代版）

- Portal 根目录：`/data/Projects/Portal`
- 共享目录：`/data/Projects/Portal/data/quant/report/v7_style_gate_research`
- 目标：年化 ≥ 15.00%，最大回撤 ≥ -15.00%
- 主基线：lgbm_ensemble_adaptive_v7c
- 辅助验证：lgbm_ensemble_adaptive_v7
- 是否读取上一轮结果：是
- 上一轮最优规则：`followup[market_ma20_smallcap]/softplus_10_30_60_100/floor[csi300>ma60=>30%]`
- 本轮候选数：基础 283 / 跟进扩展 0 / 总计 283
- 是否命中目标：否

## 当前最优规则（硬开关 + 软开关合并）

- 模式：`soft`
- 规则：`soft[market_ma20_smallcap]/softplus_10_30_60_100/floor[csi300>ma60=>20%]`
- v7c 估算：年化 8.36% / 最大回撤 -14.68% / 累计 89.52% / 平均暴露 44.90%
- v7  估算：年化 10.34% / 最大回撤 -13.12% / 累计 118.76% / 平均暴露 44.90%
- 结论：当前没有任何简单硬开关或简单软开关同时达到年化>=15%且最大回撤<=15%。先采用“回撤达标下年化最高”的规则继续观察。

## 当前最优硬开关

- 规则：`(ma20(1000)>ma60) AND (ret20(500-300)>0%)`
- v7c：年化 6.35% / 回撤 -12.40% / 累计 63.18% / 平均暴露 21.63%
- v7 ：年化 5.41% / 回撤 -18.15% / 累计 52.10% / 平均暴露 21.63%

## 当前最优软开关

- 规则：`soft[market_ma20_smallcap]/softplus_10_30_60_100/floor[csi300>ma60=>20%]`
- v7c：年化 8.36% / 回撤 -14.68% / 累计 89.52% / 平均暴露 44.90%
- v7 ：年化 10.34% / 回撤 -13.12% / 累计 118.76% / 平均暴露 44.90%

## 低回撤候选 Top 10

1. [soft] `soft[market_ma20_smallcap]/softplus_10_30_60_100/floor[csi300>ma60=>20%]` | v7c 年化 8.36% / 回撤 -14.68% / 暴露 44.90% | v7 年化 10.34% / 回撤 -13.12%
2. [soft] `soft[market_ma20_smallcap]/softplus_10_30_60_100` | v7c 年化 8.34% / 回撤 -14.68% / 暴露 44.71% | v7 年化 10.11% / 回撤 -13.12%
3. [soft] `soft[market_ma20_smallcap]/softplus_10_30_60_100/floor[csi300>ma20=>20%]` | v7c 年化 8.34% / 回撤 -14.68% / 暴露 44.71% | v7 年化 10.11% / 回撤 -13.12%
4. [soft] `soft[market_plus_small]/softplus_10_30_60_100/floor[csi300>ma20=>20%]` | v7c 年化 8.27% / 回撤 -14.91% / 暴露 44.62% | v7 年化 9.88% / 回撤 -11.62%
5. [soft] `soft[market_rel_smallcap]/softplus_10_30_60_100/floor[csi300>ma20=>20%]` | v7c 年化 8.21% / 回撤 -14.78% / 暴露 43.89% | v7 年化 9.08% / 回撤 -13.74%
6. [soft] `soft[market_rel_smallcap]/softplus_10_30_60_100` | v7c 年化 8.10% / 回撤 -14.78% / 暴露 43.65% | v7 年化 8.98% / 回撤 -13.83%
7. [soft] `soft[market_rel_smallcap]/softplus_10_30_60_100/floor[csi300>ma60=>20%]` | v7c 年化 8.10% / 回撤 -14.78% / 暴露 43.65% | v7 年化 8.98% / 回撤 -13.83%
8. [soft] `soft[market_rel_smallcap_strict]/softplus_10_30_60_100/floor[csi300>ma20=>20%]` | v7c 年化 7.74% / 回撤 -13.49% / 暴露 42.84% | v7 年化 9.68% / 回撤 -14.35%
9. [soft] `soft[market_rel_smallcap]/stepped_0_30_60_100/floor[csi300>ma20=>20%]` | v7c 年化 7.73% / 回撤 -14.80% / 暴露 42.69% | v7 年化 8.62% / 回撤 -13.34%
10. [soft] `soft[market_rel_smallcap_strict]/softplus_10_30_60_100` | v7c 年化 7.54% / 回撤 -13.52% / 暴露 42.50% | v7 年化 9.45% / 回撤 -14.35%

## 综合评分 Top 10

1. [soft] `soft[market_dualtrend_smallcap]/ladder_20_35_60_80_100` | score=0.1667 | v7c 年化 11.44% / 回撤 -16.95% / 暴露 56.68%
2. [soft] `soft[market_dualtrend_relative]/ladder_20_35_60_80_100` | score=0.1582 | v7c 年化 11.14% / 回撤 -17.15% / 暴露 56.66%
3. [soft] `soft[market_ma20_smallcap]/bullfloor_20_40_70_100` | score=0.1485 | v7c 年化 10.40% / 回撤 -16.82% / 暴露 53.70%
4. [soft] `soft[market_ma20_smallcap]/bullfloor_20_40_70_100/floor[csi300>ma60=>20%]` | score=0.1485 | v7c 年化 10.40% / 回撤 -16.82% / 暴露 53.70%
5. [soft] `soft[market_ma20_smallcap]/bullfloor_20_40_70_100/floor[csi300>ma20=>20%]` | score=0.1485 | v7c 年化 10.40% / 回撤 -16.82% / 暴露 53.70%
6. [soft] `soft[market_plus_small]/bullfloor_20_40_70_100` | score=0.1480 | v7c 年化 9.93% / 回撤 -15.80% / 暴露 53.17%
7. [soft] `soft[market_plus_small]/bullfloor_20_40_70_100/floor[csi300>ma60=>20%]` | score=0.1480 | v7c 年化 9.93% / 回撤 -15.80% / 暴露 53.17%
8. [soft] `soft[market_plus_small]/bullfloor_20_40_70_100/floor[csi300>ma20=>20%]` | score=0.1480 | v7c 年化 9.93% / 回撤 -15.80% / 暴露 53.17%
9. [soft] `soft[market_dualtrend_small1000]/ladder_20_35_60_80_100` | score=0.1460 | v7c 年化 10.95% / 回撤 -18.40% / 暴露 56.56%
10. [soft] `soft[market_dualtrend_relative]/ladder_10_30_55_80_100` | score=0.1451 | v7c 年化 9.91% / 回撤 -16.07% / 暴露 52.26%
