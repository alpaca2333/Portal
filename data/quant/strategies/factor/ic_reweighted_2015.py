"""
IC-guided factor reweighting strategy — extended backtest 2015-2026
====================================================================
Same as ic_reweighted.py, only backtest_start changed to 2015-01-01.
Covers full A-share market cycle:
  2015 牛转熊崩盘, 2016 熔断, 2017 白马牛市, 2018 贸易战熊市,
  2019-2020 科技牛, 2021 结构牛, 2022 熊市, 2023-2024 震荡, 2025 政策反转
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import StrategyConfig, FactorDef, run_pipeline
from engine.types import SelectionMode, RebalanceFreq

config = StrategyConfig(
    name="ic_reweighted_2015",
    description="IC引导因子重新定权，2015年起完整市场周期回测",
    rationale="""
## 为什么做这次回测

`ic_reweighted` 策略的回测区间从 2019 年开始，样本偏短，且缺少极端市场环境的检验。
本版将回测起点延伸至 2015 年，覆盖 A 股完整牛熊周期：

| 年份 | 市场环境 |
|------|---------|
| 2015 | 杠杆牛市 + 股灾崩盘 |
| 2016 | 熔断机制触发，全年震荡 |
| 2017 | 白马蓝筹牛市（价值风格强势）|
| 2018 | 中美贸易战，全面熊市 |
| 2019-2020 | 科技成长牛市 |
| 2021 | 结构性行情，新能源为主 |
| 2022-2024 | 震荡弱市 |
| 2025 | 政策驱动小盘反转 |

**目的**：验证因子在不同风格周期下的稳健性，避免仅针对近3年调参导致过拟合。

## 因子权重

与 `ic_reweighted` 完全一致，仅扩展了时间区间。
""",
    warm_up_start="2014-01-01",
    backtest_start="2015-01-01",
    end="2026-02-28",
    freq=RebalanceFreq.BIWEEKLY,
    mcap_keep_pct=0.70,
    selection_mode=SelectionMode.TOP_PCT,
    top_pct=0.05,
    max_per_industry=5,
    min_industry_count=5,
    min_holding=20,
    single_side_cost=0.00015,
    buffer_sigma=0.0,
)

factors = [
    FactorDef("inv_pb",   +0.25),
    FactorDef("rev_10",   +0.30),
    FactorDef("rvol_20",  -0.35),
    FactorDef("log_cap",  -0.10),
]

if __name__ == "__main__":
    run_pipeline(config, factors)
