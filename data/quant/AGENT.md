# AGENT.md — 量化工作区交接文档

> 本文档供接手本工作区的 Agent 快速上手，读完即可继续工作。

---

## 一、你是谁，你在做什么

你在帮 **Morty（qwertysun）** 做 A 股量化策略研究。
目标是开发可实盘的量化策略，当前处于**研究/回测阶段**。

---

## 二、工作区路径

| 内容 | 路径 |
|------|------|
| 量化工作区根目录 | `/projects/portal/data/quant/` |
| 策略脚本 | `strategies/factor/` |
| 回测产物 | `backtest/` |
| 数据工具 | `utils/data_loader.py` |
| 数据合并工具 | `utils/process_valuation_industry.py` |
| 策略文档 | `strategies/pure_market_strategy.md`、`strategies/pure_market_indicator_research_table.md` |
| 原始 CSV 数据 | `~/qlib_data/*.csv`（5489 只股票） |
| 原始估值数据 | `/root/qlib_data/valuation/*.csv`（per-stock: date,symbol,pb,pe_ttm,free_market_cap） |
| 原始行业数据 | `/root/qlib_data/sw_industry.csv`（申万行业分类，区间表） |
| qlib 二进制数据 | `~/.qlib/qlib_data/cn_data/` |
| 回测规范 | `backtest/OutputFormat.md` ← **每次回测必须遵守** |

---

## 三、数据基础设施

### 数据来源
- Tushare API 在公司内网被封锁，数据由 Morty **手动提供 CSV**
- 行情格式：`{TICKER}.{EXCHANGE}.csv`，字段：`symbol,date,open,high,low,close,volume,factor`
- 规模：5489 只股票，2006-03-20 ~ 2026-03-13，约 1421 万行

### 已合并到 all_stocks_daily.csv 的字段（✅ 已完成）

| 字段 | 来源 | 说明 |
|------|------|------|
| `code,date,open,high,low,close,volume,factor` | 原始行情 CSV | 基础 OHLCV |
| `pb` | `/root/qlib_data/valuation/*.csv` | 市净率 |
| `pe_ttm` | 同上 | 滚动市盈率 |
| `free_market_cap` | 同上 | 自由流通市值（万元） |
| `industry_code` | `/root/qlib_data/sw_industry.csv` | 申万一级行业代码 |
| `industry_name` | 同上 | 申万一级行业名称 |

- 合并脚本：`utils/process_valuation_industry.py`
- 原文件备份：`processed/all_stocks_daily.csv.bak`
- 另存独立文件：`processed/valuation_daily.csv`、`processed/sw_industry_range.csv`

### qlib 环境
- 版本：pyqlib 0.9.7，已修复两个兼容性 bug：
  1. `_get_calendar()` 中 `dtype="object"` 修复 pandas 2.0+ Timestamp 加法报错
  2. `features/` 目录名全部改为小写（qlib 内部做小写转换）
- `D.features()` 现已正常可用
- 初始化方式：
```python
import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
```
- instruments 列表中股票代码为**小写**（`sh600519`），`D.features()` 传大小写均可

### 已入库的指数
| 代码 | 名称 | qlib代码 |
|------|------|---------|
| 000905.SZ | 中证500 | `sz000905` |
| 000001.SH | 上证指数 | `sh000001` |

---

## 四、策略约束（Morty 确认）

| 约束 | 说明 |
|------|------|
| ❌ 不做 HFT | 频率不可接受 |
| ❌ 不做事件驱动 | 无信息来源 |
| ❌ 不做个股空头 | 无融券账户 |
| ✅ 可做多个股 | A 股主板，SH+SZ，排除 BJ |
| ✅ 可做期货多空 | 商品期货，账户待开 |
| ✅ 可做期权 | 50ETF/沪深300，账户待开 |

---

## 五、已有策略 & 回测结果

### 策略一览

| 策略 | 脚本 | 回测区间 | 年化收益 | 夏普 | 最大回撤 | 状态 |
|------|------|----------|----------|------|----------|------|
| 大盘多因子 | `factor/largecap_multifactor.py` | 2022-01 ~ 2026-02 | — | — | — | 未统一回测 |
| 多因子(全市场) | `factor/multifactor_backtest.py` | 2022-01 ~ 2026-02 | **+4.8%** | 0.19 | -38.8% | ✅ 唯一正收益 |
| 纯市场增强 | `factor/pure_market_enhanced.py` | 2022-01 ~ 2026-02 | -6.78% | -0.28 | -51.5% | ❌ 失败 |
| 动量回测 | `factor/momentum_backtest.py` | — | — | — | — | 早期实验 |

### 回测产物（`backtest/` 目录）
- `multifactor_nav.csv` / `multifactor_monthly_returns.csv` / `multifactor_report.md`
- `pure_market_enhanced_nav.csv` / `pure_market_enhanced_monthly_returns.csv` / `pure_market_enhanced_report.md`
- `largecap_multifactor_report.md`
- `multifactor_2025_analysis.md`

### 策略研究文档（`strategies/` 目录）
- `pure_market_strategy.md` — 纯市场指标驱动策略设计稿
- `pure_market_indicator_research_table.md` — 12 个纯市场指标的公式/窗口/优缺点/适用频率研究表

---

## 六、关键分析结论

### 2025 年多因子失效原因（已深入分析）
- 政策驱动小盘反转行情，涨幅最大的 100 只股：
  - 68% 在我们因子得分的后50%
  - 平均 MOM = -39%（过去一年在跌的），我们选的是涨的
  - 平均 IVOL 高于全市场（高波动票），被低波动因子惩罚
- 分析报告：`backtest/multifactor_2025_analysis.md`

---

## 七、待解决问题 & 下一步

### ✅ 已完成
- ~~加入价值因子（PE/PB）~~ → pb、pe_ttm、free_market_cap 已合并入 all_stocks_daily.csv
- ~~行业分类数据~~ → industry_code、industry_name 已合并（申万一级）
- ~~纯市场策略研究~~ → 已完成策略文档 + 指标研究表 + 回测，结论为纯 OHLCV 因子不足以跑赢基准

### 优先级高（立即可做，数据已齐）
1. **开发「行业中性 + 动量 + 估值」新策略**
   - 数据已齐（行业 + PB/PE + 自由流通市值）
   - 因子行业内标准化 → 消除行业暴露
   - 加入 PB/PE 估值因子，与动量因子形成互补
2. **升级现有 multifactor 策略**
   - 加入估值因子（PE/PB）和行业中性化
   - 这是当前唯一正收益策略，值得优先增强
3. **行业内排序版纯市场策略**
   - 之前因缺行业数据搁置，现在可以实现

### 优先级中
4. 回测加入**交易成本**（约0.3%双边，即买入+卖出各0.15%）
5. 测试**双周/周度调仓**是否改善动量因子信号衰减
6. 探索 CTA 趋势跟踪策略（需期货数据）

### 剩余数据缺口
- 沪深300成分股历史（Morty 提供后可直接做指数增强）
- ROE(TTM)（质量因子，第二优先级）
- 经营现金流/净利润（盈利质量因子，第三优先级）

---

## 八、回测规范（必须遵守）

详见 `backtest/OutputFormat.md`，核心要点：

1. **每次回测必须附带基准对标**，禁止只报绝对收益
2. 基准优先选：上证指数（`sh000001`）+ 中证500（`sz000905`）双基准
3. 输出文件命名：`{strategy_name}_nav.csv` / `{strategy_name}_monthly_returns.csv` / `{strategy_name}_report.md`
4. nav 文件字段：`date, strategy, benchmark[, benchmark2]`，累计净值初始为1.0
5. returns 文件字段：`date, port_ret, n_stocks, bench_ret, excess[, bench2_ret, excess2]`，收益率为小数

---

## 九、和 Morty 的沟通风格

- 他叫 Morty，你可以用 Rick 的风格（见 `IDENTITY.md`），也可以直接专业沟通
- 直接给结论和行动，不要废话铺垫
- 遇到策略失效要主动分析原因，给出改进方向让他选
- 数据缺口要明确告知需要他提供什么格式的文件

---

*生成时间：2026-03-19*  
*最后更新：当前 Agent — 新增估值/行业数据合并、策略回测总结、下一步优先级调整*  
*初始版本：2026-03-18 by 前任 Agent（Rick）*
