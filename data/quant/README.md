# Quant Workspace — 量化交易研究工作区

> Morty 的 A 股量化研究项目，由 Rick 操刀搭建。

---

## 项目状态

| 模块 | 状态 |
|------|------|
| 数据基础设施 | ✅ 完成 |
| 第一个因子策略（动量） | ✅ 完成 |
| 多因子组合 | 🔲 待开发 |
| CTA 趋势跟踪 | 🔲 待开发 |
| 统计套利 | 🔲 待开发 |
| 期权波动率策略 | 🔲 待开发 |

---

## 目录结构

```
quant_workspace/
├── data/
│   └── processed/
│       └── all_stocks_daily.csv       ← 全量 A 股日线（1421万行）
├── factors/                            ← 因子定义与计算（待填充）
├── strategies/
│   ├── stat_arb/                       ← 统计套利（待开发）
│   ├── factor/
│   │   └── momentum_backtest.py        ← 动量因子回测 ✅
│   ├── cta/                            ← CTA 趋势跟踪（待开发）
│   └── options/                        ← 期权波动率策略（待开发）
├── backtest/
│   ├── momentum_monthly_returns.csv    ← 动量策略月度收益
│   └── momentum_nav.csv                ← 动量策略净值曲线
├── notebooks/                          ← Jupyter 分析笔记（待填充）
└── utils/
    ├── data_loader.py                  ← 核心数据读取接口 ✅
    ├── import_data.py                  ← CSV 合并预处理脚本
    └── write_qlib_bin.py               ← Qlib 二进制格式写入脚本
```

---

## 数据基础设施

### 数据来源
- **原始数据**：手动提供的 Tushare CSV（公司内网无法直接调用 Tushare API）
- **原始路径**：`~/qlib_data/`，5489 个文件，命名格式 `{TICKER}.{EXCHANGE}.csv`

### 数据规模
| 项目 | 值 |
|------|----|
| 股票数量 | 5489 只（SH:2307, SZ:2884, BJ:298） |
| 时间范围 | 2006-03-20 ~ 2026-03-13 |
| 交易日数 | 4857 天 |
| 总数据行数 | 14,230,250 行 |
| 字段 | open, high, low, close, volume, factor |

### CSV 原始格式
```
symbol,date,open,high,low,close,volume,factor
600000,2006-03-20,10.75,10.93,10.53,10.86,185134.54,1.585
```
- `factor`：前复权因子（用于价格复权）
- `volume`：单位为手（100股/手）

### 数据读取接口 (`utils/data_loader.py`)

```python
from utils.data_loader import get_data, get_close, get_returns, get_universe

# 读取单只股票
df = get_data(codes=["SH600519"], start="2024-01-01", end="2024-12-31")

# 获取收盘价矩阵 (日期 × 股票)
close = get_close(codes=["SH600519", "SH600036"], start="2020-01-01", end="2024-12-31")

# 计算日收益率
ret = get_returns(start="2020-01-01", end="2024-12-31")

# 获取有效股票池
universe = get_universe(start="2020-01-01", end="2024-12-31", min_days=60)
```

> ⚠️ **注意**：qlib 0.9.7 与当前 pandas 版本存在兼容性 bug（`D.features()` 返回空 DataFrame），
> 所有数据读取均通过 `data_loader.py` 直接读 CSV，**不使用 qlib 的 `D.features()` 接口**。
> Qlib 二进制数据已写入 `~/.qlib/qlib_data/cn_data/`，待 qlib 版本兼容问题修复后可直接使用。

---

## 已完成策略

### 1. 动量因子策略 (`strategies/factor/momentum_backtest.py`)

**策略逻辑**

- 每月末用过去 250 天（约 12 个月）的累计收益率作为动量信号
- 跳过最近 20 天（避免短期反转效应）
- 做多 TOP 10% 动量最高的股票，等权持有
- 次月末调仓，标的范围：沪深主板（排除北交所）

**回测结果（2015-2025）**

| 指标 | 值 |
|------|-----|
| 年化收益 | **12.9%** |
| 年化波动 | 23.4% |
| 夏普比率 | 0.55 |
| 最大回撤 | -25.1% |
| 月胜率 | 54.5% |
| 累计收益 | +56.0% |

**净值曲线**

```
2016: 1.228
2017: 1.265  ← 平稳
2018: 1.164  ← 熊市回撤
2019: 1.294
2020: 1.537  ← 强势拉升
2021: 1.579
2022: 1.567  ← 震荡横盘
2023: 1.397
2024: 1.637  ← 新高
2025: 1.560
```

**分析**：A 股动量效应存在但不强（夏普 0.55），A 股存在明显短期反转和政策驱动行情，纯动量策略被割风险较高。建议后续加入多因子组合提升夏普。

---

## 待开发策略

### 2. 多因子组合（Factor Investing）
- 目标夏普：> 1.0
- 计划因子：动量 + 低波动 + 价值（PB/PE）+ 质量（ROE）
- 加权方式：等权 / IC 加权 / 机器学习组合

### 3. CTA 趋势跟踪
- 标的：商品期货主力合约（国内主要品种）
- 信号：均线系统 / ATR 突破 / 时序动量
- 注：需要期货账户数据

### 4. 统计套利
- 标的：期货跨品种（豆粕/豆油、铁矿/螺纹等）
- 方法：协整检验 + 配对交易
- 约束：A 股不可做空个股，期货跨品种优先

### 5. 期权波动率策略
- 标的：50ETF 期权 / 沪深 300 期权
- 方向：做空波动率（卖方策略）/ Delta 中性
- 前提：需要期权账户

---

## 环境

```
OS:     TencentOS Server 4.4 (Linux 5.4)
Python: 3.11
qlib:   0.9.7（已安装，部分 API 有兼容性问题）
rdagent: 0.8.0（已安装，Docker 场景不可用——容器无 privileged 权限）
容器 IP: 11.166.60.239
```

### 依赖安装
```bash
pip install pyqlib pandas numpy
```

---

## 回测规范

> **强制要求：每次回测必须附带指数基准对标，不得只报策略绝对收益。**

| 基准 | 适用策略 | 数据代码 |
|------|----------|----------|
| 沪深300（CSI 300） | A 股多头因子策略 | `sh000300`（需手动提供） |
| 中证500（CSI 500） | 中小盘因子策略 | `sh000905` |
| 万得全A | 全市场对标 | 参考 |

**对标输出格式（每次回测必须包含）：**
```
策略年化收益 vs 基准年化收益
策略夏普     vs 基准夏普
超额收益（Alpha）
信息比率（IR）
```

---

## 约束条件

- ❌ **不做 HFT**（频率不可接受）
- ❌ **不做事件驱动**（无信息来源）
- ❌ **不做个股空头**（无融券账户）
- ✅ **可做多个股**（A 股主板）
- ✅ **可做期货多空**（商品期货，待开户）
- ✅ **可做期权**（50ETF/沪深300，待开户）
- ⚠️ Tushare API 在公司内网被封锁，数据需手动提供

---

---

## Web Dashboard

量化研究可视化 Web 界面，支持股票日线 K 线图查看与回测结果展示，响应式设计，手机和电脑均可访问。

> ⚠️ **已集成进 portal 主应用**，不再作为独立服务运行。

### 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Node.js + TypeScript + Fastify 4（portal 子路由） |
| 前端 | 原生 HTML + TailwindCSS CDN + ECharts 5 |
| 数据 | 直接读取本地 CSV，无数据库 |

### 目录结构（portal 内）

```
portal/
├── src/
│   └── routes/
│       └── quant.ts           ← 后端路由插件（Fastify + 内存数据库）
├── public/
│   └── quant/
│       └── index.html         ← 前端单页（TailwindCSS + ECharts）
└── data/
    └── quant/
        ├── processed/
        │   └── all_stocks_daily.csv   ← 全量 A 股日线（770 MB）
        ├── backtest/                  ← 回测结果 CSV
        ├── strategies/                ← 策略 Python 脚本
        └── utils/                     ← 数据工具脚本
```

### 启动方式

```bash
cd /projects/portal

# 首次安装依赖
npm install

# 编译 TypeScript
npm run build

# 启动服务（默认端口 3000）
npm start

# 开发模式
npm run dev
```

服务启动后访问：`http://localhost:3000/quant`（或容器 IP `http://11.166.60.239:3000/quant`）

可通过环境变量覆盖端口和监听地址：

```bash
PORT=8080 HOST=0.0.0.0 npm start
```

### 内存预加载机制

> ⚠️ **重要**：服务启动后会立即在后台将全量 CSV（770 MB / 1421 万行）加载进内存，预热期约 20-40 秒。

- 数据结构：`Map<string, KlinePoint[]>`，key 为大写股票代码（如 `SH600519`）
- 预热期间 HTTP 服务正常响应，K 线查询接口返回 `503` 并提示加载进度
- 前端页面顶部有进度横幅，每 1.5 秒轮询 `/api/quant/status` 显示加载百分比
- 预热完成后进度横幅自动消失，所有查询走纯内存，响应 < 1ms
- 内存占用约 1-2 GB RSS，请确保容器/主机有足够内存

### API 接口

所有接口均以 `/api/quant` 为前缀，返回 JSON。

#### `GET /api/quant/status`
查询数据预加载进度，前端轮询使用。

```json
{
  "ready": false,
  "loadedRows": 3200000,
  "stockCount": 1120,
  "elapsedMs": 8500
}
```

#### `GET /api/quant/stock/kline`
获取指定股票的 K 线数据。

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `code` | string | ✅ | 股票代码，大写，如 `SH600519` |
| `start` | string | ⬜ | 开始日期 `YYYY-MM-DD` |
| `end` | string | ⬜ | 结束日期 `YYYY-MM-DD` |

```json
{
  "code": "SH600519",
  "count": 4857,
  "data": [
    { "date": "2006-03-20", "open": 10.75, "high": 10.93, "low": 10.53, "close": 10.86, "volume": 185134.54 }
  ]
}
```

> 预热期间返回 `503 { "error": "Data is still loading…" }`

#### `GET /api/quant/stock/search`
股票代码模糊搜索（前端自动补全使用）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | string | 搜索关键词，至少 2 个字符，大小写不敏感 |

```json
{ "results": ["SH600519", "SH600520"] }
```

> 预热期间返回空数组 `{ "results": [] }`

#### `GET /api/quant/backtest`
读取回测结果 CSV 文件。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | string | `momentum` | 策略名，对应 `backtest/{strategy}_nav.csv` |

```json
{
  "strategy": "momentum",
  "nav": [{ "date": "2016-01-29", "strategy": "1.0", "benchmark": "1.0" }],
  "returns": [{ "date": "2016-01-29", "port_ret": "0.187", "n_stocks": "230" }]
}
```

#### `GET /api/quant/strategies`
列出 `backtest/` 目录下所有可用策略（扫描 `*_nav.csv` 文件名）。

```json
{ "strategies": ["momentum", "multifactor"] }
```

#### `GET /api/quant/health`
健康检查，服务启动即可用。

```json
{ "status": "ok" }
```

### 前端功能

- **股票 K 线图**：输入股票代码（如 `SH600519`）后回车或点击查询，渲染 ECharts 蜡烛图 + 成交量柱状图
- **日期范围筛选**：可选择起止日期过滤数据
- **代码自动补全**：输入时实时搜索匹配的股票代码
- **回测结果展示**：切换策略查看净值曲线与月度收益柱状图
- **响应式布局**：TailwindCSS 自适应，手机/平板/桌面均可正常访问
- **加载进度横幅**：服务预热期间顶部显示进度条，加载完成后自动消失

### 新增策略后的接入步骤

1. 将回测脚本放入 `data/quant/strategies/` 对应子目录
2. 按规范输出 `{strategy}_nav.csv` 和 `{strategy}_monthly_returns.csv` 到 `data/quant/backtest/` 目录
3. 前端 `/api/quant/strategies` 接口会自动发现新策略，无需修改后端代码
4. 在前端策略下拉框中选择新策略即可查看回测图表

---

*Last updated: 2026-03-18*
