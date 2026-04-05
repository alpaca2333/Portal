# public/quant — 量化仪表盘前端

> 本文档供接手此前端子应用的 Agent 快速上手。

---

## 一、这是什么

`public/quant/index.html` 是一个**纯静态单页应用**，集成在 portal 主应用中，作为量化研究的可视化界面。
无构建步骤，直接由 Fastify 的 `@fastify/static` 托管，访问地址：`http://127.0.0.1:8080/quant/index.html`

---

## 二、功能模块

### Tab 1：📈 股票行情
- 输入股票代码（如 `SH600519`、`SZ000001`）或**名称**查询 K 线
- 支持日期范围筛选 + 快速选择（近1/3/5/10年、全部）
- 股票代码/名称自动补全（输入1字符后触发，防抖 200ms），下拉同时展示代码和名称
- 展示区间最高/最低/涨跌幅/平均成交量统计卡片
- ECharts K 线图 + 成交量子图，支持缩放
- **点击K线/成交量**可查看该日全部详情（价格、估值、财务盈利、偿债运营、同比增长、行业分类）

### Tab 2：🧪 回测结果
- 自动拉取可用策略列表，渲染策略切换按钮
- 展示累计收益、年化收益、夏普比率、最大回撤、月胜率、回测月数
- ECharts 净值曲线（支持策略 + 基准 + 基准2 三条线）
- 月度收益明细表（倒序，支持双基准列）
- 可选 Markdown 策略报告（用 `marked.js` 渲染）

### Tab 3：📚 研究报告
- 聚合展示 `data/quant/research/` 和 `data/quant/reports/` 下的所有 `.md` 文档
- 目录与文档使用和“回测结果”一致的下拉交互风格
- 默认优先打开当前目录下最近更新的文档
- 支持按文件名/相对路径搜索过滤文档
- 展示目录文档数、筛选结果数、最近更新时间
- 文档正文使用 `marked.js` 渲染，支持直接复制当前页面深链

---

## 三、与后端的接口约定

所有 API 均挂载在 `/api/quant/` 前缀下，由 `src/routes/quant.ts` 实现。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/quant/status` | GET | 数据库加载状态，返回 `{ ready, stockCount }` |
| `/api/quant/stock/kline` | GET | K 线数据，参数：`code`（必填）、`start`、`end`，返回 `{ code, name, data }` |
| `/api/quant/stock/search` | GET | 股票搜索（代码+名称），参数：`q`（≥1字符），返回 `{ results: [{ code, name }] }` |
| `/api/quant/stock/detail` | GET | **单日全字段详情**，参数：`code`、`date`，返回56个字段+行业名称 |
| `/api/quant/strategies` | GET | 可用策略列表，返回 `{ strategies: string[] }` |
| `/api/quant/backtest` | GET | 回测数据，参数：`strategy`，返回 `{ nav, returns, report }` |
| `/api/quant/documents` | GET | 列出 `research/` 与 `reports/` 下 Markdown 文档，返回 `{ sections: [{ key, label, files[] }] }` |
| `/api/quant/document` | GET | 读取单篇 Markdown，参数：`section`、`file`，返回 `{ section, file, updatedAt, content }` |

### 数据库
后端使用 `data/quant/data/quant.db`（SQLite，~40MB），通过 `better-sqlite3` 同步查询，**不加载全量数据到内存**。
所有查询均按需走索引（主键 `ts_code + trade_date`），启动即可用，前端通过轮询 `/api/quant/status` 确认 `ready=true` 后再查询。

### stock/detail 返回字段分组

| 分组 | 字段 |
|------|------|
| 价格与成交 | open, high, low, close, pre_close, change, pct_chg, vol, amount |
| 估值指标 | pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, turnover_rate, turnover_rate_f, volume_ratio, total_mv, circ_mv |
| 盈利能力 | eps, bps, cfps, revenue_ps, roe, roe_dt, roe_waa, roe_yearly, roa, roa_yearly, grossprofit_margin, netprofit_margin, profit_to_gr |
| 偿债与运营 | debt_to_assets, current_ratio, quick_ratio, inv_turn, ar_turn, ca_turn, fa_turn, assets_turn |
| 同比增长 | op_yoy, ebt_yoy, tr_yoy, or_yoy, equity_yoy |
| 行业分类 | sw_l1, sw_l1_name, sw_l2, sw_l2_name, adj_factor, is_suspended |

### 回测数据格式
- `nav` 数组字段：`date, strategy, bench_{name1}, bench_{name2}, ...`（基准列名由 `baseline/` 目录自动生成，如 `bench_000001.SH`），净值初始为 1.0
- `returns` 数组字段：`date, port_ret, n_stocks, bench_ret_{name}, excess_{name}, ...`（均为小数，如 0.05 = 5%）

---

## 四、颜色约定（A 股习惯）

```css
--green: #ef4444;   /* 红色 = 上涨（A股习惯） */
--red:   #10b981;   /* 绿色 = 下跌（A股习惯） */
```

**注意**：变量名与颜色语义是反的，这是故意的，符合 A 股红涨绿跌惯例。
`.positive` class 用 `var(--green)`（红色），`.negative` class 用 `var(--red)`（绿色）。

---

## 五、URL 路由（Hash Router）

应用使用 `hash` 路由实现 URL 友好访问，无需后端支持，刷新页面状态不丢失。

| URL | 说明 |
|-----|------|
| `/quant/index.html#/stock` | 股票行情 Tab（默认页面） |
| `/quant/index.html#/stock/SH600519` | 直接查询贵州茅台 K 线 |
| `/quant/index.html#/stock/SH600519?start=2020-01-01&end=2025-01-01` | 带日期范围的股票查询 |
| `/quant/index.html#/backtest` | 回测结果 Tab（加载默认策略） |
| `/quant/index.html#/backtest/momentum` | 直接查看动量因子策略回测 |
| `/quant/index.html#/reports` | 研究报告 Tab（默认打开最新文档） |
| `/quant/index.html#/reports?section=reports&file=stock_lead_lag_research_directions.md` | 直接打开指定 Markdown 文档 |

- Tab 切换、股票查询、策略选择时 URL 会自动更新
- 用户可以复制 URL 分享给他人，打开后自动定位到对应内容
- 浏览器前进/后退按钮可正常工作

---

## 六、关键实现细节

### 收益率正负号显示
累计收益/年化收益需判断正负再决定是否加 `+` 前缀：
```js
`${value >= 0 ? '+' : ''}${value}%`
```
**不要**直接硬编码 `+` 前缀，否则负数会显示成 `+-22.8%`。

### 策略名称映射
`strategyLabel()` 函数维护策略 ID → 中文名的映射，新增策略时需同步更新：
```js
const map = { momentum: '动量因子', multifactor: '多因子组合' };
```

### 自动补全
- 输入框 `oninput` → 防抖 200ms → `fetchAutocomplete()` → 渲染下拉列表
- 支持按代码或名称搜索，最少1字符触发
- 下拉项同时展示代码和股票名称
- 支持键盘上下键导航 + Enter 选中 + Escape 关闭
- 点击页面其他区域自动关闭

### 每日详情面板
- 点击 K 线图或成交量柱触发 ECharts `click` 事件
- 调用 `/api/quant/stock/detail` 获取该日全部字段
- 数据分6组展示（价格/估值/盈利/偿债/增长/行业），全空的分组自动隐藏
- 涨跌幅/同比增长带红绿色标，市值自动换算亿/万亿
- 请求带 AbortController，快速切换日期时自动取消上一个请求

---

## 七、外部依赖（CDN）

| 库 | 版本 | 用途 |
|----|------|------|
| Tailwind CSS | latest | 布局工具类 |
| ECharts | 5.5.0 | K 线图 + 净值曲线 |
| marked | 12.0.0 | 渲染策略报告 Markdown |

无 npm 依赖，无构建步骤，改完刷新即生效。

---

## 八、文件结构

```
public/quant/
└── index.html          # 全部前端代码（HTML + CSS + JS，约 1700 行）

src/routes/
└── quant.ts            # 后端路由（Fastify 插件）

data/quant/
├── data/
│   ├── quant.db                # SQLite 数据库（~40MB，5770只股票）
│   ├── stock_basic/            # 股票列表 CSV
│   ├── calendar/               # 交易日历 CSV
│   ├── daily/                  # 日线 OHLCV，每股一文件
│   ├── daily_basic/            # 每日基本面，每股一文件
│   ├── adj_factor/             # 复权因子，每股一文件
│   ├── fina_indicator/         # 财务指标，每股一文件
│   ├── industry/               # 申万行业分类
│   └── suspend/                # 停牌记录
├── scripts/
│   ├── download_all.py         # 总控下载脚本
│   ├── download_*.py           # 各数据源下载脚本
│   └── build_db.py             # CSV → SQLite 构建脚本
├── backtest/
│   ├── {strategy}_nav.csv
│   ├── {strategy}_monthly_returns.csv
│   └── {strategy}_report.md    # 可选，Markdown 格式
├── research/                   # 研究草稿 / 中间分析 Markdown
├── reports/                    # 研究报告 / 结论文档 Markdown
├── strategies/factor/          # Python 策略脚本
├── utils/                      # Python 数据工具
└── AGENT.md                    # 量化研究工作区交接文档
```

---

## 九、常见修改场景

**新增策略**：在 `data/quant/backtest/` 下放入 `{name}_nav.csv` 和 `{name}_monthly_returns.csv`，前端会自动发现并显示按钮，无需改代码。若有报告则额外放 `{name}_report.md`。

**新增研究文档**：把 `.md` 文件放进 `data/quant/research/` 或 `data/quant/reports/`，页面会自动扫描并按最近更新时间排序显示；支持子目录，前端展示相对路径。

**修改图表样式**：直接编辑 `index.html` 中 ECharts `option` 对象，改完刷新浏览器即可。

**调整 API 路径**：前端所有 fetch 调用均使用 `/api/quant/` 前缀，后端在 `server.ts` 中以该前缀注册路由，两处需同步修改。
