# 量化交易子应用 (Quant Trading Sub-App) - Agent 开发指南

## 1. 项目概述 (Overview)
这是一个基于 Tushare 数据源的量化交易前端展示与后端代理子应用。主要功能包括股票/指数的模糊搜索、K线图展示（集成 ECharts）以及实时基本信息面板。

## 2. 核心文件结构 (File Structure)
- **前端**:
  - `/public/quant-trading/index.html`: 包含所有的 UI 布局、CSS 样式（深色 Slate/Navy 主题）以及基于原生 JavaScript 和 ECharts 的图表渲染逻辑。
  - `/public/quant-trading/tushare.token`: 存放 Tushare API 的 Token 字符串（纯文本，无换行）。
- **后端**:
  - `/src/routes/quant-trading.ts`: Fastify 路由文件，负责与 Tushare API 进行通信，处理跨域和数据格式化。
  - `/src/server.ts`: 主服务入口，量化交易的路由被挂载在 `/api/quant` 前缀下。

## 3. 后端 API 接口 (Backend APIs)
所有接口均挂载在 `/api/quant` 前缀下：
- `GET /api/quant/search?keyword={text}`: 
  - 功能：股票/指数模糊搜索。
  - 逻辑：首次调用时会缓存 Tushare 的 `stock_basic` 列表，后续在内存中进行正则匹配。
- `GET /api/quant/kline?ts_code={code}`:
  - 功能：获取日线行情数据。
  - 逻辑：**重要**！Tushare 对股票和指数使用了不同的接口。后端会先尝试调用 `daily` 接口，如果无数据，会自动降级/切换调用 `index_daily` 接口。
- `GET /api/quant/basic_info?ts_code={code}`:
  - 功能：获取股票/指数的最新基本面和交易数据（开盘、收盘、最高、最低、成交量等）。
  - 逻辑：同样具备自动识别股票 (`daily_basic`) 和指数 (`index_dailybasic`) 的回退机制。

## 4. 给后续 Agent 的开发注意事项 (Important Notes for Future Agents)

### 4.1 Tushare API 限制与特性
- **接口区分**：Tushare 严格区分股票和指数。修改获取数据的逻辑时，务必保留或完善 `daily` -> `index_daily` 的 fallback 机制。
- **HTTPS**：请求 Tushare API 必须使用 `https://api.tushare.pro`。
- **权限问题**：部分高级数据（如分钟线、实时盘口）可能需要更高的 Tushare 积分，如果遇到接口无权限报错，请提示用户检查 Tushare 积分。

### 4.2 前端 UI 与 ECharts
- **主题颜色**：系统采用深色主题。上涨颜色统一为 `#ef4444` (红)，下跌颜色统一为 `#22c55e` (绿)。修改样式时请保持一致。
- **图表自适应**：ECharts 容器使用了 Flex 布局。如果在隐藏状态下初始化图表，会导致尺寸计算错误（变成 100px 宽/高）。必须在容器可见后初始化，并在渲染完成后调用 `chart.resize()`。
- **原生 JS**：前端没有使用 React/Vue 等框架，所有 DOM 操作和状态管理均在 `index.html` 的 `<script>` 标签中通过原生 JS 完成。

### 4.3 后端 ESM 规范
- 本项目启用了严格的 ESM 规范 (`"type": "module"`, `NodeNext`)。如果在 `/src/routes/quant-trading.ts` 中引入其他本地文件，**必须带上 `.js` 后缀**。