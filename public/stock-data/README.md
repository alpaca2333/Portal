# 股票数据子应用 (Quant Trading Sub-App) - Agent 开发指南

## 1. 项目概述 (Overview)
这是一个基于 Tushare 数据源的股票数据前端展示与后端代理子应用。主要功能包括股票/指数的模糊搜索、K线图展示（集成 ECharts）以及实时基本信息面板。

## 2. 核心文件结构 (File Structure)
- **前端**:
  - `/public/stock-data/index.html`: 包含所有的 UI 布局、CSS 样式（深色 Slate/Navy 主题）以及基于原生 JavaScript 和 ECharts 的图表渲染逻辑。
  - `/public/stock-data/tushare.token`: 存放 Tushare API 的 Token 字符串（纯文本，无换行）。
- **后端**:
  - `/src/routes/stock-data.ts`: Fastify 路由文件，负责与 Tushare API 进行通信、数据格式化以及缓存调度。
  - `/src/routes/disk-cache.ts`: 磁盘缓存工具模块，提供 `readCache` / `writeCache` 两个函数，供所有路由复用。
  - `/src/server.ts`: 主服务入口，股票数据的路由被挂载在 `/api/quant` 前缀下。
- **缓存**:
  - `/cache/stock-data/`: 缓存文件存放目录，每个缓存条目是一个独立的 JSON 文件（见第 5 节）。

## 3. 后端 API 接口 (Backend APIs)
所有接口均挂载在 `/api/quant` 前缀下，**所有接口均已接入磁盘缓存**（缓存策略见第 5 节）：
- `GET /api/quant/search?query={text}`: 
  - 功能：股票/指数模糊搜索。
  - 逻辑：读取/刷新 `stock_list.json` 缓存，在内存中对列表做字符串匹配，返回前 10 条结果。
- `GET /api/quant/kline?ts_code={code}`:
  - 功能：获取过去 5 年的日线行情数据。
  - 逻辑：**重要**！Tushare 对股票和指数使用了不同的接口。后端会先尝试调用 `daily` 接口，如果无数据，会自动降级/切换调用 `index_daily` 接口。结果以 `kline_{ts_code}.json` 缓存。
- `GET /api/quant/basic_info?ts_code={code}`:
  - 功能：获取股票/指数的最新基本面和交易数据（开盘、收盘、最高、最低、成交量等）。
  - 逻辑：同样具备自动识别股票 (`daily_basic`) 和指数 (`index_dailybasic`) 的回退机制。结果以 `basic_info_{ts_code}.json` 缓存。

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
- 本项目启用了严格的 ESM 规范 (`"type": "module"`, `NodeNext`)。如果在 `/src/routes/stock-data.ts` 中引入其他本地文件，**必须带上 `.js` 后缀**。

## 5. 磁盘缓存系统 (Disk Cache System)

### 5.1 概述
缓存功能完全在服务端实现，前端无感知。核心逻辑位于 `/src/routes/disk-cache.ts`，采用 **Stale-While-Revalidate** 策略。

### 5.2 缓存文件位置
所有缓存文件存放在项目根目录下的 `/cache/stock-data/` 中，每个键对应一个 JSON 文件：

| 缓存 Key | 文件名 | 内容 |
|---|---|---|
| `stock_list` | `stock_list.json` | 全量股票列表（`ts_code`, `symbol`, `name`）|
| `kline_{ts_code}` | `kline_{ts_code}.json` | 该标的过去 5 年日K数据 |
| `basic_info_{ts_code}` | `basic_info_{ts_code}.json` | 该标的最新基本面数据 |

文件格式：
```json
{
  "timestamp": 1741680000000,
  "data": { ... }
}
```

### 5.3 缓存有效期
- `CACHE_TTL_MS = 60 * 60 * 1000`（**1 小时**），定义在 `disk-cache.ts` 顶部，如需调整在此处修改。

### 5.4 Stale-While-Revalidate 行为

每个 API 接口对三种缓存状态的处理方式：

| 缓存状态 | 行为 |
|---|---|
| **无缓存文件**（首次请求）| 同步请求 Tushare API，写入磁盘，返回结果 |
| **缓存新鲜**（< 1 小时）| 直接读磁盘返回，不访问 API |
| **缓存过期**（> 1 小时）| **立即返回旧缓存**（页面可快速展示），同时在后台异步刷新，刷新完成后静默更新磁盘文件 |

### 5.5 路径解析注意事项
- `disk-cache.ts` 使用 `process.cwd()` 定位缓存目录，而非 `__dirname`。
- 原因：项目通过 `tsx` 直接运行 TypeScript（无 `outDir`），`import.meta.url` 推导出的 `__dirname` 层级在某些 `tsx` 版本下会发生偏移。
- **服务器必须从项目根目录启动**（即 `npm run dev`），`process.cwd()` 才能正确解析为 `/root/Projects/Portal`。