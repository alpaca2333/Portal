# Portal Web App - Agent Development Guide

## 项目概述 (Project Overview)
这是一个基于 Fastify 和 TypeScript 构建的轻量级应用门户网站。它的主要作用是作为一个统一的入口，集中管理和访问未来添加的各种子应用。

## 技术栈 (Tech Stack)
- **后端**: Node.js, Fastify (`fastify`, `@fastify/static`), TypeScript
- **前端**: 原生 HTML / CSS / JavaScript
- **运行环境**: 使用 `tsx` 运行 TypeScript，项目配置为 ESM (`"type": "module"`)

## 目录结构 (Directory Structure)
- `/src/server.ts`: 后端服务入口文件。配置了 Fastify 实例，监听 `0.0.0.0:8080`，并将 `/public` 目录作为静态资源根目录。
- `/public/index.html`: 门户网站的前端主页，采用响应式卡片布局。
- `/tsconfig.json`: TypeScript 配置文件，启用了 `strict` 模式，`module` 和 `moduleResolution` 均设置为 `NodeNext`。
- `/package.json`: 项目依赖和脚本配置。

## 运行指令 (Commands)
- 启动开发服务器: `npm run dev` 或 `npm start` (使用 `tsx` 运行)

## 给后续 Agent 的开发指南 (Guidelines for Future Agents)
1. **添加纯前端子应用**:
   - 在 `/public` 目录下创建一个新的文件夹（例如 `/public/my-app/`）。
   - 将该应用的 HTML/CSS/JS 文件放入其中。
   - 修改 `/public/index.html`，在 `<div class="app-grid">` 中添加一个新的卡片（`.app-card`），并将其点击事件或链接指向新应用的路径（如 `/my-app/index.html`）。
2. **添加后端 API 或全栈应用**:
   - 在 `/src` 目录下创建新的路由文件（例如 `/src/routes/my-app.ts`）。
   - 在 `/src/server.ts` 中引入并注册这些新路由。
   - **注意 ESM 规范**: 由于 `package.json` 中设置了 `"type": "module"`，且 `tsconfig.json` 使用了 `NodeNext`，在 TypeScript 中导入本地文件时，**必须包含 `.js` 后缀**（例如 `import { myHandler } from './myHandler.js'`）。
3. **样式与 UI**:
   - 门户主页目前使用原生 CSS 编写，保持轻量。如需修改全局样式，请直接编辑 `/public/index.html` 中的 `<style>` 标签或将其抽离为独立的 CSS 文件。
