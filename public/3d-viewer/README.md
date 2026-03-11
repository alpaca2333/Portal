# 3D Viewer Sub-Application - Agent Development Guide

## 项目概述 (Project Overview)
这是一个基于 Three.js 构建的纯前端 3D 模型查看器子应用，集成在 Portal Web App 中。它的主要功能是加载和渲染 FBX 格式的骨骼网格体（Skeletal Mesh），并支持加载配套的 FBX 动画文件进行播放。该查看器特别针对 Unreal Engine (UE) 导出的模型进行了坐标系适配。

## 目录位置 (Location)
`/public/3d-viewer/`

## 技术栈 (Tech Stack)
- **核心库**: Three.js (通过 ES Modules 和 Import Maps 引入)
- **加载器**: `FBXLoader` (依赖 `fflate` 进行解压)
- **控制器**: `OrbitControls` (用于鼠标/键盘的摄像机交互)
- **UI/样式**: 原生 HTML / CSS，悬浮在全屏 Canvas 之上

## 核心功能与逻辑 (Core Features & Logic)

1. **模型加载 (Model Loading)**:
   - 监听 `#file-input` 或拖拽事件。
   - 使用 `FBXLoader` 加载模型。
   - **UE 坐标系适配**: 加载完成后，代码会自动执行 `object.rotation.x = -Math.PI / 2;`，将 UE 的 Z-up 坐标系转换为 Three.js 的 Y-up 坐标系。

2. **动画加载 (Animation Loading)**:
   - 依赖于当前已加载的模型 (`currentModel`) 和动画混合器 (`mixer`)。
   - 监听 `#anim-input` 或 `Shift + 拖拽` 事件。
   - 加载新的 FBX 文件后，提取其 `animations[0]`，停止当前播放的动画，并将其应用到现有的 `mixer` 上。

3. **交互控制 (Controls)**:
   - 使用 `OrbitControls` 实现视角的旋转、缩放和平移。
   - 窗口大小改变时，自动更新摄像机的宽高比 (`aspect`) 和渲染器的尺寸。

## 给后续 Agent 的开发指南 (Guidelines for Future Agents)

1. **依赖管理 (Import Maps)**:
   - 本项目没有使用 Webpack/Vite 等打包工具，Three.js 及其附加组件是通过 `index.html` 中的 `<script type="importmap">` 引入的。
   - **注意**: 如果你需要引入新的 Three.js 扩展（例如 `GLTFLoader` 或其他 Post-processing 效果），**必须**先在 Import Map 中添加对应的路径映射，然后再在代码中使用 `import` 导入。

2. **状态管理 (State Variables)**:
   - `scene`, `camera`, `renderer`, `controls`, `mixer`: 全局 Three.js 核心对象。
   - `currentModel`: 保存当前加载的骨骼网格体引用。在处理动画或材质替换时，必须基于此对象进行操作。

3. **坐标系注意事项 (Coordinate System Warning)**:
   - 如果用户要求重置模型变换或修改旋转逻辑，请务必注意保留或重新应用 UE 坐标系转换逻辑（X轴旋转 -90 度），否则模型会“趴在地上”。

4. **UI 扩展 (UI Extension)**:
   - 控制面板位于 `<div id="controls">` 中。
   - 动画加载按钮 (`#btn-anim`) 默认是隐藏的 (`display: none`)，只有在成功加载第一个模型后才会显示。如果添加依赖于模型存在的新功能按钮，请遵循相同的显示/隐藏逻辑。

5. **性能与渲染 (Performance & Rendering)**:
   - 渲染循环在 `animate()` 函数中。
   - 动画更新依赖于 `THREE.Clock` 计算的 `delta` 时间 (`mixer.update(delta)`)。
   - 当前包含基础的半球光 (HemisphereLight) 和平行光 (DirectionalLight)，如果模型材质显示异常（如全黑），请优先检查灯光设置或材质的属性。