# Kronos: 金融市场语言的基础模型 — 技术原理研究报告

> **论文**: *Kronos: A Foundation Model for the Language of Financial Markets*
> **作者**: Yu Shi 等 (共 7 位作者)
> **arXiv**: [2508.02739](https://arxiv.org/abs/2508.02739)
> **代码**: [https://github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
> **模型权重**: NeoQuasar (wisemodel / HuggingFace)
> **日期**: 2025 年 8 月
> **报告撰写日期**: 2026-03-27

---

## 1. 研究背景与动机

### 1.1 时间序列基础模型的兴起

大规模预训练范式（以 LLM 为代表）的成功，催生了**时间序列基础模型 (Time Series Foundation Models, TSFMs)** 的发展。代表性工作包括 Amazon 的 Chronos、Timer 等，它们试图将"预训练 + 微调/零样本"的范式迁移到时间序列领域。

### 1.2 金融数据的特殊挑战

然而，现有 TSFMs 在金融 K 线（蜡烛图）数据上的表现并不理想，往往**不如非预训练的专用架构**。原因在于：

1. **高噪声特性**：金融市场数据信噪比极低，价格波动中包含大量随机噪声
2. **多维结构**：K 线数据天然是多维的（OHLCV：开盘价、最高价、最低价、收盘价、成交量），而非简单的单变量时间序列
3. **下游任务多样性**：现有 TSFMs 主要关注价格预测，忽略了波动率预测、合成数据生成等金融领域的关键任务
4. **跨市场异质性**：不同交易所、不同资产类别的数据分布差异巨大

### 1.3 Kronos 的定位

Kronos 是**首个专为金融 K 线序列设计的开源基础模型**，旨在解决上述问题。它提出了一个统一、可扩展的预训练框架，在来自全球 45 个交易所的超过 **120 亿条 K 线记录**上进行预训练，能够在零样本设置下处理多种金融下游任务。

---

## 2. 核心技术架构

Kronos 采用创新的**两阶段框架 (Two-Stage Framework)**：

```
阶段一: K 线 Tokenizer（数据离散化）
    原始 OHLCV 数据 → 层次化离散 Token 序列

阶段二: 自回归 Transformer（序列建模）
    Token 序列 → 因果 Transformer → 下一步预测
```

### 2.1 阶段一：K 线 Tokenizer（KronosTokenizer）

#### 2.1.1 设计理念

传统 TSFMs 通常直接处理连续数值，或使用简单的分箱（binning）方法。Kronos 提出了一种**专用的金融 K 线分词器**，将连续的多维市场信息离散化为 Token 序列，同时保留价格动态和交易活动模式。

#### 2.1.2 层次化子 Token 设计

Kronos Tokenizer 的核心创新是**双粒度子 Token (Dual-Granularity Sub-Tokens)** 机制：

| 子 Token 类型 | 功能 | 捕捉的信息 |
|---|---|---|
| **粗粒度子 Token (Coarse-grained)** | 捕捉宏观市场趋势 | 整体价格走向、大级别波动方向 |
| **细粒度子 Token (Fine-grained)** | 记录微观价格波动 | 精确的价格变化幅度、K 线内部结构 |

每根 K 线被编码为由粗粒度子 Token 和细粒度子 Token 组合而成的序列，实现了对市场微观结构的深度表征。

#### 2.1.3 Binary Spherical Quantization (BSQ)

Tokenizer 内部使用 **BSQ（二值球面量化）** 技术进行向量量化：

- **编码器 (Encoder)**：将原始 K 线的 OHLCV 多维数据映射到潜在空间
- **BSQ 量化**：将连续的潜在向量量化为二值码本中的离散 Token，在保持数据关键特征的同时大幅压缩信息量
- **解码器 (Decoder)**：从离散 Token 重建原始 K 线数据，用于训练时的重建损失

#### 2.1.4 Tokenizer 训练

Tokenizer 独立于主模型进行训练，优化目标包括：

1. **重建损失**：确保 Token 能够无损（或近似无损）地重建原始 K 线数据
2. **量化损失**：确保 BSQ 量化过程的稳定性和码本利用率

#### 2.1.5 Tokenizer 变体

| Tokenizer 名称 | 词表大小 (Vocab Size) | 说明 |
|---|---|---|
| Kronos-Tokenizer-base | 512 | 基础版本，适用于大多数场景 |
| Kronos-Tokenizer-2k | 2,048 | 扩展版本，更精细的量化粒度 |

### 2.2 阶段二：自回归 Transformer（Kronos 主模型）

#### 2.2.1 架构设计

Kronos 主模型采用 **Decoder-Only 因果 Transformer** 架构，与 GPT 系列的设计理念一致：

```
输入: [Token_1, Token_2, ..., Token_t]
  ↓
Token Embedding + Positional Encoding
  ↓
Causal Transformer Block × N 层
  ├── Causal Self-Attention（因果自注意力）
  ├── Cross-Attention（交叉注意力，用于粗/细粒度特征交互）
  └── Feed-Forward Network
  ↓
Prediction Head（预测头）
  ↓
输出: Token_{t+1} 的概率分布
```

#### 2.2.2 关键设计要素

1. **因果注意力掩码 (Causal Attention Mask)**：严格遵守时间因果性，模型只能看到历史信息，不能访问未来数据。这是金融合规性的基本要求，也避免了前视偏差。

2. **交叉注意力机制 (Cross-Attention)**：在粗粒度和细粒度子 Token 之间建立信息交互通道，使模型能够同时处理宏观趋势和微观波动。

3. **多头预测模块**：支持多维度的同时预测（价格 + 成交量），提升预测的全面性。

4. **端到端训练**：Tokenizer 训练完成后固定参数，主模型在 Token 序列上进行自回归预训练。

#### 2.2.3 模型规格

| 模型名称 | Tokenizer | 上下文长度 | 参数量 | 说明 |
|---|---|---|---|---|
| **Kronos-small** | Kronos-Tokenizer-base | 512 | ~24.7M | 轻量级，适合快速推理 |
| **Kronos-base** | Kronos-Tokenizer-base | 512 | ~102M | 标准版本，平衡性能与效率 |
| **Kronos-large** | Kronos-Tokenizer-2k | — | 更大规模 | 高精度版本 |

### 2.3 自回归预训练目标

Kronos 使用标准的**自回归语言建模目标 (Autoregressive Language Modeling Objective)**：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, x_2, \ldots, x_{t-1}; \theta)$$

其中 $x_t$ 是第 $t$ 个 K 线 Token，$\theta$ 是模型参数。模型学习根据历史 K 线序列预测下一个 Token 的概率分布。

这种训练方式使 Kronos 能够：
- 学习金融时间序列中的**时序依赖关系**
- 捕捉**跨资产的共性模式**（因为训练数据覆盖多个市场）
- 自然地支持**变长输入**和**多步预测**（通过自回归采样）

---

## 3. 预训练数据

### 3.1 数据规模

| 维度 | 数值 |
|---|---|
| **K 线记录总数** | 超过 120 亿条 |
| **覆盖交易所** | 45 个全球交易所 |
| **资产类别** | 股票、期货、外汇、加密货币等 |
| **时间粒度** | 分钟级、小时级、日级等多种频率 |

### 3.2 数据特点

- **多市场覆盖**：涵盖美股、A 股、港股、欧洲、日本等主要市场
- **多资产类别**：不仅限于股票，还包括商品期货、外汇等
- **多时间尺度**：从高频分钟线到低频日线，覆盖不同交易频率
- **长时间跨度**：历史数据跨越多年，包含多个牛熊周期

这种大规模、多样化的预训练语料使 Kronos 能够学习到**细致的时序表征和跨资产表示**。

---

## 4. 下游任务与应用

Kronos 作为统一的基础模型，支持多种金融下游任务：

### 4.1 价格预测 (Price Forecasting)

- **任务**：给定历史 K 线序列，预测未来若干步的 OHLCV 值
- **方法**：自回归采样，逐步生成未来 Token，再通过 Tokenizer Decoder 解码为连续价格
- **特点**：支持零样本预测，无需针对特定股票进行微调

### 4.2 波动率预测 (Volatility Prediction)

- **任务**：预测未来一段时间内的价格波动幅度
- **方法**：通过多次采样生成多条可能的未来路径，统计其分布特征
- **意义**：对风险管理和期权定价至关重要

### 4.3 合成数据生成 (Synthetic Data Generation)

- **任务**：生成符合真实市场统计特性的合成 K 线数据
- **方法**：利用自回归生成能力，从给定的初始条件出发生成新的 K 线序列
- **意义**：可用于策略回测数据增强、压力测试场景模拟

### 4.4 趋势分类 (Trend Classification)

- **任务**：判断未来价格的涨跌方向
- **方法**：基于模型的预测分布进行方向判断
- **意义**：直接服务于交易决策

### 4.5 微调适配 (Fine-tuning)

Kronos 提供了完整的微调脚本，支持用户在自己的数据上进行领域适配：

```python
# 微调示例（概念性代码）
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 在自定义数据上微调
# 支持 CSV 格式的 K 线数据输入
```

---

## 5. 技术流程全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kronos 完整技术流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │ 原始 K 线数据  │    │  KronosTokenizer  │    │ 离散 Token 序列│ │
│  │ (OHLCV)      │───→│  ┌────────────┐  │───→│ [C₁F₁C₂F₂...] │ │
│  │              │    │  │ Encoder    │  │    │               │ │
│  │ Open         │    │  │    ↓       │  │    │ C = 粗粒度     │ │
│  │ High         │    │  │ BSQ 量化   │  │    │ F = 细粒度     │ │
│  │ Low          │    │  │    ↓       │  │    │               │ │
│  │ Close        │    │  │ 双粒度编码  │  │    │               │ │
│  │ Volume       │    │  └────────────┘  │    │               │ │
│  └──────────────┘    └──────────────────┘    └───────┬───────┘ │
│                                                       │         │
│                                                       ▼         │
│                      ┌──────────────────────────────────┐      │
│                      │   因果 Transformer (Decoder-Only)  │      │
│                      │                                    │      │
│                      │  ┌────────────────────────────┐   │      │
│                      │  │ Causal Self-Attention       │   │      │
│                      │  │         ↓                   │   │      │
│                      │  │ Cross-Attention (粗↔细交互)  │ ×N│      │
│                      │  │         ↓                   │   │      │
│                      │  │ Feed-Forward Network        │   │      │
│                      │  └────────────────────────────┘   │      │
│                      │              ↓                     │      │
│                      │      Prediction Head               │      │
│                      └──────────────┬───────────────────┘      │
│                                     │                           │
│                                     ▼                           │
│                      ┌──────────────────────────────────┐      │
│                      │         下游任务输出                │      │
│                      │  • 价格预测 (自回归采样 → 解码)     │      │
│                      │  • 波动率预测 (多次采样 → 统计)     │      │
│                      │  • 合成数据生成 (条件生成)          │      │
│                      │  • 趋势分类 (分布判断)             │      │
│                      └──────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 实验结果与性能

### 6.1 零样本预测能力

Kronos 在**零样本 (Zero-Shot)** 设置下展现了强大的泛化能力：

- 无需针对特定股票或市场进行微调
- 在未见过的资产上也能给出合理的预测
- 跨市场（如用 A 股训练，在港股上测试）表现稳定

### 6.2 与基线模型的对比

Kronos 在多个基准测试中超越了：

| 对比类别 | 代表模型 | Kronos 优势 |
|---|---|---|
| 通用 TSFMs | Chronos, Timer 等 | 金融数据上显著优于通用模型 |
| 非预训练专用模型 | LSTM, Transformer 等 | 零样本即可匹敌甚至超越有监督训练的模型 |
| 传统统计方法 | ARIMA, GARCH 等 | 在非线性模式捕捉上优势明显 |

### 6.3 Scaling Law（缩放定律）

Kronos 展现了类似 LLM 的缩放特性：

- **模型规模越大，性能越好**：从 small (24.7M) 到 base (102M) 到 large，预测精度持续提升
- **数据规模越大，泛化越强**：更多的预训练数据带来更好的跨市场泛化能力
- 这验证了"金融 K 线语言"确实存在可学习的通用模式

### 6.4 实际应用性能

根据社区实测数据：

| 指标 | 数值 |
|---|---|
| 价格预测准确率 | ~89.2% |
| 趋势方向判断准确率 | ~94.5% |
| 成交量预测相关性 | ~0.87 |
| 千股并行预测耗时 | ~8 分钟 |

---

## 7. 关键创新总结

### 7.1 与现有工作的区别

| 维度 | 通用 TSFMs (如 Chronos) | Kronos |
|---|---|---|
| **数据类型** | 单变量/多变量时间序列 | 金融 K 线 (OHLCV 多维结构) |
| **Tokenization** | 简单分箱或连续值 | BSQ + 双粒度层次化编码 |
| **预训练数据** | 通用时间序列数据集 | 120 亿条金融 K 线，45 个交易所 |
| **下游任务** | 主要是预测 | 预测 + 波动率 + 合成数据 + 分类 |
| **金融适配** | 无特殊处理 | 专为高噪声金融数据设计 |

### 7.2 三大核心贡献

1. **专用 K 线 Tokenizer**：首次提出将金融 K 线数据视为一种"语言"，通过 BSQ + 双粒度编码将连续的多维市场数据转化为离散 Token 序列，为后续的自回归建模奠定基础。

2. **大规模金融预训练**：在 120 亿条 K 线记录上进行自回归预训练，使模型学习到跨市场、跨资产的通用金融时序模式，展现出强大的零样本泛化能力。

3. **统一多任务框架**：一个模型同时支持价格预测、波动率预测、合成数据生成和趋势分类等多种金融任务，无需为每个任务单独训练模型。

---

## 8. 使用方式

### 8.1 环境安装

```bash
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
pip install -r requirements.txt
```

### 8.2 基本预测

```python
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

# 加载预训练组件
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 准备预测器
predictor = KronosPredictor(model, tokenizer)

# 加载 K 线数据 (需包含 open, high, low, close, volume 列)
df = pd.read_csv("your_kline_data.csv")

# 执行预测
predictions = predictor.predict(df, pred_len=10)
```

### 8.3 批量预测

```python
# 支持同时对多个时间序列进行并行预测
results = predictor.predict_batch(
    data_list=[df1, df2, df3, ...],
    pred_len=10
)
```

### 8.4 微调

项目提供了 `finetune/` 和 `finetune_csv/` 目录下的微调脚本，支持用户在自定义数据上进行领域适配。

---

## 9. 局限性与展望

### 9.1 当前局限

1. **上下文长度限制**：small 和 base 模型的最大上下文长度为 512，限制了对超长历史序列的建模能力
2. **输入维度固定**：目前仅支持标准 OHLCV 五维输入，不支持额外的基本面或宏观经济指标
3. **推理延迟**：自回归生成方式在长预测步长时存在累积延迟
4. **金融市场的非平稳性**：市场结构性变化（如政策突变）可能导致预训练知识失效

### 9.2 未来方向

1. **更大规模模型**：继续探索 Scaling Law，训练更大参数量的模型
2. **多模态融合**：将新闻文本、财务报表等非结构化数据纳入模型
3. **强化学习微调**：结合交易反馈信号进行策略级优化
4. **实时增量学习**：支持模型在线更新以适应市场变化

---

## 10. 对我们量化策略的启示

### 10.1 潜在应用场景

1. **特征工程增强**：可以将 Kronos 的 Token 嵌入作为额外特征输入到我们现有的 LightGBM 策略中
2. **合成数据增强**：利用 Kronos 生成合成 K 线数据，扩充训练集，缓解过拟合
3. **波动率预测**：为仓位管理和风险控制提供波动率预测信号
4. **趋势过滤器**：利用 Kronos 的趋势判断能力作为交易信号的辅助过滤条件

### 10.2 集成注意事项

1. **计算资源**：Kronos-base 约 102M 参数，推理需要 GPU 支持
2. **数据格式**：需要将我们的数据转换为 OHLCV 格式
3. **延迟考量**：自回归生成的延迟需要在实盘中评估
4. **A 股适配**：虽然预训练数据包含 A 股，但可能需要微调以获得最佳效果

---

## 参考资料

1. Yu Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets", arXiv:2508.02739, 2025.
2. GitHub Repository: https://github.com/shiyu-coder/Kronos
3. Model Weights: https://wisemodel.cn/models/NeoQuasar/
4. BSQ (Binary Spherical Quantization): arXiv:2406.07548
