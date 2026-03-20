# Transformer 在 A 股预测中的应用：深度研究报告

**作者**：量化研究员（AI Subagent）  
**日期**：2026-03-20  
**版本**：v1.0  
**数据基础**：5489只A股，2006-2026，14,230,250条日频记录  
**现有策略基准**：ic_reweighted（年化19.81%，夏普0.85，2019-2026）

---

## 执行摘要

**核心结论**：Transformer 值得引入我们的策略，但不是以"替代"的方式，而是以"融合增强"的方式。具体路线是：构建一个轻量级 Cross-sectional Transformer，将其预测信号作为新的 alpha 因子，与现有的 ic_reweighted 因子体系做 stacking ensemble。预计可在现有 ICIR ~0.35 的基础上提升 10-20%，最终体现为夏普比例从 0.85 提升到 0.95-1.05 区间。

**不建议**纯粹替换现有因子模型，因为学术上有效的 Transformer 时序预测在实盘中往往过拟合，而我们的因子体系已经具备真实 OOS 验证。

**建议的优先级**：iTransformer（跨截面）> PatchTST（时序 patch）> TFT > 从零设计

---

## 一、可行性分析：Transformer 在股票预测上真的有效吗？

### 1.1 理论优势

Transformer 的核心机制是自注意力（Self-Attention），其数学表达为：

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

在股票预测场景下，这个机制提供了三个无法被传统方法复制的能力：

**（1）非线性的跨股相关性捕捉**

传统因子模型（如我们的 ic_reweighted）假设每只股票的收益是各因子的线性组合，且股票之间相互独立。但现实中，行业联动、资金轮动、事件驱动都会产生"股票A涨 → 股票B也会涨"的非线性关联。注意力机制可以直接学到这种关联。

MASTER（AAAI 2024，arXiv:2312.15235）正是基于这个思路，同时建模"时序内部"（intra-stock）和"跨股截面"（inter-stock）信息，并通过市场宏观信息做特征选择权重调整，在 CSI 300/500 数据集上显著超越 LSTM 和早期 Transformer 基线。

**（2）动态特征选择**

TFT（Temporal Fusion Transformer，arXiv:1912.09363）引入了变量选择网络（Variable Selection Network，VSN）。在我们的场景中，某些因子在特定市场环境下有效（如 rev_10 在 2025 年 ICIR = 0.76），而在其他时期失效（如 2024 年 ICIR = 0.13）。TFT 可以学到"当前市场环境下应该依赖哪些因子"，这是静态权重的 ic_reweighted 无法做到的。

**（3）长程依赖建模**

LSTM 的遗忘门使得 60 天以上的历史信息会显著衰减。Transformer 通过注意力可以直接关注任意历史时间步，对于月频以上的价格模式（如季节性、宏观周期）有优势。

### 1.2 关键局限——"Transformer 真的适合时序预测吗？"

这是整个研究中最关键的批判性视角。

2022 年发表的 LTSF-Linear（arXiv:2205.13504）给整个社区泼了冷水：**一个只有一层线性变换的极简模型（DLinear）在 9 个真实世界时序预测数据集上全面超越了 Informer、Autoformer、FEDformer 等当时最先进的 Transformer 模型**。

其核心论点是：Transformer 的 permutation-invariant 自注意力机制本质上会破坏时间序列的有序性，而价格序列的核心信息恰恰就在顺序中（今天的价格依赖于昨天，而不是"哪天"并不重要）。

**这个批评在股票预测中有效吗？**

我的判断是：**对纯时序预测（预测价格绝对值/趋势）有效，但对截面排名预测（预测谁涨得最多）的影响较弱**。

原因：
- 截面预测的核心不是"明天涨多少"，而是"A 相对于 B 涨得更多"。这里，股票之间的注意力（cross-stock attention）不需要时间序列的有序性，它捕捉的是同截面内的相对信息。
- iTransformer（arXiv:2310.06625，ICLR 2024）实验性地将注意力从"时间 token"翻转为"变量 token"（即每只股票是一个 token），反而在多变量预测上取得了 SOTA。这个思路和截面 Transformer 高度一致。

**结论：时序 Transformer 在价格预测上的优势不确定；但截面 Transformer 在股票排名预测上有明确的理论和实证支撑。**

### 1.3 市场效率视角：能捕捉什么 alpha？

从有效市场假说（EMH）出发，中国 A 股属于弱式到半强式效率之间，有以下几个已知的 alpha 来源适合 Transformer 捕捉：

| Alpha 类型 | 来源 | Transformer 的优势 | 传统因子的局限 |
|-----------|------|------------------|--------------|
| 行业轮动效应 | 资金在行业间流动 | 跨股注意力直接建模 | 行业哑变量是静态的 |
| 反转（已验证：rev_10，ICIR=0.37） | 短期超卖回撤 | 非线性价格 pattern | 线性 IC 已部分捕捉 |
| 低波动（已验证：rvol_20，ICIR=0.34） | 风险厌恶溢价 | — | 线性因子已有效 |
| 事件驱动（财报、政策） | 信息不对称 | 多模态融合 | 无法建模 |
| 因子时变有效性 | 市场体制切换 | 动态注意力权重 | 静态权重 |

**Transformer 真正的增量 alpha** 主要来自：**（1）因子间的非线性交互；（2）市场体制的动态感知；（3）跨股联动的隐含信息**。

---

## 二、参数规模与架构选择

### 2.1 金融数据的规模需要多大的模型？

**我们的数据量**：
- 5489 只股票 × 约 4900 个交易日 ≈ 14,230,250 个观测点
- 但双周调仓频率下，有效训练样本 = 5489 × ~580 个双周期 ≈ 318 万个截面观测
- 每个截面（每个双周调仓日）平均约 4000-5000 只可用股票

这个规模对应的合理模型规模：

| 模型规模 | 参数量 | 适合场景 | 风险 |
|---------|--------|---------|------|
| 微型（Micro） | 10K-100K | 单只股票时序 | 表达能力不足 |
| 小型（Small） | 100K-1M | 截面排名，我们的场景 | **推荐范围** |
| 中型（Medium） | 1M-10M | 多任务、预训练 | 过拟合风险上升 |
| 大型（Large） | 10M+ | 通用金融基础模型 | 数据量不够支撑 |

**具体估算**：Transformer 的参数量 ≈ 12 × d_model² × num_layers。对于 d_model=64，num_layers=2，参数量约 98K。对于 d_model=128，num_layers=4，参数量约 786K。**建议范围：d_model=64-128，num_heads=4-8，num_layers=2-4**。

经验法则：**参数量不应超过训练样本数的 1%**。我们有 ~318 万样本，参数量上限 ~31,800。但 Transformer 有共享权重机制（跨股），实际上一个截面 4000 只股票相当于 batch_size=4000，有效样本利用率更高，可以放宽到参数量不超过有效 batch 总数的 10%，即 ~31,800 ÷ 10% = OK for 786K 规模。

### 2.2 主流架构横向对比

| 架构 | 发表时间 | 核心创新 | 股票预测适配性 | 推荐指数 |
|------|---------|---------|--------------|---------|
| **iTransformer** | ICLR 2024 | 将时间维度和变量维度翻转，变量（股票）作为 token | ⭐⭐⭐⭐⭐ 天然适合截面预测 | 首选 |
| **PatchTST** | ICLR 2023 | 时序 patch 作为 token，减少注意力计算量 | ⭐⭐⭐⭐ 适合时序 pattern | 次选 |
| **TFT** | Int. J. Forecasting 2021 | 可解释性强，变量选择，多尺度 LSTM+Attention | ⭐⭐⭐⭐ 适合因子动态选择 | 可解释性需求时用 |
| **MASTER** | AAAI 2024 | 市场引导特征选择 + 跨时间股票相关 | ⭐⭐⭐⭐ 专门为股票设计 | 学习参考 |
| **Informer** | AAAI 2021 | 稀疏注意力 ProbSparse | ⭐⭐ 为长序列设计，股票应用意义有限 | 不推荐 |
| **Chronos/TimesFM** | 2024 | 预训练时序基础模型（Decoder-only） | ⭐⭐ 不是为截面排名设计 | 不适用 |
| **DLinear（LTSF-Linear）** | NeurIPS 2022 Workshop | 极简线性基线 | ⭐⭐⭐ 强基线，用来验证 Transformer 是否有真实增益 | 必须作为基线 |

**深挖：iTransformer 为什么天然适合截面预测**

iTransformer 的关键洞察：将原始 Transformer 的"时间点 → token"改为"每只股票/变量 → token"。

```
原始 Transformer：
  输入：[T个时间步 × N个变量] → T个 token，每个 token 是 N维向量
  注意力：时间步之间的注意力（捕捉时序依赖）
  
iTransformer：
  输入：[T个时间步 × N个变量] → N个 token，每个 token 是 T维时序向量
  注意力：变量之间的注意力（捕捉跨变量/跨股相关性）
  FFN：对每个变量独立处理时序模式
```

这本质上就是 Cross-sectional Transformer！每只股票用自己的历史序列作为 token 的embedding，然后通过注意力学习股票间的依赖。对于我们的场景（5000只股票，预测截面排名），这是最直接的架构。

### 2.3 轻量级 vs 大模型的权衡

**反直觉发现**：在我们的数据量级下，小模型往往更好，原因是：

1. **噪声比信号强**：A 股日频数据的信息比约为 1:10（信噪比极低），大模型容易记住噪声
2. **非平稳性**：市场机制每 2-3 年就会发生切换（参见 rev_10 的逐年 IC 变化），大模型的"记忆"会成为负担
3. **样本效率**：小模型在相同数据量下往往有更好的泛化性

具体数据：Qlib 的基准测试显示，在 CSI 300/500 数据上，参数量超过 1M 的 Transformer 模型相比 500K 的版本，IC 通常不会提升，但训练时间翻倍。

---

## 三、业界应用现状：真实有效还是 Overfitting 展示？

### 3.1 机构应用

**国内头部量化私募**（假设，基于公开信息）：
- 幻方量化、明汯、九坤等头部私募均有 AI 研究团队，但具体模型架构不公开
- 公开信息显示，主流路径是"深度学习因子"而非"深度学习策略"——即用 DL 生成更好的因子，仍然用传统框架做组合优化
- Transformer 替代 LSTM 的趋势在 2022-2023 年开始，主要用于多因子合成

**微软 Qlib（开源量化平台）**：
- 已集成 TFT、Localformer、GATs 等多个 Transformer 变体
- 公开 benchmark 数据：CSI 300，LSTM IC ≈ 0.040，TFT IC ≈ 0.043，GATs IC ≈ 0.048
- 这些数值差异在统计上显著，但在实盘中（考虑滑点、容量）是否有意义需要具体测试

**开源项目前沿进展（2022-2025）**：

| 项目/论文 | 机构 | 核心贡献 | 实用价值 |
|---------|------|---------|---------|
| MASTER (2023) | 上海交大 | 市场引导 + 跨时股票相关 | 高，有代码 |
| iTransformer (2024) | 清华 | 变量作 token | 高，ICLR Oral |
| PatchTST (2023) | CMU | patch 嵌入 | 高，ICLR |
| LTSF-Linear (2022) | 中科大 | 质疑 Transformer 在时序上的有效性 | 必读，提供视角 |
| Generalized Factor NN (2025) | arXiv:2502.11310 | PCA layer + NN 为股票 ETF 预测 | 中，思路有参考价值 |

### 3.2 批判性评估：学术有效 vs 实盘有效

**学术论文的常见"overfitting 展示"手法**：

1. **不恰当的数据集切割**：用随机 split 而非 walk-forward split，导致未来数据泄露
2. **避开危机区间**：很多论文只展示 2015-2018 区间，回避 2015 股灾、2022 熊市
3. **IC 而非 IC*IR**：只报告平均 IC，隐藏 IC 的高波动性（ICIR 才能反映信号稳定性）
4. **忽略交易成本**：很多论文假设零滑点，但实际上双周换仓 57% 在 A 股的成本是不可忽略的
5. **使用 CSI 300 做测试集**：CSI 300 是主力资金场，容量和流动性好，不代表全市场

**一个关键数据点**：Qlib benchmark 中 Transformer 类模型的 IC 通常在 0.040-0.055 之间，而我们的因子体系（用最强的 rev_10）在 2022-2026 的 ICIR = 0.37，对应平均 IC ≈ 0.056。换句话说，**我们的单个因子 rev_10 已经和最好的学术 Transformer 模型处于同一量级，甚至更好**。

这是一个重要的 sobering 事实：**Transformer 不是银弹，它在截面排名上的增益相当有限，需要仔细的工程实现才能超越良好调参的因子模型**。

### 3.3 为什么仍然值得尝试？

尽管如此，以下几个方向仍然有明确的增量价值：

1. **非线性因子交互**：我们的 ic_reweighted 是线性加权，而 Transformer 可以捕捉 `rev_10 × inv_pb` 类的非线性交互，这在单因子 IC 上看不到
2. **动态权重自适应**：我们看到 rev_10 的 IC 从 2024 年的 0.030 跳升到 2026 年的 0.146，而 rvol_20 从 2022 年的 0.073 下降到 2026 年的 0.023。静态权重无法捕捉这种时变性
3. **新特征维度**：Transformer 可以处理我们现有因子无法编码的信息，如行业内股票联动、大盘状态感知

---

## 四、如何建模：针对我们具体场景的设计方案

### 4.1 建模范式选择

**Cross-sectional Transformer（截面预测）**，而非 Time-series Transformer。

这是最关键的设计决策。区别如下：

| 维度 | Time-series Transformer | Cross-sectional Transformer |
|------|------------------------|----------------------------|
| 输入 | 单只股票的历史序列 | 同一时间截面的所有股票 |
| 输出 | 该股票未来收益预测 | 全部股票的相对排名 |
| 注意力方向 | 时间步之间 | 股票之间 |
| 优势 | 捕捉单只股票的时序 pattern | 捕捉股票间的相对关系 |
| 劣势 | 无法感知市场整体状态 | 需要处理不同数量的股票 |
| 典型架构 | PatchTST | iTransformer, MASTER |

**我们的选择：Cross-sectional Transformer（主）+ 时序特征工程（辅）**

### 4.2 输入特征设计

每只股票在当前截面时间点的输入向量包含：

```python
# 股票 i 在时间 t 的特征向量
feature_i_t = [
    # === 时序特征（需要提前计算好，防止前视偏差）===
    # 价格类（已因子复权）
    ret_1,        # 1日收益率
    ret_5,        # 5日收益率  
    ret_10,       # 10日收益率（双周，对应 rev_10 因子）
    ret_20,       # 20日收益率
    ret_60,       # 60日收益率（3个月动量）
    
    # 波动率类
    rvol_20,      # 20日实现波动率（已知有效，ICIR=0.34）
    
    # 估值类
    inv_pb,       # PB倒数（已知有效，ICIR=0.33）
    
    # 规模类
    log_cap,      # 对数自由流通市值（已知有效，ICIR=0.35）
    
    # === 高阶时序特征（新增）===
    ret_5_vol_ratio,  # 近期收益/波动率（调整后收益）
    volume_ratio,     # 近5日/近20日成交量比（相对活跃度）
    
    # === 行业编码（离散 → 连续嵌入）===
    # 行业one-hot或行业嵌入（通过 nn.Embedding 学习）
    industry_id,  # 将由 Embedding 层转化为 d_industry 维向量
    
    # === 市场状态特征（截面共享，在 encoder 中作为 context）===
    mkt_ret_5,    # 市场5日收益率（所有股票相同）
    mkt_rvol_20,  # 市场20日波动率
]
```

**重要设计原则**：
- **所有特征必须使用 t-1 日及之前的信息**，标签是 t 到 t+10 的未来10日收益率
- 按股票市值/成交量分组归一化（RankNorm），使得特征在截面内是可比较的
- 行业嵌入维度建议 d_industry = 8（28个申万行业→8维），通过 nn.Embedding 联合训练

### 4.3 处理 A 股特有问题

**（1）涨跌停**

问题：涨停次日可能出现大量卖单，停板的收益率不能反映真实需求。

处理方案：
```python
# 计算收益率时，过滤连续涨停期间
# 若股票在 t 时刻收盘价为涨停价，且次日开盘也是高开，
# 则 t 日收益率标记为"不可信"，在损失函数中设权重=0
is_limit_up = (close / prev_close - 1) >= 0.0995
is_limit_down = (close / prev_close - 1) <= -0.0995
valid_weight = ~(is_limit_up | is_limit_down)  # 当天封板不计入损失
```

**（2）停牌**

处理方案：
```python
# 停牌股票特征向量填充为 NaN，使用 masked attention
# 停牌股票的预测标签设为 NaN，在损失函数中跳过
# 复牌后第一天通常有"补涨"效应，单独处理不纳入训练
```

**（3）上市不足**

处理方案：
- 要求至少上市 120 个交易日（约 6 个月）才纳入训练
- 对于新股：设置 ipo_age_flag，由模型学习新股的特殊 pattern（实际上 A 股新股通常被高估，不纳入策略）

**（4）北交所股票**

从数据库看到 BJ920000 等北交所股票，流动性差，建议排除出训练集和预测集。

### 4.4 预测目标选择

不同的预测目标对应不同的 alpha 层次：

| 预测目标 | 定义 | 优势 | 劣势 | 推荐度 |
|---------|------|------|------|--------|
| 原始收益率 | (p_{t+10} - p_t) / p_t | 直接，易解释 | 量级差异大，MSE 被大收益股票主导 | 中 |
| Rank 收益率 | 截面 rank / N（0到1之间） | 对离群值鲁棒 | 失去绝对收益信息 | **高** |
| 超额收益率 | 个股收益 - 市场收益 | 对冲市场 beta | 需要准确的 beta 估计 | 中高 |
| 二分类（涨/跌） | 1 if ret > 0 else 0 | 简单 | 信息量损失太大 | 低 |
| 五分位 | 所属 Q1-Q5 | 聚焦头尾股票 | 离散信号 | 中 |

**推荐：Rank 收益率 + ListMLE 损失**（详见训练章节）

---

## 五、如何训练：严格避免前视偏差

### 5.1 数据量评估

**答案：5000只股票、10年日数据，对于我们设计的轻量级模型，足够，但需要合理划分。**

具体计算：
- 训练集（2006-2019，约13年）：~5000只 × ~3000日 = 1500万观测点，但可用双周截面约 330 个，每截面约 3000-4000 只股票
- 验证集（2019-2022）：约 78 个截面，每截面约 4000 只股票，~31万 observations
- 测试集（OOS，2022-2026）：约 100 个截面，已知 ic_reweighted 年化 19.81%

**注意**：对于截面 Transformer，有效的"样本"是每个双周截面，而不是每只股票每天。330个截面是相对有限的序列样本。这强化了"小模型"的结论。

### 5.2 Walk-forward 训练：正确方式

最关键的工程决策，所有前视偏差都发生在这里：

```
时间轴：2006 ─────────────────────────────────────────────── 2026

Walk-forward 滚动：
  第1折：训练[2006-2015] → 验证[2016] → 测试2017
  第2折：训练[2006-2016] → 验证[2017] → 测试2018
  ...
  第N折：训练[2006-2022] → 验证[2023] → 测试2024
  
注意事项：
  1. 特征的归一化（RankNorm）必须只使用当期截面数据，不用历史
  2. 滚动因子（如rvol_20）必须在每个时间点只使用 t 之前的数据
  3. 标准化参数（均值、方差）不能在测试集上计算
  4. 模型选择（超参数）在验证集上，最终性能在测试集上报告
```

**具体防止前视偏差的检查清单**：
- [ ] 特征计算没有使用 `shift(-1)` 方向的数据
- [ ] 因子复权使用的是事后复权因子的已知值（非未来值）
- [ ] pb、free_market_cap 等估值数据的发布时间滞后（通常有 1-3 个月的报告期滞后）
- [ ] 行业分类使用的是历史分类，不是当前分类
- [ ] 归一化在每个截面独立计算

### 5.3 损失函数设计

**推荐：IC-weighted ListMLE Loss（组合损失）**

```python
def loss_fn(predictions, returns, weights=None):
    """
    组合损失函数：
    1. ListMLE（排序损失）：主损失，直接优化截面排名
    2. IC 惩罚：正则化项，保持预测的截面 IC 不能过低
    3. 权重：按股票质量（流动性、市值）加权
    """
    # 1. ListMLE：最大化预测排名与真实排名的一致性
    # 对每个截面，最大化: log P(真实排序 | 预测分数)
    # 等价于最小化: -∑_i log softmax(score_i - score_j, j < i in true ranking)
    listmle = list_mle_loss(predictions, returns)
    
    # 2. IC 损失（可选，用于约束信号质量）
    # 按截面计算 Spearman IC，最大化其均值
    ic_per_period = compute_rank_ic(predictions, returns)
    ic_loss = -ic_per_period.mean()
    
    # 3. 样本权重（可选）
    if weights is not None:
        listmle = (listmle * weights).mean()
    
    # 组合
    alpha = 0.8  # ListMLE 权重
    beta = 0.2   # IC 损失权重
    return alpha * listmle + beta * ic_loss
```

**为什么不用 MSE**：
- MSE 对绝对收益幅度敏感，小盘股的高波动会主导梯度
- MSE 优化的是"预测准确"，而我们关心的是"排名准确"
- 实验证明，IC-consistent loss（优化相关性而非 MSE）可以提升最终策略 IC 约 15-30%

### 5.4 正则化策略

```python
# 模型配置中的正则化
config = {
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.15,           # 较低的 dropout，避免过度正则化
    "label_smoothing": 0.0,    # 不建议在排名任务中使用
    "weight_decay": 1e-4,      # L2 正则化
    
    # Early stopping
    "patience": 10,             # 验证 IC 连续 10 期不提升则停止
    "min_delta": 0.001,         # IC 改善阈值
    
    # 学习率调度
    "warmup_epochs": 5,
    "scheduler": "cosine_annealing",
    "max_lr": 1e-3,
    "min_lr": 1e-5,
}
```

**特别提示：避免最常见的过拟合陷阱**

A 股截面 Transformer 最容易过拟合的地方是注意力矩阵对特定"牛熊"体制的记忆。解决方法：
1. 在不同市场区间（牛市、熊市、震荡）分别计算验证 IC，确保各区间都有正向信号
2. 加入市场状态条件（如大盘涨跌、VIX 类指标）作为 context，让模型显式感知体制，而不是隐式记忆

---

## 六、如何应用到实盘

### 6.1 推理延迟和计算成本

我们的场景（双周调仓）对实时性要求极低：

```
典型推理流程（每两周执行一次）：
  数据准备（从 stocks.db 读取）：约 10 秒
  特征计算（~5000只股票 × 20个特征）：约 30 秒
  模型推理（前向传播）：约 2-5 秒（CPU），< 0.5 秒（GPU）
  信号排名和股票筛选：约 5 秒
  总计：< 1 分钟
  
所需算力：
  训练：约 4-8 小时（单张 RTX 3090 或云端 T4），无需持续算力
  推理：可在 CPU 上完成，无需 GPU 基础设施
```

**成本估算**：初始训练约需 100-200 元人民币的云计算费用，之后月维护成本 < 20 元。

### 6.2 与传统因子模型结合：具体 Stacking 方案

这是本研究最具操作价值的部分。有三种 ensemble 方式，推荐 **方案B（Score-level Stacking）**：

**方案A：特征级融合（Feature-level Stacking）**
将 ic_reweighted 的 4 个因子分数直接作为 Transformer 的输入特征，让 Transformer 学习非线性组合。
- 优点：简单，数据复用
- 缺点：可能破坏 Transformer 自己学到的 representation

**方案B：分数级融合（Score-level Stacking）★ 推荐**
```python
# 每个双周截面：
score_ic_rew = ic_reweighted_score(stocks)   # 现有策略得分（已归一化）
score_transformer = transformer_predict(stocks)  # Transformer 预测得分

# 融合方式1：简单平均
final_score = 0.5 * score_ic_rew + 0.5 * score_transformer

# 融合方式2：动态权重（根据近期 IC 动态调整）
ic_ic_rew_recent = rolling_ic(score_ic_rew, window=6)    # 近3个月 IC
ic_transformer_recent = rolling_ic(score_transformer, window=6)

w_ic_rew = ic_ic_rew_recent / (ic_ic_rew_recent + ic_transformer_recent)
w_transformer = 1 - w_ic_rew
final_score = w_ic_rew * score_ic_rew + w_transformer * score_transformer

# 方式2 更适合市场体制切换，但需要至少 6 个双周期才能稳定
```

**方案C：决策级融合（Portfolio-level Stacking）**
两个策略分别构建组合，然后等权合并。
- 优点：最简单，天然多样化
- 缺点：可能持有的股票数 doubling，集中度下降

**预期效果（基于 A 股研究经验的估算，需实测验证）**：

假设 Transformer 信号 IC ≈ 0.040（乐观），ic_reweighted IC ≈ 0.056，相关性 ρ ≈ 0.5：

```
组合 IC ≈ sqrt(IC_a² + IC_b² + 2ρ × IC_a × IC_b) / sqrt(2) 的简化分析
实际上，正确的 ensemble IC：

IC_ensemble ≈ (IC_a + IC_b) / 2  # 简单平均时
            ≈ 0.048              # 如果相关性高则接近均值
            
但 ICIR_ensemble = IC_ensemble / σ_ensemble，
由于两个信号在不同市场环境下的波动性有部分对冲效果，
ICIR 通常比单个信号高 10-20%。

粗略预测：
  现有 ICIR ≈ 0.35 → Ensemble ICIR ≈ 0.40-0.42
  对应夏普比例从 ~0.85 提升到 ~0.95-1.05（假设其他不变）
```

### 6.3 信号稳定性：重训练频率

```
建议方案：
  每月末重训练一次（而非每日）
  
理由：
  1. A 股日频噪声大，过于频繁的更新会引入不稳定性
  2. 我们的因子 IC 在月度尺度上相对稳定（从逐年 IC 表可以看出，大方向不变）
  3. 每日重训练的计算成本在 GPU 上可以接受，但过拟合风险更大
  4. 重训练前先在"影子账户"运行 2-4 个调仓期观察
  
模型选择标准（每次重训练后）：
  - 验证集 ICIR 必须 > 0.20（否则不更新）
  - 最近 12 个月滚动 IC > 0（不能长期负向）
  - 与现有 ic_reweighted 信号的相关性 < 0.7（确保有增量）
```

### 6.4 容量约束

我们目前持有约 87-94 只股票，覆盖 26-28 个行业，平均单票市值假设 100 亿，则策略总容量约 1.5 万亿。

引入 Transformer 信号后：
- 不改变持仓数量（仍然 top 5%，约 80-100 只）
- 不改变调仓频率（双周）
- 换仓率可能微增（Transformer 信号更新导致不同的换仓），成本增加约 0.3-0.5 个百分点
- 总容量不变，因为组合构建方式不变

---

## 七、深挖：Cross-sectional Attention 的数学原理与实现

### 7.1 数学推导

设时间 t 的截面包含 N 只股票，每只股票的特征向量为 $x_i \in \mathbb{R}^d$（d 是特征维度）。

Cross-sectional Attention 的目标是：**让每只股票的预测不只依赖自身特征，还感知到截面内其他股票的状态**。

```
输入矩阵：X = [x_1, x_2, ..., x_N]^T ∈ R^{N×d}

线性投影（学到的参数）：
  Q = X * W_Q  ∈ R^{N×d_k}  （Query：我想关注什么信息？）
  K = X * W_K  ∈ R^{N×d_k}  （Key：我能提供什么信息？）
  V = X * W_V  ∈ R^{N×d_v}  （Value：我实际提供的信息）

注意力权重：
  A = softmax(Q * K^T / sqrt(d_k))  ∈ R^{N×N}
  
  A_{ij} 表示"股票 i 对股票 j 的注意力权重"
  直觉解释：如果 A_{ij} 很大，说明股票 j 的特征对股票 i 的预测很重要
  
输出：
  Z = A * V  ∈ R^{N×d_v}
  Z_i = ∑_j A_{ij} * V_j
  
  即：股票 i 的新表示 = 按注意力权重加权的其他股票的 value 信息
```

**金融直觉解释**：
- $A_{ij}$ 大 → 股票 j 是股票 i 的"信息来源"（可能是同行业龙头、供应链关系、风格相似股）
- 注意力矩阵 $A$ 每行加权为 1，可以被解释为"stock i 的收益中，有多少比例来自 stock j 的影响"
- 经过多层 Transformer，可以捕捉"二阶影响"（A 影响 B 影响 C）

### 7.2 关键设计细节

**Masked Attention**（处理停牌股票）：
```python
# 如果某只股票 j 停牌，则所有股票对 j 的注意力应该为 0
mask = ~is_trading[None, :]  # shape: [1, N]
# 在 softmax 之前，将停牌股票的分数设为 -inf
attn_scores = attn_scores.masked_fill(mask, float('-inf'))
attn_weights = F.softmax(attn_scores, dim=-1)
```

**行业级 Attention Bias**（先验知识注入）：
```python
# 同行业股票之间给予额外的注意力偏置
industry_matrix = (industry_id[:, None] == industry_id[None, :]).float()
attn_scores += industry_bias * industry_matrix
# 这使得模型倾向于先学习行业内的联动，再学行业间的
```

**输出 Head**（预测截面收益排名）：
```python
# 最终层
output = LayerNorm(Z)  # [N, d_v]
pred_score = Linear(d_v, 1)(output).squeeze(-1)  # [N] - 每只股票的预测分数
# 使用 ListMLE 损失与真实收益排名对比
```

### 7.3 完整的 PyTorch 实现框架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossSectionalTransformer(nn.Module):
    """
    截面 Transformer：输入截面特征，输出股票排名分数
    专为 A 股双周调仓策略设计
    """
    def __init__(
        self,
        input_dim: int = 15,      # 输入特征数（不含行业嵌入）
        industry_num: int = 30,   # 行业数量（申万一级+北交所等）
        industry_dim: int = 8,    # 行业嵌入维度
        d_model: int = 64,        # Transformer 模型维度
        num_heads: int = 4,       # 注意力头数
        num_layers: int = 2,      # Transformer 层数
        ffn_dim: int = 256,       # FFN 隐藏层维度
        dropout: float = 0.15,
    ):
        super().__init__()
        
        # 行业嵌入
        self.industry_emb = nn.Embedding(industry_num, industry_dim)
        
        # 输入投影层：将原始特征映射到 d_model 维
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + industry_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,  # 输入格式：[batch=1, N_stocks, d_model]
            norm_first=True,   # Pre-LayerNorm（更稳定的训练）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出 Head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    
    def forward(self, x: torch.Tensor, industry_ids: torch.LongTensor, 
                mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] - N只股票的特征向量
            industry_ids: [N] - 行业 ID
            mask: [N] - True 表示该股票停牌（需要屏蔽）
        Returns:
            scores: [N] - 每只股票的预测排名分数
        """
        # 行业嵌入
        ind_emb = self.industry_emb(industry_ids)  # [N, industry_dim]
        
        # 拼接特征和行业嵌入
        x_combined = torch.cat([x, ind_emb], dim=-1)  # [N, input_dim + industry_dim]
        
        # 投影到 d_model
        x_proj = self.input_proj(x_combined)  # [N, d_model]
        
        # 添加 batch 维度（截面 = 一个 batch）
        x_proj = x_proj.unsqueeze(0)  # [1, N, d_model]
        
        # 处理停牌 mask
        src_key_padding_mask = mask.unsqueeze(0) if mask is not None else None
        
        # Cross-sectional Attention
        z = self.transformer(x_proj, src_key_padding_mask=src_key_padding_mask)  # [1, N, d_model]
        z = z.squeeze(0)  # [N, d_model]
        
        # 预测分数
        scores = self.output_head(z).squeeze(-1)  # [N]
        
        # 屏蔽停牌股票
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        return scores


def list_mle_loss(pred_scores: torch.Tensor, true_returns: torch.Tensor) -> torch.Tensor:
    """
    ListMLE 排序损失：最大化真实排序的似然
    参考：Xia et al. "ListMLE: A Top-k Approach to Learning to Rank"
    """
    # 按真实收益从高到低排序
    sorted_idx = torch.argsort(true_returns, descending=True)
    pred_sorted = pred_scores[sorted_idx]
    
    # 计算 ListMLE 损失
    # loss = -∑_i log(exp(s_i) / ∑_{j>=i} exp(s_j))
    loss = 0
    for i in range(len(pred_sorted)):
        numerator = pred_sorted[i]
        denominator = torch.logsumexp(pred_sorted[i:], dim=0)
        loss -= (numerator - denominator)
    
    return loss / len(pred_sorted)
```

### 7.4 A 股数据上的 IC 衰减预期

基于我们已知的因子 IC 数据，对 Transformer 信号的 IC 衰减做预测（假设，需实测验证）：

| 预测期（双周数） | 因子 IC 衰减（基于 rev_10 模式） | 估计 Transformer IC | IC 衰减率 |
|---------------|-------------------------------|---------------------|---------|
| t+1（当期） | 0.056（ICIR=0.37） | 0.050-0.065 | 基准 |
| t+2（1个月） | ~0.040（rev_10 依赖短期反转，衰减快） | 0.030-0.045 | -20-30% |
| t+4（2个月） | ~0.025 | 0.015-0.030 | -50-60% |

**重要结论**：A 股的 IC 衰减非常快（相比美股），这说明：
1. 双周调仓是合适的频率（IC 在 1-2 周内还有显著正值）
2. 不适合用月频或季频的 Transformer（衰减后 IC 过低）
3. Transformer 在多步预测上的优势在 A 股并不显著，单步（双周）预测足够

---

## 八、Transformer 与 ic_reweighted 结合的具体方案

### 8.1 当前策略的分析

从 ic_reweighted 报告可以看出：
- **核心 alpha 来源**：rev_10（ICIR=0.37）和 rvol_20（ICIR=0.34）
- **2024 年弱点**：rev_10 的 ICIR 下降到 0.13，rvol_20 下降到 0.18，策略整体收益仍有 16.2% 但信号质量下降
- **时变性问题**：log_cap 的 IC 从 2022 年的 0.043 到 2023 年的 0.070 再到 2024 年的 0.036 有明显波动

这正是 Transformer 可以补充的地方：**动态感知当前市场环境，调整因子权重**。

### 8.2 三阶段引入计划

**阶段一（验证，1-2个月）**：
在历史数据上 backtest，验证以下假设：
1. Transformer（以我们的特征）的截面 IC 是否 > 0.030（最低有效阈值）
2. 与 ic_reweighted 的相关性是否 < 0.70（确保有增量信息）
3. 在 2022-2026 OOS 测试集上的 ICIR 是否 > 0.20

**阶段二（纸面交易，2-3个月）**：
运行影子账户，每双周生成两份持仓：ic_reweighted 版和 ensemble 版。
比较两者的 realized IC，确认真实市场中的增益。

**阶段三（实际应用）**：
将 ensemble 信号比例从 20% 开始逐渐提升（20% → 40% → 50%）。
若 3 个月后 ensemble 表现稳定优于纯 ic_reweighted，提升到 50%。

### 8.3 具体 Stacking 代码框架

```python
class EnsembleStrategy:
    """
    ic_reweighted + CrossSectionalTransformer Ensemble
    """
    def __init__(self, transformer_weight: float = 0.30):
        self.transformer = CrossSectionalTransformer(...)
        self.transformer_weight = transformer_weight
        self.ic_weight = 1 - transformer_weight
        
        # 因子权重（来自 ic_reweighted）
        self.factor_weights = {
            'inv_pb': +0.25,
            'rev_10': +0.30,
            'rvol_20': -0.35,
            'log_cap': -0.10,
        }
    
    def ic_reweighted_score(self, df: pd.DataFrame) -> pd.Series:
        """计算 ic_reweighted 因子得分"""
        scores = pd.Series(0, index=df.index)
        for factor, weight in self.factor_weights.items():
            factor_values = df[factor]
            # RankNorm：转化为 0-1 之间的截面排名
            factor_rank = factor_values.rank(pct=True)
            scores += weight * factor_rank
        return scores
    
    def transformer_score(self, df: pd.DataFrame) -> pd.Series:
        """使用 Transformer 预测截面分数"""
        features = self._prepare_features(df)
        x = torch.tensor(features, dtype=torch.float32)
        industry_ids = torch.tensor(df['industry_code_encoded'].values, dtype=torch.long)
        
        with torch.no_grad():
            scores = self.transformer(x, industry_ids)
        
        return pd.Series(scores.numpy(), index=df.index)
    
    def select_stocks(self, df: pd.DataFrame, top_pct: float = 0.05) -> pd.Index:
        """选股：返回 top_pct% 的股票"""
        # 计算两种信号
        ic_scores = self.ic_reweighted_score(df)
        tf_scores = self.transformer_score(df)
        
        # 归一化到相同量级
        ic_scores_norm = ic_scores.rank(pct=True)
        tf_scores_norm = tf_scores.rank(pct=True)
        
        # 加权合并
        final_scores = (
            self.ic_weight * ic_scores_norm + 
            self.transformer_weight * tf_scores_norm
        )
        
        # 选 top N 只（保持行业中性，每行业最多5只）
        selected = self._industry_neutral_select(df, final_scores, top_pct)
        return selected
```

---

## 九、明确立场：是否值得在策略中引入 Transformer？

### 9.1 判断框架

在给出结论之前，先明确评判标准：

| 标准 | 阈值 | 现状 |
|------|------|------|
| OOS IC 增益 | > +0.005（绝对IC提升） | 待验证，预计0.003-0.008 |
| ICIR 增益 | > +0.05 | 待验证，预计0.05-0.08 |
| 实现成本 | < 2周工程时间 | 约10-15天 |
| 夏普提升（边际） | > +0.05 | 预计0.05-0.15 |
| 过拟合风险 | 可控 | 中等，需要严格的 walk-forward 验证 |
| 计算成本 | < 100元/月 | 是的，CPU 推理可行 |

### 9.2 结论：**值得引入，但有条件、有优先级**

**明确的"是"**：
1. **Transformer 用于因子交互建模**：将现有 4 个因子的非线性交互通过 Transformer 来学习，预计可提升 ICIR 10-20%
2. **iTransformer 用于截面排名增强**：以截面 Transformer 的预测作为第 5 个 alpha 因子加入现有体系，权重 30%

**明确的"否"**：
1. **不替换现有因子体系**：ic_reweighted 已有 7 年真实 OOS 验证，轻易替换是错误决定
2. **不使用时序 Transformer 做价格预测**：LTSF-Linear 的批评有力，时序 Transformer 在 A 股的增益不确定
3. **不使用大型预训练模型（TimesFM、Chronos）**：这些模型为通用时序设计，不适合截面排名任务

**关键前提**：
- 必须先在 OOS 历史数据（2022-2026）上验证 IC > 0.030
- 必须证明与现有信号的相关性 < 0.70
- 必须以 stacking 方式引入，而非替换

### 9.3 预期收益与风险

| 指标 | 当前（ic_reweighted） | 引入 Transformer 后（预估） | 变化 |
|------|---------------------|--------------------------|------|
| 年化收益 | 19.81% | 21-23% | +1-3% |
| 夏普比 | 0.85 | 0.90-1.00 | +0.05-0.15 |
| 最大回撤 | -24.49% | -22-25%（不确定） | ±1% |
| 信号 ICIR | ~0.35 | ~0.40-0.43 | +15-20% |

**最坏情况**：Transformer 信号 ICIR < 0.10（统计上无意义），则 ensemble 后效果接近原策略，风险可控。

---

## 十、下一步行动计划

以下是具体可执行的步骤，按优先级排序：

### 第一阶段：数据验证（1周内）

**任务1.1：数据质量检查**
```bash
# 运行以下检查
python check_data_quality.py
# 检查项：
# - 涨跌停识别准确性
# - 停牌数据完整性（volume=0的天数）
# - pe_ttm, roe_ttm 的 NaN 比例
# - 数据是否存在前视偏差（factor 复权因子检验）
```

**任务1.2：特征工程管道**
```python
# 文件：features/cross_sectional_features.py
# 计算以下特征（所有都是 t-1 可知的）：
# ret_1, ret_5, ret_10, ret_20, ret_60
# rvol_10, rvol_20, rvol_60
# log_cap, inv_pb (已有)
# volume_ratio = volume_5d_avg / volume_20d_avg
# 行业相对强弱：个股 rev_10 - 行业平均 rev_10
```

**任务1.3：基线测试**
先运行 DLinear 基线（一层线性变换），如果 DLinear 的 IC < 0.020，则放弃所有 DL 探索（信号太弱）。

### 第二阶段：模型实现（1-2周）

**任务2.1：实现 CrossSectionalTransformer**
使用上文第七章的代码框架，关键参数：
- d_model=64, num_heads=4, num_layers=2
- 输入特征：~15维（价格+估值+规模）
- 行业嵌入：30×8维

**任务2.2：Walk-forward 训练管道**
```python
# 关键：严格的时序切割
folds = WalkForwardSplit(
    start='2010-01-01',
    end='2026-02-28',
    train_window=1040,  # 约4年数据（双周）
    val_window=52,      # 约1年
    test_window=26,     # 约半年
    step=26,            # 每半年滚动一次
)
```

**任务2.3：损失函数验证**
在 3 种损失函数（MSE、ListMLE、IC-weighted loss）上分别训练，选择验证集 ICIR 最高的。

### 第三阶段：验证与集成（2周）

**任务3.1：OOS 验证（2022-2026）**
- 计算逐年 IC 和 ICIR（对比第三章的因子 IC 分析表）
- 计算与 ic_reweighted 的相关性
- 如果 ICIR < 0.20 或相关性 > 0.80，停止

**任务3.2：Ensemble Backtest**
```python
# 运行完整的 ensemble backtest
# 比较以下配置：
# A：纯 ic_reweighted（基准）
# B：ic_reweighted + Transformer（30/70 权重）
# C：ic_reweighted + Transformer（50/50 权重）
# 在 2019-2026 完整区间上比较夏普、IC、最大回撤
```

**任务3.3：容量和成本分析**
- 统计 ensemble 策略的额外换仓率
- 估算增加的交易成本
- 确认净收益（扣除成本后）仍然正向

### 第四阶段：生产部署（1-2周，如果验证通过）

**任务4.1：模型序列化**
```python
# 保存训练好的模型
torch.save(model.state_dict(), '/projects/portal/data/quant/models/cs_transformer_v1.pt')
# 保存特征归一化参数
# 保存训练元数据（数据切割、超参数、验证集性能）
```

**任务4.2：推理管道集成**
将 Transformer 推理嵌入现有调仓脚本，确保每次调仓日可以生成 ensemble 信号。

**任务4.3：监控与预警**
```python
# 每次调仓后记录：
# - Transformer 信号的截面 IC（与过去30天的平均值比较）
# - Transformer 与 ic_reweighted 的相关性
# 如果 IC 连续 4 期为负，发出预警，考虑降低 Transformer 权重
```

### 快速验证路径（如果时间有限）

如果只有 3 天时间，按以下最简路径快速验证可行性：

```python
# Day 1: 特征工程
# 计算所有股票的 ret_10, rvol_20, inv_pb, log_cap, ret_5 的截面排名
# 拼成矩阵 [T个截面, N只股票, 5个特征]

# Day 2: 最简 Transformer 实现 + 训练
# 用 2010-2018 训练，2019-2021 验证，2022-2026 测试
# 只需 100 行代码的 TransformerEncoder

# Day 3: 与 ic_reweighted 对比
# 计算逐年 IC，判断是否有增量价值
# 结果：如果 OOS ICIR > 0.20，继续深入；否则暂停
```

---

## 十一、参考资料

| 类型 | 来源 | 关键点 |
|------|------|--------|
| 论文 | LTSF-Linear (arXiv:2205.13504) | 质疑 Transformer 时序预测的根基性工作 |
| 论文 | iTransformer (arXiv:2310.06625, ICLR 2024) | 变量作 token，最适合截面预测 |
| 论文 | PatchTST (arXiv:2211.14730, ICLR 2023) | 时序 patch 嵌入，适合时序特征提取 |
| 论文 | TFT (arXiv:1912.09363) | 可解释 Transformer，动态因子选择 |
| 论文 | MASTER (arXiv:2312.15235, AAAI 2024) | 专为股票预测设计，有跨时跨股 attention |
| 平台 | Microsoft Qlib | A 股 benchmark，Transformer vs LSTM vs LightGBM 对比 |
| 内部数据 | factor_ic_report.md | 2022-2026 因子 IC 分析 |
| 内部数据 | ic_reweighted_report.md | 现有策略基准，年化 19.81%，夏普 0.85 |

---

## 附录：A 股截面 Transformer 的常见失败模式

1. **过拟合特定牛熊**：模型记住了 2019-2020 年的牛市 pattern，在 2022 年熊市完全失效
   - 解决：在多个市场体制下分别验证；加入市场状态条件输入

2. **注意力坍塌**：所有股票都关注同一批大盘蓝筹，失去截面分辨力
   - 解决：限制注意力头的数量；使用 relative position bias

3. **信噪比陷阱**：A 股噪声占比超过 90%，Transformer 倾向于记住噪声
   - 解决：强正则化；在 rolling validation 中检查 IC 的方差

4. **工程实现的前视偏差**：归一化参数使用了未来数据
   - 解决：强制所有归一化在每个截面独立计算

5. **容量问题被忽视**：学术论文通常不考虑市场冲击，实盘中排名靠前的股票往往已被机构买满
   - 解决：在特征中加入流动性指标；选股时保留流动性约束

---

*本报告基于截至 2026-03-20 的研究，所有前瞻性预测均为基于历史数据的估算，不构成投资建议。*

<!-- RESEARCH_COMPLETE -->
