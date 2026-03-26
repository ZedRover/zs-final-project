# 基于端到端损失函数设计的高频多信号组合优化

---

## Page 1: 研究动机与问题提出

### 从预测到决策：量化交易中的目标错配问题

在量化投资领域，alpha 信号的构建与交易执行之间存在一个被广泛忽视的结构性矛盾。现有的多信号组合方法（等权、IC 加权、岭回归等）普遍采用 **"先预测，再决策"** 的两阶段范式：首先训练模型使预测值 $\hat{y}$ 尽可能逼近真实收益 $y$，然后将 $\hat{y}$ 作为输入交给下游优化器做交易决策。

这一范式隐含了一个关键假设：**预测误差的最小化等价于交易利润的最大化**。然而在真实高频交易场景下，这一假设并不成立：

- **非对称交易成本**（买入 1.5bp vs 卖出 6.5bp）使得买卖决策的阈值不对称
- **流动性约束**（每只股票每时段可交易量有限）使得 "最优预测" 未必可执行
- **库存平衡约束**（净交易量为零）引入跨资产耦合，单只股票的预测精度失去独立意义
- **LP 优化器的非线性映射**使得 $\hat{y}$ 的微小偏差可能导致完全不同的交易方案

> **核心问题**：在存在交易摩擦与约束的条件下，如何设计损失函数使信号组合权重的学习直接服务于下游交易决策的质量，而非收益预测的精度？

---

## Page 2: 问题形式化

### Predict-then-Optimize 框架

本研究将高频信号组合建模为一个 **两阶段优化问题**：

**第一阶段：信号组合（参数化、可微）**

$$\hat{y} = \mathcal{T}(\mathbf{w}) \cdot \mathbf{x}, \quad \mathbf{x} \in \mathbb{R}^{S \times N}, \quad \mathbf{w} \in \mathbb{R}^{S}$$

其中 $S=30$ 路原始信号经截面标准化后作为输入，$\mathcal{T}(\cdot)$ 为可选的权重变换（恒等、softplus、softmax 等），组合器刻意保持线性以确保可解释性与稳定性。

**第二阶段：LP 交易决策（组合优化、不可微）**

$$\max_{\Delta w} \sum_{i=1}^{N} \left[(\hat{y}_i - c_b) \cdot b_i - (\hat{y}_i + c_s) \cdot s_i\right]$$

$$\text{s.t.} \quad 0 \le b_i \le m_b^i, \quad 0 \le s_i \le m_s^i, \quad \sum_i (b_i - s_i) = 0$$

LP 的解 $\Delta w^*(\hat{y})$ 及其对偶变量 $\lambda$（库存约束的 Lagrange 乘子）蕴含了丰富的决策结构信息。

### 核心困难

LP 求解器作为分段线性函数，其输出关于 $\hat{y}$ 几乎处处可微但在 breakpoint 处不可微，且 Jacobian 在大部分区域为零（解的活跃集不变时）。这使得**梯度无法直接穿透 LP 反向传播到组合权重**，损失函数的设计因此成为连接 "可微预测" 与 "不可微决策" 的关键桥梁。

### Oracle 解的利用

训练数据中提供了由上游规划系统求解的 **oracle 最优持仓轨迹** $h^*$，其差分 $\Delta h^{\text{opt}} = h^*_{t+1} - h^*_t$ 给出了每个时段每只股票的最优交易量。这一 oracle 信号在后续的损失函数设计中被反复利用——既作为监督标签，也作为决策质量的参考基准。

---

## Page 3: 预测导向的损失函数

### 基线方法：直接回归与排序优化

**MSE Loss — 点估计回归**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{BN}\sum_{t,i}(\hat{y}_{t,i} - y_{t,i})^2$$

MSE 将信号组合视为标准回归问题，最小化预测误差的二范数。它提供了最简单的基线，但完全不考虑下游 LP 的存在——它隐含地假设预测误差在所有资产、所有方向上同等重要，而事实上流动性高的资产对交易利润的贡献远大于流动性低的资产。

**Weighted MSE — 流动性感知的回归**

$$\mathcal{L}_{\text{WMSE}} = \frac{1}{BN}\sum_{t,i} w_{t,i} \cdot (\hat{y}_{t,i} - y_{t,i})^2, \quad w_{t,i} = \frac{m_b^{t,i} + m_s^{t,i}}{\overline{m_b + m_s}}$$

通过流动性加权，使模型更关注 "可交易" 资产的预测精度。这是从预测导向到决策导向的第一步探索：开始利用 LP 的约束信息，但仅限于调节样本权重。

**IC Loss — 排序一致性**

$$\mathcal{L}_{\text{IC}} = 1 - \frac{\sum_i(\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}{\sqrt{\sum_i(\hat{y}_i - \bar{\hat{y}})^2 \cdot \sum_i(y_i - \bar{y})^2}}$$

IC Loss 放弃绝对值预测，转而优化截面排序一致性。这一选择有其合理性：LP 求解器的买卖决策主要依赖资产间的相对排序，而非 $\hat{y}$ 的绝对量级。但 IC 仍未建模交易成本和约束，它本质上假设 "排序正确即可获利"，忽略了成本摩擦导致的决策阈值效应。

### 预测导向方法的局限

上述三种 Loss 的共同局限在于：**优化目标与决策目标之间存在不可忽略的 gap**。由于 LP 的非线性映射，$\hat{y}$ 的预测质量与 $\Delta w^*(\hat{y})$ 的决策质量之间并非单调关系。例如，一个 IC 略低但决策边界附近预测更准确的 $\hat{y}$，可能产生比高 IC 信号更好的交易结果。

---

## Page 4: 决策导向的损失函数

### 利用 LP 对偶结构的替代损失

**Dual Hinge Loss — 决策边界分类**

LP 的 KKT 条件揭示了一个简洁的决策结构：对于资产 $i$，买入当且仅当 $\hat{y}_i - c_b + \lambda > 0$，卖出当且仅当 $-\hat{y}_i - c_s - \lambda > 0$，其中 $\lambda$ 是库存约束的对偶变量。Dual Hinge Loss 将这一结构显式编码为两个 SVM 形式的分类任务：

$$\mathcal{L}_{\text{DH}} = \frac{1}{BN}\sum_{t,i}\left[\text{relu}(m - r_b \cdot s_b) + \text{relu}(m - r_s \cdot s_s)\right]$$

$$s_b = \hat{y} - c_b + \lambda, \quad s_s = -\hat{y} - c_s - \lambda, \quad r \in \{+1, -1\}$$

其中标签 $r$ 来自 oracle 解的差分 $\Delta h^{\text{opt}}$：超过阈值的正交易量标记为 "应买入"，负交易量标记为 "应卖出"。Margin 参数 $m$ 类似 SVM 的间隔，迫使模型在决策边界附近保留安全余量。

**Weighted Dual Hinge — 引入 oracle 强度与流动性**

基础 Dual Hinge 将标签二值化，丢失了 oracle 交易量的幅度信息。Weighted Dual Hinge 通过两个维度的加权来弥补：

| 维度 | 机制 | 直觉 |
|------|------|------|
| 信号强度（soft_label） | $p_b = \text{relu}(\Delta h^{\text{opt}}) / m_b$ 作为软标签概率 | oracle 交易幅度越大 → 越确信应行动 → 标签信号越强 |
| 流动性（max_buy） | $w = m_b / \overline{m_b}$ 作为样本权重 | 可交易量越大 → 对利润贡献越大 → 应分配更多学习注意力 |

两者结合（soft_mb 模式）使损失函数同时感知 "oracle 有多确信" 和 "市场允许交易多少"。

### 端到端穿透 LP 的优化

**SPO+ Loss (Elmachtoub & Grigas, 2022)**

SPO+ 从理论上解决了 LP 不可微的问题。其核心思想是构造一个关于 $\hat{y}$ 的凸代理损失，使得最小化该代理损失等价于使 LP 的决策逼近 oracle 最优：

$$\mathcal{L}_{\text{SPO+}} = \text{obj}(2\hat{y} - y,\; \Delta w_q) - 2 \cdot \text{obj}(\hat{y},\; \Delta w^*) + \text{obj}(y,\; \Delta w^*)$$

$$\nabla_{\hat{y}}\mathcal{L} = 2(\Delta w_q - \Delta w^*)$$

其中 $\Delta w_q = \text{LP}(2\hat{y} - y)$ 是扰动预测下的 LP 解，$\Delta w^* = \text{LP}(y)$ 是 oracle 解。梯度的含义直观且优美：**将预测推向使其诱导的 LP 决策更接近最优决策的方向**。

实现上，SPO+ 通过自定义 `autograd.Function` 显式定义次梯度，绕过 LP 求解器内部的不可微运算。每次前向传播需求解两次 LP，计算成本高于 Hinge 系列，但获得了理论上更优的决策对齐保证。

---

## Page 5: 多目标损失与正则化

### DHIC Loss — 决策、排序与稀疏性的统一

前述损失函数各有侧重：Hinge 优化决策正确性，IC 优化排序质量，SPO+ 优化端到端利润。一个自然的问题是：**能否将多个目标组合，使模型同时学到好的排序和好的决策？**

DHIC Loss 给出了肯定的回答：

$$\mathcal{L}_{\text{DHIC}} = \underbrace{\alpha \cdot \mathcal{L}_{\text{Hinge}}}_{\text{决策正确性}} + \underbrace{\beta \cdot \mathcal{L}_{\text{IC}}}_{\text{排序能力}} + \underbrace{\gamma \cdot \|\mathbf{w}\|_1}_{\text{权重稀疏性}}$$

- **Hinge 项**（$\alpha=1.0$）：确保买卖决策与 oracle 一致
- **IC 项**（$\beta=0.1$）：保持截面排序能力，防止 Hinge 的二值标签损失排序信息
- **L1 项**（$\gamma=10^{-4}$）：促进信号筛选，抑制噪声信号的权重

IC 项支持对不同 horizon 的收益计算相关性，并通过 valid mask 处理缺失数据。这使得 DHIC 可以同时优化短期决策边界和长期预测排序。

### 损失函数设计的统一视角

六种损失函数可以从 "对下游 LP 的感知程度" 这一维度上排列为一个连续谱：

$$\underset{\text{无感知}}{\text{MSE}} \longrightarrow \underset{\text{加权感知}}{\text{WMSE}} \longrightarrow \underset{\text{排序}}{\text{IC}} \longrightarrow \underset{\text{对偶结构}}{\text{DualHinge}} \longrightarrow \underset{\text{多目标}}{\text{DHIC}} \longrightarrow \underset{\text{端到端}}{\text{SPO+}}$$

从左到右，损失函数编码了越来越多的 LP 结构信息：从完全忽略 LP，到利用流动性权重、决策边界的 KKT 条件，直至完整穿透 LP 求解。

---

## Page 6: 训练框架与实验设计

### 数据结构与预处理

| 维度 | 规格 | 说明 |
|------|------|------|
| 信号 | $S = 30$ | 独立构建的日内 alpha 因子 |
| 时间 | $D$ 天 $\times$ 8 时段 | 日内 9 时点、8 个交易周期 |
| 资产 | $N \sim$ 数百只 | A 股，经流动性筛选 |
| AUM | 150 亿 | 所有量纲归一化的基准 |

预处理流程严格保证信息隔离：训练/验证/测试按日期切分后，**各自独立进行截面标准化**（cross-section z-score），避免未来信息泄露。Oracle 持仓差分 $\Delta h^{\text{opt}}$ 用于生成 Hinge 系列的行动标签，标签阈值可通过绝对比例或概率归一化两种方式设定。

### Rolling 窗口训练

为模拟实际投资场景中的模型更新，采用 **滚动窗口训练**：

- 每个窗口固定长度（3 / 6 / 9 个月），按月滚动
- 窗口内进行完整的训练 → 验证 → 测试流程
- 各窗口独立训练，输出可复现的权重快照
- 支持 GNU parallel 并行多窗口加速

### 评估体系

模型评估在两个层次展开：

**训练期内**：每固定 epoch 数运行一次完整交易仿真，记录 raw return、excess return、turnover、多 horizon IC 等指标，支持基于 loss / IC / agreement 的早停策略。

**训练完成后**：将所有窗口的 test 期按时间首尾相连，进行**连续仿真**（仓位不重置），最贴近实盘交易的评估方式。同时提供 checkpoint 级别的聚合/连续仿真，用于分析不同训练阶段的模型表现。

---

## Page 7: 总结与展望

### 研究贡献

1. **系统性地构建了从预测导向到决策导向的损失函数谱系**，揭示了信号组合优化中 "预测精度" 与 "决策质量" 之间的结构性矛盾
2. **将 LP 的 KKT 对偶结构引入损失函数设计**（Dual Hinge 系列），提出了一种无需穿透求解器、仅利用对偶变量 $\lambda$ 和决策边界即可对齐预测与决策的方法
3. **在高频交易场景下实现了 SPO+ 的工程落地**，通过自定义 autograd Function 实现次梯度反向传播，将理论方法与生产级 LP 求解器对接
4. **提出了 DHIC 多目标损失**，证明决策正确性、排序能力与权重稀疏性可以通过加权组合实现互补

### 各方法适用场景

| 方法 | 计算成本 | 对 LP 的依赖 | 适用场景 |
|------|:---:|:---:|------|
| MSE / IC | 低 | 无 | 快速基线、信号筛选阶段 |
| Dual Hinge | 中 | 仅推理时 | 决策边界明确、标签质量高 |
| Weighted Dual Hinge | 中 | 仅推理时 | 资产流动性差异大、需精细化加权 |
| DHIC | 中 | 仅推理时 | 需平衡排序与决策的多目标场景 |
| SPO+ | 高（2× LP/step） | 训练时 | 追求理论最优、容忍更高计算开销 |

### 未来方向

- **非线性组合器**：在保持可解释性的前提下引入交互项或浅层非线性
- **在线学习**：从固定窗口 rolling 过渡到增量更新，降低重训练成本
- **多 horizon 联合优化**：同时优化多个前瞻窗口的决策质量
- **自适应损失权重**：DHIC 各分量的权重从手动设定过渡到自动调节
