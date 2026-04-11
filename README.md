# FCT 优化项目

## 项目目标

给定网络流量场景（平均流大小、负载度），找到使 **FCT（Flow Completion Time，流完成时间）** 最小的调度阈值 T1 和 T2。

## 核心思路

**4 种流量模型是独立问题，各自训练专属子模型，推理时通过门控路由自动选择。**

```
训练阶段: 4 个独立子模型各学习 FCT = f(T_kb, load)    ← 小型 MLP 回归
推理阶段: avg_flow_size → route_model() → 子模型 → 梯度下降找最优 T  ← 门控路由 + 可微优化
```

### 为什么要拆分

4 种流量模型（server/cache/search/mine）的 `avg_flow_size` 是固定的，问题天然地按流量类型划分。拆分为独立子模型后：
- 每个子模型只需学习 2 维输入 `[T_kb, load]` 的映射，更简单精准
- 不需要用场景特征区分流量类型，避免噪声干扰
- 各子模型独立优化，互不影响

### 为什么用梯度下降

实验数据是在**离散阈值点**上测的（比如 T1 只测了 5-7 个值），直接选 FCT 最小的那个只是**离散采样中的局部最优**，真正的最优点可能在两个采样点之间（比如 T1=68KB）。

通过学习整个 FCT 函数的连续映射，可以在连续空间中找到更接近全局最优的值，同时利用 **100% 的实验数据**。

---

## 数据说明

### 来源

4 个流量模型（server/cache/search/mine）的网络仿真实验数据，位于 `dataset/cleaned_data/`。

### 数据量

| 模型 | 条数 | 实验中变了什么 | 固定了什么 | 优化目标 |
|------|------|--------------|-----------|---------|
| server | 45 | T1 (5个值 × 9个load) | T2=1024KB | T1 |
| cache | 56 | T1 (7个值 × 9个load) | T2=1024KB | T1 |
| search | 45 | T2 (5个值 × 9个load) | T1=100KB | T2 |
| mine | 45 | T2 (5个值 × 9个load) | T1=100KB | T2 |

### CSV 字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `model` | 流量模型 | `cache` |
| `load` | 网络负载度 | `0.5` |
| `t1_kb` | T1 阈值 (KB) | `50.0` |
| `t2_kb` | T2 阈值 (KB) | `1024.0` |
| `avg_fct` | 平均 FCT (微秒) | `251974.35` |

### 训练策略

4 个流量模型各自**独立训练为专属子模型**：
- 输入仅 2 维：`[T_kb, load]`（server/cache 用 t1_kb，search/mine 用 t2_kb）
- 每个子模型专注自己的数据分布，学习更精准
- 推理时通过门控路由 `route_model(avg_flow_size)` 自动选择对应子模型

---

## 阶段一：训练 — 学习 FCT 函数

### 目标

为每种流量模型训练一个独立的小型神经网络，输入 `(T_kb, load)`，输出 `FCT`。模型学到的是 FCT 随阈值和负载变化的连续函数。

### 模型架构（每个子模型）

```
输入 (2维): [T_kb, load]
  ↓
FC(64) → ReLU → Dropout(0.2)
  ↓
FC(32) → ReLU
  ↓
FC(1) → Linear
  ↓
输出: 标准化的 log(FCT)
```

### 特征工程

| 特征 | 处理 | 原因 |
|------|------|------|
| T_kb | 原值 | 学习非线性关系 |
| load | 原值 | 范围小 (0.1-0.9) |
| FCT (目标) | log 变换 + StandardScaler | 不同 load 下 FCT 差异大，压缩范围并标准化 |

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam (lr=0.001, weight_decay=1e-5) |
| 损失函数 | MSE（在标准化 log(FCT) 空间） |
| 学习率调度 | ReduceLROnPlateau (patience=20, factor=0.5) |
| 早停 | patience=30（验证集 loss 不下降则停止） |
| 数据划分 | 70% 训练 / 15% 验证 / 15% 测试 |
| batch_size | 32 |
| 最大 epoch | 500 |
| 保存策略 | 仅保存验证集 loss 最低的模型权重 |

每个子模型独立训练完成后保存到 `models/{model_name}/`：模型权重 `predictor.pth`、输入标准化器 `scaler_X.pkl`、输出标准化器 `scaler_y.pkl`。

### 训练评估指标

| 指标 | 含义 | 公式 | 期望 |
|------|------|------|------|
| MAE | 平均绝对误差 (微秒) | mean(\|y_true - y_pred\|) | 越小越好 |
| RMSE | 均方根误差 (微秒) | sqrt(mean((y_true - y_pred)²)) | 越小越好 |
| MAPE | 平均绝对百分比误差 | mean(\|y_true - y_pred\| / y_true) × 100 | < 5% 优秀，> 10% 需改进 |

### 全数据集预测检验

训练完成后，对**全部数据**做预测，输出每个数据点的 actual_fct、predicted_fct、error_rate(%)，并生成 Markdown 训练报告 `analysis/training_report_<timestamp>.md`。

---

## 阶段二：推理 — 门控路由 + 梯度下降优化

### 核心思想

```
输入: (avg_flow_size, load)
  ↓
route_model(avg_flow_size)  →  确定子模型 (server/cache/search/mine)
  ↓
加载对应子模型  →  可微的 FCT 计算器
  ↓
梯度下降: T 当作可学习参数，反复调整使 FCT 最小
  ↓
输出: (optimal_T, predicted_FCT)
```

### 门控路由

根据 `avg_flow_size` 判断流量类型：

| 区间 | 流量模型 | 典型场景 |
|------|---------|---------|
| < 200 KB | server | 小流为主 |
| 200 KB ~ 1 MB | cache | 中流为主 |
| 1 MB ~ 3 MB | search | 大流为主 |
| > 3 MB | mine | 超大流为主 |

### 统一推理接口

```python
result, fct = find_optimal_threshold(avg_flow_size, load)
# result 包含:
#   model: 路由到的模型名
#   t1 或 t2: 最优阈值
#   固定的另一个阈值
#   predicted_fct_us: 预测 FCT
```

### 梯度下降过程（以 cache 优化 T1 为例）

```python
T = torch.tensor([130.0], requires_grad=True)  # 可学习
load = 0.5                                      # 固定

for i in range(200):
    x = [T, load]
    x_norm = (x - mean) / std       # 手动标准化保持梯度图
    y_norm = model(x_norm)            # 前向传播
    y_norm.backward()                 # 求梯度 ∂FCT/∂T
    optimizer.step()                  # 更新 T
    T.data.clamp_(10, 250)          # 约束范围
```

### 各模型优化策略

| 模型 | 优化维度 | 固定阈值 | T 范围 |
|------|---------|---------|--------|
| server | T1 | T2=1024KB | 10 - 250 KB |
| cache | T1 | T2=1024KB | 10 - 250 KB |
| search | T2 | T1=100KB | 512 - 2048 KB |
| mine | T2 | T1=100KB | 512 - 2048 KB |

### 优化超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | Adam | 自适应学习率 |
| 学习率 | 10.0 | 较大 lr 加速收敛 |
| 迭代次数 | 200 | 通常 50-100 次内收敛 |

---

## 项目结构

```
FCT_optimization/
├── dataset/cleaned_data/              # 清理后的实验数据 (4个CSV, 191条)
├── models/                            # 训练产出
│   ├── server/                        # server 子模型
│   ├── cache/                         # cache 子模型
│   ├── search/                        # search 子模型
│   └── mine/                          # mine 子模型
├── analysis/                          # 分析图表 + 训练报告
├── scripts/
│   ├── model.py                       # 共享: SubModel + route_model + 配置常量
│   ├── train_fct_predictor.py         # 训练 (4 个独立子模型) + 全数据集预测 + 报告生成
│   ├── predict_optimal_threshold.py   # 推理 (门控路由 + ThresholdOptimizer)
│   └── eda.py                        # 数据探索
├── logs/
│   ├── train/{model_name}/            # TensorBoard 训练日志
│   └── inference/                     # TensorBoard 推理日志
├── docs/                              # 技术文档
├── CLAUDE.md                          # Claude Code 指引
└── README.md                          # 本文件
```

## 快速开始

```bash
conda activate clip_tsa
cd scripts

# 训练 (4 个子模型) + 全数据集预测 + 生成报告
python train_fct_predictor.py

# 推理 (门控路由 + 梯度下降)
python predict_optimal_threshold.py

# 数据探索 (可选)
python eda.py
```

## 效果评价

训练完成后，查看 `models/{model_name}/training_results.json` 中的测试集指标：

| 指标 | 含义 | 期望 |
|------|------|------|
| MAE | FCT 平均绝对误差 (微秒) | 越小越好 |
| RMSE | FCT 均方根误差 (微秒) | 越小越好 |
| MAPE | FCT 平均绝对百分比误差 | < 5% 优秀，> 10% 需改进 |

完整预测结果见 `analysis/training_report_<timestamp>.md`。

## 版本历史

- **v4.0** (2026-04-04): 增加全数据集预测检验 + Markdown 训练报告生成
- **v3.0** (2026-04-02): 拆分为 4 个独立子模型 + 门控路由，2 维输入
- **v2.0** (2026-03-31): 重构为 FCT 预测函数 + 梯度下降优化
- **v1.0** (2026-03-30): 初始版本 (直接预测阈值，已废弃)
