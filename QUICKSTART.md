# FCT 优化项目 - 快速开始

## 环境准备

使用 conda 环境 `clip_tsa`：

```bash
conda activate clip_tsa
cd scripts
```

检查依赖是否齐全：

```bash
python check_env.py
```

## 四步运行

### 步骤 1：训练 4 个子模型

```bash
python train_fct_predictor.py
```

**控制台输出**：
```
============================================================
FCT 子模型训练（4 个独立模型）
============================================================
加载数据...
  server  : 45 条记录
  cache   : 56 条记录
  search  : 45 条记录
  mine    : 45 条记录

========================================
训练子模型: SERVER (输入: [t1_kb, load])
========================================
  训练集: 31, 验证集: 7, 测试集: 7
  设备: cuda
  模型: 2 -> [64, 32] -> 1
  Epoch [ 20/500], Train Loss: 0.503449, Val Loss: 0.619709
  ...
  Early stopping at epoch 151

  SERVER 测试集评估:
    MAE:  57253 us
    RMSE: 122837 us
    MAPE: 13.70%

  保存到: models\server

(类似地训练 cache, search, mine)

============================================================
训练完成 - 汇总
============================================================
模型         MAE(us)      RMSE(us)     MAPE(%)    样本数
----------------------------------------------------
server     57253        122837       13.70      45
cache      45880        93912        4.60       56
search     27797        35223        2.34       45
mine       123415       173514       4.75       45

================================================================================
全数据集预测检验
================================================================================
(SERVER/CACHE/SEARCH/MINE 全数据集预测表格，显示每个数据点的 actual_fct, predicted_fct, error%)

训练报告已生成: analysis\training_report_<timestamp>.md
```

**产出文件**：

| 文件 | 说明 |
|------|------|
| `models/{model_name}/predictor.pth` | 最佳模型权重 |
| `models/{model_name}/scaler_X.pkl` | 输入特征标准化器 (2-dim: [T_kb, load]) |
| `models/{model_name}/scaler_y.pkl` | 目标 (log FCT) 标准化器 |
| `models/{model_name}/training_results.json` | 测试集指标 (MAE, RMSE, MAPE) |
| `models/{model_name}/loss_history.json` | 每 epoch 的 train/val loss |
| `analysis/training_report_<timestamp>.md` | 完整训练报告（含配置、全数据集预测结果） |

### 步骤 2：推理最优阈值

```bash
python predict_optimal_threshold.py
```

**控制台输出**：
```
============================================================
最优阈值预测 - 门控路由 + 梯度下降优化
============================================================
[SERVER] avg_flow_size=64KB, load=0.5, optimize: T1
  T1=250.0KB (最优), T2_KB=1024KB (固定)
  FCT=35733us (35.73ms)

[CACHE] avg_flow_size=615KB, load=0.3, optimize: T1
  T1=250.0KB (最优), T2_KB=1024KB (固定)
  FCT=247650us (247.65ms)
...

推理结果保存到: analysis\inference_results_<ts>.json
```

**产出文件**：

| 文件 | 说明 |
|------|------|
| `analysis/inference_results_<ts>.json` | 各场景最优阈值和预测 FCT |
| `analysis/fct_response_curves.png` | 各模型 FCT vs T 响应曲线 |
| `analysis/optimization_process.png` | 优化过程图 (FCT/T 随迭代变化) |

**代码中调用 (统一接口)**：
```python
from predict_optimal_threshold import find_optimal_threshold

result, fct = find_optimal_threshold(
    avg_flow_size=615 * 1024,  # 字节
    load=0.5,
)
print(f"模型: {result['model']}")
print(f"最优 T1: {result['t1']:.1f}KB, T2: {result['t2_kb']:.0f}KB")
print(f"预测 FCT: {fct:.0f}us")
```

**代码中调用 (直接指定子模型)**：
```python
from predict_optimal_threshold import ThresholdOptimizer

optimizer = ThresholdOptimizer('cache')
t_opt, fct_opt, history = optimizer.find_optimal(load=0.5)
print(f"T1={t_opt:.1f}KB, FCT={fct_opt:.0f}us")
```

### 步骤 3（可选）：数据探索

```bash
python eda.py
```

生成特征分布、相关性矩阵、FCT 分布等图表到 `analysis/`。

### 步骤 4：查看 TensorBoard

```bash
# 从项目根目录运行
tensorboard --logdir logs
```

浏览器打开 **http://localhost:6006**，可查看：

| 面板 | 内容 |
|------|------|
| **Scalars** | 各子模型 train/val loss 曲线、学习率、测试指标、推理优化过程 |
| **Histograms** | 各层权重/偏置分布 (每 50 epoch) |
| **Graphs** | SubModel 网络结构 |

---

## 数据说明

4 个流量模型的仿真实验数据，共 191 条：

| 模型 | 条数 | 变了什么 | 固定了什么 |
|------|------|---------|-----------|
| server | 45 | T1 (5个值 × 9个load) | T2=1024 |
| cache | 56 | T1 (7个值 × 9个load) | T2=1024 |
| search | 45 | T2 (5个值 × 9个load) | T1=100 |
| mine | 45 | T2 (5个值 × 9个load) | T1=100 |

每条记录 = 一次仿真实验：在某个 (T1, T2) 组合下测得的 FCT。

## 文件结构

```
FCT_optimization/
├── scripts/
│   ├── model.py                       # SubModel + route_model + 配置常量
│   ├── train_fct_predictor.py        # 训练 4 个子模型 + 全数据集预测 + 报告生成
│   ├── predict_optimal_threshold.py    # 门控路由 + 梯度下降推理
│   ├── eda.py                         # 数据探索
│   └── check_env.py                   # 环境检查
├── dataset/cleaned_data/              # 清理后的实验数据 (4 CSV, 191条)
├── models/                            # 子模型目录
│   ├── server/                        # server 子模型 (predictor.pth + scaler)
│   ├── cache/                         # cache 子模型
│   ├── search/                        # search 子模型
│   └── mine/                          # mine 子模型
├── logs/
│   ├── train/{model_name}/            # TensorBoard 训练日志
│   └── inference/                      # TensorBoard 推理日志
├── analysis/                          # 图表 + 推理结果 JSON + 训练报告
└── docs/                              # 技术文档
```

## 故障排查

| 问题 | 解决 |
|------|------|
| No module 'torch' | `conda activate clip_tsa` 激活正确环境 |
| No module 'tensorboard' | `pip install tensorboard` |
| 模型加载失败 | 先运行 `python train_fct_predictor.py` |
| 预测阈值超出范围 | 增加 `max_iter` 或调整 `lr` |
| FCT 预测误差大 | 检查各模型 `training_results.json`，MAPE>10% 说明需要更多数据 |
| TensorBoard 端口占用 | `tensorboard --logdir logs --port 6007` 换端口 |
