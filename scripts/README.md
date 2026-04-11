# 脚本说明

## 文件结构

```
scripts/
├── model.py                     # 共享模块: SubModel + route_model + MODEL_CONFIG + 路径常量
├── train_fct_predictor.py       # 训练 4 个独立子模型
├── predict_optimal_threshold.py # 推理: 门控路由 + 梯度下降优化
├── eda.py                       # 数据探索性分析
├── check_env.py                 # 环境检查
└── README.md                    # 本文件
```

## 模块依赖关系

```
model.py  (SubModel, route_model, MODEL_CONFIG, 路径常量)
  ├── train_fct_predictor.py  导入 SubModel, MODEL_CONFIG, get_submodel_dir
  ├── predict_optimal_threshold.py  导入 SubModel, MODEL_CONFIG, route_model, get_submodel_dir
  └── eda.py  导入 MODEL_CONFIG
```

修改模型结构时只需改 `model.py`，训练和推理脚本会自动同步。

## 运行顺序

### 1. 训练

```bash
python train_fct_predictor.py
```

- 输入: `dataset/cleaned_data/*.csv` (4 个 CSV, 共 191 条)
- 输出: `models/{server,cache,search,mine}/` (每个子模型的权重 + scaler + 训练结果)

### 2. 推理

```bash
python predict_optimal_threshold.py
```

- 前提: 已完成训练
- 使用门控路由: `route_model(avg_flow_size)` 自动选择子模型
- 统一接口: `find_optimal_threshold(avg_flow_size, load)`
- 输出: 控制台打印各场景最优阈值 + `analysis/` 下可视化图表

### 3. EDA (可选)

```bash
python eda.py
```

- 输入: `dataset/cleaned_data/*.csv` (直接读取，不依赖中间文件)
- 输出: `analysis/` 下分析图表
