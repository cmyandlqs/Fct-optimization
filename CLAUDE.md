# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FCT (Flow Completion Time) optimization for network scheduling thresholds. 4 independent sub-models learn `FCT = f(T_kb, load)` per traffic type, with gated routing based on `avg_flow_size`. At inference time, gradient descent finds threshold values that minimize predicted FCT.

## Commands

Run from `scripts/` directory using conda env `clip_tsa`:

```bash
conda activate clip_tsa
cd scripts
python train_fct_predictor.py          # Train 4 sub-models
python predict_optimal_threshold.py    # Inference via routing + gradient descent
python eda.py                          # EDA charts to analysis/
tensorboard --logdir ../logs           # View TensorBoard (http://localhost:6006)
```

### Dependencies

PyTorch, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, tensorboard (all in `clip_tsa` conda env).

No package manager config or test framework.

以后运行 conda 命令时使用这个路径：

D:\Miniconda3\envs\clip_tsa\python.exe

## Architecture

### Module Structure

```
scripts/
  model.py                      (SubModel class, route_model, MODEL_CONFIG, path constants)
  train_fct_predictor.py        imports SubModel, MODEL_CONFIG, RAW_DATA_DIR, LOG_DIR, get_submodel_dir
  predict_optimal_threshold.py  imports SubModel, MODEL_CONFIG, MODEL_DIR, ANALYSIS_DIR, LOG_DIR, route_model, get_submodel_dir
  eda.py                        imports MODEL_CONFIG, RAW_DATA_DIR, ANALYSIS_DIR
```

SubModel is defined once in `model.py` — do not duplicate.

### Data Flow

```
dataset/cleaned_data/*.csv (4 CSVs: server 45 + cache 56 + search 45 + mine 45)
  → train_fct_predictor.py (4 independent sub-models)
  → models/{model_name}/ (weights + scaler_X.pkl + scaler_y.pkl + training_results.json)
  → predict_optimal_threshold.py (route_model → sub-model → gradient descent)
  → analysis/ (charts)

eda.py → analysis/ (6 charts: feature distributions, FCT vs threshold, best threshold vs load, correlation matrix, FCT distributions)
```

### CSV Data Format

Each CSV contains columns: `model, load, t1_kb, t2_kb, avg_fct` (cleaned data).

### Data Structure (critical)

Each model's experiments only vary **one** threshold dimension:
- **server/cache** (101 rows): T1 varies (5-7 values × 9 loads), **T2 fixed at 1024KB**
- **search/mine** (90 rows): T2 varies (5 values × 9 loads), **T1 fixed at 100KB**

Each sub-model is trained independently on its own data with **2-dim input** `[T_kb, load]`.

### Core Model: SubModel

- **Input** (2 dims): `[T_kb, load]`
  - server/cache: T_kb = t1_kb
  - search/mine: T_kb = t2_kb
- **Architecture**: 64→32 hidden layers, ReLU, Dropout(0.2 first layer only), output 1 scalar
- **Training**: Adam (lr=0.001, weight_decay=1e-5), MSE in standardized log(FCT) space, ReduceLROnPlateau (patience=20), early stopping (patience=30), batch_size=32, up to 500 epochs
- **Normalization**: StandardScaler on both X features (`scaler_X.pkl`) and y targets (`scaler_y.pkl`). Target y is log(FCT) before standardization.

### Gate Routing: route_model

```python
def route_model(avg_flow_size):
    if avg_flow_size < 200 * 1024:        return 'server'
    elif avg_flow_size < 1 * 1024 * 1024: return 'cache'
    elif avg_flow_size < 3 * 1024 * 1024: return 'search'
    else:                                 return 'mine'
```

### Inference: ThresholdOptimizer + Unified API

```python
# Unified API
find_optimal_threshold(avg_flow_size, load) → (result_dict, predicted_fct)

# Direct usage
optimizer = ThresholdOptimizer(model_name)
t_opt, fct_opt, history = optimizer.find_optimal(load)
```

Gradient descent over learnable T through the trained sub-model. Uses PyTorch-native standardization `(x - mean) / std` to preserve gradient graph. Adam with lr=10, 200 iterations, clamps to valid range.

### Traffic Model Config (MODEL_CONFIG)

| Model   | avg_flow_size | Optimize | threshold_key | Fixed              | T Range       |
|---------|--------------|----------|---------------|--------------------|---------------|
| server  | 64 KB        | T1       | t1_kb         | t2_kb=1024         | 10 - 250 KB   |
| cache   | 630 KB       | T1       | t1_kb         | t2_kb=1024         | 10 - 250 KB   |
| search  | 1.6 MB       | T2       | t2_kb         | t1_kb=100          | 512 - 2048 KB |
| mine    | 7.4 MB       | T2       | t2_kb         | t1_kb=100          | 512 - 2048 KB |

### Saved Model Files

```
models/
  server/
    predictor.pth           # Best model weights
    scaler_X.pkl            # StandardScaler for 2-dim input [t1_kb, load]
    scaler_y.pkl            # StandardScaler for log(FCT) target
    training_results.json
    loss_history.json
  cache/
    predictor.pth
    scaler_X.pkl
    scaler_y.pkl
    training_results.json
    loss_history.json
  search/
    predictor.pth
    scaler_X.pkl
    scaler_y.pkl
    training_results.json
    loss_history.json
  mine/
    predictor.pth
    scaler_X.pkl
    scaler_y.pkl
    training_results.json
    loss_history.json
logs/
  train/{model_name}/{timestamp}/     # TensorBoard training logs per sub-model
  inference/{timestamp}/              # TensorBoard inference logs
analysis/
  inference_results_<ts>.json         # Timestamped inference results
  fct_response_curves.png             # Per-model FCT vs T curves
  optimization_process.png            # Optimization convergence plot
```

### Key Design Decisions

- `BASE_DIR` is hardcoded to `D:\sikm\Desktop\PythonProject\FCT_optimization`
- log transform on `FCT` (target only; inputs are raw T_kb and load values)
- Both X and y are standardized with StandardScaler for stable training
- Inference uses manual `(x - mean) / std` in PyTorch to keep autograd graph intact
- 4 sub-models are completely independent — no shared weights or scalers
- Gate routing based on avg_flow_size thresholds, no ML-based routing needed
