#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FCT 子模型训练

训练流程:
1. 对 4 个流量模型 (server/cache/search/mine) 各自独立训练
2. 每个子模型输入 (2 维): [T_kb, load]
3. 目标: log(FCT)
4. 保存到 models/{model_name}/ 目录

数据结构说明:
  - server/cache: 实验中只变了 T1, T2 固定为 1024KB → 子模型输入 [t1_kb, load]
  - search/mine:  实验中只变了 T2, T1 固定为 100KB  → 子模型输入 [t2_kb, load]
  - 4 个子模型独立训练，不共享权重
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import swanlab
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

from model import SubModel, MODEL_CONFIG, RAW_DATA_DIR, LOG_DIR, get_submodel_dir, MODEL_DIR, ANALYSIS_DIR

# ==================== 数据准备 ====================

def load_raw_data():
    """加载所有原始实验数据，返回按模型名分组的字典"""
    print("加载数据...")

    model_names = ['server', 'cache', 'search', 'mine']
    data = {}

    for model in model_names:
        filepath = RAW_DATA_DIR / f"fct_result_{model}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            data[model] = df
            print(f"  {model:8s}: {len(df)} 条记录")

    print(f"模型数: {len(data)}")
    return data


def prepare_training_data(df, model_name):
    """
    为单个子模型构造训练数据

    输入 (2 维): [T_kb, load]
      - server/cache: T_kb = t1_kb
      - search/mine:  T_kb = t2_kb
    输出: log(avg_fct)
    """
    config = MODEL_CONFIG[model_name]
    t_key = config['threshold_key']

    X = df[[t_key, 'load']].values
    y = np.log(df['avg_fct'].values)

    print(f"  输入维度: 2 ([{t_key}, load])")
    print(f"  样本数: {len(X)}")

    return X, y


# ==================== 数据集 ====================

class FCTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 训练 ====================

def train_submodel(model_name, X, y):
    """训练单个子模型的完整流程"""
    config = MODEL_CONFIG[model_name]
    t_key = config['threshold_key']
    save_dir = get_submodel_dir(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 40}")
    print(f"训练子模型: {model_name.upper()} (输入: [{t_key}, load])")
    print(f"{'=' * 40}")

    # 1. 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 2. 标准化
    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    X_val_norm = scaler_X.transform(X_val)
    X_test_norm = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # 3. DataLoader
    train_loader = DataLoader(FCTDataset(X_train_norm, y_train_norm), batch_size=32, shuffle=True)
    val_loader = DataLoader(FCTDataset(X_val_norm, y_val_norm), batch_size=32)

    # 4. 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SubModel(input_dim=2, hidden_dims=[64, 32], dropout_rate=0.2).to(device)
    print(f"  设备: {device}")
    print("  模型: 2 -> [64, 32] -> 1")

    # 5. SwanLab
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    swan_log_dir = LOG_DIR / "train" / model_name / run_timestamp
    run = swanlab.init(
        project="fct-optimization-train",
        experiment_name=f"{model_name}-{run_timestamp}",
        tags=["train", model_name],
        logdir=str(swan_log_dir),
        config={
            "model_name": model_name,
            "input_dim": 2,
            "hidden_dims": [64, 32],
            "dropout_rate": 0.2,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 20,
            "early_stop_patience": 30,
            "max_epoch": 500,
            "batch_size": 32,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "total_samples": len(X),
        },
        reinit=True,
    )

    # 6. 训练
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 30
    loss_history = {'train': [], 'val': []}

    for epoch in range(500):
        # 训练
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # 日志
        run.log(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            },
            step=epoch,
        )
        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)

        # 权重统计 (每 50 epoch)
        if (epoch + 1) % 50 == 0:
            for name, param in model.named_parameters():
                run.log(
                    {
                        f"weights/{name}_mean": param.data.mean().item(),
                        f"weights/{name}_std": param.data.std().item(),
                    },
                    step=epoch,
                )

        # 早停 + 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "predictor.pth")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch + 1:3d}/500], "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # 7. 加载最佳模型，在测试集评估
    model.load_state_dict(torch.load(save_dir / "predictor.pth", weights_only=True))
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_norm).to(device)
        y_pred_norm = model(X_test_tensor).cpu().numpy()

    # 反标准化 + 反对数
    y_pred_log = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true_log = scaler_y.inverse_transform(y_test_norm.reshape(-1, 1)).flatten()
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_true_log)

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n  {model_name.upper()} 测试集评估:")
    print(f"    MAE:  {mae:.0f} us")
    print(f"    RMSE: {rmse:.0f} us")
    print(f"    MAPE: {mape:.2f}%")

    run.log(
        {
            "test/MAE_us": mae,
            "test/RMSE_us": rmse,
            "test/MAPE_pct": mape,
            "test/best_val_loss": best_val_loss,
        },
        step=0,
    )

    # 记录训练过程图像
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history["train"], label="train_loss")
    ax.plot(loss_history["val"], label="val_loss")
    ax.set_title(f"Loss Curve - {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    run.log({"plots/loss_curve": swanlab.Image(fig)})
    plt.close(fig)

    run.finish()

    # 8. 保存 scaler 和结果
    joblib.dump(scaler_X, save_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, save_dir / "scaler_y.pkl")

    results = {
        'test_mae_us': float(mae),
        'test_rmse_us': float(rmse),
        'test_mape_pct': float(mape),
        'best_val_loss': float(best_val_loss),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'total_samples': len(X),
    }
    with open(save_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(save_dir / "loss_history.json", 'w') as f:
        json.dump(loss_history, f, indent=2)

    print(f"  保存到: {save_dir}")

    return results


# ==================== 全数据集预测检验 ====================

def predict_full_dataset(data):
    """
    对每个子模型用全部数据做预测，输出表格
    返回: {model_name: {rows: [...], metrics: {...}}}}
    """
    print("\n" + "=" * 80)
    print("全数据集预测检验")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for model_name in ['server', 'cache', 'search', 'mine']:
        if model_name not in data:
            continue

        df = data[model_name]
        config = MODEL_CONFIG[model_name]
        t_key = config['threshold_key']
        save_dir = get_submodel_dir(model_name)

        # 加载模型和 scaler
        model = SubModel(input_dim=2, hidden_dims=[64, 32], dropout_rate=0.2).to(device)
        model.load_state_dict(torch.load(save_dir / "predictor.pth", weights_only=True, map_location=device))
        model.eval()

        scaler_X = joblib.load(save_dir / "scaler_X.pkl")
        scaler_y = joblib.load(save_dir / "scaler_y.pkl")

        # 构造输入
        X = df[[t_key, 'load']].values
        y_actual = df['avg_fct'].values

        # 预测
        X_norm = scaler_X.transform(X)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(device)
            y_pred_norm = model(X_tensor).cpu().numpy()

        # 反标准化 + 反对数
        y_pred_log = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        y_pred = np.exp(y_pred_log)

        # 计算误差
        error_rate = np.abs(y_actual - y_pred) / y_actual * 100

        # 打印表格
        print(f"\n{model_name.upper()} 全数据集预测结果:")
        print(f"{'model':<8} {'load':<6} {'t1_kb':<8} {'t2_kb':<8} {'actual_fct':>14} {'predicted_fct':>14} {'error(%)':>10}")
        print("-" * 80)

        rows = []
        for i in range(len(df)):
            row = {
                'model': model_name,
                'load': df.iloc[i]['load'],
                't1_kb': df.iloc[i]['t1_kb'],
                't2_kb': df.iloc[i]['t2_kb'],
                'actual_fct': y_actual[i],
                'predicted_fct': y_pred[i],
                'error_rate': error_rate[i]
            }
            rows.append(row)
            print(f"{row['model']:<8} {row['load']:<6.1f} {row['t1_kb']:<8.1f} {row['t2_kb']:<8.1f} "
                  f"{row['actual_fct']:>14.2f} {row['predicted_fct']:>14.2f} {row['error_rate']:>10.2f}")

        # 统计信息
        mae = np.mean(np.abs(y_actual - y_pred))
        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        mape = np.mean(error_rate)

        print(f"\n{model_name.upper()} 整体指标:")
        print(f"  MAE:  {mae:.2f} us")
        print(f"  RMSE: {rmse:.2f} us")
        print(f"  MAPE: {mape:.2f}%")

        results[model_name] = {
            'rows': rows,
            'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape}
        }

    return results


# ==================== 生成 Markdown 报告 ====================

def generate_training_report(data, train_results, full_pred_results, timestamp):
    """
    生成训练报告 Markdown 文件
    """
    report_path = ANALYSIS_DIR / f"training_report_{timestamp}.md"

    md = []
    md.append("# FCT 子模型训练报告\n")
    md.append(f"**生成时间**: {timestamp}\n")
    md.append(f"**训练设备**: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
    md.append("\n---\n")

    # 1. 训练配置
    md.append("## 1. 训练配置\n")
    md.append("| 参数 | 值 |\n")
    md.append("|------|-----|\n")
    md.append("| 网络结构 | 2 → [64, 32] → 1 |\n")
    md.append("| 优化器 | Adam (lr=0.001, weight_decay=1e-5) |\n")
    md.append("| 损失函数 | MSE (标准化 log(FCT) 空间) |\n")
    md.append("| 学习率调度 | ReduceLROnPlateau (patience=20, factor=0.5) |\n")
    md.append("| 早停 | patience=30 |\n")
    md.append("| 数据划分 | 70% 训练 / 15% 验证 / 15% 测试 |\n")
    md.append("| batch_size | 32 |\n")
    md.append("| 最大 epoch | 500 |\n")
    md.append("\n")

    # 2. 数据概况
    md.append("## 2. 数据概况\n")
    md.append("| 模型 | 样本数 | 训练集 | 验证集 | 测试集 | 优化阈值 | 固定阈值 |\n")
    md.append("|------|--------|--------|--------|--------|----------|----------|\n")

    for model_name in ['server', 'cache', 'search', 'mine']:
        if model_name not in data:
            continue
        config = MODEL_CONFIG[model_name]
        t = train_results.get(model_name, {})
        total = len(data[model_name])
        train_n = t.get('train_samples', 0)
        val_n = t.get('val_samples', 0)
        test_n = t.get('test_samples', 0)
        opt_key = config['optimize']
        fixed = config['fixed_threshold']
        fixed_str = ', '.join([f"{k}={v}" for k, v in fixed.items()])
        md.append(f"| {model_name} | {total} | {train_n} | {val_n} | {test_n} | {opt_key} | {fixed_str} |\n")

    md.append("\n")

    # 3. 测试集评估指标
    md.append("## 3. 测试集评估指标\n")
    md.append("| 模型 | MAE (us) | RMSE (us) | MAPE (%) | 评价 |\n")
    md.append("|------|----------|-----------|---------|------|\n")

    for model_name in ['server', 'cache', 'search', 'mine']:
        if model_name not in train_results:
            continue
        t = train_results[model_name]
        mape = t['test_mape_pct']
        if mape < 5:
            rating = "优秀"
        elif mape < 10:
            rating = "良好"
        else:
            rating = "需改进"
        md.append(f"| {model_name} | {t['test_mae_us']:.0f} | {t['test_rmse_us']:.0f} | {mape:.2f} | {rating} |\n")

    md.append("\n")

    # 4. 全数据集预测结果
    md.append("## 4. 全数据集预测结果\n")

    for model_name in ['server', 'cache', 'search', 'mine']:
        if model_name not in full_pred_results:
            continue

        r = full_pred_results[model_name]
        m = r['metrics']
        rows = r['rows']

        md.append(f"### {model_name.upper()}\n")
        md.append(f"**整体指标**: MAE={m['mae']:.2f}us, RMSE={m['rmse']:.2f}us, MAPE={m['mape']:.2f}%\n\n")

        md.append("| model | load | t1_kb | t2_kb | actual_fct | predicted_fct | error(%) |\n")
        md.append("|-------|------|-------|-------|------------|---------------|----------|\n")

        for row in rows:
            md.append(f"| {row['model']} | {row['load']:.1f} | {row['t1_kb']:.1f} | {row['t2_kb']:.1f} | "
                     f"{row['actual_fct']:.2f} | {row['predicted_fct']:.2f} | {row['error_rate']:.2f} |\n")

        md.append("\n")

    # 5. 保存报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"\n训练报告已生成: {report_path}")
    return report_path


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("FCT 子模型训练（4 个独立模型）")
    print("=" * 60)

    # 1. 加载数据
    data = load_raw_data()

    # 2. 训练 4 个子模型
    all_results = {}
    for model_name in ['server', 'cache', 'search', 'mine']:
        if model_name not in data:
            print(f"跳过 {model_name}（无数据）")
            continue

        X, y = prepare_training_data(data[model_name], model_name)
        results = train_submodel(model_name, X, y)
        all_results[model_name] = results

    # 3. 汇总
    print("\n" + "=" * 60)
    print("训练完成 - 汇总")
    print("=" * 60)
    print(f"\n{'模型':<10} {'MAE(us)':<12} {'RMSE(us)':<12} {'MAPE(%)':<10} {'样本数':<8}")
    print("-" * 52)
    for name, r in all_results.items():
        print(f"{name:<10} {r['test_mae_us']:<12.0f} {r['test_rmse_us']:<12.0f} "
              f"{r['test_mape_pct']:<10.2f} {r['total_samples']:<8}")

    print(f"\n模型保存目录: {MODEL_DIR}")
    print(f"SwanLab 本地日志目录: {LOG_DIR / 'train'}")

    # 4. 全数据集预测检验
    full_pred_results = predict_full_dataset(data)

    # 5. 生成 Markdown 报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generate_training_report(data, all_results, full_pred_results, timestamp)


if __name__ == "__main__":
    main()
