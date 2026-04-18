#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优阈值推理 - 基于门控路由 + 子模型梯度下降优化

推理流程:
1. 给定 (avg_flow_size, load)
2. route_model(avg_flow_size) 路由到对应子模型
3. 梯度下降优化阈值，使预测 FCT 最小
4. 返回最优阈值和预测 FCT

统一接口:
  find_optimal_threshold(avg_flow_size, load) -> (optimal_threshold, predicted_fct)
"""

import torch
import numpy as np
import joblib
import json
import time
import swanlab
import matplotlib.pyplot as plt
from datetime import datetime

from model import SubModel, MODEL_CONFIG, ANALYSIS_DIR, LOG_DIR, route_model, get_submodel_dir


class ThresholdOptimizer:
    """阈值优化器: 用梯度下降在子模型上寻找最优阈值"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.config = MODEL_CONFIG[model_name]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        save_dir = get_submodel_dir(model_name)

        # 加载子模型
        self.model = SubModel(input_dim=2, hidden_dims=[64, 32], dropout_rate=0.2).to(self.device)
        self.model.load_state_dict(
            torch.load(save_dir / "predictor.pth", map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # 加载 scaler
        self.scaler_X = joblib.load(save_dir / "scaler_X.pkl")
        self.scaler_y = joblib.load(save_dir / "scaler_y.pkl")

        # 缓存为 torch tensor (保持梯度图不断裂)
        self.x_mean = torch.FloatTensor(self.scaler_X.mean_).to(self.device)
        self.x_std = torch.FloatTensor(self.scaler_X.scale_).to(self.device)
        self.y_mean = float(self.scaler_y.mean_[0])
        self.y_std = float(self.scaler_y.scale_[0])

        print(f"子模型加载: {model_name} ({save_dir})")

    def predict_fct(self, t_value, load):
        """预测给定阈值和负载下的 FCT (微秒)"""
        features = np.array([[t_value, load]])
        features_norm = self.scaler_X.transform(features)

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_norm).to(self.device)
            y_norm = self.model(features_tensor).item()

        log_fct = y_norm * self.y_std + self.y_mean
        fct_us = np.exp(log_fct)
        return fct_us

    def find_optimal(self, load, max_iter=100, lr=20.0, verbose=False, run=None):
        """
        用梯度下降找最优阈值

        策略: Adam + ReduceLROnPlateau + 早停
        - 初始学习率 20.0 快速靠近最优
        - 损失停滞时自动缩小学习率，精调
        - 连续 patience 步不改善则提前停止

        返回: (optimal_t, predicted_fct, history)
        """
        t_range = self.config['t_range']
        t_init = (t_range[0] + t_range[1]) / 2

        # 可学习的阈值参数
        T = torch.tensor([t_init], requires_grad=True, device=self.device)
        # 固定的 load
        load_tensor = torch.tensor([load], device=self.device)

        optimizer = torch.optim.Adam([T], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=0.1
        )

        history = {'t': [], 'fct': [], 'lr': []}

        # 早停参数
        best_fct = float('inf')
        best_t = t_init
        no_improve_count = 0
        early_stop_patience = 20

        for i in range(max_iter):
            optimizer.zero_grad()

            # 构造 2-dim 输入 [T, load]
            x = torch.cat([T, load_tensor])
            x_norm = (x - self.x_mean) / self.x_std
            x_tensor = x_norm.unsqueeze(0)  # (1, 2)

            # 预测并最小化
            y_norm = self.model(x_tensor)
            loss = y_norm

            loss.backward()
            optimizer.step()

            # 约束范围
            with torch.no_grad():
                T.data.clamp_(*t_range)

            # 记录历史
            with torch.no_grad():
                y_val = y_norm.item()
                log_fct = y_val * self.y_std + self.y_mean
                fct_us = np.exp(log_fct)
                current_lr = optimizer.param_groups[0]['lr']
                history['t'].append(T.item())
                history['fct'].append(fct_us)
                history['lr'].append(current_lr)

                if run is not None:
                    run.log(
                        {
                            f"{self.model_name}/load_{load}/fct_us": fct_us,
                            f"{self.model_name}/load_{load}/threshold_kb": T.item(),
                            f"{self.model_name}/load_{load}/learning_rate": current_lr,
                        },
                        step=i,
                    )

                # 早停判断
                if fct_us < best_fct:
                    best_fct = fct_us
                    best_t = T.item()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            # 学习率调度 (基于 loss)
            scheduler.step(loss.item())

            if verbose and (i + 1) % 20 == 0:
                print(f"  Iter {i + 1:3d}: T={T.item():.1f}KB, FCT={fct_us:.0f}us, lr={current_lr:.2f}")

            # 早停触发
            if no_improve_count >= early_stop_patience:
                if verbose:
                    print(f"  早停于第 {i + 1} 步 (连续 {early_stop_patience} 步无改善)")
                break

        fct_opt = self.predict_fct(best_t, load)

        return best_t, fct_opt, history


def find_optimal_threshold(avg_flow_size, load):
    """
    统一推理接口: 给定流量特征和负载，找到最优阈值

    参数:
        avg_flow_size: 平均流大小（字节）
        load: 网络负载 (0-1)

    返回:
        (optimal_threshold, predicted_fct)
    """
    model_name = route_model(avg_flow_size)
    optimizer = ThresholdOptimizer(model_name)
    t_opt, fct_opt, _ = optimizer.find_optimal(load)

    # 组合完整的阈值结果
    config = MODEL_CONFIG[model_name]
    result = {
        'model': model_name,
        config['optimize']: t_opt,  # t1 或 t2
        **config['fixed_threshold'],  # 固定的另一个阈值
        'predicted_fct_us': fct_opt,
    }

    return result, fct_opt


def main():
    print("=" * 60)
    print("最优阈值预测 - 门控路由 + 梯度下降优化")
    print("=" * 60)

    # SwanLab
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    inference_log_dir = LOG_DIR / "inference" / run_timestamp
    run = swanlab.init(
        project="fct-optimization-inference",
        experiment_name=f"inference-{run_timestamp}",
        tags=["inference", "threshold-optimization"],
        logdir=str(inference_log_dir),
        config={
            "optimizer": "Adam",
            "learning_rate": 20.0,
            "max_iter": 100,
            "models": ["server", "cache", "search", "mine"],
            "loads": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        reinit=True,
    )
    print(f"SwanLab 本地日志目录: {inference_log_dir}")

    # 测试场景: 4 个模型 × 5 个负载 = 20 个组合
    test_configs = [
        ('server', 65536),              # 64 KB
        ('cache',  630 * 1024),         # 630 KB
        ('search', 1.6 * 1024 * 1024),  # 1.6 MB
        ('mine',   7.4 * 1024 * 1024),  # 7.4 MB
    ]
    test_loads = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)

    inference_results = []
    # 计时统计: {model_name: [times...]}
    timing_stats = {model: [] for model, _ in test_configs}

    for model_name, avg_flow_size in test_configs:
        config = MODEL_CONFIG[model_name]
        t_key = config['optimize']
        optimizer = ThresholdOptimizer(model_name)

        print(f"\n{'=' * 60}")
        print(f"{model_name.upper()} 模型 (avg_flow_size={avg_flow_size / 1024:.0f}KB)")
        print(f"{'=' * 60}")

        for load_idx, load in enumerate(test_loads):
            # 计时开始
            start_time = time.perf_counter()
            t_opt, fct_opt, _ = optimizer.find_optimal(load, run=run)
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000

            timing_stats[model_name].append(elapsed_ms)

            # 记录最优结果
            run.log(
                {
                    f"{model_name}/optimal/threshold_kb": t_opt,
                    f"{model_name}/optimal/fct_us": fct_opt,
                    f"{model_name}/optimal/inference_time_ms": elapsed_ms,
                },
                step=load_idx,
            )

            # 显示结果
            print(f"  [{model_name.upper():6s}] load={load:.1f}, {t_key.upper()}={t_opt:.1f}KB, "
                  f"FCT={fct_opt:.0f}us ({fct_opt/1000:.2f}ms), time={elapsed_ms:.2f}ms")

            inference_results.append({
                'model': model_name,
                'avg_flow_size': avg_flow_size,
                'load': load,
                f'optimal_{config["optimize"]}_kb': round(t_opt, 2),
                **{k: v for k, v in config['fixed_threshold'].items()},
                'predicted_fct_us': round(fct_opt, 2),
                'inference_time_ms': round(elapsed_ms, 2),
            })

    # 推理耗时汇总
    print("\n" + "=" * 60)
    print("推理耗时汇总")
    print("=" * 60)
    print(f"\n{'模型':<10} {'平均耗时(ms)':<15} {'最小(ms)':<12} {'最大(ms)':<12}")
    print("-" * 50)
    for model_name in timing_stats:
        times = timing_stats[model_name]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"{model_name:<10} {avg_time:<15.2f} {min_time:<12.2f} {max_time:<12.2f}")

    # 保存推理结果
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = ANALYSIS_DIR / f"inference_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(inference_results, f, indent=2, ensure_ascii=False)
    print(f"\n推理结果保存到: {results_path}")

    # 可视化: 各模型的 FCT 响应曲线
    print("\n" + "=" * 60)
    print("可视化: FCT 响应曲线")
    print("=" * 60)

    models_to_plot = ['server', 'cache', 'search', 'mine']
    loads_to_plot = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FCT Response Curves (Per-Model Sub-Models)', fontsize=16)

    for idx, model_name in enumerate(models_to_plot):
        ax = axes[idx // 2, idx % 2]
        config = MODEL_CONFIG[model_name]
        t_range = config['t_range']

        opt = ThresholdOptimizer(model_name)

        for load in loads_to_plot:
            # 生成 FCT 响应曲线
            t_values = np.linspace(t_range[0], t_range[1], 100)
            fct_values = [opt.predict_fct(t, load) / 1000 for t in t_values]  # ms

            ax.plot(t_values, fct_values, label=f'load={load}', linewidth=2)

            # 标记最优值
            t_opt, fct_opt, _ = opt.find_optimal(load)
            ax.scatter([t_opt], [fct_opt / 1000], s=100, zorder=5,
                       edgecolors='black', linewidths=1.5)

        t_name = config['optimize'].upper()
        ax.set_title(f'{model_name.upper()} (optimize {t_name})')
        ax.set_xlabel(f'{t_name} (KB)')
        ax.set_ylabel('FCT (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fct_curve_path = ANALYSIS_DIR / "fct_response_curves.png"
    plt.savefig(fct_curve_path, dpi=300, bbox_inches='tight')
    run.log({"plots/fct_response_curves": swanlab.Image(str(fct_curve_path))})
    print(f"  FCT 响应曲线: {fct_curve_path}")
    plt.close(fig)

    # 优化过程图 (以 cache 为例)
    print("\n生成优化过程图...")
    opt = ThresholdOptimizer('cache')
    t_opt, fct_opt, history = opt.find_optimal(0.5, max_iter=100, lr=20.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history['fct'])
    axes[0].set_title('FCT Optimization (CACHE, load=0.5)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('FCT (us)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['t'])
    axes[1].set_title('T1 Convergence')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('T1 (KB)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['lr'])
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    opt_process_path = ANALYSIS_DIR / "optimization_process.png"
    plt.savefig(opt_process_path, dpi=300, bbox_inches='tight')
    run.log({"plots/optimization_process": swanlab.Image(str(opt_process_path))})
    print(f"  优化过程图: {opt_process_path}")
    plt.close(fig)

    run.finish()

    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
