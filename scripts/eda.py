#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据探索性分析 (EDA)

从 cleaned_data 读取数据，生成图表到 analysis/ 目录。
适配 4 个独立子模型架构，每模型 2 维输入 [T_kb, load]。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from model import MODEL_CONFIG, RAW_DATA_DIR, ANALYSIS_DIR

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_raw_data():
    """加载原始数据"""
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    model_names = ['server', 'cache', 'search', 'mine']
    dfs = []

    for model in model_names:
        filepath = RAW_DATA_DIR / f"fct_result_{model}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['model'] = model
            dfs.append(df)
            print(f"  {model:8s}: {len(df)} 条记录")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n合计: {len(df_all)} 条记录")
    return df_all


def data_overview(df):
    """数据概览"""
    print("\n" + "=" * 60)
    print("数据概览")
    print("=" * 60)

    print(f"\n总样本数: {len(df)}")
    print(f"字段数: {len(df.columns)}")
    print(f"\n字段列表: {list(df.columns)}")

    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("\n无缺失值")
    else:
        print(f"\n缺失值:\n{missing[missing > 0]}")

    print("\n基本统计:")
    print(df[['t1_kb', 't2_kb', 'load', 'avg_fct']].describe().to_string())


def plot_feature_distributions(df):
    """绘制各模型的特征分布图（2 维: T_kb + load）"""
    print("\n生成特征分布图...")

    models = ['server', 'cache', 'search', 'mine']
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('Feature Distributions per Model (2-Dim Input)', fontsize=16)

    for idx, model in enumerate(models):
        config = MODEL_CONFIG[model]
        t_key = config['threshold_key']
        model_df = df[df['model'] == model]

        # T_kb 分布
        ax = axes[0, idx]
        ax.hist(model_df[t_key], bins=20, alpha=0.7, edgecolor='black')
        ax.set_title(f'{model.upper()}: {t_key}')
        ax.set_xlabel(f'{t_key} (KB)')
        ax.set_ylabel('count')
        ax.grid(True, alpha=0.3)

        # Load 分布
        ax = axes[1, idx]
        ax.hist(model_df['load'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title(f'{model.upper()}: load')
        ax.set_xlabel('Load')
        ax.set_ylabel('count')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    print("  -> feature_distributions.png")


def plot_fct_by_model_and_load(df):
    """绘制各模型在不同 load 下的 FCT 分布"""
    print("\n生成 FCT-Load 关系图...")

    models = ['server', 'cache', 'search', 'mine']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FCT vs Threshold by Model', fontsize=16)

    for idx, model in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        model_df = df[df['model'] == model].copy()
        config = MODEL_CONFIG[model]

        optimize_key = config['optimize']
        t_key = f'{optimize_key}_kb'

        for load in sorted(model_df['load'].unique()):
            load_df = model_df[model_df['load'] == load]
            ax.scatter(load_df[t_key], load_df['avg_fct'] / 1000,
                       alpha=0.5, s=20, label=f'load={load:.1f}')

        ax.set_title(f'{model.upper()} (optimize {optimize_key.upper()})')
        ax.set_xlabel(f'{optimize_key.upper()} (KB)')
        ax.set_ylabel('FCT (ms)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'fct_vs_threshold_by_model.png', dpi=300, bbox_inches='tight')
    print("  -> fct_vs_threshold_by_model.png")


def plot_threshold_vs_load(df):
    """绘制各 load 下离散最优阈值"""
    print("\n生成阈值-Load 关系图...")

    models = ['server', 'cache', 'search', 'mine']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Best Threshold (discrete) vs Load', fontsize=16)

    for idx, model in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        config = MODEL_CONFIG[model]
        model_df = df[df['model'] == model].copy()

        # 每个 load 找 FCT 最小的
        best_idx = model_df.groupby('load')['avg_fct'].idxmin()
        best_df = model_df.loc[best_idx]

        optimize_key = config['optimize']
        t_key = f'{optimize_key}_kb'

        ax.plot(best_df['load'], best_df[t_key], 'o-', linewidth=2, markersize=8)
        ax.set_title(f'{model.upper()} (best {optimize_key.upper()})')
        ax.set_xlabel('Load')
        ax.set_ylabel(f'{optimize_key.upper()} (KB)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'best_threshold_vs_load.png', dpi=300, bbox_inches='tight')
    print("  -> best_threshold_vs_load.png")


def plot_correlation_matrix(df):
    """绘制各模型的相关性矩阵（2 维特征 vs log_fct）"""
    print("\n生成相关性矩阵...")

    models = ['server', 'cache', 'search', 'mine']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Per-Model Correlation: [T_kb, load] vs log(FCT)', fontsize=16)

    for idx, model in enumerate(models):
        config = MODEL_CONFIG[model]
        t_key = config['threshold_key']
        model_df = df[df['model'] == model].copy()
        model_df['log_fct'] = np.log(model_df['avg_fct'])

        corr_cols = [t_key, 'load', 'log_fct']
        corr_matrix = model_df[corr_cols].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, ax=axes[idx],
                    vmin=-1, vmax=1)
        axes[idx].set_title(f'{model.upper()}')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("  -> correlation_matrix.png")

    # 打印相关性
    for model in models:
        config = MODEL_CONFIG[model]
        t_key = config['threshold_key']
        model_df = df[df['model'] == model].copy()
        model_df['log_fct'] = np.log(model_df['avg_fct'])
        corr = model_df[[t_key, 'load', 'log_fct']].corr()
        print(f"\n{model.upper()} 与 log(FCT) 的相关性:")
        print(corr['log_fct'].sort_values(ascending=False).to_string())


def plot_fct_distributions(df):
    """绘制各模型的 FCT 分布"""
    print("\n生成 FCT 分布图...")

    models = ['server', 'cache', 'search', 'mine']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FCT Distribution by Model', fontsize=16)

    for idx, model in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        model_df = df[df['model'] == model]
        fct_ms = model_df['avg_fct'] / 1000

        ax.hist(fct_ms, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(fct_ms.mean(), color='red', linestyle='--',
                   label=f'mean: {fct_ms.mean():.0f}ms')
        ax.set_title(f'{model.upper()}')
        ax.set_xlabel('FCT (ms)')
        ax.set_ylabel('count')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'fct_distributions.png', dpi=300, bbox_inches='tight')
    print("  -> fct_distributions.png")


def main():
    print("\n" + "=" * 60)
    print("数据探索性分析 (EDA)")
    print("=" * 60)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()

    data_overview(df)
    plot_feature_distributions(df)
    plot_fct_by_model_and_load(df)
    plot_threshold_vs_load(df)
    plot_correlation_matrix(df)
    plot_fct_distributions(df)

    print("\n" + "=" * 60)
    print(f"分析完成, 图表保存在: {ANALYSIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
