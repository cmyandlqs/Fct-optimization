#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享模块 - 子模型定义、门控路由与项目配置

包含：
- SubModel 神经网络（2 维输入：[T_kb, load]）
- route_model() 门控路由函数
- MODEL_CONFIG 流量模型配置
- 项目路径常量
"""

import torch
import torch.nn as nn
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(r"D:\sikm\Desktop\PythonProject\FCT_optimization")
RAW_DATA_DIR = BASE_DIR / "dataset" / "all_data"
MODEL_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
LOG_DIR = BASE_DIR / "logs"


def get_submodel_dir(model_name: str) -> Path:
    """获取子模型目录"""
    return MODEL_DIR / model_name


# ==================== 流量模型配置 ====================
# avg_flow_size: 平均流大小（字节），用于门控路由
# optimize: 推理时优化哪个阈值 (t1 或 t2)
# threshold_key: 子模型输入中使用的阈值列名
# fixed_threshold: 推理时固定的另一个阈值
# t_range: 阈值搜索范围
#
# 数据结构（关键）:
#   server/cache: T1 变化 (5-7 值 × 9 load), T2 固定 1024KB → 优化 T1
#   search/mine:  T2 变化 (5 值 × 9 load),   T1 固定 100KB  → 优化 T2
MODEL_CONFIG = {
    'server': {
        'avg_flow_size': 65536,          # 64 KB
        'optimize': 't1',
        'threshold_key': 't1_kb',
        'fixed_threshold': {'t2_kb': 1024.0},
        't_range': (10, 250),
    },
    'cache': {
        'avg_flow_size': 615 * 1024,     # 630 KB
        'optimize': 't1',
        'threshold_key': 't1_kb',
        'fixed_threshold': {'t2_kb': 1024.0},
        't_range': (10, 250),
    },
    'search': {
        'avg_flow_size': 1.6 * 1024 * 1024,  # 1.6 MB
        'optimize': 't2',
        'threshold_key': 't2_kb',
        'fixed_threshold': {'t1_kb': 100.0},
        't_range': (512, 2048),
    },
    'mine': {
        'avg_flow_size': 7.41 * 1024 * 1024,  # 7.4 MB
        'optimize': 't2',
        'threshold_key': 't2_kb',
        'fixed_threshold': {'t1_kb': 100.0},
        't_range': (512, 2048),
    },
}

# ==================== 数据统计 ====================
# 实际数据量（从 CSV 统计）:
#   server:  45 条 (5 个 T1 值 × 9 个 load, T2 固定 1024)
#   cache:   56 条 (6-7 个 T1 值 × 9 个 load, T2 固定 1024)
#   search:  45 条 (5 个 T2 值 × 9 个 load, T1 固定 100)
#   mine:    45 条 (5 个 T2 值 × 9 个 load, T1 固定 100)

# ==================== 门控路由 ====================

def route_model(avg_flow_size: float) -> str:
    """
    根据 avg_flow_size 路由到对应子模型

    参数:
        avg_flow_size: 平均流大小（字节）
    返回:
        模型名称: 'server' / 'cache' / 'search' / 'mine'
    """
    if avg_flow_size < 200 * 1024:          # < 200 KB
        return 'server'
    elif avg_flow_size < 1 * 1024 * 1024:   # < 1 MB
        return 'cache'
    elif avg_flow_size < 3 * 1024 * 1024:   # < 3 MB
        return 'search'
    else:
        return 'mine'


# ==================== 模型定义 ====================

class SubModel(nn.Module):
    """
    子模型：学习单种流量模型的 FCT = f(T_kb, load)

    输入 (2 维): [T_kb, load]
      - server/cache: T_kb = t1_kb
      - search/mine:  T_kb = t2_kb
    输出: log(FCT)

    网络结构:
      输入(2维) -> FC(64) -> ReLU -> Dropout(0.2)
                -> FC(32) -> ReLU
                -> FC(1)  -> Linear
    """

    def __init__(self, input_dim=2, hidden_dims=None, dropout_rate=0.2):
        super(SubModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            if i == 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()
