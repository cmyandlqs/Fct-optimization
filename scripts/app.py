#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FCT 最优阈值优化可视化 - Streamlit Web UI
"""

import sys
import os
import time

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go

# 将 scripts 目录加入 path，确保能 import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SubModel, MODEL_CONFIG, get_submodel_dir


# ==================== 模型缓存 ====================

@st.cache_resource
def load_optimizer(model_name):
    """加载子模型和 scaler（缓存避免重复加载）"""
    config = MODEL_CONFIG[model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = get_submodel_dir(model_name)

    model = SubModel(input_dim=2, hidden_dims=[64, 32], dropout_rate=0.2).to(device)
    model.load_state_dict(
        torch.load(save_dir / "predictor.pth", map_location=device, weights_only=True)
    )
    model.eval()

    scaler_X = joblib.load(save_dir / "scaler_X.pkl")
    scaler_y = joblib.load(save_dir / "scaler_y.pkl")

    x_mean = torch.FloatTensor(scaler_X.mean_).to(device)
    x_std = torch.FloatTensor(scaler_X.scale_).to(device)
    y_mean = float(scaler_y.mean_[0])
    y_std = float(scaler_y.scale_[0])

    return {
        'model': model,
        'config': config,
        'device': device,
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }


def predict_fct(opt_data, t_value, load):
    """预测给定阈值和负载下的 FCT"""
    device = opt_data['device']
    features = np.array([[t_value, load]])
    features_norm = (features - opt_data['x_mean'].cpu().numpy()) / opt_data['x_std'].cpu().numpy()

    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_norm).to(device)
        y_norm = opt_data['model'](features_tensor).item()

    log_fct = y_norm * opt_data['y_std'] + opt_data['y_mean']
    return np.exp(log_fct)


def run_optimization(opt_data, model_name, load, max_iter=100, lr=20.0):
    """
    运行梯度下降优化，返回每步的历史记录
    """
    config = opt_data['config']
    device = opt_data['device']
    t_range = config['t_range']
    t_init = (t_range[0] + t_range[1]) / 2

    T = torch.tensor([t_init], requires_grad=True, device=device)
    load_tensor = torch.tensor([load], device=device)

    optimizer = torch.optim.Adam([T], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=0.1
    )

    history = {'t': [], 'fct': [], 'lr': []}
    best_fct = float('inf')
    best_t = t_init
    no_improve_count = 0
    early_stop_patience = 20
    stopped_early = False

    for i in range(max_iter):
        optimizer.zero_grad()

        x = torch.cat([T, load_tensor])
        x_norm = (x - opt_data['x_mean']) / opt_data['x_std']
        x_tensor = x_norm.unsqueeze(0)

        y_norm = opt_data['model'](x_tensor)
        loss = y_norm

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            T.data.clamp_(*t_range)

        with torch.no_grad():
            y_val = y_norm.item()
            log_fct = y_val * opt_data['y_std'] + opt_data['y_mean']
            fct_us = np.exp(log_fct)
            current_lr = optimizer.param_groups[0]['lr']

            history['t'].append(T.item())
            history['fct'].append(fct_us)
            history['lr'].append(current_lr)

            if fct_us < best_fct:
                best_fct = fct_us
                best_t = T.item()
                no_improve_count = 0
            else:
                no_improve_count += 1

        scheduler.step(loss.item())

        if no_improve_count >= early_stop_patience:
            stopped_early = True
            break

    # 最终预测
    fct_opt = predict_fct(opt_data, best_t, load)

    return {
        'history': history,
        'best_t': best_t,
        'best_fct': fct_opt,
        'total_iters': len(history['t']),
        'stopped_early': stopped_early,
    }


# ==================== Streamlit UI ====================

st.set_page_config(page_title="FCT 最优阈值优化", layout="wide")
st.title("FCT 最优阈值优化可视化")
st.markdown("选择流量类型和负载，实时观察梯度下降寻找最优阈值的过程")

# ---------- 侧边栏: 参数设置 ----------
st.sidebar.header("参数设置")

model_name = st.sidebar.selectbox(
    "流量类型",
    options=['server', 'cache', 'search', 'mine'],
    index=0,
)

load = st.sidebar.slider(
    "网络负载 (load)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
)

max_iter = st.sidebar.slider(
    "最大迭代次数",
    min_value=20,
    max_value=200,
    value=100,
    step=10,
)

init_lr = st.sidebar.slider(
    "初始学习率",
    min_value=1.0,
    max_value=50.0,
    value=20.0,
    step=1.0,
)

# 显示当前模型信息
config = MODEL_CONFIG[model_name]
t_key = config['optimize']
t_range = config['t_range']
fixed_key = list(config['fixed_threshold'].keys())[0]
fixed_val = list(config['fixed_threshold'].values())[0]

st.sidebar.markdown("---")
st.sidebar.markdown("**模型信息**")
st.sidebar.markdown(f"- 优化目标: `{t_key}`")
st.sidebar.markdown(f"- 搜索范围: [{t_range[0]}, {t_range[1]}] KB")
st.sidebar.markdown(f"- 固定: `{fixed_key}` = {fixed_val}")

# ---------- 主区域 ----------

# 开始优化按钮
if st.button("开始优化", type="primary", use_container_width=True):
    # 加载模型
    with st.spinner(f"正在加载 {model_name} 子模型..."):
        opt_data = load_optimizer(model_name)

    # 运行优化
    start_time = time.perf_counter()
    result = run_optimization(opt_data, model_name, load, max_iter=max_iter, lr=init_lr)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    history = result['history']

    # ---------- 结果卡片 ----------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最优阈值", f"{result['best_t']:.1f} KB", f"{t_key}")
    col2.metric("预测 FCT", f"{result['best_fct']:.0f} us", f"{result['best_fct']/1000:.2f} ms")
    col3.metric("迭代次数", f"{result['total_iters']}", "早停" if result['stopped_early'] else "达到上限")
    col4.metric("优化耗时", f"{elapsed_ms:.0f} ms")

    st.markdown("---")

    # ---------- 图表区域 ----------
    # 第一行: FCT 收敛 + 阈值收敛
    col_fct, col_t = st.columns(2)

    with col_fct:
        st.subheader("FCT 收敛曲线")
        fig_fct = go.Figure()
        fig_fct.add_trace(go.Scatter(
            y=history['fct'],
            mode='lines+markers',
            name='FCT',
            marker=dict(size=3),
            line=dict(color='#1f77b4', width=2),
        ))
        # 标记最优点
        best_idx = history['fct'].index(min(history['fct']))
        fig_fct.add_trace(go.Scatter(
            x=[best_idx],
            y=[history['fct'][best_idx]],
            mode='markers',
            name=f'最优 FCT={history["fct"][best_idx]:.0f}',
            marker=dict(size=12, color='red', symbol='star'),
        ))
        fig_fct.update_layout(
            xaxis_title='迭代次数',
            yaxis_title='FCT (us)',
            height=400,
        )
        st.plotly_chart(fig_fct, use_container_width=True)

    with col_t:
        st.subheader(f"阈值 {t_key.upper()} 收敛曲线")
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            y=history['t'],
            mode='lines+markers',
            name=t_key.upper(),
            marker=dict(size=3),
            line=dict(color='#ff7f0e', width=2),
        ))
        # 标记最优阈值
        fig_t.add_trace(go.Scatter(
            x=[best_idx],
            y=[history['t'][best_idx]],
            mode='markers',
            name=f'最优 {t_key.upper()}={history["t"][best_idx]:.1f}',
            marker=dict(size=12, color='red', symbol='star'),
        ))
        fig_t.update_layout(
            xaxis_title='迭代次数',
            yaxis_title=f'{t_key.upper()} (KB)',
            height=400,
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # 第二行: 学习率变化 + 迭代详情表
    col_lr, col_table = st.columns(2)

    with col_lr:
        st.subheader("学习率变化")
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(
            y=history['lr'],
            mode='lines+markers',
            name='Learning Rate',
            marker=dict(size=3),
            line=dict(color='#2ca02c', width=2),
        ))
        fig_lr.update_layout(
            xaxis_title='迭代次数',
            yaxis_title='Learning Rate',
            height=400,
        )
        st.plotly_chart(fig_lr, use_container_width=True)

    with col_table:
        st.subheader("迭代详情")
        df_history = pd.DataFrame({
            '迭代': list(range(1, len(history['t']) + 1)),
            f'{t_key.upper()} (KB)': [round(v, 2) for v in history['t']],
            'FCT (us)': [round(v, 1) for v in history['fct']],
            'FCT (ms)': [round(v / 1000, 2) for v in history['fct']],
            '学习率': [round(v, 4) for v in history['lr']],
        })
        st.dataframe(df_history, use_container_width=True, height=400)

    # ---------- FCT 响应曲线 ----------
    st.markdown("---")
    st.subheader("FCT 响应曲线（阈值 vs FCT）")

    t_values = np.linspace(t_range[0], t_range[1], 100)
    fct_values = [predict_fct(opt_data, t, load) for t in t_values]

    fig_resp = go.Figure()
    fig_resp.add_trace(go.Scatter(
        x=t_values,
        y=[v / 1000 for v in fct_values],
        mode='lines',
        name=f'load={load}',
        line=dict(width=2),
    ))
    fig_resp.add_trace(go.Scatter(
        x=[result['best_t']],
        y=[result['best_fct'] / 1000],
        mode='markers',
        name=f'最优 {t_key.upper()}={result["best_t"]:.1f}',
        marker=dict(size=14, color='red', symbol='star'),
    ))
    fig_resp.update_layout(
        xaxis_title=f'{t_key.upper()} (KB)',
        yaxis_title='FCT (ms)',
        height=400,
    )
    st.plotly_chart(fig_resp, use_container_width=True)

else:
    st.info("请设置参数后点击 **开始优化** 按钮")
