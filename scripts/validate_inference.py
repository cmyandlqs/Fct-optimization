#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理验证报告 - 对比梯度下降最优阈值与离散数据最优区间

1. 从 dataset/cleaned_data/ 读取数据，找出每个 (model, load) 下 FCT 最小的阈值区间
2. 读取 inference_results JSON
3. 判断梯度下降结果是否落在最优区间
4. 生成 Markdown 报告
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ==================== 路径配置 ====================
BASE_DIR = Path(r"D:\sikm\Desktop\PythonProject\FCT_optimization")
RAW_DATA_DIR = BASE_DIR / "dataset" / "cleaned_data"
ANALYSIS_DIR = BASE_DIR / "analysis"


def load_all_data():
    """加载所有 CSV 数据"""
    data = {}
    for model in ['server', 'cache', 'search', 'mine']:
        filepath = RAW_DATA_DIR / f"fct_result_{model}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['model'] = model
            data[model] = df
    return data


def find_discrete_optimal(data):
    """
    找出每个 (model, load) 下 FCT 最小的阈值区间
    区间定义：FCT 最优的前 3 个 t 值形成的区间
    """
    results = {}

    for model, df in data.items():
        results[model] = {}

        # 优化的阈值 key: t1_kb 或 t2_kb
        # 对于 server/cache 是 t1_kb，对于 search/mine 是 t2_kb
        if model in ['server', 'cache']:
            t_key = 't1_kb'
        else:
            t_key = 't2_kb'

        for load in sorted(df['load'].unique()):
            load_df = df[df['load'] == load].copy()

            # 按 FCT 排序，取前 3 个
            top3 = load_df.nsmallest(3, 'avg_fct')

            best_t = top3[t_key].values
            best_fct = top3['avg_fct'].values

            # 最优区间: [top3 中最小t, top3 中最大t]
            t_min = best_t.min()
            t_max = best_t.max()

            # 离散最优 (FCT 最低的那个点)
            best_idx = load_df['avg_fct'].idxmin()
            discrete_best_t = load_df.loc[best_idx, t_key]
            discrete_best_fct = load_df.loc[best_idx, 'avg_fct']

            results[model][load] = {
                't_key': t_key,
                'optimal_interval': (t_min, t_max),
                'discrete_optimal_t': discrete_best_t,
                'discrete_optimal_fct': discrete_best_fct,
                'top3_t': list(best_t),
                'top3_fct': list(best_fct),
            }

    return results


def load_inference_results():
    """加载最新的 inference_results JSON"""
    # 找最新的文件
    json_files = list(ANALYSIS_DIR.glob("inference_results_*.json"))
    if not json_files:
        raise FileNotFoundError("未找到 inference_results JSON 文件")

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    with open(latest, 'r', encoding='utf-8') as f:
        return json.load(f), latest.name


def generate_report(data, discrete_opt, inference_results, json_filename):
    """生成 Markdown 验证报告"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = ANALYSIS_DIR / f"validation_report_{timestamp}.md"

    md = []
    md.append(f"# 推理验证报告\n")
    md.append(f"**生成时间**: {timestamp}\n")
    md.append(f"**推理结果文件**: {json_filename}\n")
    md.append("\n---\n")

    # 汇总统计
    total = 0
    in_interval = 0
    out_interval = 0

    for model in ['server', 'cache', 'search', 'mine']:
        total_model = 0
        in_model = 0

        for load in sorted(discrete_opt[model].keys()):
            opt = discrete_opt[model][load]
            t_key = opt['t_key']

            # 找对应的推理结果
            inf = None
            for item in inference_results:
                if item['model'] == model and item['load'] == load:
                    inf = item
                    break

            if inf is None:
                continue

            opt_t = inf.get('optimal_t1_kb') or inf.get('optimal_t2_kb')
            t_min, t_max = opt['optimal_interval']

            # 判断是否在区间内
            in_opt = t_min <= opt_t <= t_max

            if in_opt:
                in_interval += 1
                in_model += 1
            else:
                out_interval += 1

            total += 1
            total_model += 1

        if total_model == 0:
            continue

        # 该模型命中率
        hit_rate = in_model / total_model * 100
        md.append(f"## {model.upper()}\n")
        md.append(f"**命中率**: {in_model}/{total_model} ({hit_rate:.1f}%)\n")
        md.append("\n")

    # 总体统计
    hit_rate_total = in_interval / total * 100 if total > 0 else 0
    md.append("## 总体统计\n")
    md.append("\n")
    md.append(f"| 指标 | 值 |\n")
    md.append(f"|------|-----|\n")
    md.append(f"| 总场景数 | {total} |\n")
    md.append(f"| 命中区间 | {in_interval} ({in_interval/total*100:.1f}%) |\n")
    md.append(f"| 未命中区间 | {out_interval} ({out_interval/total*100:.1f}%) |\n")
    md.append("\n")

    # 详细对比表
    md.append("## 详细对比\n")
    md.append("\n")
    md.append(f"| 模型 | load | 离散最优T | T区间 | 梯度下降T | 命中? |\n")
    md.append(f"|------|------|-----------|-------|-----------|--------|\n")

    for model in ['server', 'cache', 'search', 'mine']:
        for load in sorted(discrete_opt[model].keys()):
            opt = discrete_opt[model][load]
            t_key = opt['t_key']

            # 找对应的推理结果 (load 转 float 避免 np.float64 比较问题)
            inf = None
            for item in inference_results:
                if item['model'] == model and float(item['load']) == float(load):
                    inf = item
                    break

            if inf is None:
                continue

            opt_t = inf.get('optimal_t1_kb') or inf.get('optimal_t2_kb')
            t_min, t_max = opt['optimal_interval']
            discrete_best_t = opt['discrete_optimal_t']

            # 判断是否在区间内
            in_opt = t_min <= opt_t <= t_max
            hit_str = "✅" if in_opt else "❌"

            md.append(f"| {model} | {load:.1f} | {discrete_best_t:.1f} | [{t_min:.0f}, {t_max:.0f}] | {opt_t:.1f} | {hit_str} |\n")

    md.append("\n")

    # 详细数据表 (每个 model, load 的 top3)
    md.append("## 离散数据详情\n")
    md.append("\n")

    for model in ['server', 'cache', 'search', 'mine']:
        md.append(f"### {model.upper()}\n")
        md.append("\n")

        for load in sorted(discrete_opt[model].keys()):
            opt = discrete_opt[model][load]
            t_key = opt['t_key']
            t_min, t_max = opt['optimal_interval']

            # 找对应的推理结果 (load 转 float 避免 np.float64 比较问题)
            inf = None
            for item in inference_results:
                if item['model'] == model and float(item['load']) == float(load):
                    inf = item
                    break

            opt_t = inf.get('optimal_t1_kb') or inf.get('optimal_t2_kb') if inf else None

            md.append(f"**load={load:.1f}**\n")
            md.append(f"- 阈值类型: {t_key.upper()}\n")
            md.append(f"- 最优区间: [{t_min:.0f}, {t_max:.0f}] KB\n")
            if opt_t is not None:
                md.append(f"- 梯度下降结果: {opt_t:.1f} KB\n")
            else:
                md.append(f"- 梯度下降结果: N/A\n")
            md.append(f"- Top3 FCT 对应的阈值:\n")

            for i, (t, fct) in enumerate(zip(opt['top3_t'], opt['top3_fct'])):
                md.append(f"  {i+1}. T={t:.0f}KB, FCT={fct:.0f}us\n")

            md.append("\n")

    # 保存报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"验证报告已生成: {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("推理验证 - 对比梯度下降与离散最优")
    print("=" * 60)

    # 1. 加载数据
    print("\n加载数据...")
    data = load_all_data()
    for model, df in data.items():
        print(f"  {model}: {len(df)} 条")

    # 2. 找出离散最优区间
    print("\n计算离散最优阈值区间...")
    discrete_opt = find_discrete_optimal(data)

    # 3. 加载推理结果
    print("\n加载推理结果...")
    inference_results, json_filename = load_inference_results()
    print(f"  文件: {json_filename}")
    print(f"  记录数: {len(inference_results)}")

    # 4. 生成报告
    print("\n生成验证报告...")
    report_path = generate_report(data, discrete_opt, inference_results, json_filename)

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
