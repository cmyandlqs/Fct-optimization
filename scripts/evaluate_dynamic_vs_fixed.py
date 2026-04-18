#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态阈值（预测 FCT） vs 固定阈值基线（真实 FCT）评估脚本。

支持两阶段评估：
- v1: 仅评估 server + cache
- v2: 评估 server + cache + search + mine

默认负载粒度：0.1 ~ 0.9 全部点（步长 0.1）。

输出：
- cases.csv
- summary.json
- report.md
- plots/*.png

全部产物写入单独目录：
analysis/eval_dynamic_vs_fixed_<timestamp>/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

from model import ANALYSIS_DIR, RAW_DATA_DIR
from predict_optimal_threshold import ThresholdOptimizer


MODEL_ORDER = ["server", "cache", "search", "mine"]
MODEL_INDEX = {name: i for i, name in enumerate(MODEL_ORDER)}
PHASE_MODELS = {
    "v1": ["server", "cache"],
    "v2": ["server", "cache", "search", "mine"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dynamic threshold vs fixed baseline.")
    parser.add_argument(
        "--phase",
        choices=["v1", "v2"],
        default="v1",
        help="v1: server/cache only; v2: all four models",
    )
    parser.add_argument(
        "--loads",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="comma-separated load list, e.g. 0.1,0.2,...,0.9",
    )
    parser.add_argument(
        "--inference-file",
        type=str,
        default="",
        help="optional inference_results json; if omitted, auto uses latest",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="ignore inference file and recompute dynamic outputs for all selected cases",
    )
    return parser.parse_args()


def parse_loads(loads_arg: str) -> list[float]:
    loads = []
    for item in loads_arg.split(","):
        item = item.strip()
        if not item:
            continue
        value = round(float(item), 1)
        loads.append(value)
    if not loads:
        raise ValueError("No valid loads parsed from --loads")
    return sorted(set(loads))


def fmt_load(load: float) -> str:
    return f"{load:.1f}"


def load_inference_results(inference_file: str) -> tuple[list[dict], Path | None]:
    if inference_file:
        path = Path(inference_file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Specified inference file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), path

    files = sorted(ANALYSIS_DIR.glob("inference_results_*.json"))
    if not files:
        return [], None
    latest = files[-1]
    with latest.open("r", encoding="utf-8") as f:
        return json.load(f), latest


def build_inference_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    out = {}
    for row in rows:
        key = (row["model"], fmt_load(float(row["load"])))
        out[key] = row
    return out


def load_all_data() -> dict[str, dict[str, list[dict]]]:
    data: dict[str, dict[str, list[dict]]] = {}
    for model_name in MODEL_ORDER:
        fp = RAW_DATA_DIR / f"fct_result_{model_name}.csv"
        by_load: dict[str, list[dict]] = defaultdict(list)
        with fp.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                by_load[fmt_load(float(row["load"]))].append(row)
        data[model_name] = dict(by_load)
    return data


def improve_pct(fixed_fct: float, dynamic_fct: float) -> float:
    return (fixed_fct - dynamic_fct) / fixed_fct * 100.0


def get_fixed_policy(model_name: str) -> tuple[float, float]:
    # server/cache: 固定 t1=100, t2=1024
    # search/mine: 固定 t1=100, t2=1024
    return 100.0, 1024.0


def find_fixed_baseline_row(rows: list[dict], model_name: str) -> tuple[dict, bool, str]:
    """
    返回: (baseline_row, exact_match, note)
    若无精确点，则按固定策略近似匹配。
    """
    fixed_t1, fixed_t2 = get_fixed_policy(model_name)

    exact = [
        row
        for row in rows
        if abs(float(row["t1_kb"]) - fixed_t1) < 1e-9 and abs(float(row["t2_kb"]) - fixed_t2) < 1e-9
    ]
    if exact:
        return min(exact, key=lambda r: float(r["avg_fct"])), True, "exact"

    # 近似匹配：按当前场景的主要阈值匹配最近值
    if model_name in ("server", "cache"):
        candidates = [row for row in rows if abs(float(row["t2_kb"]) - fixed_t2) < 1e-9]
        if not candidates:
            candidates = rows
        nearest = min(candidates, key=lambda r: abs(float(r["t1_kb"]) - fixed_t1))
        note = f"approx_by_t1(nearest={float(nearest['t1_kb'])})"
    else:
        candidates = [row for row in rows if abs(float(row["t1_kb"]) - fixed_t1) < 1e-9]
        if not candidates:
            candidates = rows
        nearest = min(candidates, key=lambda r: abs(float(r["t2_kb"]) - fixed_t2))
        note = f"approx_by_t2(nearest={float(nearest['t2_kb'])})"

    return nearest, False, note


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "avg_improve_pct": 0.0,
            "median_improve_pct": 0.0,
            "win_rate": 0.0,
            "win": 0,
            "total": 0,
            "best_improve_pct": 0.0,
            "worst_improve_pct": 0.0,
        }
    win = sum(1 for v in values if v > 0)
    return {
        "avg_improve_pct": sum(values) / len(values),
        "median_improve_pct": median(values),
        "win_rate": win / len(values),
        "win": win,
        "total": len(values),
        "best_improve_pct": max(values),
        "worst_improve_pct": min(values),
    }


def plot_improve_bar(case_rows: list[dict], save_path: Path) -> None:
    labels = [f"{row['model']}-{row['load']:.1f}" for row in case_rows]
    vals = [row["improve_pct_pred_vs_fixed"] for row in case_rows]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]

    width = max(12, len(vals) * 0.55)
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Improve % (Dynamic vs Fixed)")
    ax.set_title("Scenario-level Improvement")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_improve_box(case_rows: list[dict], selected_models: list[str], save_path: Path) -> None:
    grouped = []
    for model_name in selected_models:
        vals = [
            row["improve_pct_pred_vs_fixed"]
            for row in case_rows
            if row["model"] == model_name
        ]
        grouped.append(vals)

    fig, ax = plt.subplots(figsize=(max(6, len(selected_models) * 2.2), 5))
    ax.boxplot(grouped, tick_labels=selected_models, showmeans=True)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Improve %")
    ax.set_title("Improvement Distribution by Model")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _subplot_shape(n: int) -> tuple[int, int]:
    if n <= 2:
        return 1, n
    if n <= 4:
        return 2, 2
    cols = 3
    rows = math.ceil(n / cols)
    return rows, cols


def plot_fixed_vs_dynamic_lines(case_rows: list[dict], selected_models: list[str], save_path: Path) -> None:
    rows_n, cols_n = _subplot_shape(len(selected_models))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 5.5, rows_n * 4.0), sharex=True)

    if not isinstance(axes, (list, tuple)):
        # numpy ndarray or single axes
        try:
            flat_axes = axes.ravel().tolist()
        except Exception:
            flat_axes = [axes]
    else:
        flat_axes = list(axes)

    for idx, model_name in enumerate(selected_models):
        ax = flat_axes[idx]
        rows = sorted(
            [row for row in case_rows if row["model"] == model_name],
            key=lambda r: r["load"],
        )
        x = [row["load"] for row in rows]
        fixed_ms = [row["fct_fixed_us"] / 1000 for row in rows]
        dynamic_ms = [row["fct_dynamic_pred_us"] / 1000 for row in rows]

        ax.plot(x, fixed_ms, marker="o", label="fixed")
        ax.plot(x, dynamic_ms, marker="s", label="dynamic")
        ax.set_title(model_name)
        ax.set_xlabel("load")
        ax.set_ylabel("FCT (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(selected_models), len(flat_axes)):
        flat_axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_markdown_report(
    report_path: Path,
    inference_source: str,
    case_csv_path: Path,
    summary_json_path: Path,
    summary: dict,
    case_rows: list[dict],
    plot_paths: dict[str, Path],
) -> None:
    lines = []
    lines.append("# 动态阈值 vs 固定基线 评估报告")
    lines.append("")
    lines.append(f"- 评估阶段: `{summary['config']['phase']}`")
    lines.append(f"- 模型集合: `{', '.join(summary['config']['models'])}`")
    lines.append(f"- 负载集合: `{', '.join(f'{v:.1f}' for v in summary['config']['loads'])}`")
    lines.append(f"- 推理来源: `{inference_source}`")
    lines.append("- 固定基线定义:")
    lines.append("  - server/cache: t1_kb=100, t2_kb=1024")
    lines.append("  - search/mine: t1_kb=100, t2_kb=1024")
    lines.append("- 说明: 动态真实FCT暂缺，当前使用动态预测FCT进行对比")
    lines.append("")
    lines.append("## 汇总结果")
    lines.append("")
    lines.append("| 维度 | 平均降幅(%) | 中位数降幅(%) | Win Rate | 最优降幅(%) | 最差降幅(%) |")
    lines.append("|------|-------------|---------------|----------|-------------|-------------|")

    ov = summary["overall"]
    lines.append(
        f"| Overall | {ov['avg_improve_pct']:.3f} | {ov['median_improve_pct']:.3f} | "
        f"{ov['win']}/{ov['total']} ({ov['win_rate']*100:.1f}%) | "
        f"{ov['best_improve_pct']:.3f} | {ov['worst_improve_pct']:.3f} |"
    )

    for model_name in summary["config"]["models"]:
        sm = summary["by_model"][model_name]
        lines.append(
            f"| {model_name} | {sm['avg_improve_pct']:.3f} | {sm['median_improve_pct']:.3f} | "
            f"{sm['win']}/{sm['total']} ({sm['win_rate']*100:.1f}%) | "
            f"{sm['best_improve_pct']:.3f} | {sm['worst_improve_pct']:.3f} |"
        )

    lines.append("")
    lines.append("## 场景明细（Top 10 按降幅排序）")
    lines.append("")
    lines.append("| model | load | fct_fixed_us | fct_dynamic_pred_us | improve_pct | dynamic_source | baseline_note |")
    lines.append("|------|------|--------------|---------------------|-------------|----------------|---------------|")

    top_rows = sorted(case_rows, key=lambda r: r["improve_pct_pred_vs_fixed"], reverse=True)[:10]
    for row in top_rows:
        lines.append(
            f"| {row['model']} | {row['load']:.1f} | {row['fct_fixed_us']:.2f} | "
            f"{row['fct_dynamic_pred_us']:.2f} | {row['improve_pct_pred_vs_fixed']:.3f}% | "
            f"{row['dynamic_source']} | {row['baseline_note']} |"
        )

    lines.append("")
    lines.append("## 可视化")
    lines.append("")
    lines.append(f"1. 场景降幅柱状图: `{plot_paths['bar']}`")
    lines.append(f"2. 分模型降幅箱线图: `{plot_paths['box']}`")
    lines.append(f"3. 固定 vs 动态FCT折线图: `{plot_paths['line']}`")
    lines.append("")
    lines.append("## 产物文件")
    lines.append("")
    lines.append(f"- case明细: `{case_csv_path}`")
    lines.append(f"- summary: `{summary_json_path}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def get_dynamic_from_inference_or_compute(
    model_name: str,
    load: float,
    inference_map: dict[tuple[str, str], dict],
    optimizers: dict[str, ThresholdOptimizer],
    force_recompute: bool,
) -> tuple[float, float, float, str]:
    """
    返回: (dynamic_t1_kb, dynamic_t2_kb, predicted_fct_us, source)
    """
    key = (model_name, fmt_load(load))

    if (not force_recompute) and key in inference_map:
        item = inference_map[key]
        if model_name in ("server", "cache"):
            t1 = float(item["optimal_t1_kb"])
            t2 = 1024.0
        else:
            t1 = 100.0
            t2 = float(item["optimal_t2_kb"])
        return t1, t2, float(item["predicted_fct_us"]), "inference_json"

    optimizer = optimizers[model_name]
    if model_name in ("server", "cache"):
        t_opt, fct_pred, _ = optimizer.find_optimal(load)
        return float(t_opt), 1024.0, float(fct_pred), "recomputed"

    t_opt, fct_pred, _ = optimizer.find_optimal(load)
    return 100.0, float(t_opt), float(fct_pred), "recomputed"


def main() -> None:
    args = parse_args()
    selected_models = PHASE_MODELS[args.phase]
    selected_loads = parse_loads(args.loads)

    inference_rows, inference_path = load_inference_results(args.inference_file)
    inference_map = build_inference_map(inference_rows)
    all_data = load_all_data()

    # 仅为需要的模型初始化优化器
    optimizers = {model: ThresholdOptimizer(model) for model in selected_models}

    case_rows = []

    for model_name in selected_models:
        for load in selected_loads:
            load_key = fmt_load(load)
            if load_key not in all_data[model_name]:
                print(f"[跳过] {model_name} load={load_key}: 数据集中无该负载")
                continue

            rows = all_data[model_name][load_key]
            baseline_row, baseline_exact, baseline_note = find_fixed_baseline_row(rows, model_name)
            fct_fixed_us = float(baseline_row["avg_fct"])

            dyn_t1, dyn_t2, fct_dynamic_pred_us, dyn_source = get_dynamic_from_inference_or_compute(
                model_name=model_name,
                load=load,
                inference_map=inference_map,
                optimizers=optimizers,
                force_recompute=args.force_recompute,
            )

            improve = improve_pct(fct_fixed_us, fct_dynamic_pred_us)
            case_rows.append(
                {
                    "model": model_name,
                    "load": load,
                    "fixed_t1_kb": 100.0,
                    "fixed_t2_kb": 1024.0,
                    "dynamic_t1_kb": dyn_t1,
                    "dynamic_t2_kb": dyn_t2,
                    "fct_fixed_us": fct_fixed_us,
                    "fct_dynamic_pred_us": fct_dynamic_pred_us,
                    "fct_dynamic_real_us": "",
                    "improve_pct_pred_vs_fixed": improve,
                    "improve_pct_real_vs_fixed": "",
                    "baseline_exact": baseline_exact,
                    "baseline_note": baseline_note,
                    "dynamic_source": dyn_source,
                }
            )

    case_rows = sorted(
        case_rows,
        key=lambda r: (MODEL_INDEX[r["model"]], r["load"]),
    )

    overall_vals = [row["improve_pct_pred_vs_fixed"] for row in case_rows]
    by_model = {}
    for model_name in selected_models:
        vals = [
            row["improve_pct_pred_vs_fixed"]
            for row in case_rows
            if row["model"] == model_name
        ]
        by_model[model_name] = summarize(vals)

    summary = {
        "config": {
            "phase": args.phase,
            "models": selected_models,
            "loads": selected_loads,
            "force_recompute": args.force_recompute,
            "inference_file_arg": args.inference_file,
        },
        "inference_source": str(inference_path) if inference_path else "none",
        "overall": summarize(overall_vals),
        "by_model": by_model,
        "notes": {
            "dynamic_fct_source": "predicted_fct_us (from inference json or recomputed)",
            "dynamic_real_fct": "not available yet, left blank",
            "baseline_policy": {
                "server_cache": "t1_kb=100, t2_kb=1024",
                "search_mine": "t1_kb=100, t2_kb=1024",
            },
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = ANALYSIS_DIR / f"eval_dynamic_vs_fixed_{args.phase}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    case_csv_path = run_dir / "cases.csv"
    summary_json_path = run_dir / "summary.json"
    report_md_path = run_dir / "report.md"

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    bar_plot = plot_dir / "improvement_bar.png"
    box_plot = plot_dir / "improvement_box_by_model.png"
    line_plot = plot_dir / "fixed_vs_dynamic_lines.png"

    # 保存 case csv
    fieldnames = [
        "model",
        "load",
        "fixed_t1_kb",
        "fixed_t2_kb",
        "dynamic_t1_kb",
        "dynamic_t2_kb",
        "fct_fixed_us",
        "fct_dynamic_pred_us",
        "fct_dynamic_real_us",
        "improve_pct_pred_vs_fixed",
        "improve_pct_real_vs_fixed",
        "baseline_exact",
        "baseline_note",
        "dynamic_source",
    ]
    with case_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    # 保存 summary json
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成可视化
    if case_rows:
        plot_improve_bar(case_rows, bar_plot)
        plot_improve_box(case_rows, selected_models, box_plot)
        plot_fixed_vs_dynamic_lines(case_rows, selected_models, line_plot)

    # 生成 markdown 报告
    generate_markdown_report(
        report_path=report_md_path,
        inference_source=str(inference_path) if inference_path else "none",
        case_csv_path=case_csv_path,
        summary_json_path=summary_json_path,
        summary=summary,
        case_rows=case_rows,
        plot_paths={"bar": bar_plot, "box": box_plot, "line": line_plot},
    )

    print("=" * 72)
    print("动态阈值 vs 固定基线 评估完成")
    print("=" * 72)
    print(f"评估阶段: {args.phase}")
    print(f"模型: {', '.join(selected_models)}")
    print(f"负载: {', '.join(f'{v:.1f}' for v in selected_loads)}")
    print(f"推理来源: {summary['inference_source']}")
    print(f"评估输出目录: {run_dir}")
    print(f"Case 明细: {case_csv_path}")
    print(f"汇总结果: {summary_json_path}")
    print(f"评估报告: {report_md_path}")
    print(f"可视化目录: {plot_dir}")
    print()
    print("[整体结果]")
    print(
        f"avg={summary['overall']['avg_improve_pct']:.3f}% | "
        f"median={summary['overall']['median_improve_pct']:.3f}% | "
        f"win={summary['overall']['win']}/{summary['overall']['total']} "
        f"({summary['overall']['win_rate'] * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
