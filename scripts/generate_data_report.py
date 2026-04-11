"""
FCT数据集统计报告生成脚本
读取四个CSV文件并生成markdown格式的中文统计报告
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

# 基础目录
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "dataset" / "cleaned_data"

# 输出目录
REPORT_DIR = BASE_DIR / "analysis"
REPORT_DIR.mkdir(exist_ok=True)

# 模型文件映射
MODEL_FILES = {
    "server": "fct_result_server.csv",
    "cache": "fct_result_cache.csv",
    "search": "fct_result_search.csv",
    "mine": "fct_result_mine.csv",
}

# 模型中文名称
MODEL_NAMES_CN = {
    "server": "服务器流量 (小流量)",
    "cache": "缓存流量 (中小流量)",
    "search": "搜索流量 (中等流量)",
    "mine": "挖矿流量 (大流量)",
}


def analyze_model(model_name, csv_path):
    """Analyze a single model's data."""
    df = pd.read_csv(csv_path)

    # Basic info
    total_rows = len(df)
    loads = sorted(df["load"].unique())
    num_loads = len(loads)

    # Determine which threshold varies
    t1_values = sorted(df["t1_kb"].unique())
    t2_values = sorted(df["t2_kb"].unique())

    if len(t1_values) > 1:
        varying_threshold = "t1_kb"
        fixed_threshold = "t2_kb"
        varying_values = t1_values
        fixed_value = t2_values[0]
    else:
        varying_threshold = "t2_kb"
        fixed_threshold = "t1_kb"
        varying_values = t2_values
        fixed_value = t1_values[0]

    # FCT statistics
    fct_min = df["avg_fct"].min()
    fct_max = df["avg_fct"].max()
    fct_mean = df["avg_fct"].mean()

    # Per-load threshold values
    load_thresholds = {}
    for load in loads:
        load_df = df[df["load"] == load]
        load_thresholds[load] = sorted(load_df[varying_threshold].unique())

    return {
        "model": model_name,
        "total_rows": total_rows,
        "loads": loads,
        "num_loads": num_loads,
        "varying_threshold": varying_threshold,
        "fixed_threshold": fixed_threshold,
        "varying_values": varying_values,
        "fixed_value": fixed_value,
        "num_varying_values": len(varying_values),
        "fct_min": fct_min,
        "fct_max": fct_max,
        "fct_mean": fct_mean,
        "load_thresholds": load_thresholds,
    }


def generate_markdown_report(analyses):
    """生成markdown格式的中文统计报告"""
    lines = []
    lines.append("# FCT数据集统计报告")
    lines.append("")
    lines.append("## 概览")
    lines.append("")
    lines.append("本报告汇总了四种流量模型的实验数据：")
    lines.append("- **server**: 服务器流量 (小流量, 平均 ~64 KB)")
    lines.append("- **cache**: 缓存流量 (中小流量, 平均 ~630 KB)")
    lines.append("- **search**: 搜索流量 (中等流量, 平均 ~1.6 MB)")
    lines.append("- **mine**: 挖矿流量 (大流量, 平均 ~7.4 MB)")
    lines.append("")

    # 汇总表格
    lines.append("## 数据集汇总")
    lines.append("")
    lines.append("| 模型 | 行数 | 负载数 | 变化阈值 | 固定值 | 阈值取值 | FCT范围 |")
    lines.append("|------|------|--------|----------|--------|----------|---------|")
    for a in analyses:
        fct_range = f"{a['fct_min']:.2f} - {a['fct_max']:.2f}"
        fixed_str = f"{a['fixed_threshold']}={a['fixed_value']}"
        thresh_vals = ", ".join([str(v) for v in a["varying_values"]])
        lines.append(
            f"| {a['model']} | {a['total_rows']} | {a['num_loads']} | {a['varying_threshold']} | {fixed_str} | {thresh_vals} | {fct_range} |"
        )
    lines.append("")

    # 各模型详细分析
    for a in analyses:
        model_cn = MODEL_NAMES_CN.get(a['model'], a['model'])
        lines.append(f"## {a['model'].upper()} 模型 - {model_cn}")
        lines.append("")
        lines.append(f"- **样本总数**: {a['total_rows']}")
        lines.append(f"- **负载取值**: {', '.join([str(l) for l in a['loads']])}")
        lines.append(f"- **变化阈值**: `{a['varying_threshold']}`")
        lines.append(f"- **固定阈值**: `{a['fixed_threshold']} = {a['fixed_value']}`")
        lines.append(f"- **变化阈值取值**: {', '.join([str(v) for v in a['varying_values']])} (共{len(a['varying_values'])}个值)")
        lines.append("")
        lines.append(f"### FCT统计信息")
        lines.append(f"- **最小值**: {a['fct_min']:.2f}")
        lines.append(f"- **最大值**: {a['fct_max']:.2f}")
        lines.append(f"- **平均值**: {a['fct_mean']:.2f}")
        lines.append("")

        lines.append(f"### 各负载下的阈值取值")
        lines.append("")
        lines.append("| 负载 | 阈值取值 | 数量 |")
        lines.append("|------|----------|------|")
        for load in a["loads"]:
            vals = a["load_thresholds"][load]
            vals_str = ", ".join([str(v) for v in vals])
            lines.append(f"| {load} | {vals_str} | {len(vals)} |")
        lines.append("")

    # 跨模型对比
    lines.append("## 跨模型对比")
    lines.append("")
    lines.append("### 阈值设计")
    lines.append("")
    lines.append("- **server/cache**: 优化T1阈值 (小到中等流量阈值)")
    lines.append("  - T2固定为 1024 KB")
    lines.append("  - T1范围: 根据模型不同而变化")
    lines.append("")
    lines.append("- **search/mine**: 优化T2阈值 (中等到大流量阈值)")
    lines.append("  - T1固定为 100 KB")
    lines.append("  - T2范围: 500-1500 KB")
    lines.append("")

    lines.append("### 负载分布")
    lines.append("")
    lines.append("所有模型均使用相同的9个负载值: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9")
    lines.append("")

    # 数据质量说明
    lines.append("## 数据质量说明")
    lines.append("")
    lines.append(f"- **server**: 设计一致 - {len(analyses[0]['varying_values'])}个T1值 × 9个负载 = {analyses[0]['total_rows']}行")
    lines.append(f"- **cache**: 略有不规则 - 共{analyses[1]['total_rows']}行 (部分负载的T1取值不同)")
    lines.append(f"- **search**: 设计一致 - {len(analyses[2]['varying_values'])}个T2值 × 9个负载 = {analyses[2]['total_rows']}行")
    lines.append(f"- **mine**: 设计一致 - {len(analyses[3]['varying_values'])}个T2值 × 9个负载 = {analyses[3]['total_rows']}行")
    lines.append("")

    return "\n".join(lines)


def main():
    """主函数：生成统计报告"""
    print("正在读取CSV文件...")

    analyses = []
    for model_name, filename in MODEL_FILES.items():
        csv_path = RAW_DATA_DIR / filename
        print(f"  - {filename}")
        analysis = analyze_model(model_name, csv_path)
        analyses.append(analysis)

    print("\n正在生成Markdown报告...")
    markdown = generate_markdown_report(analyses)

    # 保存报告
    report_path = REPORT_DIR / "data_statistics_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n报告已保存至: {report_path}")

    # 打印摘要
    print("\n=== 统计摘要 ===")
    for a in analyses:
        print(f"\n{a['model'].upper()}:")
        print(f"  - 行数: {a['total_rows']}")
        print(f"  - 负载数: {a['num_loads']} ({', '.join([str(l) for l in a['loads']])})")
        print(f"  - 变化阈值: {a['varying_threshold']} ({len(a['varying_values'])}个值: {', '.join([str(v) for v in a['varying_values']])})")
        print(f"  - 固定阈值: {a['fixed_threshold']} = {a['fixed_value']}")


if __name__ == "__main__":
    main()
