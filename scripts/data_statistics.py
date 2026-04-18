"""
数据统计脚本 - 统计cleaned_data和new_data的数据分布情况
"""

import pandas as pd
from pathlib import Path

# 定义路径
BASE_DIR = Path(__file__).parent.parent
CLEANED_DATA_DIR = BASE_DIR / "dataset" / "cleaned_data"
NEW_DATA_DIR = BASE_DIR / "dataset" / "new_data"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


def load_data_from_directory(data_dir):
    """从指定目录加载所有CSV文件"""
    data = {}
    for csv_file in sorted(data_dir.glob("*.csv")):
        model_name = csv_file.stem.replace("fct_result_", "").replace("_new", "")
        df = pd.read_csv(csv_file)
        data[model_name] = df
    return data


def analyze_threshold_values(df, model_name):
    """分析阈值变化情况"""
    stats = {}

    # server和cache主要变化t1_kb，search和mine主要变化t2_kb
    if model_name in ['server', 'cache']:
        threshold_col = 't1_kb'
        fixed_col = 't2_kb'
    else:
        threshold_col = 't2_kb'
        fixed_col = 't1_kb'

    # 变化的阈值统计
    threshold_values = df[threshold_col].unique()
    threshold_values_sorted = sorted(threshold_values)

    stats['threshold_column'] = threshold_col
    stats['threshold_values'] = threshold_values_sorted
    stats['threshold_count'] = len(threshold_values)
    stats['threshold_min'] = float(min(threshold_values))
    stats['threshold_max'] = float(max(threshold_values))

    # 固定阈值统计
    fixed_values = df[fixed_col].unique()
    stats['fixed_column'] = fixed_col
    stats['fixed_values'] = sorted(fixed_values.tolist())
    stats['fixed_is_constant'] = len(fixed_values) == 1

    return stats


def compute_statistics(df, model_name, dataset_name):
    """计算单个数据集的统计信息"""
    stats = {
        'dataset': dataset_name,
        'model': model_name,
        'total_rows': len(df),
        'loads': sorted(df['load'].unique().tolist()),
        'load_count': len(df['load'].unique()),
    }

    # 每个load的样本数
    samples_per_load = df.groupby('load').size().to_dict()
    stats['samples_per_load'] = {float(k): int(v) for k, v in sorted(samples_per_load.items())}

    # FCT统计
    stats['fct'] = {
        'min': float(df['avg_fct'].min()),
        'max': float(df['avg_fct'].max()),
        'mean': float(df['avg_fct'].mean()),
        'std': float(df['avg_fct'].std()),
    }

    # 阈值分析
    threshold_stats = analyze_threshold_values(df, model_name)
    stats.update(threshold_stats)

    return stats


def print_statistics(all_stats):
    """打印统计信息到控制台"""
    print("=" * 80)
    print("FCT数据集统计报告")
    print("=" * 80)
    print()

    for dataset_name, models in all_stats.items():
        print(f"{'=' * 80}")
        print(f"数据集: {dataset_name}")
        print(f"{'=' * 80}")
        print()

        for model_name, stats in models.items():
            print(f"--- 场景: {model_name} ---")
            print(f"总行数: {stats['total_rows']}")
            print(f"负载数量: {stats['load_count']} ({', '.join(map(str, stats['loads']))})")
            print(f"每个负载的样本数: {stats['samples_per_load']}")
            print(f"\n变化的阈值: {stats['threshold_column']}")
            print(f"  - 阈值数量: {stats['threshold_count']}")
            print(f"  - 阈值范围: [{stats['threshold_min']}, {stats['threshold_max']}]")
            print(f"  - 阈值列表: {stats['threshold_values']}")
            print(f"\n固定阈值: {stats['fixed_column']}")
            print(f"  - 固定值: {stats['fixed_values']}")
            print(f"  - 是否恒定: {stats['fixed_is_constant']}")
            print("\nFCT统计 (avg_fct):")
            print(f"  - 最小值: {stats['fct']['min']:.2f}")
            print(f"  - 最大值: {stats['fct']['max']:.2f}")
            print(f"  - 平均值: {stats['fct']['mean']:.2f}")
            print(f"  - 标准差: {stats['fct']['std']:.2f}")
            print()


def save_comparison_report(all_stats):
    """生成对比报告并保存到文件"""
    report_lines = []
    report_lines.append("# FCT数据集对比报告")
    report_lines.append("")
    report_lines.append("## 数据集概览")
    report_lines.append("")
    report_lines.append("| 数据集 | 场景 | 总行数 | 负载数 | 变化阈值 | 阈值范围 | 固定阈值 |")
    report_lines.append("|--------|------|--------|--------|----------|----------|----------|")

    for dataset_name, models in all_stats.items():
        for model_name, stats in models.items():
            threshold_range = f"[{stats['threshold_min']}, {stats['threshold_max']}]"
            fixed_val = str(stats['fixed_values'])
            report_lines.append(
                f"| {dataset_name} | {model_name} | {stats['total_rows']} | "
                f"{stats['load_count']} | {stats['threshold_column']} | {threshold_range} | {fixed_val} |"
            )

    report_lines.append("")
    report_lines.append("## 详细统计")
    report_lines.append("")

    for dataset_name, models in all_stats.items():
        report_lines.append(f"### {dataset_name}")
        report_lines.append("")

        for model_name, stats in models.items():
            report_lines.append(f"#### {model_name}")
            report_lines.append("")
            report_lines.append(f"- **总行数**: {stats['total_rows']}")
            report_lines.append(f"- **负载列表**: {', '.join(map(str, stats['loads']))}")
            report_lines.append("- **每个负载的样本数**:")
            for load, count in stats['samples_per_load'].items():
                report_lines.append(f"  - load={load}: {count} 个样本")

            report_lines.append(f"- **变化的阈值** ({stats['threshold_column']}):")
            report_lines.append(f"  - 数量: {stats['threshold_count']}")
            report_lines.append(f"  - 范围: [{stats['threshold_min']}, {stats['threshold_max']}]")
            report_lines.append(f"  - 值列表: {stats['threshold_values']}")

            report_lines.append(f"- **固定阈值** ({stats['fixed_column']}): {stats['fixed_values']}")

            report_lines.append("- **FCT统计**:")
            report_lines.append(f"  - 最小值: {stats['fct']['min']:.2f}")
            report_lines.append(f"  - 最大值: {stats['fct']['max']:.2f}")
            report_lines.append(f"  - 平均值: {stats['fct']['mean']:.2f}")
            report_lines.append(f"  - 标准差: {stats['fct']['std']:.2f}")

            report_lines.append("")

    # 保存报告
    report_path = ANALYSIS_DIR / "data_statistics_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    return report_path


def save_summary_csv(all_stats):
    """保存汇总统计到CSV"""
    summary_data = []

    for dataset_name, models in all_stats.items():
        for model_name, stats in models.items():
            row = {
                'dataset': dataset_name,
                'model': model_name,
                'total_rows': stats['total_rows'],
                'load_count': stats['load_count'],
                'loads': ','.join(map(str, stats['loads'])),
                'threshold_column': stats['threshold_column'],
                'threshold_count': stats['threshold_count'],
                'threshold_min': stats['threshold_min'],
                'threshold_max': stats['threshold_max'],
                'threshold_values': str(stats['threshold_values']),
                'fixed_column': stats['fixed_column'],
                'fixed_values': str(stats['fixed_values']),
                'fct_min': stats['fct']['min'],
                'fct_max': stats['fct']['max'],
                'fct_mean': stats['fct']['mean'],
                'fct_std': stats['fct']['std'],
            }
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    csv_path = ANALYSIS_DIR / "data_statistics_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    return csv_path


def main():
    """主函数"""
    print("正在加载数据...")

    # 加载两个数据集
    cleaned_data = load_data_from_directory(CLEANED_DATA_DIR)
    new_data = load_data_from_directory(NEW_DATA_DIR)

    print(f"cleaned_data: {list(cleaned_data.keys())}")
    print(f"new_data: {list(new_data.keys())}")
    print()

    # 计算统计信息
    all_stats = {
        'cleaned_data': {},
        'new_data': {}
    }

    for model_name, df in cleaned_data.items():
        all_stats['cleaned_data'][model_name] = compute_statistics(df, model_name, 'cleaned_data')

    for model_name, df in new_data.items():
        all_stats['new_data'][model_name] = compute_statistics(df, model_name, 'new_data')

    # 打印统计信息
    print_statistics(all_stats)

    # 保存报告
    print("正在生成报告...")
    report_path = save_comparison_report(all_stats)
    print(f"Markdown报告已保存: {report_path}")

    csv_path = save_summary_csv(all_stats)
    print(f"CSV汇总已保存: {csv_path}")

    print()
    print("=" * 80)
    print("统计完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
