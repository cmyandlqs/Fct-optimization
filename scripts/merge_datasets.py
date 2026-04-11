"""
数据集合并脚本 - 将cleaned_data和new_data合并为all_data
- server/cache: 直接合并所有数据
- mine/search: 只合并new_data中t1_kb=100.0的数据（与cleaned_data匹配）
"""

import pandas as pd
from pathlib import Path

# 定义路径
BASE_DIR = Path(__file__).parent.parent
CLEANED_DATA_DIR = BASE_DIR / "dataset" / "cleaned_data"
NEW_DATA_DIR = BASE_DIR / "dataset" / "new_data"
ALL_DATA_DIR = BASE_DIR / "dataset" / "all_data"
ALL_DATA_DIR.mkdir(exist_ok=True)


def load_csv(data_dir, model_name):
    """加载指定模型的CSV文件"""
    if model_name in ['server', 'cache', 'mine', 'search']:
        cleaned_file = data_dir / f"fct_result_{model_name}.csv"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if cleaned_file.exists():
        return pd.read_csv(cleaned_file)
    return None


def load_new_csv(data_dir, model_name):
    """加载new_data中指定模型的CSV文件"""
    new_file = data_dir / f"fct_result_{model_name}_new.csv"

    if new_file.exists():
        return pd.read_csv(new_file)
    return None


def merge_server_or_cache(model_name):
    """合并server或cache场景（直接合并所有数据）"""
    print(f"处理 {model_name}...")

    # 加载数据
    cleaned_df = load_csv(CLEANED_DATA_DIR, model_name)
    new_df = load_new_csv(NEW_DATA_DIR, model_name)

    if cleaned_df is None and new_df is None:
        print(f"  警告: {model_name} 没有找到任何数据")
        return None

    # 合并数据
    dfs = []
    if cleaned_df is not None:
        dfs.append(cleaned_df)
        print(f"  cleaned_data: {len(cleaned_df)} 行")
    if new_df is not None:
        dfs.append(new_df)
        print(f"  new_data: {len(new_df)} 行")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  合并后: {len(merged)} 行")

    # 去重（基于model, load, t1_kb, t2_kb）
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=['model', 'load', 't1_kb', 't2_kb'], keep='first')
    after_dedup = len(merged)
    if before_dedup != after_dedup:
        print(f"  去重: 移除了 {before_dedup - after_dedup} 行重复数据")

    # 在每个load内按t1_kb升序排列
    merged = merged.sort_values(['load', 't1_kb']).reset_index(drop=True)

    return merged


def merge_mine_or_search(model_name):
    """合并mine或search场景（只合并new_data中t1_kb=100.0的数据）"""
    print(f"处理 {model_name}...")

    # 加载数据
    cleaned_df = load_csv(CLEANED_DATA_DIR, model_name)
    new_df = load_new_csv(NEW_DATA_DIR, model_name)

    if cleaned_df is None and new_df is None:
        print(f"  警告: {model_name} 没有找到任何数据")
        return None

    dfs = []
    if cleaned_df is not None:
        dfs.append(cleaned_df)
        print(f"  cleaned_data: {len(cleaned_df)} 行")

    # 对于new_data，只选择t1_kb=100.0的行
    if new_df is not None:
        # 检查cleaned_data的t1_kb值
        if cleaned_df is not None:
            t1_values = cleaned_df['t1_kb'].unique()
            print(f"  cleaned_data的t1_kb值: {t1_values}")
            target_t1 = t1_values[0]  # 获取cleaned_data的t1_kb值（应该是100.0）
        else:
            target_t1 = 100.0

        # 筛选new_data中t1_kb=target_t1的行
        new_df_filtered = new_df[new_df['t1_kb'] == target_t1].copy()
        dfs.append(new_df_filtered)
        print(f"  new_data: {len(new_df)} 行 -> 筛选t1_kb={target_t1}后: {len(new_df_filtered)} 行")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  合并后: {len(merged)} 行")

    # 去重
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=['model', 'load', 't1_kb', 't2_kb'], keep='first')
    after_dedup = len(merged)
    if before_dedup != after_dedup:
        print(f"  去重: 移除了 {before_dedup - after_dedup} 行重复数据")

    # 在每个load内按t2_kb升序排列
    merged = merged.sort_values(['load', 't2_kb']).reset_index(drop=True)

    return merged


def main():
    """主函数"""
    print("=" * 60)
    print("数据集合并脚本")
    print("=" * 60)
    print()

    models = ['server', 'cache', 'search', 'mine']

    for model in models:
        if model in ['server', 'cache']:
            merged_df = merge_server_or_cache(model)
        else:  # mine, search
            merged_df = merge_mine_or_search(model)

        if merged_df is not None:
            # 保存合并后的数据（命名与cleaned_data保持一致）
            output_file = ALL_DATA_DIR / f"fct_result_{model}.csv"
            merged_df.to_csv(output_file, index=False)
            print(f"  已保存: {output_file}")
            print()

    print("=" * 60)
    print("合并完成!")
    print(f"输出目录: {ALL_DATA_DIR}")
    print("=" * 60)

    # 输出最终统计
    print()
    print("最终统计:")
    for csv_file in sorted(ALL_DATA_DIR.glob("fct_result_*.csv")):
        df = pd.read_csv(csv_file)
        model_name = csv_file.stem.replace("fct_result_", "")
        loads = sorted(df['load'].unique())
        print(f"  {model_name}: {len(df)} 行, loads: {loads}")


if __name__ == "__main__":
    main()
