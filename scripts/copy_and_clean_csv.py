#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 raw_data 的 CSV 复制到 cleaned_data，只保留有用列。

保留: model, load, t1_kb, t2_kb, avg_fct, flow_count
删除: file, topo, scenario, t1_bytes, t2_bytes, skipped_lines
"""

import shutil
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "dataset" / "raw_data"
DST_DIR = Path(__file__).resolve().parent.parent / "dataset" / "cleaned_data"
FILES = ["fct_result_cache.csv", "fct_result_server.csv",
         "fct_result_search.csv", "fct_result_mine.csv"]

# 保留的列（按原始顺序）
KEEP_COLS = ["model", "load", "t1_kb", "t2_kb", "avg_fct"]

DST_DIR.mkdir(parents=True, exist_ok=True)

for name in FILES:
    src = SRC_DIR / name
    dst = DST_DIR / name
    shutil.copy2(src, dst)

    with open(dst, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        lines = f.readlines()

    # 建立列名 -> 索引的映射
    cols = header.split(",")
    col_idx = {col: i for i, col in enumerate(cols)}

    # 验证必需列都存在
    missing = [c for c in KEEP_COLS if c not in col_idx]
    if missing:
        print(f"  [警告] {name}: 缺少列 {missing}")
        continue

    # 写新文件
    new_header = ",".join(KEEP_COLS) + "\n"
    new_lines = []
    for line in lines:
        parts = line.strip().split(",")
        row = {cols[i]: parts[i] for i in range(len(cols))}
        new_parts = []
        for c in KEEP_COLS:
            val = row[c]
            if c == "avg_fct":
                # 保留两位小数
                val = f"{float(val):.2f}"
            new_parts.append(val)
        new_lines.append(",".join(new_parts) + "\n")

    with open(dst, "w", encoding="utf-8") as f:
        f.write(new_header)
        f.writelines(new_lines)

    print(f"  {name}: 12列 → 6列 ({len(new_lines)} 行)")

print(f"\n完成，输出目录: {DST_DIR}")
