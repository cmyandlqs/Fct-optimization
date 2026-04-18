#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查当前 Python 环境是否满足项目依赖，输出各库版本。
"""

import sys

REQUIRED = {
    "torch":            "PyTorch (深度学习框架)",
    "pandas":           "pandas (数据处理)",
    "numpy":            "numpy (数值计算)",
    "sklearn":          "scikit-learn (标准化器、数据划分)",
    "matplotlib":       "matplotlib (绑图)",
    "seaborn":          "seaborn (统计可视化)",
    "joblib":           "joblib (保存 scaler)",
    "swanlab":          "swanlab (实验追踪与可视化)",
}

# import 名 -> pip 包名 的映射 (不一致的)
PIP_NAME = {
    "sklearn": "scikit-learn",
    "swanlab": "swanlab",
}

def main():
    print(f"Python:  {sys.version}")
    print(f"路径:    {sys.executable}")
    print("=" * 50)

    missing = []
    for mod, desc in REQUIRED.items():
        try:
            imported = __import__(mod)
            ver = getattr(imported, "__version__", "未知")
            print(f"  [OK]  {desc:40s} {ver}")
        except ImportError:
            pip_name = PIP_NAME.get(mod, mod)
            missing.append(pip_name)
            print(f"  [缺失] {desc:40s} 未安装")

    print("=" * 50)
    if missing:
        print("\n安装缺失的库:")
        print(f"  pip install {' '.join(missing)}")
    else:
        print("\n所有依赖已满足。")


if __name__ == "__main__":
    main()
