# 动态阈值 vs 固定基线 评估报告

- 推理结果来源: `/home/sikm/Project/Fct-optimization/analysis/inference_results_20260418_170008.json`
- 固定基线定义:
  - server/cache: t1_kb=100, t2_kb=1024
  - search/mine: t1_kb=100, t2_kb=1024
- 说明: 动态真实FCT暂缺，当前使用动态预测FCT进行对比

## 汇总结果

| 维度 | 平均降幅(%) | 中位数降幅(%) | Win Rate | 最优降幅(%) | 最差降幅(%) |
|------|-------------|---------------|----------|-------------|-------------|
| Overall | 3.719 | 2.087 | 13/20 (65.0%) | 26.132 | -15.227 |
| server | 11.066 | 12.001 | 4/5 (80.0%) | 26.132 | -5.456 |
| cache | 8.241 | 5.869 | 5/5 (100.0%) | 21.970 | 0.706 |
| search | -1.227 | -2.333 | 2/5 (40.0%) | 4.039 | -5.100 |
| mine | -3.205 | -0.143 | 2/5 (40.0%) | 3.825 | -15.227 |

## 场景明细（Top 8 按降幅排序）

| model | load | fct_fixed_us | fct_dynamic_pred_us | improve_pct | baseline_note |
|------|------|--------------|---------------------|-------------|---------------|
| server | 0.9 | 871681.27 | 643893.99 | 26.132% | exact |
| cache | 0.1 | 197227.90 | 153897.62 | 21.970% | exact |
| server | 0.5 | 53438.68 | 46725.15 | 12.563% | exact |
| server | 0.1 | 25975.50 | 22858.13 | 12.001% | exact |
| cache | 0.9 | 2291338.43 | 2050704.41 | 10.502% | exact |
| server | 0.7 | 118747.39 | 106765.63 | 10.090% | exact |
| cache | 0.7 | 889598.78 | 837391.82 | 5.869% | exact |
| search | 0.1 | 419921.73 | 402960.73 | 4.039% | exact |

## 可视化

1. 场景降幅柱状图: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_20260418_173758/plots/improvement_bar.png`
2. 分模型降幅箱线图: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_20260418_173758/plots/improvement_box_by_model.png`
3. 固定 vs 动态FCT折线图: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_20260418_173758/plots/fixed_vs_dynamic_lines.png`

## 产物文件

- case明细: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_20260418_173758/cases.csv`
- summary: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_20260418_173758/summary.json`