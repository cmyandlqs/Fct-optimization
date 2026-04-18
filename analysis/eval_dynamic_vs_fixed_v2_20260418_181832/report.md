# 动态阈值 vs 固定基线 评估报告

- 评估阶段: `v2`
- 模型集合: `server, cache, search, mine`
- 负载集合: `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`
- 推理来源: `/home/sikm/Project/Fct-optimization/analysis/inference_results_20260418_170008.json`
- 固定基线定义:
  - server/cache: t1_kb=100, t2_kb=1024
  - search/mine: t1_kb=100, t2_kb=1024
- 说明: 动态真实FCT暂缺，当前使用动态预测FCT进行对比

## 汇总结果

| 维度 | 平均降幅(%) | 中位数降幅(%) | Win Rate | 最优降幅(%) | 最差降幅(%) |
|------|-------------|---------------|----------|-------------|-------------|
| Overall | 4.069 | 2.087 | 24/36 (66.7%) | 26.132 | -15.227 |
| server | 12.107 | 12.563 | 8/9 (88.9%) | 26.132 | -5.456 |
| cache | 6.150 | 2.444 | 9/9 (100.0%) | 21.970 | 0.706 |
| search | -0.592 | -1.864 | 3/9 (33.3%) | 6.405 | -5.100 |
| mine | -1.388 | -0.143 | 4/9 (44.4%) | 7.072 | -15.227 |

## 场景明细（Top 10 按降幅排序）

| model | load | fct_fixed_us | fct_dynamic_pred_us | improve_pct | dynamic_source | baseline_note |
|------|------|--------------|---------------------|-------------|----------------|---------------|
| server | 0.9 | 871681.27 | 643893.99 | 26.132% | inference_json | exact |
| cache | 0.1 | 197227.90 | 153897.62 | 21.970% | inference_json | exact |
| server | 0.8 | 261174.74 | 217532.16 | 16.710% | recomputed | exact |
| server | 0.6 | 75297.33 | 63173.43 | 16.101% | recomputed | exact |
| server | 0.4 | 44409.62 | 38060.30 | 14.297% | recomputed | exact |
| server | 0.5 | 53438.68 | 46725.15 | 12.563% | inference_json | exact |
| server | 0.1 | 25975.50 | 22858.13 | 12.001% | inference_json | exact |
| cache | 0.9 | 2291338.43 | 2050704.41 | 10.502% | inference_json | exact |
| server | 0.7 | 118747.39 | 106765.63 | 10.090% | inference_json | exact |
| cache | 0.2 | 217231.27 | 197262.81 | 9.192% | recomputed | exact |

## 可视化

1. 场景降幅柱状图: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_v2_20260418_181832/plots/improvement_bar.png`
2. 分模型降幅箱线图: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_v2_20260418_181832/plots/improvement_box_by_model.png`
3. 固定 vs 动态FCT折线图: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_v2_20260418_181832/plots/fixed_vs_dynamic_lines.png`

## 产物文件

- case明细: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_v2_20260418_181832/cases.csv`
- summary: `/home/sikm/Project/Fct-optimization/analysis/eval_dynamic_vs_fixed_v2_20260418_181832/summary.json`