# 动态阈值 vs 固定基线：评估方案（简化版）

## 1. 评估目标

验证当前动态阈值算法相对原有固定阈值策略是否有收益，核心关注：
1. 动态阈值对应的 FCT 预测值是否低于固定基线。
2. 降低幅度（百分比）是多少。
3. 收益是否在不同模型/负载下稳定。

> 说明：当前动态阈值只有预测 FCT，没有真实实测 FCT；真实值列先留空，后续补实验数据。

---

## 2. 对比对象

## 2.1 动态阈值方案（待评估）
1. 输入：`(model, load)`。
2. 输出：动态最优阈值（`optimal_t1_kb` 或 `optimal_t2_kb`）和 `predicted_fct_us`。
3. 数据来源：`analysis/inference_results_<timestamp>.json`。

## 2.2 固定阈值基线（对照组）
1. `server/cache`：固定 `t1_kb=100`，`t2_kb=1024`。
2. `search/mine`：固定 `t2_kb=1024`，`t1_kb=100`。

---

## 3. 场景集合

建议按统一场景集合评估：
1. 模型：`server / cache / search / mine`
2. 负载：`0.1, 0.3, 0.5, 0.7, 0.9`
3. 共 `4 × 5 = 20` 个场景

---

## 4. 指标定义

## 4.1 单场景降幅（核心指标）

对每个场景 \(s\)：

\[
\text{improve\_pct}(s)=\frac{FCT_{\text{fixed}}(s)-FCT_{\text{dynamic,pred}}(s)}{FCT_{\text{fixed}}(s)}\times 100\%
\]

符号说明：
1. \(FCT_{\text{fixed}}(s)\)：固定阈值基线 FCT（真实值，来自数据表或历史测量）。
2. \(FCT_{\text{dynamic,pred}}(s)\)：动态阈值预测 FCT（当前已有）。

解释：
1. `> 0`：动态方案优于固定基线。
2. `= 0`：两者持平。
3. `< 0`：动态方案劣于固定基线。

## 4.2 汇总指标（建议同时展示）
1. **平均降幅**：所有场景 `improve_pct` 的平均值。
2. **中位数降幅**：抗异常值，更稳健。
3. **Win Rate**：`improve_pct > 0` 的场景占比。
4. **最优/最差场景降幅**：用于定位收益边界。
5. **分模型平均降幅**：server/cache/search/mine 各自统计。

---

## 5. 结果表格模板

## 5.1 场景明细表（主表）

| model | load | fixed_t1_kb | fixed_t2_kb | dynamic_t1_kb | dynamic_t2_kb | fct_fixed_us | fct_dynamic_pred_us | fct_dynamic_real_us(可空) | improve_pct_pred_vs_fixed | improve_pct_real_vs_fixed(可空) |
|------|------|-------------|-------------|---------------|---------------|--------------|---------------------|---------------------------|---------------------------|----------------------------------|
| cache | 0.1 | 100 | 1024 | 205.49 | 1024 | ... | ... |  | ... |  |

字段说明：
1. `fct_dynamic_real_us`：你后续补真实实验值。
2. `improve_pct_real_vs_fixed`：后续补真实降幅。
3. 当前阶段主要看 `improve_pct_pred_vs_fixed`。

## 5.2 汇总表（建议）

| 维度 | 平均降幅(%) | 中位数降幅(%) | Win Rate | 最优降幅(%) | 最差降幅(%) |
|------|-------------|---------------|----------|-------------|-------------|
| Overall | ... | ... | ... | ... | ... |
| server | ... | ... | ... | ... | ... |
| cache | ... | ... | ... | ... | ... |
| search | ... | ... | ... | ... | ... |
| mine | ... | ... | ... | ... | ... |

---

## 6. 可视化建议（简洁且有说服力）

1. **柱状图（核心）**：
   - x 轴：`model-load`（20 个场景）
   - y 轴：`improve_pct_pred_vs_fixed`
   - 0 轴以上为收益，以下为退化

2. **分模型箱线图**：
   - x 轴：模型
   - y 轴：降幅百分比
   - 展示不同模型收益分布和稳定性

3. **固定 vs 动态 FCT 对比图（分模型）**：
   - x 轴：load
   - 两条线：`fct_fixed_us` 与 `fct_dynamic_pred_us`
   - 直观看每个负载下是否整体下降

4. **Win/Loss 饼图或柱状图（可选）**：
   - 提升场景数 vs 未提升场景数

---

## 7. 输出产物建议

1. `analysis/eval_dynamic_vs_fixed_cases_<timestamp>.csv`
2. `analysis/eval_dynamic_vs_fixed_summary_<timestamp>.json`
3. `analysis/eval_dynamic_vs_fixed_plots/*.png`
4. `analysis/eval_dynamic_vs_fixed_report_<timestamp>.md`

---

## 8. 备注（本轮暂不做）

1. 暂不与“离散最优阈值（oracle）”对比；仅在后续扩展评估中加入。
2. 当前结论属于“预测层收益”，最终建议补充动态阈值真实实验值，形成“真实收益闭环”。
3. 若后续做正式报告，建议增加置信区间/显著性检验（例如 bootstrap）。

