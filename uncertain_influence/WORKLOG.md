# uncertain_influence worklog

## 2026-04-11

### 这次做了什么

- 新增分析脚本 `tools/uncertain_influence.py`
- 明确主误差定义为 `error_scalar = 1 - IoU_bev`
- 新增核心图:
  - `uncertainty_vs_error_bin.png`
- 新增辅助图:
  - `uncertainty_vs_error_scatter.png`
  - `distance_uncertainty_error.png`
- 新增输出文件:
  - `analysis.log`
  - `analysis_summary.json`
  - `analysis_report.md`
  - `uncertainty_error_rows.jsonl`

### 这次怎么定义 error

- 主定义:
  - `error_scalar = 1 - IoU_bev`
- 辅助字段:
  - `center_error = L2(pred_xy, gt_xy)`

### 这次怎么做匹配

- 按 frame 读取预测和 GT
- 只在同类别之间匹配
- 按 score 从高到低做 greedy matching
- 每个 GT 最多匹配一次
- 默认 `match_iou_min = 0.1`

### 这次发现的数据问题

- 当前已有的
  - `output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/result.pkl`
  是标准检测输出
- 它没有:
  - `uncertainty_xyz`
  - `pred_uncertainty`
- 所以它本身不能直接支撑 `uncertainty vs error` 分析

### 为了解这个问题改了什么

- 修改 `tools/eval_utils/eval_utils_gaussian.py`
- 去掉了只跑前 20 个 batch 的 debug 截断
- 保证后续重新导出时能拿到完整验证集的 uncertainty 结果

### 现在的结论

- 分析链已经补齐
- 标准 `result.pkl` 不能直接证明 uncertainty 是否有效
- 下一步需要先重新导出:
  - `result_with_uncertainty_epoch_80.pkl`

### 如果后面跑出来没效果

优先按这个顺序排查:

1. uncertainty 是否真的成功导出
2. uncertainty 是否已经解码到米尺度
3. `POST_SCORE_THRESH=0.81` 是否太高, 导致只剩高分样本, 误差分布被截断
4. 是否要改成按 matched predictions 单独分析
5. 是否要增加 `center_error` 作为第二条证据线

## 2026-04-11 结果复核

### 已完成数据分析

- 成功读取:
  - `output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/final_result/data/result_with_uncertainty_epoch_80.pkl`
- 成功输出:
  - `uncertain_influence/results/uncertainty_vs_error_bin.png`
  - `uncertain_influence/results/uncertainty_vs_error_scatter.png`
  - `uncertain_influence/results/distance_uncertainty_error.png`
  - `uncertain_influence/results/analysis_summary.json`
  - `uncertain_influence/results/analysis_report.md`

### 核心结果

- 总预测数: `14619`
- 成功匹配 GT 的预测: `12314`
- Pearson(`uncertainty`, `error_scalar`): `0.4045`
- Spearman(`uncertainty`, `error_scalar`): `0.5356`
- Pearson(`distance`, `uncertainty`): `0.4855`
- Pearson(`distance`, `error_scalar`): `0.3165`
- 5 个 uncertainty bin 的平均误差单调上升
- 当前结论:
  - `effect_supported = True`

### 更细一点的解释

- `matched` 子集上:
  - 平均 uncertainty: `0.2082 m`
  - 平均 error_scalar: `0.1367`
  - `corr(uncertainty, error) = 0.4270`
- `unmatched` 子集上:
  - 平均 uncertainty 更高: `0.2731 m`
  - error_scalar 固定接近 `1.0`

这说明:

- uncertainty 高的预测, 更容易成为高误差或未匹配预测
- 这个趋势不是偶然噪声, 而是有统计支撑的

### 论文呈现建议

- 正文主图:
  - `uncertainty_vs_error_bin.png`
- 正文主文案:
  - uncertainty bin 从低到高时, 平均 error 从 `0.1290` 增长到 `0.4880`
- 正文配套数字:
  - Pearson `0.4045`
  - Spearman `0.5356`
- 补充材料:
  - `uncertainty_vs_error_scatter.png`
  - `distance_uncertainty_error.png`

### 当前判断

- 这组结果是合理的
- 可以支撑:
  - uncertainty 确实在反映误差
  - distance 增大时 uncertainty 也会增大
- 这一节已经基本可以写进论文

## 2026-04-11 图形美化

### 调整内容

- 重做了 `uncertain_influence` 的三张主图样式
- 统一了:
  - 字体大小
  - 轴标题
  - 网格和边框
  - 配色和透明度
- 在图里直接加入了关键统计:
  - Pearson
  - Spearman
  - monotonic bins
- 在 scatter 图里区分了:
  - `matched`
  - `unmatched`

### 当前更适合直接放论文的图

- `uncertainty_vs_error_bin.png`
- `distance_uncertainty_error.png`
