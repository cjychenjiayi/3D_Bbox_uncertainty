# uncertain_influence

这一组内容专门回答一个问题:

`bbox uncertainty` 到底有没有真的反映检测误差?

## 核心定义

推荐先用最简单的版本:

- `error_scalar = 1 - IoU_bev`

解释:

- 如果预测框和 GT 在 BEV 上越接近, `IoU_bev` 越大。
- 那么 `1 - IoU_bev` 越小, 代表误差越小。
- 这样它天然落在 `[0, 1]` 左右, 很适合做趋势图。

脚本里也同时保留了 `center_error` 作为辅助字段, 但主图默认只看 `error_scalar`。

## 唯一核心图

最关键的是:

- `uncertainty_vs_error_bin.png`

它的形式是:

- x 轴: uncertainty 分成 5 个 bin
- y 轴: 每个 bin 的平均 `error_scalar`

目标趋势是:

- uncertainty 越高, 平均 error 越高

只要这条趋势成立, 这一节就比较站得住。

## 文件结构

- `tools/uncertain_influence.py`
  - 主分析脚本
- `uncertain_influence/README.md`
  - 方法说明
- `uncertain_influence/WORKLOG.md`
  - 这次实现和修改记录
- `uncertain_influence/results/`
  - 分析输出目录

## 推荐运行方式

先导出带 uncertainty 的预测结果:

```bash
conda activate <env_name>

cd tools
python test_gaussian.py \
  --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml \
  --ckpt ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth \
  --save_to_file
```

然后做 `uncertain_influence` 分析:

```bash
cd ..
python tools/uncertain_influence.py \
  --pred_pkl output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/final_result/data/result_with_uncertainty_epoch_80.pkl \
  --save_dir uncertain_influence/results
```

## 当前注意事项

- 标准 `result.pkl` 是常规检测输出, 不带 uncertainty 字段。
- 所以如果直接拿它做 `uncertainty vs error`, 是算不出来的。
- 必须用 `result_with_uncertainty_epoch_*.pkl` 这种带 `uncertainty_xyz` 的导出结果。

## 示例结果

基于:

- `output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/final_result/data/result_with_uncertainty_epoch_80.pkl`

在:

- `uncertain_influence/results/`

一次验证运行得到的核心统计是:

- `num_predictions = 14619`
- `matched_predictions = 12314`
- `pearson(uncertainty, error) = 0.4045`
- `spearman(uncertainty, error) = 0.5356`
- `pearson(distance, uncertainty) = 0.4855`
- `monotonic bins = 4/4`
- `effect_supported = True`

对应的 5-bin 主图里, 平均 error 随 uncertainty 单调上升:

- `[0.103, 0.156] -> 0.1290`
- `[0.156, 0.182] -> 0.1513`
- `[0.182, 0.221] -> 0.2255`
- `[0.221, 0.278] -> 0.3701`
- `[0.278, 0.570] -> 0.4880`

这说明该 uncertainty 输出至少不是随机噪声, 而是和真实误差存在明显正相关。

## 论文里建议放什么

### 正文最推荐

1. `uncertainty_vs_error_bin.png`
   - 这是最核心的一张图
   - 最容易支撑“uncertainty 越大, error 越大”

2. 一个简短表格或正文一句话
   - `Pearson = 0.4045`
   - `Spearman = 0.5356`
   - `5/5 bins monotonic increasing` 或者 `4/4 monotonic steps`

### 正文可选第二张

- `distance_uncertainty_error.png` 左图中的 `distance vs uncertainty`
  - 它能支持“远距离目标更不确定”这个直观现象

### 更适合放补充材料

- `uncertainty_vs_error_scatter.png`
  - 信息量大, 但视觉上更乱
  - 更适合 supplementary

## 结果怎么解读更稳

- 当前结果可以作为 uncertainty-error 正相关的初步证据。
- 但散点图中有一条 `error = 1` 的横线, 这对应很多 unmatched predictions。
- 所以论文正文建议用 bin plot 做主证据, 不建议单独把 scatter 当唯一证据。
- 如果后续你要更严谨, 可以再加一个 matched-only 版本放补充材料。
