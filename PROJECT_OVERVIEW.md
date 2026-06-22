# GLENet Project Overview

This document summarizes the project and **explicitly separates**:

- **Original GLENet innovations** (from the original GLENet pipeline/repo)
- **Added Gaussian bbox uncertainty work** (this fork)

It is intended as a short attribution and technical overview for open-source release notes.

---

## 1) What this project does

GLENet is a two-stage uncertainty-aware 3D detection framework:

1. **Stage A (Generative uncertainty estimation)**: train a CVAE-style object generator to estimate label uncertainty from repeated stochastic predictions.
2. **Stage B (Probabilistic detector training)**: inject uncertainty labels into 3D detectors (e.g., SECOND / Voxel R-CNN variants) for uncertainty-aware regression/classification/post-processing.

---

## 2) Original GLENet innovations (not my added work)

### 2.1 Generative label uncertainty estimation

For each object, the model learns latent uncertainty with prior/posterior distributions and KL regularization:

$$
\mathcal{L}_{KL} = D_{KL}(q_\psi(z\mid x,y)\|p_\phi(z\mid x))
$$

A practical training objective combines regression, direction, and KL terms:

$$
\mathcal{L}_{A}=\mathcal{L}_{reg}+\lambda_{dir}\mathcal{L}_{dir}+\lambda_{kl}\mathcal{L}_{KL}
$$

### 2.2 Uncertainty label construction by repeated stochastic inference

For one object, run multiple stochastic predictions and compute variance:

$$
u = \operatorname{Var}(\{\hat y^{(k)}\}_{k=1}^{K})
$$

Then write this uncertainty vector back into dataset info/dbinfo files.

### 2.3 Uncertainty-guided detector optimization

In KL-label heads, predicted uncertainty and label uncertainty are jointly used in regression weighting (code-level variants exist in anchor and ROI heads). A common form is:

$$
\mathcal{L}_{reg}^{KL}
= e^{-s_p}\,\ell_{reg}
+ e^{(s_l-s_p)}
- \frac{1}{2}(s_l-s_p)
$$

where $s_p$ is predicted log-variance and $s_l$ comes from label uncertainty.

### 2.4 Uncertainty-aware post-processing

The custom NMS path supports variance-aware box fusion (`new_nms_gpu`) instead of pure hard suppression.

---

## 3) Added Gaussian bbox uncertainty work

The following items describe the added prediction-time 3D bbox uncertainty extension.

### 3.1 Gaussian uncertainty extension for Voxel R-CNN

1. Added detector class:
   - `pcdet/models/detectors/voxel_rcnn_gaussian.py` (`VoxelRCNNGaussian`)
2. Added/used Gaussian uncertainty head implementation in:
   - `pcdet/models/roi_heads/voxelrcnn_gaussian_iou_head.py`
3. Added config for Gaussian training:
   - `tools/cfgs/kitti_models/GLENet_VR_gaussian.yaml`
4. Added uncertainty visualization tool:
   - `tools/vis_pic_uncertain.py`

### 3.2 Uncertainty head behavior (as documented)

- Predict both regression output (`rcnn_reg`) and log-variance (`rcnn_reg_std`).
- Use an uncertainty MLP gate to modulate classification confidence:

$$
\text{rcnn\_cls} = \sigma(\text{ori\_rcnn\_cls}) \cdot \text{uncertainty\_coefficient}
$$

### 3.3 Aleatoric Gaussian regression loss (as documented)

$$
\mathcal{L}_{reg}=\frac{1}{2}e^{-s}\|y_{gt}-y_{pred}\|^2 + \frac{1}{2}s
$$

- $s$ is predicted log-variance (`rcnn_reg_std`).
- Implemented stability clamp:

$$
s \in [-5, 5]
$$

### 3.4 Gaussian post-processing output (as documented)

- Extract uncertainty from prediction path and return it as `pred_uncertainty` in post-processing output.

---

## 4) Training workflow

## 4.1 Original GLENet pipeline (two-stage)

1. Train uncertainty generator (`cvae_uncertainty`, multi-fold).
2. Run repeated prediction per fold.
3. Map per-object variance and inject into infos/dbinfos.
4. Train probabilistic detector with updated infos/dbinfos.

Representative commands (from project docs):

```bash
cd cvae_uncertainty
GPU_IDS="${GPU_IDS:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
for iter in `seq 0 9`; do
  sed "s@# FOLD_IDX: 0@FOLD_IDX: ${iter}@" cfgs/exp20_gen_ori.yaml > cfgs/exp20_gen.yaml
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" bash scripts/dist_train.sh "${NUM_GPUS}" --cfg_file cfgs/exp20_gen.yaml --extra_tag fold_${iter}
  sh predict.sh exp20_gen fold_${iter} 400 0
done
python mapping_uncertainty.py
python change_gt_infos.py

cd ../tools
python train.py --cfg_file cfgs/kitti_models/GLENet_VR.yaml
```

## 4.2 Gaussian workflow

Training:

```bash
conda activate <env_name>
cd tools
export CUDA_VISIBLE_DEVICES=<gpu_ids>
bash scripts/dist_train.sh 2 --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml --batch_size 4 --epochs 80 --extra_tag full_gaussian_run
```

Evaluation:

```bash
export CUDA_VISIBLE_DEVICES=<gpu_ids>
bash scripts/dist_test.sh 2 --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml --batch_size 4 --ckpt ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth
```

Single-GPU test option:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml --batch_size 4 --ckpt ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth
```

---

## 5) Visualization and monitoring

From `README_gaussian.md`:

- Logs: check `log_train_xxxx.txt` in output directory.
- TensorBoard:

```bash
tensorboard --logdir ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/tensorboard
```

- Visualization command:

```bash
python vis_pic_uncertain.py --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml \
    --ckpt ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth \
    --uncertainty_scale 1.0 \
    --save_dir ../visualization_outputs/vis_full_uncertainty
```

---

## 6) Attribution note

- **Original GLENet innovations** are listed as project/pipeline innovations.
- **Added Gaussian work** is limited to the prediction-time bbox uncertainty extension described above.
- No additional modules, formulas, or engineering changes are claimed as my own beyond that scope.
