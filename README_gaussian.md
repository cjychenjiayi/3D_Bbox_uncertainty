# Gaussian Uncertainty Training Guide (GLENet)

This document outlines the steps to train and evaluate the GLENet model with the newly implemented Gaussian Uncertainty Head (Aleatoric Uncertainty). It also details the technical modifications made to the original GLENet architecture.

## 1. Environment Setup

Ensure the environment is active:
```bash
conda activate <env_name>
cd tools
```

## 2. Configuration

The configuration file is located at:
`tools/cfgs/kitti_models/GLENet_VR_gaussian.yaml`

Key settings for Full Training:
- **Batch Size:** 4 per GPU.
- **GPUs:** 2 GPUs.
- **Total Batch Size:** 8.
- **Optimizer:** Adam OneCycle.
- **Learning Rate:** 0.01 (Standard scaling).

## 3. Training

We use `dist_train.sh` for multi-GPU training.

**Command:**
```bash
export CUDA_VISIBLE_DEVICES=<gpu_ids>
bash scripts/dist_train.sh 2 --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml --batch_size 4 --epochs 80 --extra_tag full_gaussian_run
```

- `scripts/dist_train.sh 2`: Tells the script to use 2 processes.
- `--batch_size 4`: Batch size **per GPU**.
- `--extra_tag full_gaussian_run`: Creates a unique output folder `output/kitti_models/GLENet_VR_gaussian/full_gaussian_run` to store logs and checkpoints.

## 4. Evaluation

After training is complete, evaluate the model on the validation set.

**Command:**
```bash
export CUDA_VISIBLE_DEVICES=<gpu_ids>

# Use dist_test.sh for parallel evaluation (faster) or test.py for single card
bash scripts/dist_test.sh 2 --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml --batch_size 4 --ckpt ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth
```

Or for single GPU evaluation:
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml --batch_size 4 --ckpt ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth
```

## 5. Output Monitoring

- **Logs:** Check `log_train_xxxx.txt` in the output directory.
- **Tensorboard:**
  ```bash
  tensorboard --logdir ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/tensorboard
  ```
- **Loss Expectation:** The Total Loss should stabilize below 5.0 after a few epochs. The `rcnn_loss_reg_log` component represents the uncertainty regularization.

## 6. Technical Implementation Details

This section details the modifications made to the codebase to support Gaussian Uncertainty Estimation.

### A. Modified Files
The following files were created or modified:

1.  **Architecture**:
    -   `pcdet/models/detectors/voxel_rcnn_gaussian.py`: New detector class `VoxelRCNNGaussian` inheriting from `VoxelRCNN`.
    -   `pcdet/models/roi_heads/voxelrcnn_gaussian_iou_head.py`: Implements the Gaussian uncertainty head `VoxelRCNNGaussianIoUHead`.

2.  **Configuration**:
    -   `tools/cfgs/kitti_models/GLENet_VR_gaussian.yaml`: Configuration file enabling the Gaussian model.

3.  **Visualization**:
    -   `tools/vis_pic_uncertain.py`: Tool to visualize 3D bounding boxes with their predicted uncertainty.

### B. Uncertainty Head Architecture
The uncertainty estimation is implemented in the `VoxelRCNNGaussianIoUHead`.

1.  **Regression & Uncertainty Branch**:
    -   The head predicts both the box regression parameters (`rcnn_reg`) and their associated log-variance (`rcnn_reg_std`).
    -   **Layer Structure**:
        ```python
        self.reg_std_layer = nn.Linear(pre_channel, box_coder.code_size * num_class) # Predicts log variance (s)
        ```
    -   **Confidence Modulation**:
        -   The uncertainty information is also passed through a small MLP (`FC -> BN -> ReLU -> FC -> Sigmoid`) to produce a modulation coefficient.
        -   This coefficient scales the classification score, effectively lowering the confidence of uncertain predictions.
        ```python
        rcnn_cls = sigmoid(ori_rcnn_cls) * uncertainty_coefficient
        ```

2.  **Aleatoric Uncertainty Loss**:
    -   We replaced the standard Smooth-L1 Loss with a negative log-likelihood loss for a Gaussian distribution.
    -   **Loss Formula**:
        $$ \mathcal{L}_{reg} = \frac{1}{2} e^{-s} ||y_{gt} - y_{pred}||^2 + \frac{1}{2} s $$
        where $s$ is the predicted log-variance (`rcnn_reg_std`).
    -   **Interpretation**:
        -   The first term penalizes regression errors, weighted by the inverse variance ($e^{-s}$). When uncertainty ($s$) is high, the weight decreases.
        -   The second term ($0.5 s$) penalizes high uncertainty to prevent the model from predicting infinite variance for all samples.
    -   **Stability**: The log-variance $s$ is clamped to the range `[-5.0, 5.0]` to prevent numerical instability.

### C. Post-Processing
In `VoxelRCNNGaussian.post_processing`:
-   The uncertainty predictions (`batch_box_std_preds`) are extracted alongside box predictions.
-   These are passed through Non-Maximum Suppression (NMS) and returned in the final dictionary key `pred_uncertainty`.

## 7. Visualization

A dedicated visualization tool is provided to inspect the uncertainty predictions.

**Command:**
```bash
python vis_pic_uncertain.py --cfg_file cfgs/kitti_models/GLENet_VR_gaussian.yaml \
    --ckpt ../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth \
    --uncertainty_scale 1.0 \
    --save_dir ../visualization_outputs/vis_full_uncertainty
```

- **`--uncertainty_scale`**: Adjusts the visual size/transparency of the uncertainty indicators.
