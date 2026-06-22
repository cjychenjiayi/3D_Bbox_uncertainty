# Gaussian Uncertainty Implementation Notes

## 2026-02-04 - Gaussian Uncertainty Implementation

### 1. Model Architecture Modifications
- **File:** `pcdet/models/roi_heads/voxelrcnn_kl_label_iou_head.py`
- **Class:** `VoxelRCNNKLLabelIoUHead`
- **Changes:**
    - Augmented the ROI Head to include a variance prediction branch (`reg_std_layer`).
    - Structure: `Linear` (Output 7 dim: $x, y, z, dx, dy, dz, \theta$) -> `BatchNorm1d` -> `ReLU` -> `Linear(64)` -> `BatchNorm1d` -> `ReLU` -> `Linear(1)` -> `Sigmoid`.
        - Note: The primary output `rcnn_reg_std` (Log Variance) comes from the first `Linear` layer.
        - The subsequent layers produce a coefficient used to modulate the classification score (uncertainty-aware classification), ensuring high uncertainty predictions have lower confidence scores.
    - Added `init_reg_std_layer_weights` method:
        - Initialized `reg_std_layer.weight` with mean=0, std=0.0001.
        - **Crucial Fix:** Initialized `reg_std_layer.bias` to `0.1` (positive value) to ensure initial variance predictions are not effectively zero (which would cause infinite loss), promoting stability early in training.

### 2. Loss Function Implementation
- **File:** `pcdet/models/roi_heads/voxelrcnn_kl_label_iou_head.py`
- **Method:** `get_box_reg_layer_loss`
- **Changes:**
    - Replaced the previous experimental KL-divergence loss with **Aleatoric Uncertainty Loss** (Unsupervised).
    - **Formula:**
    $$ \mathcal{L}_{reg} = \frac{1}{2} e^{-s} ||y_{gt} - y_{pred}||^2 + \frac{1}{2} s $$
    where $s$ is the predicted log variance (`rcnn_reg_std`).
    - **Stability Constraints:**
        - Clamped `rcnn_reg_std` (Log Variance) to the range `[-5.0, 5.0]` to prevent numerical instability (explosion of $e^{-s}$ or $s$).
        - Removed dependency on `label_var_log` (Ground Truth Uncertainty), making the uncertainty learning purely data-driven (unsupervised).

### 3. Configuration Updates
- **File:** `tools/cfgs/kitti_models/GLENet_VR_gaussian.yaml`
- **Changes:**
    - Created a dedicated configuration for the Gaussian model.
    - Set `MODEL.NAME` to `VoxelRCNNGaussian`.
    - **Optimization:**
        - Reverted to `adam_onecycle` optimizer for optimal convergence.
        - Set `BATCH_SIZE_PER_GPU: 4`.
        - Adjusted `LR` to `0.01` (standard Pytorch/OpenPCDet setting for this schedule).

### 4. Debugging & Stabilization
- **Issues Resolved:**
    - Fixed `ValueError: loaded state dict has a different number of parameter groups` by ensuring clean restart.
    - Fixed **Loss Explosion** (Loss > 50) observed with standard regression loss + exp terms.
    - Fixed **NameError** in `tb_dict` logging.
    - Cleaned up duplicate method definitions in source code.

## 2026-04-11 - Uncertainty-Error Analysis Pipeline

### 5. uncertain_influence Analysis Pipeline
- **Files:**
    - `tools/uncertain_influence.py`
    - `uncertain_influence/README.md`
    - `uncertain_influence/WORKLOG.md`
- **Changes:**
    - Added a dedicated analysis pipeline for validating whether predicted uncertainty correlates with actual detection error.
    - Defined the primary scalar error as:
      - `error_scalar = 1 - IoU_bev`
    - Added the main plot requested for analysis:
      - `uncertainty_vs_error_bin.png`
    - Added auxiliary plots for interpretation:
      - uncertainty vs error scatter
      - distance vs uncertainty / error
    - Added structured outputs:
      - `analysis.log`
      - `analysis_summary.json`
      - `analysis_report.md`
      - `uncertainty_error_rows.jsonl`

### 6. Gaussian Export Fix For Analysis
- **File:** `tools/eval_utils/eval_utils_gaussian.py`
- **Changes:**
    - Removed the temporary debug cutoff that stopped Gaussian evaluation after 20 batches.
    - This ensures the uncertainty export can cover the full validation set, which is necessary for trustworthy `uncertainty vs error` analysis.

### 7. Data Finding Logged
- The currently existing `result.pkl` under:
  - `output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/result.pkl`
  does **not** contain uncertainty fields.
- Therefore, it is not sufficient by itself to validate the uncertainty-effect relationship.
- The intended input for the new analysis is:
  - `result_with_uncertainty_epoch_*.pkl`

### 8. uncertainty_influence Result Validation
- **Input used:**
  - `output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/final_result/data/result_with_uncertainty_epoch_80.pkl`
- **Observed results:**
  - `num_predictions = 14619`
  - `matched_predictions = 12314`
  - `pearson(uncertainty, error) = 0.4045`
  - `spearman(uncertainty, error) = 0.5356`
  - `pearson(distance, uncertainty) = 0.4855`
  - `effect_supported = True`
- **Interpretation:**
  - The Gaussian uncertainty output shows a clear positive correlation with BEV error.
  - The 5-bin `uncertainty vs error` curve is monotonically increasing, which supports the intended paper claim.
