
import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import copy

import numpy as np
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

from pcdet.config import cfg as cfg1, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils, calibration_kitti

# Use a separate config object for the second model to avoid conflicts
from easydict import EasyDict

def cfg_from_yaml_file_to_dict(cfg_file, cfg_dict):
    import yaml
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

    # Merge into the provided EasyDict or dict
    def merge_new_config(config, new_config):
        for key, val in new_config.items():
            if not isinstance(val, dict):
                config[key] = val
                continue
            if key not in config:
                config[key] = EasyDict()
            merge_new_config(config[key], val)

    merge_new_config(cfg_dict, new_config)
    return cfg_dict

# Re-use visualization utils from vis_pic_uncertain.py (copying the relevant parts to be standalone)
def to_cpu_numpy(x):
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    # Debug print if shape is unexpected
    if len(qs.shape) != 2 or qs.shape[1] != 2:
        print(f"Warning: Invalid qs shape {qs.shape}")
        return image

    for k in range(0, 4):
        # Reflected parallel lines
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def draw_projected_box3d_uncertainty(image, qs, uncertainty, color=(0, 255, 0), thickness=2):
    # This is a placeholder for uncertainty visualization if needed
    # For now, we will just draw the box with a specific color or thickness
    return draw_projected_box3d(image, qs, color, thickness)


def main():
    # -------------------------------------------------------------------------
    # Hardcoded Configuration for Reproducibility
    # -------------------------------------------------------------------------
    # Detect Project Root (assuming script is in GLENet/tools/compare_models.py)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Ensure we run from 'tools' directory for relative config paths to work correctyly locally
    # (pcdet configs often use relative paths for _BASE_CONFIG_)
    os.chdir(PROJECT_ROOT / 'tools')
    print(f"Working Directory set to: {os.getcwd()}")

    # Model 1: Baseline (Pure Detection)
    CFG1_PATH = PROJECT_ROOT / 'tools/cfgs/kitti_models/GLENet_VR.yaml'
    CKPT1_PATH = PROJECT_ROOT / 'provide/GLENet_VR.pth'
    TAG1 = 'GLENet_VR'

    # Model 2: Gaussian (Uncertainty)
    CFG2_PATH = PROJECT_ROOT / 'tools/cfgs/kitti_models/GLENet_VR_gaussian.yaml'
    CKPT2_PATH = PROJECT_ROOT / 'output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_79.pth'
    TAG2 = 'GLENet_G'

    # Output Settings
    SAVE_DIR = PROJECT_ROOT / 'visualization_outputs' / 'vis_compare'
    SCORE_THRESH = 0.3
    MAX_SAMPLES = -1 # Set to -1 for all, or integer (e.g. 10) for quick test

    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='Compare two models (Hardcoded Paths)')
    parser.add_argument('--max_samples', type=int, default=MAX_SAMPLES, help='Override max samples')
    args = parser.parse_args()

    # Logger
    logger = common_utils.create_logger()
    logger.info(f"Project Root: {PROJECT_ROOT}")

    # -------------------------------------------------------------------------
    # Load Model 1
    # -------------------------------------------------------------------------
    logger.info(f"Loading Model 1: {TAG1}")
    cfg_from_yaml_file(str(CFG1_PATH), cfg1)

    # Force Val Dataset/Test Split
    cfg1.DATA_CONFIG.DATA_SPLIT['test'] = 'val'

    dataset, dataloader, sampler = build_dataloader(
        dataset_cfg=cfg1.DATA_CONFIG,
        class_names=cfg1.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1, logger=logger, training=False
    )

    model1 = build_network(model_cfg=cfg1.MODEL, num_class=len(cfg1.CLASS_NAMES), dataset=dataset)
    model1.load_params_from_file(filename=str(CKPT1_PATH), logger=logger, to_cpu=True)
    model1.cuda()
    model1.eval()

    # -------------------------------------------------------------------------
    # Load Model 2
    # -------------------------------------------------------------------------
    logger.info(f"Loading Model 2: {TAG2}")

    # We need a fresh config for model 2.
    # Since pcdet uses a global 'cfg', we have to be careful.
    # We will manually load the second config into a new EasyDict
    from pcdet.config import cfg as cfg_template
    import copy
    cfg2 = copy.deepcopy(cfg_template)
    cfg2 = cfg_from_yaml_file_to_dict(str(CFG2_PATH), cfg2)

    # [Fix] Ensure the model is built with the correct config
    # Some pcdet versions might rely on the global cfg during build_network if not passed explicitly in sub-modules
    # But usually providing model_cfg is enough.

    model2 = build_network(model_cfg=cfg2.MODEL, num_class=len(cfg1.CLASS_NAMES), dataset=dataset)
    model2.load_params_from_file(filename=str(CKPT2_PATH), logger=logger, to_cpu=True)
    model2.cuda()
    model2.eval()

    # -------------------------------------------------------------------------
    # Inference Loop
    # -------------------------------------------------------------------------
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, batch_dict in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            if args.max_samples != -1 and idx >= args.max_samples:
                break
            # Warning: batch_dict is modified in-place. We need a deep copy for the second model?
            # Actually, tensors are on CPU/GPU. Deepcopying tensors might be slow but necessary if model modifies them.
            # VoxelRCNN usually adds keys. It doesn't modify input points/voxels destructively mostly.
            # But let's be safe.

            # Since deepcopy of tensors is heavy, we just reload or re-collate if possible.
            # But dataloader yields one.
            # Let's try just copying the dict structure, assuming tensors aren't mutated in a way that breaks the second model.

            batch_dict1 = batch_dict
            # We can't easily deepcopy the batch if it has GPU tensors already? No, load_data_to_gpu happens later.
            # Wait, dataloader output is CPU.

            # Simple shallow copy of dict is not enough if values (dicts/lists) are modified.
            # Let's use a function to collate it again? No, we don't have the raw data here easily.

            # Strategy:
            # 1. Load data to gpu for model 1
            # 2. Run model 1
            # 3. Restore or Re-load for model 2?
            # Actually, most keys added are specific to the model instance (pred keys).
            # The input keys (points, voxels, coords) are usually read-only.
            # BUT: 'batch_dict' often accumulates 'rois', 'roi_labels' etc which might conflict if both models write to same keys.

            # Solution: Create a fresh dict for model 2 with same input tensors.
            # Since we are single-batch (batch_size=1), the input structure is simple.

            load_data_to_gpu(batch_dict1)

            # Create a shallow copy for Model 2 sharing the same tensors
            batch_dict2 = {}
            for k, v in batch_dict1.items():
                batch_dict2[k] = v

            # --- Model 1 Inference ---
            with torch.no_grad():
                pred_dicts1, _ = model1.forward(batch_dict1)

            # --- Model 2 Inference ---
            # Reload data for model 2 just in case model 1 modified it in place (e.g. rois)
            # Since we only have batch_dict, let's just make sure we don't reuse the modified batch_dict1
            # But tensors are shared.
            # If Model 1 modifies tensors in place, we are in trouble.
            # Usually inference doesn't modify input tensors (points, voxels).

            with torch.no_grad():
                pred_dicts2, _ = model2.forward(batch_dict2)


            # ---------------------------------------------------------------------
            # Visualization
            # ---------------------------------------------------------------------

            frame_id = batch_dict['frame_id'][0]
            if not isinstance(frame_id, str):
                frame_id = f"{frame_id:06d}"

            # Use frame_id (string) to get image, not idx (int)
            # frame_id is e.g. '000001'

            # Get Image
            # Using KittiDataset internals
            if hasattr(dataset, 'get_image'):
                image = dataset.get_image(frame_id)
            else:
                # Fallback implementation for KITTI
                img_file = dataset.root_split_path / 'image_2' / (f'{frame_id}.png')
                image = cv2.imread(str(img_file))

            # Get Calibration
            if hasattr(dataset, 'get_calib'):
                calib = dataset.get_calib(frame_id)
            else:
                calib_file = dataset.root_split_path / 'calib' / (f'{frame_id}.txt')
                calib = calibration_kitti.Calibration(calib_file)


            # --- Draw Model 1 (Baseline) ---
            # Color: Green (0, 255, 0)

            pred1 = pred_dicts1[0]
            boxes_lidar1 = to_cpu_numpy(pred1['pred_boxes'])
            scores1 = to_cpu_numpy(pred1['pred_scores'])
            mask1 = scores1 > SCORE_THRESH
            boxes_lidar1 = boxes_lidar1[mask1]
            scores1 = scores1[mask1]

            # --- Draw Model 2 (Gaussian) ---
            # Color: Red (0, 0, 255) because BGR

            pred2 = pred_dicts2[0]
            boxes_lidar2 = to_cpu_numpy(pred2['pred_boxes'])
            scores2 = to_cpu_numpy(pred2['pred_scores'])
            mask2 = scores2 > SCORE_THRESH
            boxes_lidar2 = boxes_lidar2[mask2]
            scores2 = scores2[mask2]

            # Save object level results to text file
            result_file = SAVE_DIR / f"{frame_id}.txt"
            with open(result_file, 'w') as f:
                f.write(f"Frame: {frame_id}\n")
                f.write(f"Model 1 ({TAG1}) Detections:\n")
                for i in range(len(boxes_lidar1)):
                    box = boxes_lidar1[i]
                    score = scores1[i]
                    f.write(f"Box: {box.tolist()}, Score: {score:.4f}\n")

                f.write(f"\nModel 2 ({TAG2}) Detections:\n")
                uncertainty2 = None
                if 'pred_uncertainty' in pred2 and pred2['pred_uncertainty'] is not None:
                    uncertainty2 = to_cpu_numpy(pred2['pred_uncertainty'])[mask2]
                elif 'pred_std' in pred2 and pred2['pred_std'] is not None:
                     uncertainty2 = to_cpu_numpy(pred2['pred_std'])[mask2]

                for i in range(len(boxes_lidar2)):
                    box = boxes_lidar2[i]
                    score = scores2[i]
                    u_str = ""
                    if uncertainty2 is not None:
                        u_str = f", Uncertainty: {uncertainty2[i].tolist()}"
                    f.write(f"Box: {box.tolist()}, Score: {score:.4f}{u_str}\n")



            # ---------------------------------------------------------------------
            # Plotting
            # ---------------------------------------------------------------------

            # Draw Baseline (Green)
            img_baseline = image.copy()
            for box in boxes_lidar1:
                corners_3d = box_utils.boxes_to_corners_3d(box[None, :7])[0]
                corners_2d, _ = calib.corners3d_to_img_boxes(corners_3d.reshape(1, 8, 3))
                # corners_2d is [N, 8, 2] -> [1, 8, 2]
                if corners_2d is not None:
                     if corners_2d[0].shape == (8, 2):
                         draw_projected_box3d(img_baseline, corners_2d[0], color=(0, 255, 0), thickness=2)

            img_gaussian = image.copy()
            # Loop for Gaussian
            if boxes_lidar2 is not None:
                for i in range(len(boxes_lidar2)):
                    box = boxes_lidar2[i]
                    corners_3d = box_utils.boxes_to_corners_3d(box[None, :7])[0]
                    corners_2d, _ = calib.corners3d_to_img_boxes(corners_3d.reshape(1, 8, 3))

                    if corners_2d is not None:
                        # Check valid 2D box
                        if corners_2d[0].shape != (8, 2):
                             continue

                        # Color based on uncertainty? Or just Red?
                        # Let's stick to Red for detection
                        draw_projected_box3d(img_gaussian, corners_2d[0], color=(0, 0, 255), thickness=2)

                        if uncertainty2 is not None and i < len(uncertainty2):
                            # Draw uncertainty value
                            u_val = uncertainty2[i]
                            # Assuming u_val has 3 dims (x, y, z) or 1 dim
                            if u_val.size > 1:
                                u_text = f"{u_val.mean():.2f}"
                            else:
                                u_text = f"{u_val.item():.2f}"

                            # Find top-left corner of 2D box for text
                            x_min = int(np.min(corners_2d[0][:, 0]))
                            y_min = int(np.min(corners_2d[0][:, 1]))
                            # Ensure coordinates are within image bounds
                            x_min = max(0, min(x_min, img_gaussian.shape[1] - 1))
                            y_min = max(0, min(y_min, img_gaussian.shape[0] - 1))

                            cv2.putText(img_gaussian, f"U:{u_text}", (x_min, y_min - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Stack images vertically or save separate?
            # Let's stack them: Top=Baseline, Bottom=Gaussian

            # Add Labels
            cv2.putText(img_baseline, f"Model 1: {TAG1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_gaussian, f"Model 2: {TAG2}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Helper to resize if shapes don't match (unlikely for same dataset)
            if img_baseline.shape != img_gaussian.shape:
                img_gaussian = cv2.resize(img_gaussian, (img_baseline.shape[1], img_baseline.shape[0]))

            combined_img = np.vstack((img_baseline, img_gaussian))

            # Ensure safe save path
            save_path = SAVE_DIR / f"{frame_id}_cmp.png"

            # Convert to uint8 before saving to avoid cv2 warning "Unsupported depth image for selected encoder is fallbacked to CV_8U"
            # It seems the image is float32 (0-1) from KittiDataset, but cv2.line/putText might have worked on it naturally?
            # Or maybe KittiDataset returns float32 0-1?
            # Let's check combined_img dtype.

            if combined_img.dtype != np.uint8:
                # If float 0-1, scale to 0-255
                if combined_img.max() <= 1.1:
                     combined_img = (combined_img * 255).astype(np.uint8)
                else:
                     combined_img = combined_img.astype(np.uint8)

            cv2.imwrite(str(save_path), combined_img)


            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(dataloader)}")

    logger.info("Done.")

if __name__ == '__main__':
    main()
