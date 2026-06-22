import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, box_utils, calibration_kitti

def box3d(h,w,l):
    return np.array([
        [ l/2,0, w/2],[ l/2,0,-w/2],[-l/2,0,-w/2],[-l/2,0, w/2],
        [ l/2,-h, w/2],[ l/2,-h,-w/2],[-l/2,-h,-w/2],[-l/2,-h, w/2]
    ], dtype=np.float32)

def Ry(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def proj(pts,P):
    pts = np.c_[pts, np.ones((pts.shape[0],1), dtype=np.float32)] @ P.T
    pts[:,0] /= pts[:,2]
    pts[:,1] /= pts[:,2]
    return pts[:,:2]

def draw3d(img, pts2d, color):
    # Same as vis_pic.py
    E=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    img_h, img_w = img.shape[:2]
    pts2d = pts2d.astype(int)

    # Clip points to be within image slightly to avoid crazy lines
    # (Optional, but often needed)

    for i,j in E:
        pt1 = tuple(pts2d[i])
        pt2 = tuple(pts2d[j])
        cv2.line(img, pt1, pt2, color, 2)

def draw3d_alpha(img, pts2d, color, alpha=0.5):
    """Draw translucent 3D box wireframe"""
    overlay = img.copy()
    E=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts2d = pts2d.astype(int)

    # Clip points roughly to image size to avoid huge drawing issues
    h, w = img.shape[:2]

    valid = True
    for pt in pts2d:
        if not (-w < pt[0] < 2*w and -h < pt[1] < 2*h):
            # If any point is way outside, drawing lines might be slow or weird, but cv2 handles it usually.
            pass

    for i,j in E:
        pt1 = tuple(pts2d[i])
        pt2 = tuple(pts2d[j])
        cv2.line(overlay, pt1, pt2, color, 2) # Same thickness

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def to_cpu_numpy(x):
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/GLENet_VR_gaussian.yaml', help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='../output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth', help='specify the pretrained model')
    parser.add_argument('--save_dir', type=str, default='../visualization_outputs/vis_full_uncertainty', help='directory to save visualization results')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='score threshold')
    parser.add_argument('--max_samples', type=int, default=-1, help='max samples to visualize (-1 for all)')
    parser.add_argument('--uncertainty_scale', type=float, default=1.0, help='scale factor for uncertainty visualization')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    # Force Val Dataset
    dataset_cfg = cfg.DATA_CONFIG
    dataset_cfg.DATA_SPLIT['test'] = 'val'
    dataset_cfg.INFO_PATH['test'] = ['kitti_infos_val.pkl']

    dataset, dataloader, sampler = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Visualizing to {args.save_dir}...")

    with torch.no_grad():
        for idx, batch_dict in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model.forward(batch_dict)

            # --- Inference Results ---
            pred = pred_dicts[0]
            boxes_lidar = to_cpu_numpy(pred['pred_boxes'])
            scores = to_cpu_numpy(pred['pred_scores'])
            labels = to_cpu_numpy(pred['pred_labels'])

            # Extract Uncertainty
            # For VoxelRCNNGaussian, pred_uncertainty should be in the dict
            uncertainty = None
            if 'pred_uncertainty' in pred and pred['pred_uncertainty'] is not None:
                uncertainty = to_cpu_numpy(pred['pred_uncertainty'])

            # Filter by score
            mask = scores > args.score_thresh
            boxes_lidar = boxes_lidar[mask]
            scores = scores[mask]
            if uncertainty is not None:
                uncertainty = uncertainty[mask]

            # --- Metadata ---
            frame_id = batch_dict['frame_id'][0]
            if isinstance(frame_id, (int, np.integer)):
                 frame_id = f"{frame_id:06d}"

            # Retrieve Image and Calib
            # Use dataset's method to get paths usually, but accessing internal list is easier if public method missing
            # KittiDataset has get_image and get_calib

            # Note: build_dataloader wraps dataset. We need the underlying dataset.
            # But the 'dataset' variable holds it.

            # KittiDataset stores info in self.kitti_infos but we don't have direct index mapping from dataloader batch easily
            # unless we trust the order. But batch_dict['frame_id'] is reliable.

            # Get calibration
            calib = dataset.get_calib(frame_id)
            image_path = dataset.root_split_path / 'image_2' / ('%s.png' % frame_id)
            img = cv2.imread(str(image_path))

            if img is None:
                print(f"Could not load image for {frame_id}")
                continue

            cam3d = img.copy()

            # --- Draw Prediction on Image (Red/Orange) ---
            if len(boxes_lidar) > 0:
                # boxes_lidar: (N, 7) [x, y, z, dx, dy, dz, rot]

                # 1. Draw Mean Prediction (Solid)
                corners_lidar = box_utils.boxes_to_corners_3d(boxes_lidar) # (N, 8, 3)
                corners_lidar = corners_lidar.reshape(-1, 3)
                corners_rect = calib.lidar_to_rect(corners_lidar)
                corners_rect = corners_rect.reshape(-1, 8, 3)

                for i in range(len(boxes_lidar)):
                    pts2d = proj(corners_rect[i], calib.P2)
                    draw3d(cam3d, pts2d, (0, 165, 255)) # Orange for Mean

                    # 2. Draw Uncertainty Samples (Transparent) using args.uncertainty_scale
                    if uncertainty is not None and args.uncertainty_scale > 0:
                        # Decode uncertainty
                        x, y, z, dx, dy, dz, rot = boxes_lidar[i]
                        l, w, h = dx, dy, dz
                        diag = np.sqrt(l**2 + w**2)

                        ux_enc, uy_enc, uz_enc = uncertainty[i]
                        ux = ux_enc * diag * args.uncertainty_scale
                        uy = uy_enc * diag * args.uncertainty_scale
                        uz = uz_enc * h * args.uncertainty_scale

                        # Generate Samples from Gaussian
                        num_samples = 10
                        # Independent errors in x, y, z
                        # (N_samples, 3)
                        noise = np.random.randn(num_samples, 3)
                        dx_noise = noise[:, 0] * ux
                        dy_noise = noise[:, 1] * uy
                        dz_noise = noise[:, 2] * uz

                        # Create sampled boxes
                        # We only perturb x, y, z (center).
                        # Dimensions and rotation are kept constant for simple position uncertainty viz.
                        sample_boxes = np.tile(boxes_lidar[i], (num_samples, 1))
                        sample_boxes[:, 0] += dx_noise
                        sample_boxes[:, 1] += dy_noise
                        sample_boxes[:, 2] += dz_noise

                        # Project and Draw
                        s_corners_lidar = box_utils.boxes_to_corners_3d(sample_boxes)
                        s_corners_lidar = s_corners_lidar.reshape(-1, 3)
                        s_corners_rect = calib.lidar_to_rect(s_corners_lidar)
                        s_corners_rect = s_corners_rect.reshape(-1, 8, 3)

                        for k in range(num_samples):
                            s_pts2d = proj(s_corners_rect[k], calib.P2)
                            # Draw with low alpha (e.g. 0.1)
                            draw3d_alpha(cam3d, s_pts2d, (0, 255, 255), alpha=0.15)

            cv2.imwrite(os.path.join(args.save_dir, f"{frame_id}_3d.png"), cam3d)


            # --- Draw BEV with Uncertainty ---
            # LiDAR frame: x-forward, y-left.
            fig, ax = plt.subplots(figsize=(6,6), dpi=100)

            # Draw GT if available (Green)
            # We can get GT from 'gt_boxes' in batch_dict if we enable it, but test dataloader might not load it unless configured?
            # 'gt_boxes' is in batch_dict if Info provided it.
            if 'gt_boxes' in batch_dict:
                 gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()
                 # Remove padding (zeros)
                 # usually shape is (M, 8) last dim is class?
                 # count valid
                 mask_gt = (gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0) # w, l > 0
                 gt_boxes = gt_boxes[mask_gt]

                 for i in range(len(gt_boxes)):
                     box = gt_boxes[i]
                     x, y, z, dx, dy, dz, rot = box[:7]
                     c, s = np.cos(rot), np.sin(rot)
                     R = np.array([[c, -s],[s, c]])
                     # In LiDAR frame, x is forward, y is left.
                     # But commonly plotted as X horizontal, Y vertical.
                     # Wait, usually for car, X is forward.
                     # Let's plot X horizontal, Y vertical.

                     corners = np.array([
                        [dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2], [-dx/2, dy/2]
                     ]) @ R.T
                     corners += np.array([x, y])
                     poly = np.vstack([corners, corners[0]])
                     ax.plot(poly[:,0], poly[:,1], 'g-', linewidth=2, label='GT' if i==0 else "")

            # Draw Pred (Red)
            if len(boxes_lidar) > 0:
                 for i in range(len(boxes_lidar)):
                     box = boxes_lidar[i]
                     x, y, z, dx, dy, dz, rot = box # LiDAR

                     c, s = np.cos(rot), np.sin(rot)
                     R = np.array([[c, -s],[s, c]])
                     corners = np.array([
                        [dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2], [-dx/2, dy/2]
                     ]) @ R.T
                     corners += np.array([x, y])
                     poly = np.vstack([corners, corners[0]])
                     ax.plot(poly[:,0], poly[:,1], 'r-', linewidth=2, label='Pred' if i==0 else "")

                     # Uncertainty Bubble
                     if uncertainty is not None:
                         # uncertainty is std dev (sigma) in encoded space.
                         # Need to scale by box size to get meters.
                         # pred_boxes: x, y, z, dx, dy, dz, rot
                         # dx, dy, dz are dimensions (l, w, h).
                         # Note: pcdet format is (l, w, h) or (dx, dy, dz). Usually (l, w, h).
                         # Diagonal d = sqrt(l^2 + w^2)

                         l, w, h = dx, dy, dz
                         diag = np.sqrt(l**2 + w**2)

                         # Encoded uncertainty: ux_enc, uy_enc, uz_enc
                         ux_enc, uy_enc, uz_enc = uncertainty[i]

                         # Decode to meters
                         ux = ux_enc * diag
                         uy = uy_enc * diag
                         # uz = uz_enc * h  # Z not shown in BEV

                         # Draw Gradient/Cloud effect (1-sigma, 2-sigma, 3-sigma)
                         # 1-sigma (68% conf): alpha 0.4
                         # 2-sigma (95% conf): alpha 0.2
                         # 3-sigma (99% conf): alpha 0.1

                         sigmas = [3.0, 2.0, 1.0]
                         alphas = [0.1, 0.2, 0.4]

                         for sigma, alpha in zip(sigmas, alphas):
                             width = 2 * sigma * ux * args.uncertainty_scale  # Scale up
                             height = 2 * sigma * uy * args.uncertainty_scale

                             ell = Ellipse((x, y), width=width, height=height, angle=0,
                                           facecolor='orange', alpha=alpha, edgecolor=None)
                             ax.add_patch(ell)

                         # Center dot
                         ax.scatter(x, y, c='red', s=5, zorder=10)

            # Setup Plot Limits
            # KITTI LiDAR range: x [0, 70], y [-40, 40]
            # Plot X as vertical (up), Y as horizontal (left-right)?
            # Or standard map: X right, Y up?
            # Usually BEV: X (forward) up, Y (left) left.
            # So map X_lidar -> Y_plot, Y_lidar -> -X_plot ?
            # Let's just plot X as X, Y as Y.

            ax.set_xlim(0, 70)
            ax.set_ylim(-40, 40)
            ax.set_aspect('equal')
            ax.set_title(f"BEV - Frame {frame_id}")
            if idx == 0:
                ax.legend()

            fig.savefig(os.path.join(args.save_dir, f"{frame_id}_bev_uncertainty.png"))
            plt.close(fig)

            # Limit number of samples
            if args.max_samples > 0 and (idx + 1) >= args.max_samples:
                break

if __name__ == '__main__':
    main()
