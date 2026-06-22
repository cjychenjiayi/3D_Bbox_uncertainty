import _init_path
import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.patches import Ellipse, Polygon

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_2d_array(x, width, dtype=np.float32):
    arr = to_numpy(x)
    if arr is None:
        return np.zeros((0, width), dtype=dtype)
    arr = np.asarray(arr, dtype=dtype)
    if arr.size == 0:
        return np.zeros((0, width), dtype=dtype)
    if arr.ndim == 1 and arr.size % width == 0:
        arr = arr.reshape(-1, width)
    else:
        arr = np.atleast_2d(arr)
    if arr.shape[1] < width:
        padded = np.zeros((arr.shape[0], width), dtype=dtype)
        padded[:, : arr.shape[1]] = arr
        return padded
    return arr[:, :width]


def as_1d_array(x, dtype=np.float32):
    arr = to_numpy(x)
    if arr is None:
        return np.zeros((0,), dtype=dtype)
    arr = np.asarray(arr, dtype=dtype).reshape(-1)
    return arr


def valid_frame_id(frame_id):
    if isinstance(frame_id, bytes):
        frame_id = frame_id.decode("utf-8")
    if isinstance(frame_id, (int, np.integer)):
        return f"{int(frame_id):06d}"
    return str(frame_id)


def bev_corners(boxes_lidar):
    centers = boxes_lidar[:, 0:2]
    dx = boxes_lidar[:, 3]
    dy = boxes_lidar[:, 4]
    heading = boxes_lidar[:, 6]

    template = np.array(
        [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )
    local = template[None, :, :] * np.stack([dx, dy], axis=1)[:, None, :]

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rot = np.stack(
        [
            np.stack([cos_h, -sin_h], axis=1),
            np.stack([sin_h, cos_h], axis=1),
        ],
        axis=1,
    )
    return np.matmul(local, np.transpose(rot, (0, 2, 1))) + centers[:, None, :]


def decode_uncertainty_to_meters(boxes_lidar, uncertainty_xyz):
    if len(boxes_lidar) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if uncertainty_xyz is None:
        return np.zeros((len(boxes_lidar), 3), dtype=np.float32)

    dims = np.maximum(boxes_lidar[:, 3:6], 1e-3)
    diag = np.sqrt(dims[:, 0] ** 2 + dims[:, 1] ** 2)

    unc_m = np.zeros_like(uncertainty_xyz, dtype=np.float32)
    unc_m[:, 0] = uncertainty_xyz[:, 0] * diag
    unc_m[:, 1] = uncertainty_xyz[:, 1] * diag
    unc_m[:, 2] = uncertainty_xyz[:, 2] * dims[:, 2]
    return unc_m


def summarize_rows(rows):
    if not rows:
        return {
            "num_detections": 0,
            "pearson_range_xy_vs_unc_mean": None,
            "pearson_range_xy_vs_unc_bev": None,
        }

    range_xy = np.array([row["range_xy_m"] for row in rows], dtype=np.float32)
    unc_mean = np.array([row["unc_mean_m"] for row in rows], dtype=np.float32)
    unc_bev = np.array([row["unc_bev_m"] for row in rows], dtype=np.float32)

    def safe_corr(a, b):
        if len(a) < 2 or np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
            return None
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "num_detections": int(len(rows)),
        "pearson_range_xy_vs_unc_mean": safe_corr(range_xy, unc_mean),
        "pearson_range_xy_vs_unc_bev": safe_corr(range_xy, unc_bev),
        "range_xy_mean_m": float(range_xy.mean()),
        "unc_mean_mean_m": float(unc_mean.mean()),
        "unc_bev_mean_m": float(unc_bev.mean()),
    }


def normalize_predictions(boxes, scores, labels, uncertainty_xyz, score_thresh):
    boxes = as_2d_array(boxes, 7, dtype=np.float32)
    scores = as_1d_array(scores, dtype=np.float32)
    labels = as_1d_array(labels, dtype=np.int32)
    if len(labels) == 0 and len(boxes) > 0:
        labels = np.zeros((len(boxes),), dtype=np.int32)
    if uncertainty_xyz is None:
        uncertainty_xyz = np.zeros((len(boxes), 3), dtype=np.float32)
    else:
        uncertainty_xyz = as_2d_array(uncertainty_xyz, 3, dtype=np.float32)

    num_preds = min(len(boxes), len(scores), len(labels), len(uncertainty_xyz))
    boxes = boxes[:num_preds]
    scores = scores[:num_preds]
    labels = labels[:num_preds]
    uncertainty_xyz = uncertainty_xyz[:num_preds]

    if num_preds == 0:
        return boxes, scores, labels, np.zeros((0, 3), dtype=np.float32)

    keep = np.isfinite(scores) & (scores >= score_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    uncertainty_xyz = uncertainty_xyz[keep]
    unc_m = decode_uncertainty_to_meters(boxes, uncertainty_xyz)
    return boxes, scores, labels, unc_m


def rows_from_predictions(frame_id, boxes, scores, labels, unc_m):
    rows = []
    for obj_idx in range(len(boxes)):
        box = boxes[obj_idx]
        unc_obj = unc_m[obj_idx]
        rows.append(
            {
                "frame_id": frame_id,
                "label": int(labels[obj_idx]),
                "score": float(scores[obj_idx]),
                "range_xy_m": float(np.linalg.norm(box[:2])),
                "distance_3d_m": float(np.linalg.norm(box[:3])),
                "x_m": float(box[0]),
                "y_m": float(box[1]),
                "z_m": float(box[2]),
                "unc_x_m": float(unc_obj[0]),
                "unc_y_m": float(unc_obj[1]),
                "unc_z_m": float(unc_obj[2]),
                "unc_bev_m": float(np.mean(unc_obj[:2])),
                "unc_mean_m": float(np.mean(unc_obj)),
            }
        )
    return rows


def draw_bev_frame(
    save_path,
    frame_id,
    pred_boxes,
    pred_scores,
    pred_labels,
    pred_unc_m,
    gt_boxes=None,
    class_names=None,
    uncertainty_scale=1.0,
):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=140)

    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_corners = bev_corners(gt_boxes[:, :7])
        for corners in gt_corners:
            poly = np.vstack([corners, corners[0]])
            ax.plot(poly[:, 0], poly[:, 1], color="#2e8b57", linewidth=1.8, alpha=0.85)

    if len(pred_boxes) > 0:
        unc_metric = pred_unc_m[:, :2].mean(axis=1)
        max_unc = max(float(unc_metric.max()), 1e-3)
        cmap = plt.get_cmap("turbo")
        corners_all = bev_corners(pred_boxes)

        for idx, corners in enumerate(corners_all):
            color = cmap(float(unc_metric[idx] / max_unc))
            polygon = Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=2.0,
                alpha=0.95,
            )
            ax.add_patch(polygon)

            center_x, center_y = pred_boxes[idx, 0], pred_boxes[idx, 1]
            ell = Ellipse(
                (center_x, center_y),
                width=max(pred_unc_m[idx, 0] * 2.0 * uncertainty_scale, 0.05),
                height=max(pred_unc_m[idx, 1] * 2.0 * uncertainty_scale, 0.05),
                angle=0.0,
                facecolor=color,
                edgecolor=None,
                alpha=0.18,
            )
            ax.add_patch(ell)
            ax.scatter(center_x, center_y, color=color, s=12, zorder=5)

            class_name = str(pred_labels[idx])
            if class_names is not None:
                cls_idx = int(pred_labels[idx]) - 1
                if 0 <= cls_idx < len(class_names):
                    class_name = class_names[cls_idx]
            range_xy = np.linalg.norm(pred_boxes[idx, :2])
            text = (
                f"{class_name} {pred_scores[idx]:.2f}\n"
                f"r={range_xy:.1f}m u={unc_metric[idx]:.2f}m"
            )
            ax.text(
                center_x + 0.5,
                center_y + 0.5,
                text,
                fontsize=7,
                color=color,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            )

    ax.set_xlim(0, 75)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_xlabel("X forward (m)")
    ax.set_ylabel("Y left (m)")
    ax.set_title(f"Frame {frame_id}: bbox uncertainty in BEV")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def bin_mean_std(x, y, num_bins):
    bins = np.linspace(0.0, max(70.0, float(x.max()) + 1.0), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    per_bin_mean = np.full(num_bins, np.nan, dtype=np.float32)
    per_bin_std = np.full(num_bins, np.nan, dtype=np.float32)
    for bin_idx, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (x >= start) & (x < end)
        if mask.any():
            per_bin_mean[bin_idx] = float(y[mask].mean())
            per_bin_std[bin_idx] = float(y[mask].std())
    return bin_centers, per_bin_mean, per_bin_std


def draw_global_summary(rows, save_dir, num_bins):
    range_xy = np.array([row["range_xy_m"] for row in rows], dtype=np.float32)
    score = np.array([row["score"] for row in rows], dtype=np.float32)
    unc_x = np.array([row["unc_x_m"] for row in rows], dtype=np.float32)
    unc_y = np.array([row["unc_y_m"] for row in rows], dtype=np.float32)
    unc_z = np.array([row["unc_z_m"] for row in rows], dtype=np.float32)
    unc_bev = np.array([row["unc_bev_m"] for row in rows], dtype=np.float32)
    unc_mean = np.array([row["unc_mean_m"] for row in rows], dtype=np.float32)
    labels = np.array([row["label"] for row in rows], dtype=np.int32)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    ax_scatter = axes[0, 0]
    ax_bin = axes[0, 1]
    ax_score = axes[1, 0]
    ax_hist = axes[1, 1]

    scatter = ax_scatter.scatter(
        range_xy,
        unc_mean,
        c=score,
        s=12,
        alpha=0.45,
        cmap="winter",
        linewidths=0,
    )
    ax_scatter.set_title("Distance vs mean bbox uncertainty")
    ax_scatter.set_xlabel("Range in BEV (m)")
    ax_scatter.set_ylabel("Mean XYZ uncertainty (m)")
    cbar = fig.colorbar(scatter, ax=ax_scatter)
    cbar.set_label("Detection score")

    bin_centers, per_bin_mean, per_bin_std = bin_mean_std(range_xy, unc_mean, num_bins)
    ax_bin.plot(bin_centers, per_bin_mean, color="#d94801", linewidth=2.2)
    ax_bin.fill_between(
        bin_centers,
        per_bin_mean - per_bin_std,
        per_bin_mean + per_bin_std,
        color="#fdae6b",
        alpha=0.35,
    )
    ax_bin.set_title("Binned uncertainty trend")
    ax_bin.set_xlabel("Range in BEV (m)")
    ax_bin.set_ylabel("Mean XYZ uncertainty (m)")
    ax_bin.grid(alpha=0.25, linestyle="--")

    ax_score.scatter(score, unc_bev, c=range_xy, s=12, alpha=0.45, cmap="winter", linewidths=0)
    ax_score.set_title("Score vs BEV uncertainty")
    ax_score.set_xlabel("Detection score")
    ax_score.set_ylabel("BEV uncertainty (m)")

    car_mask = labels == 1
    ped_mask = labels == 2
    cyc_mask = labels == 3
    has_class_hist = False
    if car_mask.any():
        ax_hist.hist(unc_mean[car_mask], bins=35, alpha=0.55, label="Car")
        has_class_hist = True
    if ped_mask.any():
        ax_hist.hist(unc_mean[ped_mask], bins=35, alpha=0.55, label="Pedestrian")
        has_class_hist = True
    if cyc_mask.any():
        ax_hist.hist(unc_mean[cyc_mask], bins=35, alpha=0.55, label="Cyclist")
        has_class_hist = True
    if not has_class_hist:
        ax_hist.hist(unc_mean, bins=35, alpha=0.55, label="All detections")
    ax_hist.set_title("Uncertainty distribution by class")
    ax_hist.set_xlabel("Mean XYZ uncertainty (m)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend()

    fig.tight_layout()
    fig.savefig(save_dir / "uncertainty_summary.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=150)
    ax2.scatter(range_xy, unc_x, s=10, alpha=0.25, label="sigma_x")
    ax2.scatter(range_xy, unc_y, s=10, alpha=0.25, label="sigma_y")
    ax2.scatter(range_xy, unc_z, s=10, alpha=0.25, label="sigma_z")
    ax2.set_title("Per-axis uncertainty vs distance")
    ax2.set_xlabel("Range in BEV (m)")
    ax2.set_ylabel("Uncertainty (m)")
    ax2.legend()
    ax2.grid(alpha=0.25, linestyle="--")
    fig2.tight_layout()
    fig2.savefig(save_dir / "uncertainty_axes_vs_distance.png")
    plt.close(fig2)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize GLENet Gaussian uncertainty trends")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="tools/cfgs/kitti_models/GLENet_VR_gaussian.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/ckpt/checkpoint_epoch_80.pth",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="visualization_outputs/vis_uncertainty_trend",
    )
    parser.add_argument(
        "--pred_pkl",
        type=str,
        default=None,
        help="Optional Gaussian result pickle generated by test_gaussian.py / eval_utils_gaussian.py",
    )
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--max_frames", type=int, default=150)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--num_bins",
        type=int,
        default=12,
        choices=range(1, 101),
        metavar="[1-100]",
    )
    parser.add_argument("--uncertainty_scale", type=float, default=1.0)
    parser.add_argument(
        "--save_bev_frames",
        action="store_true",
        help="Save per-frame BEV images in addition to the global trend plots",
    )
    return parser.parse_args()


def collect_rows_from_prediction_records(records, score_thresh, max_frames=-1):
    rows = []
    frame_payloads = []
    for frame_idx, record in enumerate(records):
        if max_frames > 0 and frame_idx >= max_frames:
            break
        frame_id = valid_frame_id(record.get("frame_id", "unknown"))
        boxes, scores, labels, unc_m = normalize_predictions(
            record.get("boxes_3d", record.get("pred_boxes", [])),
            record.get("scores", record.get("pred_scores", [])),
            record.get("labels", record.get("pred_labels", [])),
            record.get("uncertainty_xyz", record.get("pred_uncertainty", None)),
            score_thresh,
        )
        rows.extend(rows_from_predictions(frame_id, boxes, scores, labels, unc_m))

        frame_payloads.append(
            {
                "frame_id": frame_id,
                "pred_boxes": boxes,
                "pred_scores": scores,
                "pred_labels": labels,
                "pred_unc_m": unc_m,
            }
        )
    return rows, frame_payloads


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    cfg_file = Path(args.cfg_file)
    ckpt_file = Path(args.ckpt)
    pred_pkl_file = Path(args.pred_pkl) if args.pred_pkl else None
    save_dir = Path(args.save_dir)

    if not cfg_file.is_absolute():
        cfg_file = (project_root / cfg_file).resolve()
    if not ckpt_file.is_absolute():
        ckpt_file = (project_root / ckpt_file).resolve()
    if pred_pkl_file is not None and not pred_pkl_file.is_absolute():
        pred_pkl_file = (project_root / pred_pkl_file).resolve()
    if not save_dir.is_absolute():
        save_dir = (project_root / save_dir).resolve()

    # PCDet config inheritance uses paths relative to tools/.
    os.chdir(project_root / "tools")

    cfg_from_yaml_file(str(cfg_file), cfg)
    logger = common_utils.create_logger()

    dataset_cfg = cfg.DATA_CONFIG
    dataset_cfg.DATA_SPLIT["test"] = "val"
    dataset_cfg.INFO_PATH["test"] = ["kitti_infos_val.pkl"]

    save_dir.mkdir(parents=True, exist_ok=True)
    bev_dir = save_dir / "bev_frames"
    if args.save_bev_frames:
        bev_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if pred_pkl_file is not None:
        with open(pred_pkl_file, "rb") as f:
            prediction_records = pickle.load(f)
        rows, frame_payloads = collect_rows_from_prediction_records(
            prediction_records, args.score_thresh, args.max_frames
        )
        if args.save_bev_frames:
            for payload in frame_payloads:
                draw_bev_frame(
                    bev_dir / f"{payload['frame_id']}_bev.png",
                    frame_id=payload["frame_id"],
                    pred_boxes=payload["pred_boxes"],
                    pred_scores=payload["pred_scores"],
                    pred_labels=payload["pred_labels"],
                    pred_unc_m=payload["pred_unc_m"],
                    gt_boxes=None,
                    class_names=cfg.CLASS_NAMES,
                    uncertainty_scale=args.uncertainty_scale,
                )
    else:
        dataset, dataloader, _ = build_dataloader(
            dataset_cfg=dataset_cfg,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=False,
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=str(ckpt_file), logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

        with torch.no_grad():
            for frame_idx, batch_dict in enumerate(tqdm.tqdm(dataloader, total=len(dataloader))):
                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    break

                frame_id = valid_frame_id(batch_dict["frame_id"][0])
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = model.forward(batch_dict)
                pred = pred_dicts[0]

                pred_boxes, pred_scores, pred_labels, pred_unc_m = normalize_predictions(
                    pred.get("pred_boxes", []),
                    pred.get("pred_scores", []),
                    pred.get("pred_labels", []),
                    pred.get("pred_uncertainty", None),
                    args.score_thresh,
                )
                rows.extend(
                    rows_from_predictions(
                        frame_id, pred_boxes, pred_scores, pred_labels, pred_unc_m
                    )
                )

                if args.save_bev_frames:
                    gt_boxes = None
                    if "gt_boxes" in batch_dict:
                        gt_raw = to_numpy(batch_dict["gt_boxes"][0])
                        if gt_raw is not None and len(gt_raw) > 0:
                            valid_gt = (
                                (gt_raw[:, 3] > 0)
                                & (gt_raw[:, 4] > 0)
                                & (gt_raw[:, 5] > 0)
                            )
                            gt_boxes = gt_raw[valid_gt][:, :7]
                    draw_bev_frame(
                        bev_dir / f"{frame_id}_bev.png",
                        frame_id=frame_id,
                        pred_boxes=pred_boxes,
                        pred_scores=pred_scores,
                        pred_labels=pred_labels,
                        pred_unc_m=pred_unc_m,
                        gt_boxes=gt_boxes,
                        class_names=cfg.CLASS_NAMES,
                        uncertainty_scale=args.uncertainty_scale,
                    )

    if not rows:
        logger.warning("No detections survived the score threshold. Nothing was saved.")
        return

    draw_global_summary(rows, save_dir, args.num_bins)

    summary = summarize_rows(rows)
    summary["cfg_file"] = args.cfg_file
    summary["ckpt"] = args.ckpt
    summary["pred_pkl"] = args.pred_pkl
    summary["score_thresh"] = args.score_thresh
    summary["max_frames"] = args.max_frames
    summary["num_frames_with_detections"] = int(len({row["frame_id"] for row in rows}))

    with open(save_dir / "uncertainty_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(save_dir / "uncertainty_points.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Saved uncertainty visualizations to %s", save_dir)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
