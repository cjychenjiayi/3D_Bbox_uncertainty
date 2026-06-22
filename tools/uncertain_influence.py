import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from pcdet.config import cfg, cfg_from_yaml_file


TITLE_FONTSIZE = 36
AXIS_LABEL_FONTSIZE = 32
TICK_LABEL_FONTSIZE = 27
LEGEND_FONTSIZE = 25
ANNOTATION_FONTSIZE = 24
COLORBAR_LABEL_FONTSIZE = 31
COLORBAR_TICK_FONTSIZE = 26


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#111111",
            "axes.titleweight": "bold",
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "legend.frameon": True,
            "legend.edgecolor": "#cccccc",
            "legend.fontsize": LEGEND_FONTSIZE,
            "savefig.bbox": "tight",
            "lines.linewidth": 3.0,
        }
    )


def add_axis_style(ax):
    ax.grid(True, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE, width=1.2, length=6)


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_frame_id(frame_id):
    if isinstance(frame_id, bytes):
        frame_id = frame_id.decode("utf-8")
    if isinstance(frame_id, (int, np.integer)):
        return f"{int(frame_id):06d}"
    return str(frame_id)


def make_logger(save_dir):
    logger = logging.getLogger("uncertain_influence")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(save_dir / "analysis.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def bev_corners(box):
    x, y, _, dx, dy, _, heading = box[:7]
    template = np.array(
        [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )
    corners = template * np.array([dx, dy], dtype=np.float32)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([x, y], dtype=np.float32)


def bev_iou(box_a, box_b):
    poly_a = bev_corners(box_a).astype(np.float32)
    poly_b = bev_corners(box_b).astype(np.float32)
    area_a = float(abs(cv2.contourArea(poly_a)))
    area_b = float(abs(cv2.contourArea(poly_b)))
    if area_a <= 0 or area_b <= 0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    inter_area = float(inter_area) if inter_area is not None else 0.0
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return max(0.0, min(1.0, inter_area / union_area))


def decode_uncertainty_to_metric(boxes_lidar, uncertainty_xyz):
    if uncertainty_xyz is None:
        return None
    dims = np.maximum(boxes_lidar[:, 3:6], 1e-3)
    diag = np.sqrt(dims[:, 0] ** 2 + dims[:, 1] ** 2)
    unc_metric = np.zeros_like(uncertainty_xyz, dtype=np.float32)
    unc_metric[:, 0] = uncertainty_xyz[:, 0] * diag
    unc_metric[:, 1] = uncertainty_xyz[:, 1] * diag
    unc_metric[:, 2] = uncertainty_xyz[:, 2] * dims[:, 2]
    return unc_metric


def rankdata(values):
    order = np.argsort(values)
    ranks = np.zeros(len(values), dtype=np.float32)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def safe_corr(x, y):
    if len(x) < 2:
        return None
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    if len(x) < 2:
        return None
    return safe_corr(rankdata(x), rankdata(y))


def load_gt_by_frame(gt_info_path, class_names):
    with open(gt_info_path, "rb") as f:
        infos = pickle.load(f)

    gt_by_frame = {}
    class_name_set = set(class_names)
    for info in infos:
        frame_id = normalize_frame_id(info["point_cloud"]["lidar_idx"])
        annos = info.get("annos", {})
        names_all = np.asarray(annos.get("name", np.array([], dtype=str)))
        boxes = np.asarray(annos.get("gt_boxes_lidar", np.zeros((0, 7), dtype=np.float32)))

        # KITTI infos often keep `DontCare` in `name`, while `gt_boxes_lidar`
        # only stores real objects. Align them the same way as dataset loading.
        names_valid = names_all[names_all != "DontCare"]
        if len(names_valid) != len(boxes):
            names_valid = names_valid[: len(boxes)]
            boxes = boxes[: len(names_valid)]

        keep = np.array([name in class_name_set for name in names_valid], dtype=bool)
        gt_by_frame[frame_id] = {
            "names": names_valid[keep],
            "boxes_lidar": boxes[keep],
        }
    return gt_by_frame


def load_prediction_records(pred_pkl_path):
    with open(pred_pkl_path, "rb") as f:
        records = pickle.load(f)
    return records


def find_best_match(pred_box, pred_class_name, gt_names, gt_boxes, used_gt):
    best_iou = 0.0
    best_idx = -1
    for gt_idx in range(len(gt_boxes)):
        if used_gt[gt_idx]:
            continue
        if str(gt_names[gt_idx]) != pred_class_name:
            continue
        cur_iou = bev_iou(pred_box, gt_boxes[gt_idx])
        if cur_iou > best_iou:
            best_iou = cur_iou
            best_idx = gt_idx
    return best_idx, best_iou


def build_rows(records, gt_by_frame, class_names, score_thresh, match_iou_min):
    rows = []
    issues = []

    for record in records:
        frame_id = normalize_frame_id(record.get("frame_id", "unknown"))
        boxes = to_numpy(record.get("boxes_3d", record.get("pred_boxes", [])))
        scores = to_numpy(record.get("scores", record.get("pred_scores", [])))
        labels = to_numpy(record.get("labels", record.get("pred_labels", [])))
        uncertainty = to_numpy(record.get("uncertainty_xyz", record.get("pred_uncertainty")))

        if boxes is None:
            boxes = np.zeros((0, 7), dtype=np.float32)
        if scores is None:
            scores = np.zeros((0,), dtype=np.float32)
        if labels is None:
            labels = np.zeros((0,), dtype=np.int32)
        if uncertainty is None:
            issues.append(f"{frame_id}: prediction file has no uncertainty field")
            continue

        keep = scores >= score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        uncertainty = uncertainty[keep]

        if len(boxes) == 0:
            continue

        uncertainty_metric = decode_uncertainty_to_metric(boxes, uncertainty)
        gt_payload = gt_by_frame.get(frame_id, {"names": np.array([]), "boxes_lidar": np.zeros((0, 7), dtype=np.float32)})
        gt_names = gt_payload["names"]
        gt_boxes = gt_payload["boxes_lidar"]
        used_gt = np.zeros(len(gt_boxes), dtype=bool)

        order = np.argsort(-scores)
        for idx in order:
            label = int(labels[idx])
            class_name = class_names[label - 1] if 0 < label <= len(class_names) else str(label)
            matched_gt_idx, best_iou = find_best_match(boxes[idx], class_name, gt_names, gt_boxes, used_gt)
            matched = matched_gt_idx >= 0 and best_iou >= match_iou_min
            if matched:
                used_gt[matched_gt_idx] = True
                gt_box = gt_boxes[matched_gt_idx]
                center_error = float(np.linalg.norm(boxes[idx, :2] - gt_box[:2]))
            else:
                gt_box = None
                center_error = None

            error_scalar = 1.0 - float(best_iou)
            unc_metric = uncertainty_metric[idx]
            rows.append(
                {
                    "frame_id": frame_id,
                    "label": label,
                    "class_name": class_name,
                    "score": float(scores[idx]),
                    "range_xy_m": float(np.linalg.norm(boxes[idx, :2])),
                    "unc_x_m": float(unc_metric[0]),
                    "unc_y_m": float(unc_metric[1]),
                    "unc_z_m": float(unc_metric[2]),
                    "unc_bev_m": float(np.mean(unc_metric[:2])),
                    "unc_mean_m": float(np.mean(unc_metric)),
                    "best_iou_bev": float(best_iou),
                    "error_scalar": error_scalar,
                    "center_error": center_error,
                    "matched": bool(matched),
                    "matched_gt_available": bool(gt_box is not None),
                }
            )
    return rows, issues


def make_bins(values, num_bins):
    if len(values) == 0:
        return None
    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.quantile(values, quantiles)
    bins[0] = min(bins[0], values.min())
    bins[-1] = max(bins[-1], values.max())
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6
    return bins


def draw_core_bin_plot(rows, save_path, num_bins):
    uncertainty = np.array([row["unc_mean_m"] for row in rows], dtype=np.float32)
    error = np.array([row["error_scalar"] for row in rows], dtype=np.float32)

    bins = make_bins(uncertainty, num_bins)
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []
    bin_labels = []

    for left, right in zip(bins[:-1], bins[1:]):
        mask = (uncertainty >= left) & (uncertainty < right)
        if right == bins[-1]:
            mask = (uncertainty >= left) & (uncertainty <= right)
        bin_counts.append(int(mask.sum()))
        bin_labels.append(f"{left:.2f}-{right:.2f}")
        if mask.any():
            bin_centers.append(float((left + right) * 0.5))
            bin_means.append(float(error[mask].mean()))
            bin_stds.append(float(error[mask].std()))
        else:
            bin_centers.append(float((left + right) * 0.5))
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=200)
    x = np.arange(len(bin_means))

    # Blue-green color scheme
    line_color = "#1A8C7A"
    band_color = "#A8D8D1"

    # Main line with larger markers
    ax.plot(x, bin_means, color=line_color, marker="o", linewidth=3.5, markersize=12,
            markerfacecolor="white", markeredgewidth=3, markeredgecolor=line_color, zorder=5)

    # Std band with gradient feel
    ax.fill_between(
        x,
        np.array(bin_means) - np.array(bin_stds),
        np.array(bin_means) + np.array(bin_stds),
        color=band_color,
        alpha=0.50,
        zorder=2,
    )
    # Thin border for the band
    ax.plot(x, np.array(bin_means) - np.array(bin_stds), color=band_color, linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)
    ax.plot(x, np.array(bin_means) + np.array(bin_stds), color=band_color, linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=0, ha="center", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel("Uncertainty Bin (m)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=14)
    ax.set_ylabel("Mean Error", fontsize=AXIS_LABEL_FONTSIZE, labelpad=14)
    ax.set_title("Uncertainty Tracks Error", fontsize=TITLE_FONTSIZE, pad=20)
    add_axis_style(ax)
    ax.set_ylim(bottom=0.0, top=max(0.55, float(np.nanmax(bin_means) + np.nanmax(bin_stds) * 1.15)))

    # Annotation box with larger font
    annotation = (
        f"Pearson  r = {safe_corr(uncertainty, error):.3f}\n"
        f"Spearman ρ = {safe_spearman(uncertainty, error):.3f}\n"
        f"Monotonic bins = {sum(np.diff(np.array([v for v in bin_means if not np.isnan(v)])) >= 0)}/"
        f"{max(0, len([v for v in bin_means if not np.isnan(v)]) - 1)}"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        fontfamily="monospace",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.5", "alpha": 0.95},
    )

    # Data labels with larger font
    for xi, yi, count in zip(x, bin_means, bin_counts):
        if not np.isnan(yi):
            ax.text(
                xi,
                yi + 0.022,
                f"{yi:.3f}\nn={count}",
                ha="center",
                va="bottom",
                fontsize=21,
                fontweight="bold",
                color="#1A6B5E",
            )

    fig.tight_layout(pad=1.2)
    fig.savefig(save_path)
    plt.close(fig)

    return {
        "bin_labels": bin_labels,
        "bin_mean_error": [None if np.isnan(v) else float(v) for v in bin_means],
        "bin_std_error": [None if np.isnan(v) else float(v) for v in bin_stds],
        "bin_counts": bin_counts,
    }


def draw_scatter_plot(rows, save_path):
    uncertainty = np.array([row["unc_mean_m"] for row in rows], dtype=np.float32)
    error = np.array([row["error_scalar"] for row in rows], dtype=np.float32)
    distance = np.array([row["range_xy_m"] for row in rows], dtype=np.float32)
    matched = np.array([row["matched"] for row in rows], dtype=bool)

    range_cmap = LinearSegmentedColormap.from_list(
        "scientific_range_purple_blue_green_yellow",
        [
            (0.00, "#3B0F70"),
            (0.35, "#3288BD"),
            (0.68, "#66C2A5"),
            (1.00, "#FDE725"),
        ],
    )

    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    if (~matched).any():
        ax.scatter(
            uncertainty[~matched],
            error[~matched],
            s=20,
            alpha=0.14,
            color="#c7c7c7",
            linewidths=0,
            label="Unmatched",
            zorder=2,
        )
    scatter = ax.scatter(
        uncertainty[matched],
        error[matched],
        c=distance[matched],
        s=26,
        alpha=0.58,
        cmap=range_cmap,
        linewidths=0,
        label="Matched",
        zorder=3,
    )
    ax.set_xlabel("Mean XYZ Uncertainty (m)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=14)
    ax.set_ylabel("Detection Error (1 − IoU$_{BEV}$)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=14)
    ax.set_title("Uncertainty vs Detection Error", fontsize=TITLE_FONTSIZE, pad=22)
    add_axis_style(ax)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Range in BEV (m)", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=12)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, width=1.2, length=6)
    ax.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    ax.text(
        0.43,
        0.91,
        "Unmatched: error = 1",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=22,
        color="#666666",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    fig.tight_layout(pad=1.5)
    fig.savefig(save_path)
    plt.close(fig)


def draw_distance_uncertainty_plot(rows, save_path):
    distance = np.array([row["range_xy_m"] for row in rows], dtype=np.float32)
    uncertainty = np.array([row["unc_mean_m"] for row in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=200)
    ax.scatter(distance, uncertainty, s=30, alpha=0.35, color="#2E86AB", linewidths=0, zorder=3)
    ax.set_xlabel("Range in BEV (m)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=12)
    ax.set_ylabel("Mean XYZ Uncertainty (m)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=12)
    ax.set_title("Distance vs Uncertainty", fontsize=TITLE_FONTSIZE, pad=20)
    add_axis_style(ax)
    ax.text(
        0.04,
        0.95,
        f"Pearson r = {safe_corr(distance, uncertainty):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        fontfamily="monospace",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.4", "alpha": 0.95},
    )
    fig.tight_layout(pad=1.6)
    fig.savefig(save_path)
    plt.close(fig)


def draw_distance_error_plot(rows, save_path):
    distance = np.array([row["range_xy_m"] for row in rows], dtype=np.float32)
    error = np.array([row["error_scalar"] for row in rows], dtype=np.float32)
    matched = np.array([row["matched"] for row in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=200)
    if (~matched).any():
        ax.scatter(
            distance[~matched],
            error[~matched],
            s=22,
            alpha=0.15,
            color="#bdbdbd",
            linewidths=0,
            label="Unmatched",
            zorder=2,
        )
    ax.scatter(
        distance[matched],
        error[matched],
        s=30,
        alpha=0.35,
        color="#C0392B",
        linewidths=0,
        label="Matched",
        zorder=3,
    )
    ax.set_xlabel("Range in BEV (m)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=12)
    ax.set_ylabel("Detection Error (1 − IoU$_{BEV}$)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=12)
    ax.set_title("Distance vs Error", fontsize=TITLE_FONTSIZE, pad=20)
    add_axis_style(ax)
    ax.text(
        0.04,
        0.95,
        f"Pearson r = {safe_corr(distance, error):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        fontfamily="monospace",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.4", "alpha": 0.95},
    )
    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9)

    fig.tight_layout(pad=1.6)
    fig.savefig(save_path)
    plt.close(fig)


def compute_summary(rows, bin_stats):
    uncertainty = np.array([row["unc_mean_m"] for row in rows], dtype=np.float32)
    error = np.array([row["error_scalar"] for row in rows], dtype=np.float32)
    uncertainty_bev = np.array([row["unc_bev_m"] for row in rows], dtype=np.float32)
    distance = np.array([row["range_xy_m"] for row in rows], dtype=np.float32)

    valid_bin_means = [value for value in bin_stats["bin_mean_error"] if value is not None]
    monotonic_steps = 0
    monotonic_total = max(0, len(valid_bin_means) - 1)
    for prev, cur in zip(valid_bin_means[:-1], valid_bin_means[1:]):
        if cur >= prev:
            monotonic_steps += 1

    return {
        "num_predictions": int(len(rows)),
        "matched_predictions": int(sum(row["matched"] for row in rows)),
        "pearson_uncertainty_error": safe_corr(uncertainty, error),
        "spearman_uncertainty_error": safe_spearman(uncertainty, error),
        "pearson_distance_uncertainty": safe_corr(distance, uncertainty),
        "pearson_distance_error": safe_corr(distance, error),
        "mean_uncertainty_m": float(uncertainty.mean()),
        "mean_bev_uncertainty_m": float(uncertainty_bev.mean()),
        "mean_error_scalar": float(error.mean()),
        "monotonic_bin_steps": monotonic_steps,
        "monotonic_bin_total": monotonic_total,
        "effect_supported": bool(
            safe_corr(uncertainty, error) is not None
            and safe_corr(uncertainty, error) > 0
            and monotonic_total > 0
            and monotonic_steps >= max(1, monotonic_total // 2)
        ),
    }


def write_report(save_dir, args, summary, issues, bin_stats, pred_pkl_path):
    lines = [
        "# uncertain_influence report",
        "",
        "## Setup",
        f"- Prediction file: `{pred_pkl_path}`",
        f"- GT info file: `{args.gt_info_pkl}`",
        f"- Error scalar definition: `1 - IoU_bev`",
        f"- Match IoU minimum: `{args.match_iou_min}`",
        f"- Score threshold: `{args.score_thresh}`",
        f"- Core figure: `uncertainty_vs_error_bin.png`",
        f"- Scatter figure: `uncertainty_vs_error_scatter.png`",
        f"- Distance uncertainty figure: `distance_uncertainty.png`",
        f"- Distance error figure: `distance_error.png`",
        "",
        "## Summary",
        f"- Number of predictions used: `{summary['num_predictions']}`",
        f"- Matched predictions: `{summary['matched_predictions']}`",
        f"- Pearson(uncertainty, error): `{summary['pearson_uncertainty_error']}`",
        f"- Spearman(uncertainty, error): `{summary['spearman_uncertainty_error']}`",
        f"- Pearson(distance, uncertainty): `{summary['pearson_distance_uncertainty']}`",
        f"- Pearson(distance, error): `{summary['pearson_distance_error']}`",
        f"- Mean uncertainty: `{summary['mean_uncertainty_m']:.4f} m`",
        f"- Mean error scalar: `{summary['mean_error_scalar']:.4f}`",
        f"- Monotonic bins: `{summary['monotonic_bin_steps']}/{summary['monotonic_bin_total']}`",
        f"- Effect supported: `{summary['effect_supported']}`",
        "",
        "## Bin Means",
    ]
    for label, mean_error, count in zip(
        bin_stats["bin_labels"], bin_stats["bin_mean_error"], bin_stats["bin_counts"]
    ):
        lines.append(f"- {label}: mean_error=`{mean_error}`, count=`{count}`")

    lines.extend(["", "## Notes"])
    if issues:
        for issue in issues[:20]:
            lines.append(f"- {issue}")
        if len(issues) > 20:
            lines.append(f"- ... and {len(issues) - 20} more issues")
    else:
        lines.append("- No parsing issues were detected.")

    if not summary["effect_supported"]:
        lines.extend(
            [
                "",
                "## Discussion",
                "- Current data does not yet give a strong positive `uncertainty -> error` trend.",
                "- First check whether the prediction file truly contains Gaussian uncertainty exports.",
                "- If the trend is weak even with valid exports, we should discuss calibration or scoring changes next.",
            ]
        )

    (save_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def resolve_default_pred_pkl(project_root):
    candidates = [
        project_root / "output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/result_with_uncertainty_epoch_80.pkl",
        project_root / "output/kitti_models/GLENet_VR_gaussian/full_gaussian_run/eval/eval_with_train/epoch_80/val/final_result/data/result_with_uncertainty_epoch_80.pkl",
        project_root / "output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/result.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze uncertainty influence on detection error")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="tools/cfgs/kitti_models/GLENet_VR_gaussian.yaml",
    )
    parser.add_argument(
        "--pred_pkl",
        type=str,
        default=None,
        help="Prediction pickle, preferably result_with_uncertainty_epoch_*.pkl",
    )
    parser.add_argument(
        "--gt_info_pkl",
        type=str,
        default="data/kitti/kitti_infos_val.pkl",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="uncertain_influence/results",
    )
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--match_iou_min", type=float, default=0.1)
    parser.add_argument("--num_bins", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_plot_style()
    project_root = Path(__file__).resolve().parent.parent
    cfg_file = Path(args.cfg_file)
    gt_info_pkl = Path(args.gt_info_pkl)
    pred_pkl = Path(args.pred_pkl) if args.pred_pkl else resolve_default_pred_pkl(project_root)
    save_dir = Path(args.save_dir)

    if not cfg_file.is_absolute():
        cfg_file = (project_root / cfg_file).resolve()
    if not gt_info_pkl.is_absolute():
        gt_info_pkl = (project_root / gt_info_pkl).resolve()
    if not pred_pkl.is_absolute():
        pred_pkl = (project_root / pred_pkl).resolve()
    if not save_dir.is_absolute():
        save_dir = (project_root / save_dir).resolve()

    save_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(save_dir)

    os.chdir(project_root / "tools")
    cfg_from_yaml_file(str(cfg_file), cfg)

    logger.info("uncertain_influence started")
    logger.info("cfg_file=%s", cfg_file)
    logger.info("pred_pkl=%s", pred_pkl)
    logger.info("gt_info_pkl=%s", gt_info_pkl)
    logger.info("error_scalar = 1 - IoU_bev")

    if not pred_pkl.exists():
        message = (
            f"Prediction file not found: {pred_pkl}. "
            "Please export Gaussian uncertainty first with tools/test_gaussian.py."
        )
        logger.error(message)
        (save_dir / "analysis_report.md").write_text(
            "# uncertain_influence report\n\n- " + message + "\n",
            encoding="utf-8",
        )
        return

    records = load_prediction_records(pred_pkl)
    if not isinstance(records, list) or len(records) == 0:
        logger.error("Prediction file is empty or unsupported.")
        return

    first_record = records[0]
    if "uncertainty_xyz" not in first_record and "pred_uncertainty" not in first_record:
        logger.warning("Prediction file does not contain uncertainty fields.")
        (save_dir / "analysis_report.md").write_text(
            "\n".join(
                [
                    "# uncertain_influence report",
                    "",
                    f"- Loaded `{pred_pkl}` but it has no `uncertainty_xyz` / `pred_uncertainty` field.",
                    "- Current `result.pkl` is standard detection output only, so uncertainty-vs-error cannot be validated yet.",
                    "- Next step: run `tools/test_gaussian.py` to produce `result_with_uncertainty_epoch_*.pkl`, then rerun this script.",
                ]
            ),
            encoding="utf-8",
        )
        return

    gt_by_frame = load_gt_by_frame(gt_info_pkl, cfg.CLASS_NAMES)
    rows, issues = build_rows(records, gt_by_frame, cfg.CLASS_NAMES, args.score_thresh, args.match_iou_min)

    if not rows:
        logger.warning("No valid rows were constructed from the current data.")
        return

    with open(save_dir / "uncertainty_error_rows.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bin_stats = draw_core_bin_plot(rows, save_dir / "uncertainty_vs_error_bin.png", args.num_bins)
    draw_scatter_plot(rows, save_dir / "uncertainty_vs_error_scatter.png")
    draw_distance_uncertainty_plot(rows, save_dir / "distance_uncertainty.png")
    draw_distance_error_plot(rows, save_dir / "distance_error.png")
    summary = compute_summary(rows, bin_stats)
    summary.update(
        {
            "cfg_file": str(cfg_file),
            "pred_pkl": str(pred_pkl),
            "gt_info_pkl": str(gt_info_pkl),
            "error_scalar_definition": "1 - IoU_bev",
        }
    )

    with open(save_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_report(save_dir, args, summary, issues, bin_stats, pred_pkl)
    logger.info("Rows written: %d", len(rows))
    logger.info("Effect supported: %s", summary["effect_supported"])
    logger.info("Results saved to %s", save_dir)


if __name__ == "__main__":
    main()
