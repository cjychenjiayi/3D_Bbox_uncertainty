import argparse
import json
import pickle
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, Polygon
from pcdet.utils import box_utils, calibration_kitti


DEFAULT_FRAMES = [
    "007458",
    "003219",
    "000590",
    "002050",
    "006833",
    "006163",
    "005426",
    "006034",
]

UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list(
    "paper_uncertainty",
    ["#D8B400", "#4DAA57", "#1558A8", "#2D004B"],
)
FALLBACK_VIEW_FRUSTUM_ANGLES_DEG = (7.5, 0.0, -7.5)
EGO_LABEL_FONTSIZE = 22
RANGE_LABEL_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 32
TICK_LABEL_FONTSIZE = 27
LEGEND_FONTSIZE = 25
COLORBAR_LABEL_FONTSIZE = 28
COLORBAR_TICK_FONTSIZE = 24
GRID_COLOR = "#7f95aa"
GRID_LINEWIDTH = 1.65


def to_frame_id(frame_id):
    if isinstance(frame_id, bytes):
        frame_id = frame_id.decode("utf-8")
    if isinstance(frame_id, (int, np.integer)):
        return f"{int(frame_id):06d}"
    return str(frame_id)


def decode_uncertainty(boxes, uncertainty):
    if len(boxes) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    dims = np.maximum(boxes[:, 3:6], 1e-3)
    diag = np.sqrt(dims[:, 0] ** 2 + dims[:, 1] ** 2)
    unc_m = np.zeros_like(uncertainty, dtype=np.float32)
    unc_m[:, 0] = uncertainty[:, 0] * diag
    unc_m[:, 1] = uncertainty[:, 1] * diag
    unc_m[:, 2] = uncertainty[:, 2] * dims[:, 2]
    return unc_m


def bev_corners(box):
    x, y, _, dx, dy, _, heading = box[:7]
    template = np.array(
        [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )
    local = template * np.array([dx, dy], dtype=np.float32)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def load_records(pred_pkl):
    with open(pred_pkl, "rb") as f:
        records = pickle.load(f)
    return {to_frame_id(record.get("frame_id")): record for record in records}


def filter_prediction(record, score_thresh):
    boxes = np.asarray(record.get("boxes_3d", []), dtype=np.float32)
    scores = np.asarray(record.get("scores", []), dtype=np.float32).reshape(-1)
    labels = np.asarray(record.get("labels", []), dtype=np.int32).reshape(-1)
    unc = np.asarray(record.get("uncertainty_xyz", []), dtype=np.float32)
    if boxes.size == 0 or scores.size == 0 or labels.size == 0 or unc.size == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
        )

    count = min(len(boxes), len(scores), len(labels), len(unc))
    boxes = boxes[:count]
    scores = scores[:count]
    labels = labels[:count]
    unc = unc[:count]
    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    unc = unc[keep]
    return boxes, scores, labels, decode_uncertainty(boxes, unc)


def read_gt_boxes_lidar(frame_id, gt_dir, calib_dir):
    if gt_dir is None:
        return np.zeros((0, 7), dtype=np.float32)
    label_path = gt_dir / f"{frame_id}.txt"
    calib_path = calib_dir / f"{frame_id}.txt"
    if not label_path.exists() or not calib_path.exists():
        return np.zeros((0, 7), dtype=np.float32)

    calib = calibration_kitti.Calibration(str(calib_path))
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[0] != "Car":
            continue
        h, w, l = map(float, parts[8:11])
        x, y, z = map(float, parts[11:14])
        ry = float(parts[14])
        loc_lidar = calib.rect_to_lidar(np.array([[x, y, z]], dtype=np.float32))[0]
        loc_lidar[2] += h / 2.0
        heading = -(np.pi / 2.0 + ry)
        boxes.append([loc_lidar[0], loc_lidar[1], loc_lidar[2], l, w, h, heading])
    if not boxes:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def camera_bev_fov_angles(calib_path, image_path):
    if not calib_path.exists() or not image_path.exists():
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    calib = calibration_kitti.Calibration(str(calib_path))
    image_width = image.shape[1]
    lidar_origin = calib.rect_to_lidar(np.zeros((1, 3), dtype=np.float32))[0]

    def angle_for_u(u):
        ray_rect = np.array(
            [[(float(u) - float(calib.cu)) / float(calib.fu), 0.0, 1.0]],
            dtype=np.float32,
        )
        ray_lidar = calib.rect_to_lidar(ray_rect)[0] - lidar_origin
        return float(np.rad2deg(np.arctan2(ray_lidar[1], ray_lidar[0])))

    left_angle = angle_for_u(0.0)
    center_angle = angle_for_u(float(calib.cu))
    right_angle = angle_for_u(float(image_width - 1))
    return (left_angle, center_angle, right_angle)


def uncertainty_norm(unc_scalar):
    return Normalize(vmin=0.2, vmax=0.4, clip=True)


def draw_projected_box3d(image, corners, color, thickness=2):
    corners = np.asarray(corners, dtype=np.int32)
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for start, end in edges:
        p1 = tuple(corners[start])
        p2 = tuple(corners[end])
        cv2.line(image, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    return image


def color_to_bgr(color):
    rgba = np.asarray(color[:3]) * 255.0
    return tuple(int(v) for v in rgba[::-1])


def draw_3d_from_prediction(
    record,
    frame_id,
    image_dir,
    calib_dir,
    save_path,
    score_thresh=0.3,
    max_boxes=45,
):
    image_path = image_dir / f"{frame_id}.png"
    calib_path = calib_dir / f"{frame_id}.txt"
    image = cv2.imread(str(image_path))
    if image is None or not calib_path.exists():
        return False

    boxes, scores, labels, unc_m = filter_prediction(record, score_thresh)
    if len(boxes) == 0:
        return False

    unc_scalar = unc_m[:, :2].mean(axis=1)
    norm = uncertainty_norm(unc_scalar)
    cmap = UNCERTAINTY_CMAP
    calib = calibration_kitti.Calibration(str(calib_path))

    order = np.argsort(boxes[:, 0])[::-1]
    if max_boxes > 0:
        order = order[:max_boxes]

    corners_lidar = box_utils.boxes_to_corners_3d(boxes[:, :7])
    corners_rect = calib.lidar_to_rect(corners_lidar.reshape(-1, 3)).reshape(-1, 8, 3)
    pts_img, pts_depth = calib.rect_to_img(corners_rect.reshape(-1, 3))
    pts_img = pts_img.reshape(-1, 8, 2)
    pts_depth = pts_depth.reshape(-1, 8)

    h, w = image.shape[:2]
    overlay = image.copy()
    for idx in order:
        if np.nanmax(pts_depth[idx]) <= 0:
            continue
        box2d = pts_img[idx]
        if (
            np.nanmax(box2d[:, 0]) < 0
            or np.nanmin(box2d[:, 0]) > w - 1
            or np.nanmax(box2d[:, 1]) < 0
            or np.nanmin(box2d[:, 1]) > h - 1
        ):
            continue

        color = color_to_bgr(cmap(norm(float(unc_scalar[idx]))))
        thickness = 2 if float(unc_scalar[idx]) < norm.vmax * 0.72 else 3
        draw_projected_box3d(image, box2d, color=color, thickness=thickness)

        x1, y1 = np.nanmin(box2d, axis=0)
        x2, y2 = np.nanmax(box2d, axis=0)
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        if x2 > x1 and y2 > y1:
            alpha = 0.05 + 0.11 * norm(float(unc_scalar[idx]))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
            image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
            overlay = image.copy()

    cv2.imwrite(str(save_path), image)
    return True


def draw_ego(ax):
    car = np.array([[-1.0, -0.9], [1.6, -0.9], [2.4, 0.0], [1.6, 0.9], [-1.0, 0.9]])
    ax.add_patch(
        Polygon(
            car,
            closed=True,
            facecolor="#1f2933",
            edgecolor="white",
            linewidth=1.2,
            zorder=10,
        )
    )
    ax.scatter([0.0], [0.0], s=16, color="white", zorder=11)
    ax.text(
        0.0,
        -1.65,
        "Ego",
        color="#D73027",
        fontsize=EGO_LABEL_FONTSIZE,
        fontweight="bold",
        ha="center",
        va="top",
        zorder=12,
    )


def frustum_endpoint(max_range, angle_deg):
    angle = np.deg2rad(angle_deg)
    return np.array([max_range * np.cos(angle), max_range * np.sin(angle)])


def draw_view_frustum(ax, max_range=72.0, fov_angles_deg=None):
    left_angle, center_angle, right_angle = fov_angles_deg or FALLBACK_VIEW_FRUSTUM_ANGLES_DEG
    left = frustum_endpoint(max_range, left_angle)
    center = frustum_endpoint(max_range, center_angle)
    right = frustum_endpoint(max_range, right_angle)
    arc_angles = np.linspace(min(left_angle, right_angle), max(left_angle, right_angle), 96)
    arc_points = np.array([frustum_endpoint(max_range, angle) for angle in arc_angles])
    frustum = np.vstack([[0.0, 0.0], arc_points])
    ax.add_patch(
        Polygon(
            frustum,
            closed=True,
            facecolor="#dbe6f0",
            edgecolor="none",
            alpha=0.92,
            zorder=0,
        )
    )
    ax.plot([0, left[0]], [0, left[1]], color="#526d86", linewidth=1.35, alpha=0.82, zorder=1)
    ax.plot([0, right[0]], [0, right[1]], color="#526d86", linewidth=1.35, alpha=0.82, zorder=1)
    ax.plot([0, center[0]], [0, center[1]], color="#718da6", linewidth=1.0, alpha=0.78, zorder=1)


def draw_bev(
    record,
    frame_id,
    save_path,
    score_thresh=0.3,
    uncertainty_scale=3.2,
    gt_boxes=None,
    view_frustum_angles_deg=None,
):
    boxes, scores, labels, unc_m = filter_prediction(record, score_thresh)
    if len(boxes) == 0:
        return None

    unc_scalar = unc_m[:, :2].mean(axis=1)
    norm = uncertainty_norm(unc_scalar)
    cmap = UNCERTAINTY_CMAP

    fig = plt.figure(figsize=(13.2, 4.75), dpi=220)
    ax = fig.add_axes([0.074, 0.150, 0.818, 0.780])
    ax.set_facecolor("#fbfcfd")

    draw_view_frustum(ax, fov_angles_deg=view_frustum_angles_deg)

    for dist in [10, 20, 30, 40, 50, 60, 70]:
        ax.axvline(dist, color=GRID_COLOR, linewidth=GRID_LINEWIDTH, linestyle="--", alpha=0.95, zorder=1)
        ax.text(dist, -12.7, f"{dist}m", color="#52677a", fontsize=RANGE_LABEL_FONTSIZE, ha="center", va="bottom")

    for y in [-10, -5, 0, 5, 10]:
        ax.axhline(
            y,
            color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH,
            linestyle="-" if y == 0 else "--",
            alpha=0.95,
            zorder=1,
        )

    line_segments = []
    line_colors = []
    order = np.argsort(boxes[:, 0])
    for idx in order:
        x, y = float(boxes[idx, 0]), float(boxes[idx, 1])
        color = cmap(norm(float(unc_scalar[idx])))
        line_segments.append([(0.0, 0.0), (x, y)])
        line_colors.append(color)

    if line_segments:
        rays = LineCollection(line_segments, colors=line_colors, linewidths=0.9, alpha=0.24, zorder=2)
        ax.add_collection(rays)

    gt_boxes = np.asarray(gt_boxes if gt_boxes is not None else [], dtype=np.float32).reshape(-1, 7)
    for gt_box in gt_boxes:
        corners = bev_corners(gt_box)
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor="none",
                edgecolor="#D73027",
                linewidth=2.5,
                alpha=0.98,
                zorder=4,
            )
        )

    for idx in order:
        box = boxes[idx]
        x, y = float(box[0]), float(box[1])
        ux = max(float(unc_m[idx, 0]) * uncertainty_scale, 0.16)
        uy = max(float(unc_m[idx, 1]) * uncertainty_scale, 0.16)
        color = cmap(norm(float(unc_scalar[idx])))

        for sigma, alpha in [(2.6, 0.08), (1.7, 0.13), (1.0, 0.22)]:
            ax.add_patch(
                Ellipse(
                    (x, y),
                    width=2.0 * sigma * ux,
                    height=2.0 * sigma * uy,
                    angle=0.0,
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=3,
                )
            )

        corners = bev_corners(box)
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor="none",
                edgecolor=color,
                linewidth=2.2,
                alpha=0.98,
                zorder=5,
            )
        )
        ax.scatter(x, y, s=20, color=color, edgecolor="white", linewidth=0.45, zorder=6)

    draw_ego(ax)

    ax.set_xlim(-6.0, 74.0)
    ax.set_ylim(-14.0, 14.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")
    ax.set_xlabel("Forward range (m)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=9)
    ax.set_ylabel("Lateral (m)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=-6)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE, colors="#455a69", length=6, width=1.05)
    for spine in ax.spines.values():
        spine.set_color("#a6b7c8")
        spine.set_linewidth(1.0)

    if len(gt_boxes) > 0:
        gt_proxy = plt.Line2D([0], [0], color="#D73027", linewidth=2.5, label="GT")
        pred_proxy = plt.Line2D([0], [0], color="#2166AC", linewidth=2.4, label="Prediction")
        ax.legend(
            handles=[gt_proxy, pred_proxy],
            loc="upper right",
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#d9e2ec",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.898, 0.150, 0.020, 0.780])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("BEV uncertainty (m)", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=12)
    cbar.set_ticks([0.20, 0.25, 0.30, 0.35, 0.40])
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "num_predictions": int(len(boxes)),
        "mean_uncertainty_bev_m": float(unc_scalar.mean()),
        "max_uncertainty_bev_m": float(unc_scalar.max()),
        "max_range_m": float(np.linalg.norm(boxes[:, :2], axis=1).max()),
        "view_frustum_angles_deg": [float(v) for v in (view_frustum_angles_deg or FALLBACK_VIEW_FRUSTUM_ANGLES_DEG)],
    }


def resize_to_width(image, width):
    h, w = image.shape[:2]
    scale = width / float(w)
    return cv2.resize(image, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def make_composite(frame_id, image_3d_path, bev_path, save_path, width=1800):
    image_3d = cv2.imread(str(image_3d_path))
    image_bev = cv2.imread(str(bev_path))
    if image_3d is None or image_bev is None:
        return False

    image_3d = resize_to_width(image_3d, width)
    target_bev_height = int(round(image_3d.shape[0] * 1.10))
    bev_side_pad = 34
    image_bev = cv2.resize(
        image_bev,
        (width - 2 * bev_side_pad, target_bev_height),
        interpolation=cv2.INTER_AREA,
    )
    image_bev = cv2.copyMakeBorder(
        image_bev,
        24,
        0,
        bev_side_pad,
        bev_side_pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    separator = np.full((18, width, 3), 255, dtype=np.uint8)
    composite = np.vstack([image_3d, separator, image_bev])
    cv2.imwrite(str(save_path), composite)
    return True


def resolve_source_3d_path(source_dir, frame_id):
    for suffix in ("_3d.png", "_3d_paper.png"):
        source_3d_path = source_dir / f"{frame_id}{suffix}"
        if source_3d_path.exists():
            return source_3d_path
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build paper-ready 3D+BEV uncertainty case figures.")
    parser.add_argument(
        "--pred_pkl",
        type=str,
        default="output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/final_result/data/result_with_uncertainty_epoch_80.pkl",
    )
    parser.add_argument("--source_dir", type=str, default="visualization_outputs/vis_full_uncertainty_v2")
    parser.add_argument("--save_dir", type=str, default="visualization_outputs/vis_full_uncertainty_v2_paper_cases")
    parser.add_argument("--image_dir", type=str, default="data/kitti/testing/image_2")
    parser.add_argument("--calib_dir", type=str, default="data/kitti/testing/calib")
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--draw_3d_from_pkl", action="store_true")
    parser.add_argument(
        "--frames",
        type=str,
        default=",".join(DEFAULT_FRAMES),
        help="Comma-separated frame ids, or 'all' to render every available frame.",
    )
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--summary_path", type=str, default="selected_frames.json")
    return parser.parse_args(argv)


def available_frame_ids(records, source_dir, score_thresh, require_source_3d=True):
    frame_ids = []
    for frame_id in sorted(records):
        if require_source_3d and resolve_source_3d_path(source_dir, frame_id) is None:
            continue
        boxes, _, _, _ = filter_prediction(records[frame_id], score_thresh)
        if len(boxes) == 0:
            continue
        frame_ids.append(frame_id)
    return frame_ids


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    pred_pkl = Path(args.pred_pkl)
    source_dir = Path(args.source_dir)
    save_dir = Path(args.save_dir)
    image_dir = Path(args.image_dir)
    calib_dir = Path(args.calib_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    summary_path = Path(args.summary_path)

    if not pred_pkl.is_absolute():
        pred_pkl = root / pred_pkl
    if not source_dir.is_absolute():
        source_dir = root / source_dir
    if not save_dir.is_absolute():
        save_dir = root / save_dir
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    if not calib_dir.is_absolute():
        calib_dir = root / calib_dir
    if gt_dir is not None and not gt_dir.is_absolute():
        gt_dir = root / gt_dir
    if not summary_path.is_absolute():
        summary_path = save_dir / summary_path

    bev_dir = save_dir / "bev"
    image_3d_dir = save_dir / "3d"
    combo_dir = save_dir / "stacked"
    bev_dir.mkdir(parents=True, exist_ok=True)
    image_3d_dir.mkdir(parents=True, exist_ok=True)
    combo_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(pred_pkl)
    if args.frames.strip().lower() == "all":
        frame_ids = available_frame_ids(
            records,
            source_dir,
            args.score_thresh,
            require_source_3d=not args.draw_3d_from_pkl,
        )
    else:
        frame_ids = [frame.strip() for frame in args.frames.split(",") if frame.strip()]
    summary = {}

    for pos, frame_id in enumerate(frame_ids, start=1):
        if frame_id not in records:
            print(f"skip {frame_id}: no prediction record", flush=True)
            continue

        source_3d_path = resolve_source_3d_path(source_dir, frame_id)
        generated_3d_path = image_3d_dir / f"{frame_id}_3d_paper.png"
        if args.draw_3d_from_pkl:
            made_3d = draw_3d_from_prediction(
                records[frame_id],
                frame_id,
                image_dir,
                calib_dir,
                generated_3d_path,
                score_thresh=args.score_thresh,
            )
            if not made_3d:
                print(f"skip {frame_id}: could not draw 3D visualization", flush=True)
                continue
            image_3d_path = generated_3d_path
        else:
            if source_3d_path is None:
                print(f"skip {frame_id}: no 3D visualization", flush=True)
                continue
            image_3d_path = source_3d_path

        gt_boxes = read_gt_boxes_lidar(frame_id, gt_dir, calib_dir)
        view_frustum_angles = camera_bev_fov_angles(
            calib_dir / f"{frame_id}.txt",
            image_dir / f"{frame_id}.png",
        )
        if view_frustum_angles is None:
            print(f"warning {frame_id}: could not read camera FOV, using fallback frustum", flush=True)
        bev_path = bev_dir / f"{frame_id}_bev_paper.png"
        stats = draw_bev(
            records[frame_id],
            frame_id,
            bev_path,
            score_thresh=args.score_thresh,
            gt_boxes=gt_boxes,
            view_frustum_angles_deg=view_frustum_angles,
        )
        if stats is None:
            print(f"skip {frame_id}: no predictions after threshold", flush=True)
            continue
        stats["num_gt"] = int(len(gt_boxes))
        combo_path = combo_dir / f"{frame_id}_paper_stack.png"
        made = make_composite(frame_id, image_3d_path, bev_path, combo_path)
        stats["stacked"] = made
        summary[frame_id] = stats
        if pos <= 10 or pos % 100 == 0 or pos == len(frame_ids):
            print(f"saved {pos}/{len(frame_ids)} {frame_id}: {stats}", flush=True)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved paper cases to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
