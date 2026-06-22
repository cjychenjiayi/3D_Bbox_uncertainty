import sys

from make_paper_uncertainty_cases import main


if __name__ == "__main__":
    defaults = [
        "--pred_pkl",
        "output/kitti_models/GLENet_VR_gaussian/default/eval/epoch_80/val/default/final_result/data/result_with_uncertainty_epoch_80.pkl",
        "--save_dir",
        "visualization_outputs/vis_full_uncertainty_val_paper_cases",
        "--image_dir",
        "data/kitti/training/image_2",
        "--calib_dir",
        "data/kitti/training/calib",
        "--gt_dir",
        "data/kitti/training/label_2",
        "--frames",
        "all",
        "--draw_3d_from_pkl",
        "--score_thresh",
        "0.3",
    ]
    main(defaults + sys.argv[1:])
