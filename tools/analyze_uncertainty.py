
import pickle
import numpy as np
import argparse
import sys
from pathlib import Path

def analyze_uncertainty(file_path):
    print(f"Loading results from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    print(f"Total number of frames: {len(data)}")
    if len(data) == 0:
        print("Empty data file.")
        return

    # Statistics containers
    all_sigmas = []
    class_sigmas = {}

    for frame in data:
        # frame keys: 'frame_id', 'boxes_3d', 'scores', 'labels', 'uncertainty_xyz'
        # labels are usually 1-based class indices in PCDET pkls
        unc = frame['uncertainty_xyz']
        labels = frame['labels']

        if len(unc) > 0:
            all_sigmas.append(unc)

            for i, label in enumerate(labels):
                if label not in class_sigmas:
                    class_sigmas[label] = []
                class_sigmas[label].append(unc[i])

    if all_sigmas:
        all_sigmas_concat = np.concatenate(all_sigmas, axis=0)
        print(f"\nOverall Statistics (detected objects):")
        print(f"  Count: {all_sigmas_concat.shape[0]}")
        print(f"  Mean Sigma (X): {np.mean(all_sigmas_concat[:, 0]):.4f}")
        print(f"  Mean Sigma (Y): {np.mean(all_sigmas_concat[:, 1]):.4f}")
        print(f"  Mean Sigma (Z): {np.mean(all_sigmas_concat[:, 2]):.4f}")
        print(f"  Min Sigma: {np.min(all_sigmas_concat):.4f}")
        print(f"  Max Sigma: {np.max(all_sigmas_concat):.4f}")

    print("\nPer Class Statistics:")
    for cls_id, sigmas in class_sigmas.items():
        sigmas_np = np.array(sigmas)
        print(f"  Class {cls_id}:")
        print(f"    Count: {sigmas_np.shape[0]}")
        print(f"    Mean Sigma (X,Y,Z): {np.mean(sigmas_np, axis=0)}")

    print("\n--- Sample View (First 3 frames with objects) ---")
    count_viz = 0
    for i in range(len(data)):
        frame = data[i]
        num_obj = len(frame['boxes_3d'])
        if num_obj == 0:
            continue

        print(f"Frame ID: {frame['frame_id']}")
        print(f"  Object Count: {num_obj}")
        for j in range(min(5, num_obj)):
            box = frame['boxes_3d'][j]
            unc = frame['uncertainty_xyz'][j]
            score = frame['scores'][j]
            label = frame['labels'][j]
            # box is usually [x, y, z, dx, dy, dz, heading]
            print(f"    Obj {j} (Class {label}): Score={score:.3f}")
            print(f"      Pos: [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}]")
            print(f"      Unc (StdDev): [{unc[0]:.4f}, {unc[1]:.4f}, {unc[2]:.4f}]")

        count_viz += 1
        if count_viz >= 3:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to the result pickle file')
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File {args.file} does not exist.")
        sys.exit(1)

    analyze_uncertainty(args.file)
