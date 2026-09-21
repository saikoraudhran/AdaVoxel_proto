#!/usr/bin/env python3
"""
compare_pointclouds.py

Visualizes a raw SemanticKITTI .bin scan next to its preprocessed .npz
counterpart, so you can see the effect of remapping/downsampling.

Two view modes:
  --side_by_side   opens two separate Open3D windows (raw, then preprocessed)
  --overlay        (default) shows the preprocessed cloud only, colored by
                    class, which is usually more informative than comparing
                    raw vs preprocessed visually (raw has no color info)

Also prints basic stats: point counts, per-class point counts, and the
downsampling ratio if applicable.

Usage:
    python compare_pointclouds.py \
        --raw_bin  "C:\\Users\\dhanu\\SIH\\Point_Clouds\\000000.bin" \
        --prep_npz "C:\\Users\\dhanu\\SIH\\Prep_Point_Clouds\\000000.npz" \
        --side_by_side
"""

import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


# Class id -> RGB color (0-1 range), matching the reduced scheme from
# preprocess_semantickitti.py
CLASS_COLORS = {
    0: [0.5, 0.5, 0.5],   # ignore/unlabeled -> gray
    1: [0.2, 0.8, 0.2],   # drivable -> green
    2: [0.8, 0.8, 0.2],   # non_drivable_terrain -> yellow
    3: [0.8, 0.2, 0.2],   # static_obstacle -> red
    4: [0.2, 0.2, 0.9],   # dynamic_object -> blue
}
CLASS_NAMES = {
    0: "ignore", 1: "drivable", 2: "non_drivable_terrain",
    3: "static_obstacle", 4: "dynamic_object",
}


def load_raw_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_prep_npz(path):
    data = np.load(path)
    return data["points"], data["sem_label"]


def print_stats(raw_points, prep_points, sem_label):
    print(f"\nRaw points:          {raw_points.shape[0]:,}")
    print(f"Preprocessed points: {prep_points.shape[0]:,}")
    if raw_points.shape[0] > 0:
        ratio = prep_points.shape[0] / raw_points.shape[0] * 100
        print(f"Kept:                {ratio:.1f}% (downsampling removed the rest)")

    print("\nClass breakdown (preprocessed):")
    for cls_id, name in CLASS_NAMES.items():
        count = int(np.sum(sem_label == cls_id))
        if count > 0:
            pct = count / len(sem_label) * 100
            print(f"  {cls_id} ({name:22s}): {count:8,} points  ({pct:5.1f}%)")


def make_o3d_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def colors_from_labels(sem_label):
    colors = np.zeros((len(sem_label), 3), dtype=np.float64)
    for cls_id, rgb in CLASS_COLORS.items():
        colors[sem_label == cls_id] = rgb
    return colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_bin", required=True, help="Path to original .bin file")
    parser.add_argument("--prep_npz", required=True, help="Path to preprocessed .npz file")
    parser.add_argument("--side_by_side", action="store_true",
                         help="Open two separate windows: raw first, then preprocessed")
    args = parser.parse_args()

    if o3d is None:
        print("Open3D is not installed. Run: pip install open3d")
        return

    raw_points = load_raw_bin(args.raw_bin)
    prep_points, sem_label = load_prep_npz(args.prep_npz)

    print_stats(raw_points, prep_points, sem_label)

    prep_colors = colors_from_labels(sem_label)

    if args.side_by_side:
        print("\nOpening RAW point cloud (uncolored) -- close window to continue...")
        raw_cloud = make_o3d_cloud(raw_points)
        raw_cloud.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.visualization.draw_geometries([raw_cloud], window_name="Raw (.bin)")

        print("Opening PREPROCESSED point cloud (colored by class)...")
        prep_cloud = make_o3d_cloud(prep_points, prep_colors)
        o3d.visualization.draw_geometries([prep_cloud], window_name="Preprocessed (.npz)")
    else:
        print("\nOpening preprocessed point cloud, colored by class:")
        print("  gray=ignore  green=drivable  yellow=non-drivable terrain  red=static obstacle  blue=dynamic object")
        prep_cloud = make_o3d_cloud(prep_points, prep_colors)
        o3d.visualization.draw_geometries([prep_cloud], window_name="Preprocessed (.npz) - colored by class")


if __name__ == "__main__":
    main()
