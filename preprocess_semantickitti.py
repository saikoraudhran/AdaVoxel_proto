#!/usr/bin/env python3
"""
preprocess_semantickitti.py

Preprocesses raw SemanticKITTI velodyne (.bin) + label (.label) files into
a reduced, ML-ready class scheme, and writes the result into an output
directory as compressed .npz files (one per scan), mirroring the original
sequences/<seq>/ structure.

Expected input layout (standard SemanticKITTI extraction):

    <dataset_root>/
        sequences/
            00/
                velodyne/000000.bin ...
                labels/000000.label ...   (only present for seq 00-10)
                poses.txt
            01/
            ...
            21/                            (test seqs, no labels/ folder)

Output layout:

    <output_dir>/
        sequences/
            00/
                000000.npz
                000001.npz
                ...

Each .npz contains:
    points      (N, 4) float32   -> x, y, z, remission (intensity)
    sem_label   (N,)   uint8     -> remapped class id (see CLASS_MAP below)
    inst_label  (N,)   uint32    -> original instance id (0 if none/test set)

Usage:
    python preprocess_semantickitti.py \
        --dataset_root /path/to/dataset \
        --output_dir   /path/to/output \
        --sequences 00 01 02 \
        --voxel_size 0.05 \
        --workers 8
"""

import argparse
import os
import numpy as np
from functools import partial
from multiprocessing import Pool

# --------------------------------------------------------------------------
# 1. Raw SemanticKITTI label id -> class name (official semantic-kitti.yaml)
# --------------------------------------------------------------------------
RAW_LABELS = {
    0: "unlabeled", 1: "outlier",
    10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 16: "on-rails",
    18: "truck", 20: "other-vehicle",
    30: "person", 31: "bicyclist", 32: "motorcyclist",
    40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
    50: "building", 51: "fence", 52: "other-structure",
    60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain",
    80: "pole", 81: "traffic-sign", 99: "other-object",
    252: "moving-car", 253: "moving-bicyclist", 254: "moving-person",
    255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
    258: "moving-truck", 259: "moving-other-vehicle",
}

# --------------------------------------------------------------------------
# 2. Reduced class scheme for the foveated-grid pipeline.
#    Adjust freely -- this is the part you'll want to tune per your model.
# --------------------------------------------------------------------------
NEW_CLASSES = {
    0: "ignore",              # unlabeled / outlier / other-object / noise
    1: "drivable",            # road, parking, lane-marking
    2: "non_drivable_terrain",# sidewalk, terrain, other-ground
    3: "static_obstacle",     # buildings, fences, poles, vegetation, etc.
    4: "dynamic_object",      # vehicles + people, moving or not
}

NAME_TO_NEW_ID = {
    "unlabeled": 0, "outlier": 0, "other-object": 0, "other-structure": 3,

    "road": 1, "parking": 1, "lane-marking": 1,

    "sidewalk": 2, "terrain": 2, "other-ground": 2,

    "building": 3, "fence": 3, "pole": 3, "traffic-sign": 3,
    "vegetation": 3, "trunk": 3,

    "car": 4, "bicycle": 4, "bus": 4, "motorcycle": 4, "on-rails": 4,
    "truck": 4, "other-vehicle": 4, "person": 4, "bicyclist": 4,
    "motorcyclist": 4,
    "moving-car": 4, "moving-bicyclist": 4, "moving-person": 4,
    "moving-motorcyclist": 4, "moving-on-rails": 4, "moving-bus": 4,
    "moving-truck": 4, "moving-other-vehicle": 4,
}

# Build a fast lookup table: raw_label_id -> new_class_id
_MAX_RAW_ID = max(RAW_LABELS.keys()) + 1
LOOKUP_TABLE = np.zeros(_MAX_RAW_ID, dtype=np.uint8)
for raw_id, name in RAW_LABELS.items():
    LOOKUP_TABLE[raw_id] = NAME_TO_NEW_ID[name]


# --------------------------------------------------------------------------
# 3. IO helpers
# --------------------------------------------------------------------------
def load_velodyne_bin(path):
    """Returns (N, 4) float32 array: x, y, z, remission."""
    scan = np.fromfile(path, dtype=np.float32)
    return scan.reshape(-1, 4)


def load_label_file(path, num_points):
    """
    Returns (sem_label, inst_label), each shape (N,).
    Lower 16 bits = semantic class, upper 16 bits = instance id.
    """
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.shape[0] != num_points:
        raise ValueError(
            f"Label/point count mismatch in {path}: "
            f"{raw.shape[0]} labels vs {num_points} points"
        )
    sem_label = (raw & 0xFFFF).astype(np.int32)
    inst_label = (raw >> 16).astype(np.uint32)
    return sem_label, inst_label


def remap_semantic_labels(sem_label):
    """Vectorized remap: raw id -> reduced class id (0-4)."""
    sem_label = np.clip(sem_label, 0, _MAX_RAW_ID - 1)
    return LOOKUP_TABLE[sem_label]


def voxel_downsample(points, sem_label, inst_label, voxel_size):
    """
    Simple voxel-grid downsample: keep one point per occupied voxel
    (the point nearest the voxel's centroid). Cuts point count while
    preserving spatial coverage -- useful before feeding into the grid
    engine so you're not paying for redundant near-duplicate points.
    """
    if voxel_size is None or voxel_size <= 0:
        return points, sem_label, inst_label

    coords = points[:, :3]
    voxel_idx = np.floor(coords / voxel_size).astype(np.int64)

    # Hash each voxel index to a single integer key
    keys = (
        voxel_idx[:, 0].astype(np.int64) * 73856093 ^
        voxel_idx[:, 1].astype(np.int64) * 19349663 ^
        voxel_idx[:, 2].astype(np.int64) * 83492791
    )

    order = np.argsort(keys)
    keys_sorted = keys[order]
    # first occurrence of each unique key after sorting = one representative
    # point per voxel (arbitrary but deterministic pick)
    unique_mask = np.empty(len(keys_sorted), dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
    keep_idx = order[unique_mask]

    return points[keep_idx], sem_label[keep_idx], inst_label[keep_idx]


# --------------------------------------------------------------------------
# 4. Per-scan processing
# --------------------------------------------------------------------------
def process_scan(bin_path, label_path, out_path, voxel_size):
    points = load_velodyne_bin(bin_path)

    if label_path is not None and os.path.exists(label_path):
        sem_label, inst_label = load_label_file(label_path, points.shape[0])
        sem_label = remap_semantic_labels(sem_label)
    else:
        # Test-set scans (sequences 11-21) ship with no labels.
        sem_label = np.zeros(points.shape[0], dtype=np.uint8)
        inst_label = np.zeros(points.shape[0], dtype=np.uint32)

    points, sem_label, inst_label = voxel_downsample(
        points, sem_label, inst_label, voxel_size
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        points=points.astype(np.float32),
        sem_label=sem_label.astype(np.uint8),
        inst_label=inst_label.astype(np.uint32),
    )
    return out_path


def _process_scan_worker(args, voxel_size):
    bin_path, label_path, out_path = args
    try:
        process_scan(bin_path, label_path, out_path, voxel_size)
        return None
    except Exception as e:
        return f"FAILED: {bin_path} -> {e}"


# --------------------------------------------------------------------------
# 5. Driver
# --------------------------------------------------------------------------
def collect_jobs(dataset_root, output_dir, sequences):
    jobs = []
    for seq in sequences:
        velodyne_dir = os.path.join(dataset_root, "sequences", seq, "velodyne")
        labels_dir = os.path.join(dataset_root, "sequences", seq, "labels")
        out_seq_dir = os.path.join(output_dir, "sequences", seq)

        if not os.path.isdir(velodyne_dir):
            print(f"[skip] no velodyne dir for sequence {seq}: {velodyne_dir}")
            continue

        has_labels = os.path.isdir(labels_dir)
        if not has_labels:
            print(f"[info] sequence {seq} has no labels/ (test-set scan)")

        for fname in sorted(os.listdir(velodyne_dir)):
            if not fname.endswith(".bin"):
                continue
            scan_id = fname[:-4]
            bin_path = os.path.join(velodyne_dir, fname)
            label_path = (
                find_label_path(labels_dir, scan_id) if has_labels else None
            )
            out_path = os.path.join(out_seq_dir, scan_id + ".npz")
            jobs.append((bin_path, label_path, out_path))
    return jobs


def find_label_path(labels_dir, scan_id):
    """
    Look for this scan's label file under either the standard '.label'
    extension or a '.bin' extension (some re-packaged downloads name
    them .bin, identical to the raw binary format underneath).
    Returns None if neither is found.
    """
    for ext in (".label", ".bin"):
        candidate = os.path.join(labels_dir, scan_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def collect_jobs_flat(input_dir, labels_dir, output_dir):
    """
    Use this instead of collect_jobs() when your .bin files sit directly
    in one folder (no sequences/<seq>/velodyne/ nesting).

    If labels_dir is given, it looks for a same-named label file per
    .bin file inside it (e.g. 000000.bin -> 000000.label or 000000.bin),
    trying both extensions. If labels_dir is None, scans are treated as
    unlabeled (test-set style). Any poses.txt or other non-matching file
    in labels_dir is simply ignored.
    """
    jobs = []
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    has_labels = labels_dir is not None and os.path.isdir(labels_dir)
    if labels_dir is not None and not has_labels:
        print(f"[warn] labels_dir given but not found: {labels_dir}")

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".bin"):
            continue
        scan_id = fname[:-4]
        bin_path = os.path.join(input_dir, fname)
        label_path = find_label_path(labels_dir, scan_id) if has_labels else None
        if has_labels and label_path is None:
            print(f"[warn] no label found for {fname} (looked for {scan_id}.label / {scan_id}.bin)")
        out_path = os.path.join(output_dir, scan_id + ".npz")
        jobs.append((bin_path, label_path, out_path))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root",
                         help="Path to SemanticKITTI root (contains 'sequences/'). "
                              "Use this OR --input_dir, not both.")
    parser.add_argument("--input_dir",
                         help="A flat folder containing .bin files directly "
                              "(no sequences/<seq>/velodyne/ nesting). "
                              "Use this OR --dataset_root, not both.")
    parser.add_argument("--labels_dir", default=None,
                         help="Only used with --input_dir: a flat folder of "
                              "matching .label files (same base filenames). "
                              "Omit if you have no labels (e.g. test scans).")
    parser.add_argument("--output_dir", required=True,
                         help="Where preprocessed .npz files will be written")
    parser.add_argument("--sequences", nargs="+",
                         default=[f"{i:02d}" for i in range(22)],
                         help="Only used with --dataset_root: sequence ids to "
                              "process, e.g. 00 01 02 (default: all 00-21)")
    parser.add_argument("--voxel_size", type=float, default=None,
                         help="Optional voxel size in meters for downsampling "
                              "(e.g. 0.05). Omit to keep raw point density.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                         help="Number of parallel worker processes")
    args = parser.parse_args()

    if not args.dataset_root and not args.input_dir:
        parser.error("Provide either --dataset_root or --input_dir")
    if args.dataset_root and args.input_dir:
        parser.error("Use only one of --dataset_root or --input_dir, not both")

    if args.input_dir:
        jobs = collect_jobs_flat(args.input_dir, args.labels_dir, args.output_dir)
        print(f"Found {len(jobs)} scans to process in {args.input_dir}")
    else:
        jobs = collect_jobs(args.dataset_root, args.output_dir, args.sequences)
        print(f"Found {len(jobs)} scans to process across sequences {args.sequences}")

    if not jobs:
        print("Nothing to do -- check --dataset_root and --sequences.")
        return

    worker_fn = partial(_process_scan_worker, voxel_size=args.voxel_size)
    failures = []
    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_fn, jobs), 1):
            if result is not None:
                failures.append(result)
            if i % 500 == 0 or i == len(jobs):
                print(f"  processed {i}/{len(jobs)}")

    print(f"Done. {len(jobs) - len(failures)} succeeded, {len(failures)} failed.")
    for f in failures[:20]:
        print(" ", f)


if __name__ == "__main__":
    main()