#!/usr/bin/env python3
"""
Generate dataset splits s2..s10 from s1.

Treats s1 as the complete dataset (already split into train/val/test).
Recombines all scenes per dataset (SEAN, SSUB) and re-shuffles them
into new scene-level splits with different seeds, preserving the same
split ratios and structure.

Structure of each split directory:
  s{N}/
    train/
      sean/{scene_name}/temporal_*.pt
      ssub/{scene_name}/episode_*.json
    val/
      sean/{scene_name}/...
      ssub/{scene_name}/...
    test/
      sean/{scene_name}/...
      (no ssub test — ssub_test_ratio=0)

Usage (inside container):
  python3 src/generate_splits.py \\
      --source /data/dataset_splits/s1 \\
      --output-root /data/dataset_splits \\
      --splits 2 10
"""

import argparse
import os
import shutil
import numpy as np


def collect_all_scenes(source_dir, dataset_type):
    """
    Collect all scene directories for a given dataset type (sean or ssub)
    across train/val/test splits. Returns a sorted list of
    (scene_name, source_path) tuples.
    """
    scenes = {}  # scene_name -> source_path
    for split in ["train", "val", "test"]:
        dataset_dir = os.path.join(source_dir, split, dataset_type)
        if not os.path.isdir(dataset_dir):
            continue
        for scene_name in os.listdir(dataset_dir):
            scene_path = os.path.join(dataset_dir, scene_name)
            if os.path.isdir(scene_path):
                if scene_name in scenes:
                    raise ValueError(
                        f"Duplicate scene '{scene_name}' for {dataset_type} "
                        f"found in multiple splits"
                    )
                scenes[scene_name] = scene_path
    return sorted(scenes.items())


def scene_split(scene_names, val_ratio, test_ratio, seed):
    """
    Split scene names into train/val/test using the same logic as
    UnifiedGraphDataset.scene_split.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(scene_names).tolist()
    n = len(shuffled)
    n_test = max(1, round(test_ratio * n)) if test_ratio > 0 else 0
    n_val = max(1, round(val_ratio * n))

    test_scenes = shuffled[:n_test]
    val_scenes = shuffled[n_test:n_test + n_val]
    train_scenes = shuffled[n_test + n_val:]

    return train_scenes, val_scenes, test_scenes


def create_split(source_dir, output_dir, seed):
    """
    Create one split directory by recombining s1 scenes and re-shuffling.
    Uses symlinks to avoid copying data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- SEAN: ~80/16.6/3.3 split (same as s1: 48/10/2 out of 60) --
    sean_scenes = collect_all_scenes(source_dir, "sean")
    sean_names = [name for name, _ in sean_scenes]
    sean_paths = {name: path for name, path in sean_scenes}

    sean_train, sean_val, sean_test = scene_split(
        sean_names,
        val_ratio=0.16,
        test_ratio=0.04,
        seed=seed,
    )

    print(f"  SEAN: {len(sean_names)} scenes -> "
          f"train={len(sean_train)} val={len(sean_val)} test={len(sean_test)}")

    # -- SSUB: ~85/15/0 split (same as s1: 191/34/0 out of 225) --
    ssub_scenes = collect_all_scenes(source_dir, "ssub")
    ssub_names = [name for name, _ in ssub_scenes]
    ssub_paths = {name: path for name, path in ssub_scenes}

    ssub_train, ssub_val, ssub_test = scene_split(
        ssub_names,
        val_ratio=0.15,
        test_ratio=0.0,
        seed=seed,
    )

    print(f"  SSUB: {len(ssub_names)} scenes -> "
          f"train={len(ssub_train)} val={len(ssub_val)} test={len(ssub_test)}")

    # -- Create directory structure with symlinks --
    assignments = [
        ("sean", "train", sean_train, sean_paths),
        ("sean", "val", sean_val, sean_paths),
        ("sean", "test", sean_test, sean_paths),
        ("ssub", "train", ssub_train, ssub_paths),
        ("ssub", "val", ssub_val, ssub_paths),
    ]
    # Only add ssub test if there are test scenes
    if ssub_test:
        assignments.append(("ssub", "test", ssub_test, ssub_paths))

    for dataset_type, split, scenes, paths in assignments:
        target_dir = os.path.join(output_dir, split, dataset_type)
        os.makedirs(target_dir, exist_ok=True)
        for scene_name in scenes:
            src = os.path.abspath(paths[scene_name])
            dst = os.path.join(target_dir, scene_name)
            os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset splits s2..s10 from s1"
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to the s1 split directory (e.g. /data/dataset_splits/s1)"
    )
    parser.add_argument(
        "--output-root", required=True,
        help="Root directory for output splits (e.g. /data/dataset_splits)"
    )
    parser.add_argument(
        "--splits", nargs=2, type=int, default=[2, 10],
        metavar=("START", "END"),
        help="Range of split numbers to generate (inclusive, default: 2 10)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing split directories"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    start, end = args.splits
    for split_num in range(start, end + 1):
        split_dir = os.path.join(args.output_root, f"s{split_num}")

        if os.path.exists(split_dir):
            if args.force:
                print(f"Removing existing {split_dir}")
                shutil.rmtree(split_dir)
            else:
                print(f"Skipping s{split_num} (already exists, use --force to overwrite)")
                continue

        # Use split_num as the seed so each split is different but reproducible
        seed = split_num
        print(f"\nGenerating s{split_num} (seed={seed}):")
        create_split(args.source, split_dir, seed=seed)
        print(f"  -> {split_dir}")

    # Print summary
    print("\n=== Summary ===")
    for split_num in range(start, end + 1):
        split_dir = os.path.join(args.output_root, f"s{split_num}")
        if not os.path.isdir(split_dir):
            continue
        sean_train = len(os.listdir(os.path.join(split_dir, "train", "sean")))
        sean_val = len(os.listdir(os.path.join(split_dir, "val", "sean")))
        sean_test_dir = os.path.join(split_dir, "test", "sean")
        sean_test = len(os.listdir(sean_test_dir)) if os.path.isdir(sean_test_dir) else 0
        ssub_train = len(os.listdir(os.path.join(split_dir, "train", "ssub")))
        ssub_val = len(os.listdir(os.path.join(split_dir, "val", "ssub")))
        ssub_test_dir = os.path.join(split_dir, "test", "ssub")
        ssub_test = len(os.listdir(ssub_test_dir)) if os.path.isdir(ssub_test_dir) else 0
        print(
            f"  s{split_num}: SEAN train={sean_train} val={sean_val} test={sean_test} | "
            f"SSUB train={ssub_train} val={ssub_val} test={ssub_test}"
        )


if __name__ == "__main__":
    main()
