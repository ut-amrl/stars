"""
Generate 10 independent scene-based train/val/test splits from combined_v1.h5
for the label-efficiency experiment.

Each split seed produces a different random assignment of scenes to train/val/test.
Output files have the same format as split_indices_final.pt (indices into combined_v1.h5)
so train_unified_gnn_vae_v3.py accepts them unchanged via --split_file.

SEAN:  80/16/4  scene split  (60 scenes  → ~48 train, ~10 val, ~2 test)
SNSB:  85/15/0  scene split  (225 scenes → ~191 train, ~34 val, 0 test)

Usage:
    python generate_label_eff_splits.py
    python generate_label_eff_splits.py --h5_path dataloaders/combined_v1.h5 \
        --out_dir checkpoints/label_eff_splits --n_seeds 10
"""

import argparse
import os
import h5py
import numpy as np
import torch

DATASET_SEAN = 0
DATASET_SNSB = 1

SEAN_VAL_RATIO  = 0.16
SEAN_TEST_RATIO = 0.04
SNSB_VAL_RATIO  = 0.15
SNSB_TEST_RATIO = 0.00


def scene_split(indices, scenes, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    unique_scenes = list(dict.fromkeys(scenes))
    unique_scenes = rng.permutation(unique_scenes).tolist()
    n = len(unique_scenes)
    n_test = max(1, round(test_ratio * n)) if test_ratio > 0 else 0
    n_val  = max(1, round(val_ratio  * n))
    test_scenes  = set(unique_scenes[:n_test])
    val_scenes   = set(unique_scenes[n_test: n_test + n_val])
    train_scenes = set(unique_scenes[n_test + n_val:])
    train_idx, val_idx, test_idx = [], [], []
    for i, scene in zip(indices, scenes):
        if scene in test_scenes:
            test_idx.append(i)
        elif scene in val_scenes:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx, train_scenes, val_scenes, test_scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path",  default="/data/combined_v1.h5")
    parser.add_argument("--out_dir",  default="checkpoints/label_eff_splits")
    parser.add_argument("--n_seeds",  type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading {args.h5_path} ...")
    with h5py.File(args.h5_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        sean_indices, sean_scenes = [], []
        snsb_indices, snsb_scenes = [], []
        for pos, k in enumerate(keys):
            ds  = int(f[k].attrs.get("dataset", -1))
            sc  = str(f[k].attrs.get("scene", ""))
            if ds == DATASET_SEAN:
                sean_indices.append(pos)
                sean_scenes.append(sc)
            elif ds == DATASET_SNSB:
                snsb_indices.append(pos)
                snsb_scenes.append(sc)

    print(f"SEAN: {len(sean_indices)} samples, "
          f"{len(set(sean_scenes))} unique scenes")
    print(f"SNSB: {len(snsb_indices)} samples, "
          f"{len(set(snsb_scenes))} unique scenes")

    for seed in range(args.n_seeds):
        sean_tr, sean_va, sean_te, sean_tr_sc, sean_va_sc, sean_te_sc = scene_split(
            sean_indices, sean_scenes,
            SEAN_VAL_RATIO, SEAN_TEST_RATIO, seed
        )
        snsb_tr, snsb_va, snsb_te, snsb_tr_sc, snsb_va_sc, snsb_te_sc = scene_split(
            snsb_indices, snsb_scenes,
            SNSB_VAL_RATIO, SNSB_TEST_RATIO, seed
        )

        train = sorted(sean_tr + snsb_tr)
        val   = sorted(sean_va + snsb_va)
        test  = sorted(sean_te + snsb_te)

        out_path = os.path.join(args.out_dir, f"split_seed{seed}.pt")
        torch.save({
            "train": train,
            "val":   val,
            "test":  test,
            "sean_train_scenes": sorted(sean_tr_sc),
            "sean_val_scenes":   sorted(sean_va_sc),
            "sean_test_scenes":  sorted(sean_te_sc),
            "snsb_train_scenes": sorted(snsb_tr_sc),
            "snsb_val_scenes":   sorted(snsb_va_sc),
            "seed": seed,
            "h5_path": args.h5_path,
        }, out_path)

        print(
            f"seed={seed} | "
            f"SEAN train={len(sean_tr)}({len(sean_tr_sc)}sc) "
            f"val={len(sean_va)}({len(sean_va_sc)}sc) "
            f"test={len(sean_te)}({len(sean_te_sc)}sc) | "
            f"SNSB train={len(snsb_tr)}({len(snsb_tr_sc)}sc) "
            f"val={len(snsb_va)}({len(snsb_va_sc)}sc) "
            f"→ {out_path}"
        )

    print(f"\nDone. {args.n_seeds} split files in {args.out_dir}/")


if __name__ == "__main__":
    main()
