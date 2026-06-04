"""
One-time script: extract scene-based SEAN split from combined_v1.h5
and map indices to sean_together_v2.h5 ordering.

Output: checkpoints/split_indices_sean_snsb_hetero_v1.pt
  keys: train, val, test (indices into sean_together_v2.h5)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import torch

COMBINED_H5   = "dataloaders/combined_v1.h5"
SEAN_H5       = "dataloaders/sean_together_v2.h5"
OUT_PATH      = "checkpoints/split_indices_sean_snsb_hetero_v1.pt"
VAL_RATIO     = 0.16
TEST_RATIO    = 0.04
SEED          = 42
DATASET_SEAN  = 0


def scene_split(keys, scenes, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    unique_scenes = list(dict.fromkeys(scenes))
    unique_scenes = rng.permutation(unique_scenes).tolist()
    n = len(unique_scenes)
    n_test = max(1, round(test_ratio * n))
    n_val  = max(1, round(val_ratio  * n))
    test_scenes  = set(unique_scenes[:n_test])
    val_scenes   = set(unique_scenes[n_test: n_test + n_val])
    train_idx, val_idx, test_idx = [], [], []
    for i, scene in enumerate(scenes):
        if scene in test_scenes:
            test_idx.append(i)
        elif scene in val_scenes:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx, test_scenes, val_scenes


# ── Load SEAN keys + scenes from combined_v1.h5 ──────────────────────────────
with h5py.File(COMBINED_H5, "r") as fc:
    combined_all_keys = list(fc.keys())          # insertion order
    sean_positions = []                           # positions in combined_all_keys
    sean_keys_ordered = []
    sean_scenes = []
    for i, k in enumerate(combined_all_keys):
        if fc[k].attrs.get("dataset", -1) == DATASET_SEAN:
            sean_positions.append(i)
            sean_keys_ordered.append(k)
            sean_scenes.append(str(fc[k].attrs["scene"]))

print(f"SEAN graphs in combined: {len(sean_keys_ordered)}")

# ── Run scene_split ───────────────────────────────────────────────────────────
train_local, val_local, test_local, test_scenes, val_scenes = scene_split(
    sean_keys_ordered, sean_scenes, VAL_RATIO, TEST_RATIO, SEED
)

# ── Build key→index map for sean_together_v2.h5 ───────────────────────────────
with h5py.File(SEAN_H5, "r") as fs:
    sean_h5_keys = list(fs.keys())
key_to_sean_idx = {k: i for i, k in enumerate(sean_h5_keys)}

# ── Map local SEAN indices → sean_together_v2.h5 indices ─────────────────────
def map_indices(local_idx_list, keys_ordered, key_to_idx):
    return [key_to_idx[keys_ordered[i]] for i in local_idx_list]

train_idx = map_indices(train_local, sean_keys_ordered, key_to_sean_idx)
val_idx   = map_indices(val_local,   sean_keys_ordered, key_to_sean_idx)
test_idx  = map_indices(test_local,  sean_keys_ordered, key_to_sean_idx)

print(f"Scene split (seed={SEED}): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
print(f"Unique scenes: {len(set(sean_scenes))}  test_scenes={len(test_scenes)}  val_scenes={len(val_scenes)}")

os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "train": train_idx,
    "val":   val_idx,
    "test":  test_idx,
    "train_scenes": sorted(set(sean_scenes) - test_scenes - val_scenes),
    "val_scenes":   sorted(val_scenes),
    "test_scenes":  sorted(test_scenes),
    "source": COMBINED_H5,
    "seed":   SEED,
}, OUT_PATH)
print(f"Saved -> {OUT_PATH}")
