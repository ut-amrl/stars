"""
Parse raw SEAN_TOGETHER .pt files into a flat HDF5 graph dataset.

Each sample = one timestep from one trajectory.
Graph structure:
  Nodes  : robot (type 0), follower (type 1), valid nearby people (type 2)
  NodeFeat: [x, y, cos_theta, sin_theta, node_type]  shape [N, 5]
  Edges  : fully connected (no self-loops)
  EdgeFeat: [distance, rel_dir_x, rel_dir_y]          shape [E, 3]
  Global : behavior label at this timestep             shape [1]
  OccGrid: occupancy grid normalised to [0,1]          shape [72, 72]
  Label  : true_label (3 integer class labels)         shape [3]

Usage:
  python sean_to_hdf5.py
  python sean_to_hdf5.py --sean_root SEAN_TOGETHER_Dataset/SEAN_TOGETHER --output sean_together.h5
"""

import os
import argparse
import numpy as np
import torch
import h5py
from pathlib import Path


def _to_np(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    return np.asarray(v)


def _valid_nearby_mask(nearby_t):
    """nearby_t: [16, 4]. A person is present if their orientation is non-zero."""
    ori_sq = nearby_t[:, 2] ** 2 + nearby_t[:, 3] ** 2
    return ori_sq > 0.1


def build_graph(robot_ori_t, follower_t, nearby_t):
    """
    Build node features, edge_index, edge_attr for one timestep.

    Args:
        robot_ori_t : [2]    cos/sin orientation of robot (always at origin)
        follower_t  : [4]    x, y, cos, sin of follower
        nearby_t    : [16,4] x, y, cos, sin of nearby people (zeros = absent)

    Returns:
        node_feat  : [N, 5]   float32
        edge_index : [2, E]   int64
        edge_attr  : [E, 3]   float32
    """
    nodes = []

    # Robot — always at (0, 0)
    nodes.append([0.0, 0.0, robot_ori_t[0], robot_ori_t[1], 0.0])

    # Follower
    nodes.append([follower_t[0], follower_t[1], follower_t[2], follower_t[3], 1.0])

    # Valid nearby people
    mask = _valid_nearby_mask(nearby_t)
    for i in np.where(mask)[0]:
        nodes.append([nearby_t[i, 0], nearby_t[i, 1],
                      nearby_t[i, 2], nearby_t[i, 3], 2.0])

    node_feat = np.array(nodes, dtype=np.float32)  # [N, 5]
    N = len(nodes)

    # Fully connected edges (no self-loops)
    src, dst = zip(*[(i, j) for i in range(N) for j in range(N) if i != j])
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)       # [2, E]

    # Edge features: euclidean distance + normalised relative direction
    edge_attr = []
    for i, j in zip(src, dst):
        diff = node_feat[j, :2] - node_feat[i, :2]
        dist = float(np.linalg.norm(diff))
        rel = diff / (dist + 1e-8)
        edge_attr.append([dist, rel[0], rel[1]])
    edge_attr = np.array(edge_attr, dtype=np.float32)  # [E, 3]

    return node_feat, edge_index, edge_attr


def parse_sean_to_hdf5(sean_root: str, output_path: str):
    sean_root = Path(sean_root)
    folders = sorted([p for p in sean_root.iterdir() if p.is_dir()])

    print(f"Found {len(folders)} scene folders in {sean_root}")

    sample_idx = 0
    skipped = 0

    with h5py.File(output_path, 'w') as hf:
        for folder in folders:
            pt_files = sorted(
                folder.glob("temporal_*.pt"),
                key=lambda p: int(p.stem.split('_')[1])
            )

            for pt_file in pt_files:
                try:
                    x = torch.load(str(pt_file), weights_only=False)
                except Exception as e:
                    print(f"  SKIP {pt_file.name}: {e}")
                    skipped += 1
                    continue

                robot_ori  = _to_np(x["spatial"]["robot"])     # (40, 2)
                follower   = _to_np(x["spatial"]["follower"])  # (40, 4)
                nearby     = _to_np(x["spatial"]["nearby"])    # (40, 16, 4)
                behavior   = _to_np(x["other"]["behavior"])    # (40, 1)
                occ_grids  = x["viz"]["map"]                   # list of (72,72)

                true_label = _to_np(x["true_label"]).astype(np.float32)
                if true_label.ndim == 0:
                    true_label = np.array([float(true_label), 0.0, 0.0], dtype=np.float32)

                T = robot_ori.shape[0]
                for t in range(T):
                    node_feat, edge_index, edge_attr = build_graph(
                        robot_ori[t], follower[t], nearby[t]
                    )

                    occ = np.asarray(occ_grids[t], dtype=np.float32) / 100.0
                    global_attr = behavior[t].astype(np.float32).reshape(1)

                    grp = hf.create_group(f"sample_{sample_idx}")
                    grp.create_dataset("node_features",   data=node_feat,   compression="gzip")
                    grp.create_dataset("edge_index",      data=edge_index,  compression="gzip")
                    grp.create_dataset("edge_attr",       data=edge_attr,   compression="gzip")
                    grp.create_dataset("occupancy_grid",  data=occ,         compression="gzip")
                    grp.create_dataset("global_attr",     data=global_attr)
                    grp.create_dataset("label",           data=true_label)

                    sample_idx += 1

            print(f"  {folder.name}: {len(pt_files)} files → {len(pt_files)*T} samples")

    print(f"\nDone. Total samples: {sample_idx}  |  Skipped files: {skipped}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sean_root", default="SEAN_TOGETHER_Dataset/SEAN_TOGETHER")
    parser.add_argument("--output",    default="sean_together.h5")
    args = parser.parse_args()
    parse_sean_to_hdf5(args.sean_root, args.output)
