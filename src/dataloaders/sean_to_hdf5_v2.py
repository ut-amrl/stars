"""
Build the v2 HDF5 temporal heterogeneous graph dataset.

Current storage format:
  - 1 graph per .pt file (~2,969 samples)
  - Fixed 18 nodes: robot(1) + follower(1) + bystander slots(16)
  - Node type ids: 0=robot, 1=follower, 2=bystander
  - node_features [18, 592]:
      robot    : 512D mean map_resnet18 + 80D flattened goal trajectory
      follower : optional 15D RF output in the first dims, else zeros
      bystander: zeros (downstream model uses learnable embeddings)
  - edge_attr [306, 40, 8]:
      [rel_x, rel_y, cos_rel, sin_rel, d_rel_x, d_rel_y, d_cos, d_sin]
      zeros where either node is absent at that timestep
  - valid_mask [18, 40]: which nodes are present at each timestep

Edge feature details (SE(2) in 2D):
  rel_x, rel_y    : position of dst in src's local frame
  cos_rel, sin_rel: relative orientation cos(theta_dst - theta_src), sin(theta_dst - theta_src)
  d_cos, d_sin    : change in relative orientation from t-1

Usage:
  cd dataloaders/
  python sean_to_hdf5_v2.py
  python sean_to_hdf5_v2.py --sean_root SEAN_TOGETHER_Dataset/SEAN_TOGETHER --output sean_together_v2.h5
"""

import argparse
import numpy as np
import torch
import h5py
from pathlib import Path

NODE_FEATURE_STORAGE_DIM = 592   # unified storage dim for x [18, 592] in HDF5
                                 # robot    x[0]  : 512 map_resnet18 mean + 80 goal trajectory [40,2] flattened
                                 # follower x[1]  : first 15D filled post RF training, rest zeros
                                 # bystander x[2:]: all zeros — model uses learnable embed, ignores x
N_BYSTANDERS     = 16
N_NODES          = 1 + 1 + N_BYSTANDERS   # 18
N_TIMESTEPS      = 40
EDGE_FEATURE_DIM = 8   # [rel_x, rel_y, cos_rel, sin_rel, d_rel_x, d_rel_y, d_cos, d_sin]


def _to_np(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    return np.asarray(v)


def se2_relative_pose(pos_i, ori_i, pos_j, ori_j):
    """
    Compute pose of node j expressed in node i's local frame.

    Args:
        pos_i, pos_j : [2]  positions in robot global frame
        ori_i, ori_j : [2]  (cos θ, sin θ) orientations

    Returns: [rel_x, rel_y, cos_rel, sin_rel]
    """
    cos_i, sin_i = float(ori_i[0]), float(ori_i[1])
    dx = float(pos_j[0]) - float(pos_i[0])
    dy = float(pos_j[1]) - float(pos_i[1])

    # Rotate displacement into i's local frame (inverse rotation by θ_i)
    rel_x =  cos_i * dx + sin_i * dy
    rel_y = -sin_i * dx + cos_i * dy

    # Relative orientation: cos(θ_j - θ_i), sin(θ_j - θ_i)
    cos_j, sin_j = float(ori_j[0]), float(ori_j[1])
    cos_rel = cos_j * cos_i + sin_j * sin_i
    sin_rel = sin_j * cos_i - cos_j * sin_i

    return np.array([rel_x, rel_y, cos_rel, sin_rel], dtype=np.float32)  # [4]


def build_temporal_graph(robot_ori, follower, nearby, map_r18, goal):
    """
    Build one temporal graph from a full 40-timestep trajectory.

    Args:
        robot_ori : [40, 2]     cos/sin orientation of robot (always at origin)
        follower  : [40, 4]     x, y, cos, sin of follower
        nearby    : [40, 16, 4] x, y, cos, sin of bystander slots
        map_r18   : [40, 512]   ResNet18 map embeddings
        goal      : [40, 2]     goal trajectory in robot frame

    Returns:
        node_features : [18, 592]
        node_type     : [18]      int64
        valid_mask    : [18, 40]  float32 (1=present, 0=absent)
        edge_index    : [2, 306]  int64
        edge_attr     : [306, 40, 8] float32
    """
    T = N_TIMESTEPS

    # ---- Node features ----
    node_features = np.zeros((N_NODES, NODE_FEATURE_STORAGE_DIM), dtype=np.float32)
    node_type     = np.zeros(N_NODES, dtype=np.int64)

    node_features[0, :592] = np.concatenate([
        map_r18.mean(axis=0),            # [512] mean map embedding
        goal.astype(np.float32).flatten()  # [80] full goal trajectory [40, 2]
    ])
    node_type[0] = 0
    node_type[1] = 1                           # follower: zeros placeholder
    node_type[2:] = 2                          # bystanders: zeros placeholder

    # ---- Valid mask ----
    valid_mask = np.zeros((N_NODES, T), dtype=np.float32)
    valid_mask[0, :] = 1.0   # robot always present
    valid_mask[1, :] = 1.0   # follower always present
    for i in range(N_BYSTANDERS):
        for t in range(T):
            ori_sq = nearby[t, i, 2]**2 + nearby[t, i, 3]**2
            valid_mask[2 + i, t] = 1.0 if ori_sq > 0.1 else 0.0

    # ---- Positions and orientations per timestep ----
    positions    = np.zeros((N_NODES, T, 2), dtype=np.float32)
    orientations = np.zeros((N_NODES, T, 2), dtype=np.float32)

    positions[0, :, :]    = 0.0              # robot always at origin
    orientations[0, :, :] = robot_ori        # [40, 2]

    positions[1, :, :]    = follower[:, :2]
    orientations[1, :, :] = follower[:, 2:]

    for i in range(N_BYSTANDERS):
        positions[2 + i, :, :]    = nearby[:, i, :2]
        orientations[2 + i, :, :] = nearby[:, i, 2:]

    # ---- Edge index: fully connected, no self-loops ----
    pairs      = [(i, j) for i in range(N_NODES) for j in range(N_NODES) if i != j]
    src_arr    = np.array([p[0] for p in pairs], dtype=np.int64)
    dst_arr    = np.array([p[1] for p in pairs], dtype=np.int64)
    edge_index = np.stack([src_arr, dst_arr], axis=0)   # [2, 306]
    E          = len(pairs)

    # ---- Temporal edge features [E, 40, 8] ----
    edge_attr = np.zeros((E, T, EDGE_FEATURE_DIM), dtype=np.float32)

    for e_idx, (i, j) in enumerate(pairs):
        prev_rel_x  = None
        prev_rel_y  = None
        prev_cos_rel = None
        prev_sin_rel = None

        for t in range(T):
            if not (valid_mask[i, t] and valid_mask[j, t]):
                # Either node absent → zeros, reset velocity tracking
                prev_rel_x   = None
                prev_rel_y   = None
                prev_cos_rel = None
                prev_sin_rel = None
                continue

            pose = se2_relative_pose(
                positions[i, t], orientations[i, t],
                positions[j, t], orientations[j, t]
            )
            rel_x, rel_y, cos_rel, sin_rel = pose

            d_rel_x = (rel_x - prev_rel_x)   if prev_rel_x   is not None else 0.0
            d_rel_y = (rel_y - prev_rel_y)   if prev_rel_y   is not None else 0.0
            d_cos   = (cos_rel - prev_cos_rel) if prev_cos_rel is not None else 0.0
            d_sin   = (sin_rel - prev_sin_rel) if prev_sin_rel is not None else 0.0

            edge_attr[e_idx, t] = [rel_x, rel_y, cos_rel, sin_rel, d_rel_x, d_rel_y, d_cos, d_sin]
            prev_rel_x   = rel_x
            prev_rel_y   = rel_y
            prev_cos_rel = cos_rel
            prev_sin_rel = sin_rel

    return node_features, node_type, valid_mask, edge_index, edge_attr


def parse_sean_to_hdf5_v2(sean_root: str, output_path: str, rf_output_path: str = None):
    sean_root = Path(sean_root)
    folders   = sorted([p for p in sean_root.iterdir() if p.is_dir()])
    print(f"Found {len(folders)} scene folders in {sean_root}")

    rf_output = None
    if rf_output_path is not None:
        rf_output = np.load(rf_output_path)   # [N, 15]
        print(f"Loaded RF output from {rf_output_path}  shape={rf_output.shape}")

    sample_idx = 0
    skipped    = 0

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

                robot_ori  = _to_np(x["spatial"]["robot"])      # (40, 2)
                follower   = _to_np(x["spatial"]["follower"])   # (40, 4)
                nearby     = _to_np(x["spatial"]["nearby"])     # (40, 16, 4)
                map_r18    = _to_np(x["physical"]["map_resnet18"]).astype(np.float32)  # (40, 512)
                goal       = _to_np(x["spatial"]["goal"]).astype(np.float32)           # (40, 2)

                true_label = _to_np(x["true_label"]).astype(np.float32)
                if true_label.ndim == 0:
                    true_label = np.array([float(true_label), 0.0, 0.0], dtype=np.float32)
                node_features, node_type, valid_mask, edge_index, edge_attr = \
                    build_temporal_graph(robot_ori, follower, nearby, map_r18, goal)

                # Optionally inject RF output into follower node (first 15 dims)
                if rf_output is not None:
                    node_features[1, :15] = rf_output[sample_idx]

                grp = hf.create_group(f"sample_{sample_idx}")
                grp.create_dataset("node_features", data=node_features, compression="gzip")
                grp.create_dataset("node_type",     data=node_type)
                grp.create_dataset("valid_mask",    data=valid_mask,    compression="gzip")
                grp.create_dataset("edge_index",    data=edge_index)
                grp.create_dataset("edge_attr",     data=edge_attr,     compression="gzip")
                grp.create_dataset("label",         data=true_label)

                sample_idx += 1

            print(f"  {folder.name}: {len(pt_files)} samples")

    print(f"\nDone. Total samples: {sample_idx}  |  Skipped: {skipped}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sean_root",     default="SEAN_TOGETHER_Dataset/SEAN_TOGETHER")
    parser.add_argument("--output",        default="sean_together_v2.h5")
    parser.add_argument("--rf_output",     default=None,
                        help="path to rf_output.npy — if provided, injects 15D follower features")
    args = parser.parse_args()
    parse_sean_to_hdf5_v2(args.sean_root, args.output, args.rf_output)
