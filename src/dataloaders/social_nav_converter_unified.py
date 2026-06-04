import argparse
import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from pathlib import Path

SSUB_TIMESTEPS     = 20
UNIFIED_TIMESTEPS  = 40
EDGE_FEATURE_DIM   = 8
NODE_FEATURE_DIM   = 592   
N_BYSTANDER_SLOTS  = 16
N_SEAN_NODES_FULL  = 18    
DATASET_SEAN       = 0
DATASET_SSUB_REAL  = 1
DATASET_SSUB_SYNTH = 2
NODE_TYPE_ROBOT  = 0
NODE_TYPE_HUMAN  = 1
IGNORE_LABEL   = -100
ACTION_CLASSES = ("avoid", "follow", "overtake", "yield", "not_consider")
ACTION_TO_IDX  = {name: idx for idx, name in enumerate(ACTION_CLASSES)}


def _to_np(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    return np.asarray(v)


def se2_relative_pose(pos_i, ori_i, pos_j, ori_j):
    cos_i, sin_i = float(ori_i[0]), float(ori_i[1])
    dx = float(pos_j[0]) - float(pos_i[0])
    dy = float(pos_j[1]) - float(pos_i[1])
    rel_x =  cos_i * dx + sin_i * dy
    rel_y = -sin_i * dx + cos_i * dy
    cos_j, sin_j = float(ori_j[0]), float(ori_j[1])
    cos_rel = cos_j * cos_i + sin_j * sin_i
    sin_rel = sin_j * cos_i - cos_j * sin_i
    return np.array([rel_x, rel_y, cos_rel, sin_rel], dtype=np.float32)


def parse_ssub_episode(data):
    robot_raw = np.array(data["robot_trajectory"], dtype=np.float32)[:, 1:]  
    people_ids = sorted(data["people_trajectories"].keys())
    if people_ids:
        people_raw = np.stack([
            np.array(data["people_trajectories"][pid], dtype=np.float32)[:, 1:]
            for pid in people_ids
        ])  # [P, T, 3]
    else:
        T = robot_raw.shape[0]
        people_raw = np.zeros((0, T, 3), dtype=np.float32)
    return robot_raw, people_raw, people_ids


def build_positions_orientations(robot_raw, people_raw):
    T = robot_raw.shape[0]
    P = people_raw.shape[0]
    N = 1 + P
    positions    = np.zeros((N, T, 2), dtype=np.float32)
    orientations = np.zeros((N, T, 2), dtype=np.float32)
    positions[0]          = robot_raw[:, :2]
    orientations[0, :, 0] = np.cos(robot_raw[:, 2])
    orientations[0, :, 1] = np.sin(robot_raw[:, 2])
    for i in range(P):
        positions[1 + i]          = people_raw[i, :, :2]
        orientations[1 + i, :, 0] = np.cos(people_raw[i, :, 2])
        orientations[1 + i, :, 1] = np.sin(people_raw[i, :, 2])
    return positions, orientations


def build_node_type(N):
    node_type = np.zeros(N, dtype=np.int64)
    node_type[1:] = NODE_TYPE_HUMAN
    return node_type


def build_edge_index(N):
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    edge_index = np.stack([
        np.array([p[0] for p in pairs], dtype=np.int64),
        np.array([p[1] for p in pairs], dtype=np.int64),
    ], axis=0)   
    return edge_index, pairs


def build_edge_attr(pairs, positions, orientations, T):
    E = len(pairs)
    edge_attr = np.zeros((E, T, EDGE_FEATURE_DIM), dtype=np.float32)
    for e_idx, (i, j) in enumerate(pairs):
        prev = None
        for t in range(T):
            rel_x, rel_y, cos_rel, sin_rel = se2_relative_pose(
                positions[i, t], orientations[i, t],
                positions[j, t], orientations[j, t],
            )
            if prev is not None:
                d_rel_x = rel_x  - prev[0]; d_rel_y = rel_y  - prev[1]
                d_cos   = cos_rel - prev[2]; d_sin   = sin_rel - prev[3]
            else:
                d_rel_x = d_rel_y = d_cos = d_sin = 0.0
            edge_attr[e_idx, t] = [rel_x, rel_y, cos_rel, sin_rel,
                                    d_rel_x, d_rel_y, d_cos, d_sin]
            prev = (rel_x, rel_y, cos_rel, sin_rel)
    return edge_attr


def build_ssub_action_labels(data, people_ids):
    N = 1 + len(people_ids)
    action_label = np.full(N, IGNORE_LABEL, dtype=np.int64)
    labels_by_id = data["people_action_labels"]
    for i, pid in enumerate(people_ids):
        action_label[1 + i] = ACTION_TO_IDX[labels_by_id[pid]]
    return action_label


def interpolate_edge_attr(edge_attr, target_size=UNIFIED_TIMESTEPS):
    E, T, D = edge_attr.shape
    if E == 0:
        return np.zeros((0, target_size, D), dtype=np.float32)
    if T == target_size:
        return edge_attr.astype(np.float32)
    t = torch.from_numpy(
        edge_attr.transpose(0, 2, 1).reshape(1, E * D, T).astype(np.float32)
    )
    t = F.interpolate(t, size=target_size, mode="linear", align_corners=True)
    return t.numpy().reshape(E, D, target_size).transpose(0, 2, 1).astype(np.float32)


def process_sean_file(pt_path):
    pt_path = Path(pt_path)
    scene   = pt_path.parent.name
    x = torch.load(str(pt_path), weights_only=False)

    robot_ori = _to_np(x["spatial"]["robot"]).astype(np.float32)      
    follower  = _to_np(x["spatial"]["follower"]).astype(np.float32)   
    nearby    = _to_np(x["spatial"]["nearby"]).astype(np.float32)  
    map_r18   = _to_np(x["physical"]["map_resnet18"]).astype(np.float32) 
    goal      = _to_np(x["spatial"]["goal"]).astype(np.float32)     

    T = UNIFIED_TIMESTEPS

    valid_mask_full = np.zeros((N_SEAN_NODES_FULL, T), dtype=np.float32)
    valid_mask_full[0, :] = 1.0
    valid_mask_full[1, :] = 1.0
    for i in range(N_BYSTANDER_SLOTS):
        for t in range(T):
            ori_sq = nearby[t, i, 2] ** 2 + nearby[t, i, 3] ** 2
            valid_mask_full[2 + i, t] = 1.0 if ori_sq > 0.1 else 0.0

    keep_mask    = valid_mask_full.sum(axis=1) > 0
    keep_indices = np.where(keep_mask)[0]
    N_kept       = len(keep_indices)

    orig_type_full = np.array([0, 1] + [2] * N_BYSTANDER_SLOTS, dtype=np.int64)
    node_type = np.where(orig_type_full[keep_indices] == 0, NODE_TYPE_ROBOT, NODE_TYPE_HUMAN)

    valid_mask = valid_mask_full[keep_indices, :]

    node_features = np.zeros((N_kept, NODE_FEATURE_DIM), dtype=np.float32)
    node_features[0, :512] = map_r18.mean(axis=0)
    node_features[0, 512:] = goal.astype(np.float32).flatten()

    positions    = np.zeros((N_SEAN_NODES_FULL, T, 2), dtype=np.float32)
    orientations = np.zeros((N_SEAN_NODES_FULL, T, 2), dtype=np.float32)
    positions[0, :, :]    = 0.0
    orientations[0, :, :] = robot_ori
    positions[1, :, :]    = follower[:, :2]
    orientations[1, :, :] = follower[:, 2:]
    for i in range(N_BYSTANDER_SLOTS):
        positions[2 + i, :, :]    = nearby[:, i, :2]
        orientations[2 + i, :, :] = nearby[:, i, 2:]

    pairs = [(i, j) for i in range(N_kept) for j in range(N_kept) if i != j]
    E = len(pairs)
    edge_index = np.stack([
        np.array([p[0] for p in pairs], dtype=np.int64),
        np.array([p[1] for p in pairs], dtype=np.int64),
    ], axis=0)

    edge_attr = np.zeros((E, T, EDGE_FEATURE_DIM), dtype=np.float32)
    for e_idx, (new_i, new_j) in enumerate(pairs):
        old_i, old_j = keep_indices[new_i], keep_indices[new_j]
        prev_rx = prev_ry = prev_cos = prev_sin = None
        for t in range(T):
            if not (valid_mask[new_i, t] and valid_mask[new_j, t]):
                prev_rx = prev_ry = prev_cos = prev_sin = None
                continue
            rel_x, rel_y, cos_rel, sin_rel = se2_relative_pose(
                positions[old_i, t], orientations[old_i, t],
                positions[old_j, t], orientations[old_j, t],
            )
            d_rel_x = (rel_x  - prev_rx)  if prev_rx  is not None else 0.0
            d_rel_y = (rel_y  - prev_ry)  if prev_ry  is not None else 0.0
            d_cos   = (cos_rel - prev_cos) if prev_cos is not None else 0.0
            d_sin   = (sin_rel - prev_sin) if prev_sin is not None else 0.0
            edge_attr[e_idx, t] = [rel_x, rel_y, cos_rel, sin_rel,
                                    d_rel_x, d_rel_y, d_cos, d_sin]
            prev_rx, prev_ry, prev_cos, prev_sin = rel_x, rel_y, cos_rel, sin_rel

    true_label = _to_np(x["true_label"]).astype(np.float32)
    if true_label.ndim == 0:
        true_label = np.array([float(true_label), 0.0, 0.0], dtype=np.float32)
    emotion_label = true_label[:3].astype(np.float32)
    action_label  = np.full(N_kept, IGNORE_LABEL, dtype=np.int64)

    return (node_features, node_type, valid_mask, edge_index, edge_attr,
            emotion_label, action_label, DATASET_SEAN, scene)


def write_combined_sample(hf, sample_idx,
                          node_features, node_type, valid_mask,
                          edge_index, edge_attr,
                          emotion_label, action_label, dataset_id, scene):
    grp = hf.create_group(f"sample_{sample_idx}")
    grp.create_dataset("node_features", data=node_features, compression="gzip")
    grp.create_dataset("node_type",     data=node_type)
    grp.create_dataset("valid_mask",    data=valid_mask,    compression="gzip")
    grp.create_dataset("edge_index",    data=edge_index)
    grp.create_dataset("edge_attr",     data=edge_attr,     compression="gzip")
    grp.create_dataset("emotion_label", data=emotion_label)
    grp.create_dataset("action_label",  data=action_label)
    grp.attrs["dataset"] = dataset_id
    grp.attrs["scene"]   = scene


def write_ssub_sample(hf, group_name, robot_raw, people_raw, people_ids,
                      node_type, edge_index, edge_attr, action_label,
                      scene, is_real):
    grp = hf.create_group(group_name)
    grp.create_dataset("robot_trajectory",    data=robot_raw)
    grp.create_dataset("people_trajectories", data=people_raw)
    grp.create_dataset("node_type",           data=node_type)
    grp.create_dataset("edge_index",          data=edge_index)
    grp.create_dataset("edge_attr",           data=edge_attr, compression="gzip")
    grp.create_dataset("action_label",        data=action_label)
    dt = h5py.string_dtype()
    grp.create_dataset("people_ids",
                       data=np.array(people_ids, dtype=object), dtype=dt)
    grp.create_dataset("action_class_names",
                       data=np.array(ACTION_CLASSES, dtype=object), dtype=dt)
    grp.attrs["scene"] = scene
    grp.attrs["real"]  = 1 if is_real else 0


def convert_ssub(train_dir, output_path, synth_dir=None):
    episodes = []

    scene_dirs = sorted(
        path for path in glob.glob(os.path.join(train_dir, "*"))
        if os.path.isdir(path)
    )
    print(f"Found {len(scene_dirs)} real scene directories")
    for scene_dir in scene_dirs:
        for jf in sorted(glob.glob(os.path.join(scene_dir, "*.json"))):
            episodes.append((jf, True))

    if synth_dir is not None:
        synth_files = sorted(glob.glob(os.path.join(synth_dir, "*.json")))
        print(f"Found {len(synth_files)} synthetic episodes")
        for jf in synth_files:
            episodes.append((jf, False))

    print(f"Total episodes to process: {len(episodes)}")

    total_real = 0
    total_synth = 0
    timestep_counts = []
    people_counts = []

    with h5py.File(output_path, "w") as hf:
        for jf, is_real in episodes:
            with open(jf) as f:
                data = json.load(f)

            robot_raw, people_raw, people_ids = parse_ssub_episode(data)
            N = 1 + len(people_ids)
            T_raw = robot_raw.shape[0]

            positions, orientations = build_positions_orientations(robot_raw, people_raw)
            node_type = build_node_type(N)
            edge_index, pairs = build_edge_index(N)
            edge_attr = build_edge_attr(pairs, positions, orientations, T_raw)
            action_label = build_ssub_action_labels(data, people_ids)

            name = data["episode_name"]
            if not is_real:
                name = f"synth/{name}"
            if name in hf:
                counter = 1
                while f"{name}_{counter}" in hf:
                    counter += 1
                name = f"{name}_{counter}"

            write_ssub_sample(hf, name, robot_raw, people_raw, people_ids,
                              node_type, edge_index, edge_attr, action_label,
                              scene=data["scene"], is_real=is_real)

            timestep_counts.append(T_raw)
            people_counts.append(people_raw.shape[0])
            if is_real:
                total_real += 1
            else:
                total_synth += 1

    print(f"\nDone. Real episodes: {total_real}, Synthetic episodes: {total_synth}, "
          f"Total: {total_real + total_synth}")
    if timestep_counts:
        unique_t = sorted(set(timestep_counts))
        print(f"Timesteps per episode: min={min(timestep_counts)} "
              f"max={max(timestep_counts)} unique={unique_t}")
    if people_counts:
        print(f"People per episode: min={min(people_counts)} "
              f"max={max(people_counts)} avg={sum(people_counts)/len(people_counts):.1f}")
    print(f"Saved to: {output_path}")


def convert_both(sean_root, ssub_train_dir, ssub_synth_dir, output_path,
                 skip_sean=False, skip_ssub=False):
    sample_idx = 0
    counts     = {DATASET_SEAN: 0, DATASET_SSUB_REAL: 0, DATASET_SSUB_SYNTH: 0}
    skipped    = 0

    with h5py.File(output_path, "w") as hf:

        if not skip_sean and sean_root is not None:
            sean_root_p = Path(sean_root)
            folders     = sorted(p for p in sean_root_p.iterdir() if p.is_dir())
            print(f"[SEAN] {len(folders)} scene folders in {sean_root_p}")
            for folder in folders:
                pt_files = sorted(
                    folder.glob("temporal_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]),
                )
                for pt_file in pt_files:
                    try:
                        result = process_sean_file(pt_file)
                    except Exception as e:
                        print(f"  SKIP {pt_file.name}: {e}")
                        skipped += 1
                        continue
                    write_combined_sample(hf, sample_idx, *result)
                    counts[DATASET_SEAN] += 1
                    sample_idx += 1
                print(f"  {folder.name}: {len(pt_files)} samples")

        if not skip_ssub and ssub_train_dir is not None:
            scene_dirs = sorted(
                d for d in glob.glob(os.path.join(ssub_train_dir, "*"))
                if os.path.isdir(d)
            )
            print(f"[SSUB real] {len(scene_dirs)} scene directories in {ssub_train_dir}")
            for scene_dir in scene_dirs:
                for jf in sorted(glob.glob(os.path.join(scene_dir, "*.json"))):
                    try:
                        with open(jf) as f:
                            data = json.load(f)

                        robot_raw, people_raw, people_ids = parse_ssub_episode(data)
                        N = 1 + len(people_ids)
                        T_raw = robot_raw.shape[0]

                        positions, orientations = build_positions_orientations(robot_raw, people_raw)
                        node_type = build_node_type(N)
                        edge_index, pairs = build_edge_index(N)
                        edge_attr_raw = build_edge_attr(pairs, positions, orientations, T_raw)
                        action_label = build_ssub_action_labels(data, people_ids)

                        edge_attr = interpolate_edge_attr(edge_attr_raw, UNIFIED_TIMESTEPS)
                        node_features = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
                        valid_mask = np.ones((N, UNIFIED_TIMESTEPS), dtype=np.float32)
                        emotion_label = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

                        write_combined_sample(hf, sample_idx,
                                              node_features, node_type, valid_mask,
                                              edge_index, edge_attr,
                                              emotion_label, action_label,
                                              DATASET_SSUB_REAL, data["scene"])
                        counts[DATASET_SSUB_REAL] += 1
                        sample_idx += 1
                    except Exception as e:
                        print(f"  SKIP {os.path.basename(jf)}: {e}")
                        skipped += 1

        if not skip_ssub and ssub_synth_dir is not None:
            synth_files = sorted(glob.glob(os.path.join(ssub_synth_dir, "*.json")))
            print(f"[SSUB synth] {len(synth_files)} synthetic episodes in {ssub_synth_dir}")
            for jf in synth_files:
                try:
                    with open(jf) as f:
                        data = json.load(f)

                    robot_raw, people_raw, people_ids = parse_ssub_episode(data)
                    N = 1 + len(people_ids)
                    T_raw = robot_raw.shape[0]

                    positions, orientations = build_positions_orientations(robot_raw, people_raw)
                    node_type = build_node_type(N)
                    edge_index, pairs = build_edge_index(N)
                    edge_attr_raw = build_edge_attr(pairs, positions, orientations, T_raw)
                    action_label = build_ssub_action_labels(data, people_ids)

                    edge_attr = interpolate_edge_attr(edge_attr_raw, UNIFIED_TIMESTEPS)
                    node_features = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
                    valid_mask = np.ones((N, UNIFIED_TIMESTEPS), dtype=np.float32)
                    emotion_label = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

                    write_combined_sample(hf, sample_idx,
                                          node_features, node_type, valid_mask,
                                          edge_index, edge_attr,
                                          emotion_label, action_label,
                                          DATASET_SSUB_SYNTH, data["episode_name"])
                    counts[DATASET_SSUB_SYNTH] += 1
                    sample_idx += 1
                except Exception as e:
                    print(f"  SKIP {os.path.basename(jf)}: {e}")
                    skipped += 1

    print(f"\nDone.")
    print(f"  SEAN:       {counts[DATASET_SEAN]}")
    print(f"  SSUB real:  {counts[DATASET_SSUB_REAL]}")
    print(f"  SSUB synth: {counts[DATASET_SSUB_SYNTH]}")
    print(f"  Total:      {sample_idx}  |  Skipped: {skipped}")
    print(f"Saved to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build SSUB-only or combined SEAN+SSUB HDF5 dataset")
    parser.add_argument("--mode", choices=["ssub", "both"], required=True,
                        help="'ssub' for SSUB only, 'both' for SEAN+SSUB combined")
    parser.add_argument("--train_dir",  default="train3",
                        help="SSUB real data directory (scene subdirectories)")
    parser.add_argument("--synth_dir",  default="synthetic_dataset2",
                        help="SSUB synthetic data directory (flat JSON)")
    parser.add_argument("--sean_root",  default="SEAN_TOGETHER_Dataset/SEAN_TOGETHER",
                        help="SEAN Together root (only used with --mode both)")
    parser.add_argument("--output",     default=None,
                        help="Output HDF5 file path (default: mode-dependent)")
    parser.add_argument("--skip_sean",  action="store_true",
                        help="Skip SEAN data (only for --mode both)")
    parser.add_argument("--skip_ssub",  action="store_true",
                        help="Skip SSUB data (only for --mode both)")
    args = parser.parse_args()

    output = args.output or ("social_nav_v3.h5" if args.mode == "ssub" else "combined_v1.h5")

    if args.mode == "ssub":
        convert_ssub(args.train_dir, output, synth_dir=args.synth_dir)
    else:
        convert_both(
            sean_root     = args.sean_root     if not args.skip_sean else None,
            ssub_train_dir = args.train_dir    if not args.skip_ssub else None,
            ssub_synth_dir = args.synth_dir    if not args.skip_ssub else None,
            output_path   = output,
            skip_sean     = args.skip_sean,
            skip_ssub     = args.skip_ssub,
        )
