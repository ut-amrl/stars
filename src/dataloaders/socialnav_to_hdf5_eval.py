import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F


UNIFIED_TIMESTEPS = 40
EDGE_FEATURE_DIM  = 8
NODE_FEATURE_DIM  = 592

DATASET_SSUB_REAL = 1

NODE_TYPE_ROBOT = 0
NODE_TYPE_HUMAN = 1

ACTION_CLASSES = ("avoid", "follow", "overtake", "yield", "not_consider")
ACTION_TO_IDX  = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
IGNORE_LABEL   = -100


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


def interpolate_edge_attr(edge_attr, target_size=UNIFIED_TIMESTEPS):
    """
    Upsample [E, T_raw, 8] → [E, target_size, 8] via linear interpolation.
    Uses F.interpolate to stay consistent with combined_to_hdf5.py.
    """
    E, T_raw, D = edge_attr.shape
    if E == 0:
        return np.zeros((0, target_size, D), dtype=np.float32)
    if T_raw == target_size:
        return edge_attr.astype(np.float32)
    t = torch.from_numpy(
        edge_attr.transpose(0, 2, 1).reshape(1, E * D, T_raw).astype(np.float32)
    )
    t = F.interpolate(t, size=target_size, mode="linear", align_corners=True)
    return t.numpy().reshape(E, D, target_size).transpose(0, 2, 1).astype(np.float32)


def parse_episode(data):
    robot_raw = np.array(data["robot_trajectory"], dtype=np.float32)[:, 1:]

    people_ids = sorted(data["people_trajectories"].keys())
    if people_ids:
        people_raw = np.stack([
            np.array(data["people_trajectories"][pid], dtype=np.float32)[:, 1:]
            for pid in people_ids
        ])
    else:
        T = robot_raw.shape[0]
        people_raw = np.zeros((0, T, 3), dtype=np.float32)

    return robot_raw, people_raw, people_ids


def build_graph(robot_raw, people_raw):
    T = robot_raw.shape[0]
    N_people = people_raw.shape[0]
    N = 1 + N_people

    node_type = np.empty(N, dtype=np.int64)
    node_type[0]  = NODE_TYPE_ROBOT
    node_type[1:] = NODE_TYPE_HUMAN

    positions    = np.zeros((N, T, 2), dtype=np.float32)
    orientations = np.zeros((N, T, 2), dtype=np.float32)

    positions[0]       = robot_raw[:, :2]
    orientations[0, :, 0] = np.cos(robot_raw[:, 2])
    orientations[0, :, 1] = np.sin(robot_raw[:, 2])

    for i in range(N_people):
        positions[1 + i]          = people_raw[i, :, :2]
        orientations[1 + i, :, 0] = np.cos(people_raw[i, :, 2])
        orientations[1 + i, :, 1] = np.sin(people_raw[i, :, 2])

    pairs      = [(i, j) for i in range(N) for j in range(N) if i != j]
    src_arr    = np.array([p[0] for p in pairs], dtype=np.int64)
    dst_arr    = np.array([p[1] for p in pairs], dtype=np.int64)
    edge_index = np.stack([src_arr, dst_arr], axis=0)
    E          = len(pairs)

    edge_attr_raw = np.zeros((E, T, EDGE_FEATURE_DIM), dtype=np.float32)
    for e_idx, (i, j) in enumerate(pairs):
        prev = None
        for t in range(T):
            pose = se2_relative_pose(
                positions[i, t], orientations[i, t],
                positions[j, t], orientations[j, t],
            )
            rel_x, rel_y, cos_rel, sin_rel = pose
            if prev is not None:
                d_rel_x = rel_x  - prev[0]
                d_rel_y = rel_y  - prev[1]
                d_cos   = cos_rel - prev[2]
                d_sin   = sin_rel - prev[3]
            else:
                d_rel_x = d_rel_y = d_cos = d_sin = 0.0
            edge_attr_raw[e_idx, t] = [rel_x, rel_y, cos_rel, sin_rel,
                                        d_rel_x, d_rel_y, d_cos, d_sin]
            prev = (rel_x, rel_y, cos_rel, sin_rel)

    return node_type, edge_index, edge_attr_raw


def build_action_targets(data, people_ids):
    N = 1 + len(people_ids)
    action_label = np.full(N, IGNORE_LABEL, dtype=np.int64)
    action_probs = np.zeros((N, len(ACTION_CLASSES)), dtype=np.float32)

    labels_by_id = data["people_action_labels"]
    probs_by_id  = data["people_action_probabilities"]

    for i, pid in enumerate(people_ids):
        action_label[1 + i] = ACTION_TO_IDX[labels_by_id[pid]]
        for cls_idx, cls in enumerate(ACTION_CLASSES):
            action_probs[1 + i, cls_idx] = probs_by_id[pid].get(cls, 0.0)

    return action_label, action_probs


def convert(oracle_dir, output_path):
    scene_dirs = sorted(
        path for path in glob.glob(os.path.join(oracle_dir, "*"))
        if os.path.isdir(path)
    )

    if not scene_dirs:
        print(f"No scene directories found in {oracle_dir}")
        return

    print(f"Found {len(scene_dirs)} scenes in {oracle_dir}")

    sample_idx      = 0
    timestep_counts = []
    people_counts   = []

    with h5py.File(output_path, "w") as hf:
        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)

            for jf in sorted(glob.glob(os.path.join(scene_dir, "*.json"))):
                with open(jf) as f:
                    data = json.load(f)

                robot_raw, people_raw, people_ids = parse_episode(data)
                T_raw    = robot_raw.shape[0]
                N        = 1 + len(people_ids)

                timestep_counts.append(T_raw)
                people_counts.append(len(people_ids))

                node_type, edge_index, edge_attr_raw = build_graph(robot_raw, people_raw)
                edge_attr = interpolate_edge_attr(edge_attr_raw, UNIFIED_TIMESTEPS)

                action_label, action_probs = build_action_targets(data, people_ids)

                # Fields required by CombinedGraphDataset / HGNNCombinedAE
                node_features = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
                valid_mask    = np.ones((N, UNIFIED_TIMESTEPS), dtype=np.float32)
                emotion_label = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

                grp = hf.create_group(f"sample_{sample_idx}")
                grp.create_dataset("node_features", data=node_features, compression="gzip")
                grp.create_dataset("node_type",     data=node_type)
                grp.create_dataset("valid_mask",    data=valid_mask,    compression="gzip")
                grp.create_dataset("edge_index",    data=edge_index)
                grp.create_dataset("edge_attr",     data=edge_attr,     compression="gzip")
                grp.create_dataset("emotion_label", data=emotion_label)
                grp.create_dataset("action_label",  data=action_label)

                # Extra fields for evaluation (not read by CombinedGraphDataset)
                grp.create_dataset("action_probs", data=action_probs)
                dt = h5py.string_dtype()
                grp.create_dataset(
                    "action_class_names",
                    data=np.array(ACTION_CLASSES, dtype=object),
                    dtype=dt,
                )

                grp.attrs["dataset"]      = DATASET_SSUB_REAL
                grp.attrs["scene"]        = scene_name
                grp.attrs["raw_timesteps"] = T_raw

                sample_idx += 1

    print(f"\nDone. Total episodes: {sample_idx}")
    if timestep_counts:
        unique_t = sorted(set(timestep_counts))
        print(
            f"Raw timesteps per episode: "
            f"min={min(timestep_counts)} max={max(timestep_counts)} "
            f"unique={unique_t}"
        )
        print(f"Edge attr upsampled to {UNIFIED_TIMESTEPS} timesteps")
    if people_counts:
        print(
            f"People per episode: "
            f"min={min(people_counts)} max={max(people_counts)} "
            f"avg={sum(people_counts) / len(people_counts):.1f}"
        )
    print(f"Saved to: {output_path}")


def stats(oracle_dir):
    json_files = sorted(glob.glob(os.path.join(oracle_dir, "*", "*.json")))
    if not json_files:
        print(f"No episodes found in {oracle_dir}")
        return

    people_counts   = []
    timestep_counts = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        people_counts.append(len(data["people_trajectories"]))
        timestep_counts.append(
            np.array(data["robot_trajectory"], dtype=np.float32).shape[0]
        )
    print(
        f"human_oracle: {len(json_files)} episodes, "
        f"people min={min(people_counts)} max={max(people_counts)} "
        f"avg={sum(people_counts)/len(people_counts):.1f}, "
        f"timesteps min={min(timestep_counts)} max={max(timestep_counts)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_convert = sub.add_parser("convert")
    p_convert.add_argument("--oracle_dir", default="ssub_test/human_oracle")
    p_convert.add_argument("--output", default="social_nav_eval_combined.h5")

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--oracle_dir", default="ssub_test/human_oracle")

    args = parser.parse_args()
    if args.cmd == "stats":
        stats(args.oracle_dir)
    elif args.cmd == "convert":
        convert(args.oracle_dir, args.output)
    else:
        convert("ssub_test/human_oracle", "social_nav_eval_combined.h5")
