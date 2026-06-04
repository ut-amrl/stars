"""
Info about the created HDF5 file (v3)
**All cords are relative to robot's initial position**
**Assumes all human nodes have a label**

Changes from V2:
- Writes `goal_position` (the (gx, gy) destination point in robot's initial
  frame) as a per-group dataset when present in the source JSON. v2 silently
  dropped this field.

Fields (per group)
- robot_trajectory: [T, 3] where each timestep includes x, y, and theta of the robot.
- people_trajectories: [P, T, 3] where P is total people and each timestep includes x, y, and theta of the person.
- people_ids: Storing the human_ids from json in the same order as people_trajectories to map graph nodes to people
- node_type: [1+P] stores the node type (robot=0, human=1). No padding nodes.
- edge_index: [2, (1+P)*P] defines all of the edges in the fully connected graph
- edge_attr: [(1+P)*P, T, 8] represents the spatial information between two nodes for a timestep
- action_label: [1+P] the label for each node (robot gets IGNORE_LABEL=-100)
- action_probs: [1+P, 5] optional normalized action-label distribution when
  `people_action_probabilities` is present in the source JSON
- action_class_names: Stores the string labels for each of the classes
- goal_position: [2] (gx, gy) the robot's goal in its initial frame (NEW in v3)
- scene: The scene name so we can split by scene if needed
- real: Whether the data is from the real (1) or synthetic (0) dataset
"""

import argparse
import glob
import json
import os
import h5py
import numpy as np


EXPECTED_TIMESTEPS = 20
EDGE_FEATURE_DIM = 8
NODE_TYPE_ROBOT = 0
NODE_TYPE_HUMAN = 1
ACTION_CLASSES = ("avoid", "follow", "overtake", "yield", "not_consider")
ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
IGNORE_LABEL = -100


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
    node_type[0] = NODE_TYPE_ROBOT
    node_type[1:] = NODE_TYPE_HUMAN

    positions = np.zeros((N, T, 2), dtype=np.float32)
    orientations = np.zeros((N, T, 2), dtype=np.float32)

    positions[0] = robot_raw[:, :2]
    orientations[0, :, 0] = np.cos(robot_raw[:, 2])
    orientations[0, :, 1] = np.sin(robot_raw[:, 2])

    for i in range(N_people):
        positions[1 + i] = people_raw[i, :, :2]
        orientations[1 + i, :, 0] = np.cos(people_raw[i, :, 2])
        orientations[1 + i, :, 1] = np.sin(people_raw[i, :, 2])

    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    src_arr = np.array([p[0] for p in pairs], dtype=np.int64)
    dst_arr = np.array([p[1] for p in pairs], dtype=np.int64)
    edge_index = np.stack([src_arr, dst_arr], axis=0)
    E = len(pairs)

    edge_attr = np.zeros((E, T, EDGE_FEATURE_DIM), dtype=np.float32)

    for e_idx, (i, j) in enumerate(pairs):
        prev = None
        for t in range(T):
            pose = se2_relative_pose(
                positions[i, t], orientations[i, t],
                positions[j, t], orientations[j, t]
            )
            rel_x, rel_y, cos_rel, sin_rel = pose

            if prev is not None:
                d_rel_x = rel_x - prev[0]
                d_rel_y = rel_y - prev[1]
                d_cos = cos_rel - prev[2]
                d_sin = sin_rel - prev[3]
            else:
                d_rel_x = d_rel_y = d_cos = d_sin = 0.0

            edge_attr[e_idx, t] = [rel_x, rel_y, cos_rel, sin_rel, d_rel_x, d_rel_y, d_cos, d_sin]
            prev = (rel_x, rel_y, cos_rel, sin_rel)

    return node_type, edge_index, edge_attr


def build_action_targets(data, people_ids):
    N = 1 + len(people_ids)
    action_label = np.full(N, IGNORE_LABEL, dtype=np.int64)
    action_probs = None

    labels_by_id = data["people_action_labels"]
    probs_by_id = data.get("people_action_probabilities")
    if probs_by_id is not None:
        action_probs = np.zeros((N, len(ACTION_CLASSES)), dtype=np.float32)

    for i, pid in enumerate(people_ids):
        action_label[1 + i] = ACTION_TO_IDX[labels_by_id[pid]]
        if action_probs is not None:
            for cls, cls_idx in ACTION_TO_IDX.items():
                action_probs[1 + i, cls_idx] = probs_by_id[pid].get(cls, 0.0)

    return action_label, action_probs


def convert(train_dir, output_path, synth_dir=None):
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

            robot_raw, people_raw, people_ids = parse_episode(data)
            name = data["episode_name"]
            if not is_real:
                name = f"synth/{name}"

            if name in hf:
                counter = 1
                while f"{name}_{counter}" in hf:
                    counter += 1
                name = f"{name}_{counter}"

            timestep_counts.append(robot_raw.shape[0])
            people_counts.append(people_raw.shape[0])

            node_type, edge_index, edge_attr = build_graph(robot_raw, people_raw)
            action_label, action_probs = build_action_targets(data, people_ids)

            grp = hf.create_group(name)
            grp.create_dataset("robot_trajectory", data=robot_raw)
            grp.create_dataset("people_trajectories", data=people_raw)
            grp.create_dataset("node_type", data=node_type)
            grp.create_dataset("edge_index", data=edge_index)
            grp.create_dataset("edge_attr", data=edge_attr, compression="gzip")
            grp.create_dataset("action_label", data=action_label)
            if action_probs is not None:
                grp.create_dataset("action_probs", data=action_probs)

            if "goal_position" in data:
                grp.create_dataset(
                    "goal_position",
                    data=np.array(data["goal_position"], dtype=np.float32),
                )

            dt = h5py.string_dtype()
            grp.create_dataset("people_ids", data=np.array(people_ids, dtype=object), dtype=dt)
            grp.create_dataset("action_class_names", data=np.array(ACTION_CLASSES, dtype=object), dtype=dt)

            grp.attrs["scene"] = data["scene"]
            grp.attrs["real"] = 1 if is_real else 0

            if is_real:
                total_real += 1
            else:
                total_synth += 1

    print(f"\nDone. Real episodes: {total_real}, Synthetic episodes: {total_synth}, Total: {total_real + total_synth}")
    if timestep_counts:
        unique_t = sorted(set(timestep_counts))
        print(
            "Timesteps per episode: "
            f"min={min(timestep_counts)} max={max(timestep_counts)} "
            f"unique={unique_t}"
        )
        if unique_t != [EXPECTED_TIMESTEPS]:
            print(f"WARNING: Not all episodes have exactly {EXPECTED_TIMESTEPS} timesteps.")
    if people_counts:
        print(
            "People per episode: "
            f"min={min(people_counts)} max={max(people_counts)} "
            f"avg={sum(people_counts) / len(people_counts):.1f}"
        )
    print(f"Saved to: {output_path}")


def stats(train_dir, synth_dir=None):
    json_files = sorted(glob.glob(os.path.join(train_dir, "*", "*.json")))
    if synth_dir is not None:
        json_files += sorted(glob.glob(os.path.join(synth_dir, "*.json")))

    counts = []
    timestep_counts = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        counts.append(len(data["people_trajectories"]))
        timestep_counts.append(np.array(data["robot_trajectory"], dtype=np.float32).shape[0])
    print(f"Episodes: {len(counts)}")
    print(f"Max people: {max(counts)}")
    print(f"Avg people: {sum(counts) / len(counts):.1f}")
    print(f"Min people: {min(counts)}")
    print(f"Timesteps: min={min(timestep_counts)} max={max(timestep_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_convert = sub.add_parser("convert")
    p_convert.add_argument("--train_dir", default="train3")
    p_convert.add_argument("--synth_dir", default="synthetic_dataset2")
    p_convert.add_argument("--no_synth", action="store_true",
                           help="Only convert --train_dir; ignore --synth_dir.")
    p_convert.add_argument("--output", default="socialnav_v3.h5")

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--train_dir", default="train3")
    p_stats.add_argument("--synth_dir", default="synthetic_dataset2")
    p_stats.add_argument("--no_synth", action="store_true",
                         help="Only inspect --train_dir; ignore --synth_dir.")

    args = parser.parse_args()
    if args.cmd == "stats":
        stats(args.train_dir, synth_dir=None if args.no_synth else args.synth_dir)
    elif args.cmd == "convert":
        convert(args.train_dir, args.output, synth_dir=None if args.no_synth else args.synth_dir)
    else:
        convert("train3", "socialnav_v3.h5", synth_dir="synthetic_dataset2")
