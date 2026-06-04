"""
Heterogeneous GNN-VAE pipeline — SNSB hetero graph dataset.

v1: HeteroData loader for SocialNav-SUB (single-source, e.g. human_oracle).

Mirrors sean_graph_dataset_v4.py for the SocialNav-SUB hetero topology:
  - 2 node types:  'robot', 'human'
  - 3 edge types:  ('robot','to','human'), ('human','to','robot'),
                   ('human','to','human')

Per-sample HeteroData layout:
  data['robot'].x      [1, T*3]   flattened robot trajectory (x, y, theta) for T steps
  data['robot'].traj   [1, T, 3]  same trajectory in temporal form
  data['robot'].goal   [1, 2]     robot's (gx, gy) goal in its initial frame
  data['human'].x      [P, T*3]   flattened per-human trajectory
  data['human'].traj   [P, T, 3]  same per-human trajectory in temporal form
  data['human'].y      [P]        per-human action class index in {0..4}, dtype long

  data[(rt, 'to', dt)].edge_index  [2, E_t]
  data[(rt, 'to', dt)].edge_attr   [E_t, T, 8]    SE(2) relative pose + temporal deltas

Extras (per-graph metadata; PyG will attach them as plain Python attributes):
  data.scene           str         scene id (e.g. "129_Spot_7")
  data.episode_key     str         HDF5 group key (== episode name)
  data.real            int         1 for real, 0 for synthetic (always 1 here)
  data.num_humans      int         P
  data.num_timesteps   int         T (10 for SocialNav-SUB)

If the HDF5 lacks `goal_position` (older converters), data['robot'].goal is
omitted and callers can branch on `hasattr(data['robot'], 'goal')`.

Expected HDF5 source: output of `socialnav_to_hdf5_v3.py convert ...`.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch_geometric.data import HeteroData, Dataset


EDGE_TYPES = [
    ('robot', 'to', 'human'),
    ('human', 'to', 'robot'),
    ('human', 'to', 'human'),
]

NUM_ACTION_CLASSES = 5  # avoid, follow, overtake, yield, not_consider
IGNORE_LABEL = -100     # robot's action_label entry; we drop it before exposing y
EDGE_FEATURE_DIM = 8


class SocialNavGraphDatasetV1(Dataset):
    def __init__(self, h5_path, transform=None, action_probs_json_dir=None, preload=True):
        super().__init__(root=None, transform=transform)
        self.h5_path = h5_path
        self.action_probs_json_dir = action_probs_json_dir

        with h5py.File(h5_path, 'r') as f:
            self.keys = self._collect_sample_keys(f)
            self.action_classes = self._read_action_classes(f)
            T_set, P_set = self._collect_shape_stats(f)

        if len(self.keys) == 0:
            raise ValueError(
                f"No valid sample groups in {h5_path}. "
                "Expected each group to have: robot_trajectory, people_trajectories, "
                "node_type, edge_index, edge_attr, action_label."
            )

        self.timesteps = next(iter(T_set)) if len(T_set) == 1 else None
        if self.timesteps is None:
            print(f"[SocialNavGraphDatasetV1] WARNING: variable timesteps across episodes: {sorted(T_set)}")

        print(
            f"[SocialNavGraphDatasetV1] {len(self.keys)} graphs | "
            f"T={sorted(T_set)} | "
            f"humans/graph min={min(P_set)} max={max(P_set)} avg={np.mean(list(P_set)):.1f} | "
            f"classes={self.action_classes}"
        )

        # Preload raw tensors into RAM to avoid reopening HDF5 for every sample.
        self._cache = None
        if preload:
            print(f"[SocialNavGraphDatasetV1] Preloading {len(self.keys)} graphs into RAM ...")
            self._cache = []
            with h5py.File(h5_path, 'r') as f:
                for key in self.keys:
                    self._cache.append(self._read_group(f[key]))
            print("[SocialNavGraphDatasetV1] Preload complete.")

    @staticmethod
    def _collect_sample_keys(h5_file):
        required = {
            'robot_trajectory', 'people_trajectories', 'node_type',
            'edge_index', 'edge_attr', 'action_label',
        }
        valid = []
        def _walk(group, prefix=''):
            for key in group.keys():
                path = f'{prefix}/{key}' if prefix else key
                obj = group[key]
                if isinstance(obj, h5py.Group):
                    if required.issubset(set(obj.keys())):
                        valid.append(path)
                    else:
                        _walk(obj, path)
        _walk(h5_file)
        return sorted(valid)

    @staticmethod
    def _read_action_classes(h5_file):
        def _find(group):
            for key in group.keys():
                obj = group[key]
                if isinstance(obj, h5py.Group):
                    if 'action_class_names' in obj:
                        names = obj['action_class_names'][:]
                        return [s.decode() if isinstance(s, bytes) else str(s) for s in names]
                    result = _find(obj)
                    if result is not None:
                        return result
            return None
        return _find(h5_file) or ["avoid", "follow", "overtake", "yield", "not_consider"]

    def _collect_shape_stats(self, h5_file):
        T_set, P_set = set(), []
        for key in self.keys:
            grp = h5_file[key]
            T_set.add(int(grp['robot_trajectory'].shape[0]))
            P_set.append(int(grp['people_trajectories'].shape[0]))
        return T_set, P_set

    def len(self):
        return len(self.keys)

    @staticmethod
    def _read_group(grp):
        robot_traj   = torch.tensor(grp['robot_trajectory'][:],    dtype=torch.float)
        people_traj  = torch.tensor(grp['people_trajectories'][:], dtype=torch.float)
        edge_index   = torch.tensor(grp['edge_index'][:],          dtype=torch.long)
        edge_attr    = torch.tensor(grp['edge_attr'][:],           dtype=torch.float)
        action_label = torch.tensor(grp['action_label'][:],        dtype=torch.long)
        action_probs = (
            torch.tensor(grp['action_probs'][:], dtype=torch.float)
            if 'action_probs' in grp else None
        )
        people_ids_raw = grp['people_ids'][:] if 'people_ids' in grp else []
        people_ids = [
            p.decode() if isinstance(p, bytes) else str(p)
            for p in people_ids_raw
        ]
        scene = str(grp.attrs.get('scene', ''))
        real  = int(grp.attrs.get('real', 1))
        goal  = (
            torch.tensor(grp['goal_position'][:], dtype=torch.float).reshape(1, 2)
            if 'goal_position' in grp else None
        )
        return (
            robot_traj, people_traj, edge_index, edge_attr, action_label,
            action_probs, people_ids, scene, real, goal,
        )

    def _load_action_probs_from_json(self, key, scene, people_ids):
        if self.action_probs_json_dir is None:
            return None

        episode_name = os.path.basename(key)
        frame = episode_name.rsplit("_", 1)[-1]
        candidates = [
            os.path.join(self.action_probs_json_dir, scene, f"episode_{frame}.json"),
            os.path.join(self.action_probs_json_dir, scene, f"{episode_name}.json"),
        ]

        json_path = next((p for p in candidates if os.path.exists(p)), None)
        if json_path is None:
            return None

        with open(json_path) as f:
            data = json.load(f)

        probs_by_id = data.get("people_action_probabilities")
        if probs_by_id is None:
            return None

        action_probs = torch.zeros(1 + len(people_ids), NUM_ACTION_CLASSES, dtype=torch.float)
        for i, pid in enumerate(people_ids):
            pid_probs = probs_by_id.get(pid)
            if pid_probs is None:
                continue
            for cls_idx, cls in enumerate(self.action_classes):
                action_probs[1 + i, cls_idx] = float(pid_probs.get(cls, 0.0))
        return action_probs

    def get(self, idx):
        key = self.keys[idx]
        if self._cache is not None:
            (
                robot_traj, people_traj, edge_index, edge_attr, action_label,
                action_probs, people_ids, scene, real, goal,
            ) = self._cache[idx]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                (
                    robot_traj, people_traj, edge_index, edge_attr, action_label,
                    action_probs, people_ids, scene, real, goal,
                ) = self._read_group(f[key])

        T = robot_traj.shape[0]
        P = people_traj.shape[0]
        if action_probs is None:
            action_probs = self._load_action_probs_from_json(key, scene, people_ids)

        # --- Node features (flat trajectory) ---
        robot_x = robot_traj.reshape(1, T * 3)        # [1, T*3]
        human_x = people_traj.reshape(P, T * 3) if P > 0 else torch.zeros(0, T * 3)

        # --- Per-human action labels (drop robot's IGNORE_LABEL entry) ---
        human_y = action_label[1:] if P > 0 else torch.zeros(0, dtype=torch.long)

        # --- Split edges by (src_type, dst_type) using global indices ---
        # global node 0 = robot; global nodes 1..P = humans (local idx = global-1)
        srcs  = {et: [] for et in EDGE_TYPES}
        dsts  = {et: [] for et in EDGE_TYPES}
        attrs = {et: [] for et in EDGE_TYPES}

        for e in range(edge_index.shape[1]):
            sg, dg = int(edge_index[0, e]), int(edge_index[1, e])
            s_is_robot = (sg == 0)
            d_is_robot = (dg == 0)
            if s_is_robot and not d_is_robot:
                et = ('robot', 'to', 'human')
                sl, dl = 0, dg - 1
            elif not s_is_robot and d_is_robot:
                et = ('human', 'to', 'robot')
                sl, dl = sg - 1, 0
            elif not s_is_robot and not d_is_robot:
                et = ('human', 'to', 'human')
                sl, dl = sg - 1, dg - 1
            else:
                continue  # robot->robot self loop; not in the converter, skip defensively
            srcs[et].append(sl)
            dsts[et].append(dl)
            attrs[et].append(edge_attr[e])

        # --- Build HeteroData ---
        data = HeteroData()
        data['robot'].x    = robot_x          # [1, T*3]
        data['robot'].traj = robot_traj.unsqueeze(0)  # [1, T, 3]
        if goal is not None:
            data['robot'].goal = goal         # [1, 2]

        if P > 0:
            data['human'].x    = human_x      # [P, T*3]
            data['human'].traj = people_traj  # [P, T, 3]
            data['human'].y    = human_y      # [P]
            if action_probs is not None:
                data['human'].action_probs = action_probs[1:]  # [P, 5]
        else:
            data['human'].num_nodes = 0
            data['human'].x    = torch.zeros(0, T * 3)
            data['human'].traj = torch.zeros(0, T, 3)
            data['human'].y    = torch.zeros(0, dtype=torch.long)
            if action_probs is not None:
                data['human'].action_probs = torch.zeros(0, NUM_ACTION_CLASSES)

        for et in EDGE_TYPES:
            if len(srcs[et]) > 0:
                data[et].edge_index = torch.tensor([srcs[et], dsts[et]], dtype=torch.long)
                data[et].edge_attr  = torch.stack(attrs[et])  # [E_t, T, 8]
            else:
                data[et].edge_index = torch.zeros(2, 0, dtype=torch.long)
                data[et].edge_attr  = torch.zeros(0, T, EDGE_FEATURE_DIM)

        data.scene         = scene
        data.episode_key   = key
        data.real          = real
        data.num_humans    = P
        data.num_timesteps = T
        return data
