"""
Heterogeneous GNN-VAE pipeline — SEAN-only v4 graph dataset.

v4: HeteroData loader - follower treated like bystander (no GT labels as node features).

Changes from v3:
  follower.x removed - follower has no input features.
  follower.num_nodes = 1 (encoder uses a learnable embed, like bystander).
  This makes Stage 1 fully self-supervised: no GT labels anywhere in encoder input.
  No masking needed at probe time.

Everything else identical to v3 (robot normalization, edge masking, bystander handling).
"""

import h5py
import numpy as np
import re
import torch
from torch_geometric.data import HeteroData, Dataset

EDGE_TYPES = [
    ('robot',     'to', 'follower'),
    ('follower',  'to', 'robot'),
    ('robot',     'to', 'bystander'),
    ('bystander', 'to', 'robot'),
    ('follower',  'to', 'bystander'),
    ('bystander', 'to', 'follower'),
    ('bystander', 'to', 'bystander'),
]


class SEANGraphDatasetV4(Dataset):
    def __init__(self, h5_path, transform=None, robot_mean=None, robot_std=None, preload=True):
        super().__init__(root=None, transform=transform)
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.keys = self._collect_sample_keys(f)

        if len(self.keys) == 0:
            raise ValueError(
                f"No valid sample groups found in HDF5 file: {h5_path}. "
                "Expected each sample group to contain: "
                "node_features, valid_mask, edge_index, edge_attr, label."
            )

        self.robot_mean = robot_mean
        self.robot_std  = robot_std

        self._print_edge_stats()

        # Preload all raw tensors into RAM — eliminates per-sample HDF5 reads
        self._cache = None
        if preload:
            print(f"[SEANGraphDatasetV4] Preloading {len(self.keys)} graphs into RAM ...")
            self._cache = []
            with h5py.File(h5_path, 'r') as f:
                for key in self.keys:
                    grp = f[key]
                    self._cache.append((
                        torch.tensor(grp['node_features'][:], dtype=torch.float),
                        torch.tensor(grp['valid_mask'][:],    dtype=torch.float),
                        torch.tensor(grp['edge_index'][:],    dtype=torch.long),
                        torch.tensor(grp['edge_attr'][:],     dtype=torch.float),
                        torch.tensor(grp['label'][:],         dtype=torch.float),
                    ))
            print(f"[SEANGraphDatasetV4] Preload complete.")

    @staticmethod
    def _parse_key_index(key):
        """Extract a numeric index for stable sorting when available."""
        m = re.search(r'_(\d+)$', key)
        if m:
            return int(m.group(1))
        m = re.search(r'(\d+)$', key)
        if m:
            return int(m.group(1))
        return None

    def _collect_sample_keys(self, h5_file):
        required = {'node_features', 'valid_mask', 'edge_index', 'edge_attr', 'label'}
        valid = []

        for key in h5_file.keys():
            obj = h5_file[key]
            if not isinstance(obj, h5py.Group):
                continue
            if required.issubset(set(obj.keys())):
                valid.append(key)

        # Prefer numeric ordering if keys include an index; otherwise lexical fallback.
        return sorted(valid, key=lambda k: (self._parse_key_index(k) is None,
                                            self._parse_key_index(k) or 0,
                                            k))

    def set_robot_stats(self, train_indices):
        """Compute and set normalization stats from training indices only."""
        self.robot_mean, self.robot_std = self._compute_robot_stats(train_indices)

    def _compute_robot_stats(self, indices=None):
        """Compute mean/std from a subset of indices (train only) or all samples if None."""
        idxs = list(indices) if indices is not None else list(range(len(self.keys)))
        all_robot_x = []
        if self._cache is not None:
            for i in idxs:
                all_robot_x.append(self._cache[i][0][0].numpy())
        else:
            keys = [self.keys[i] for i in idxs]
            with h5py.File(self.h5_path, 'r') as f:
                for key in keys:
                    all_robot_x.append(f[key]['node_features'][0])
        all_robot_x = np.stack(all_robot_x)
        mean = torch.tensor(all_robot_x.mean(axis=0), dtype=torch.float)
        std  = torch.tensor(all_robot_x.std(axis=0),  dtype=torch.float)
        tiny_std = std < 1e-6
        if tiny_std.any():
            # Constant train dimensions should stay centered, not explode on
            # validation/test values that were absent from the train split.
            std = std.clone()
            std[tiny_std] = 1.0
        print(
            f"[SEANGraphDatasetV4] Robot feature stats (n={len(idxs)}): "
            f"mean [{mean.min():.2f}, {mean.max():.2f}]  "
            f"std [{std.min():.4f}, {std.max():.2f}]  "
            f"clamped_std_dims={int(tiny_std.sum())}"
        )
        return mean, std

    def _print_edge_stats(self):
        edge_counts      = []
        bystander_counts = []
        with h5py.File(self.h5_path, 'r') as f:
            for key in self.keys:
                grp        = f[key]
                valid_mask = np.array(grp['valid_mask'][:])
                edge_index = np.array(grp['edge_index'][:])
                presence   = valid_mask.sum(axis=1) > 0
                keep       = presence[edge_index[0]] & presence[edge_index[1]]
                edge_counts.append(int(keep.sum()))
                bystander_counts.append(int(presence[2:].sum()))
        edge_counts      = np.array(edge_counts)
        bystander_counts = np.array(bystander_counts)
        print(
            f"[SEANGraphDatasetV4] {len(self.keys)} graphs | "
            f"edges min={edge_counts.min()} max={edge_counts.max()} mean={edge_counts.mean():.1f} | "
            f"bystanders mean={bystander_counts.mean():.1f}/16"
        )

    def len(self):
        return len(self.keys)

    def get(self, idx):
        if self._cache is not None:
            node_features, valid_mask, edge_index_all, edge_attr_all, y = self._cache[idx]
        else:
            key = self.keys[idx]
            with h5py.File(self.h5_path, 'r') as f:
                grp            = f[key]
                node_features  = torch.tensor(grp['node_features'][:], dtype=torch.float)  # [18, 592]
                valid_mask     = torch.tensor(grp['valid_mask'][:],    dtype=torch.float)  # [18, 40]
                edge_index_all = torch.tensor(grp['edge_index'][:],    dtype=torch.long)   # [2, 306]
                edge_attr_all  = torch.tensor(grp['edge_attr'][:],     dtype=torch.float)  # [306, 40, 8]
                y              = torch.tensor(grp['label'][:],         dtype=torch.float)  # [3]

        # --- Normalize robot node features ---
        if self.robot_mean is None or self.robot_std is None:
            raise RuntimeError("robot_mean/robot_std not set - call dataset.set_robot_stats(train_indices) after splitting.")
        robot_x = node_features[[0]]
        robot_x = (robot_x - self.robot_mean) / (self.robot_std + 1e-8)

        # --- Filter edges by valid_mask ---
        node_presence = valid_mask.sum(dim=1) > 0
        keep          = node_presence[edge_index_all[0]] & node_presence[edge_index_all[1]]
        edge_index_f  = edge_index_all[:, keep]
        edge_attr_f   = edge_attr_all[keep]

        # --- Build local index mapping ---
        present_b_globals = torch.where(node_presence[2:])[0] + 2
        n_bystanders      = len(present_b_globals)
        b_global_to_local = {int(g): i for i, g in enumerate(present_b_globals.tolist())}

        def get_type_local(g):
            g = int(g)
            if g == 0: return 'robot',     0
            if g == 1: return 'follower',  0
            return 'bystander', b_global_to_local[g]

        # --- Split edges by type ---
        srcs  = {et: [] for et in EDGE_TYPES}
        dsts  = {et: [] for et in EDGE_TYPES}
        attrs = {et: [] for et in EDGE_TYPES}

        for e in range(edge_index_f.shape[1]):
            sg, dg = int(edge_index_f[0, e]), int(edge_index_f[1, e])
            st, sl = get_type_local(sg)
            dt, dl = get_type_local(dg)
            et     = (st, 'to', dt)
            srcs[et].append(sl)
            dsts[et].append(dl)
            attrs[et].append(edge_attr_f[e])

        # --- Build HeteroData ---
        data = HeteroData()

        data['robot'].x             = robot_x        # [1, 592] normalized
        data['follower'].num_nodes  = 1              # no features, encoder uses a learnable embed
        data['bystander'].num_nodes = n_bystanders

        for et in EDGE_TYPES:
            if len(srcs[et]) > 0:
                data[et].edge_index = torch.tensor([srcs[et], dsts[et]], dtype=torch.long)
                data[et].edge_attr  = torch.stack(attrs[et])

        data['robot'].y = y.unsqueeze(0)   # [1, 3] raw Likert, for linear probe targets

        return data
