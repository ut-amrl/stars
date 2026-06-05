"""
Heterogeneous GNN-VAE pipeline — combined SEAN+SNSB dataloader wrappers.

Unified dataloader wrappers for the SEAN + SocialNav-SUB GNN-VAE.

These wrappers keep the source dataset loaders unchanged, then remap their
native HeteroData objects into the shared UnifiedGNNVAE contract:
  - node types: robot, bystander
  - edge types: robot->bystander, bystander->robot, bystander->bystander
  - dataset_id: 0 for SEAN, 1 for SocialNav-SUB

The wrappers do not normalize features themselves. Source loaders remain
responsible for their own train-split normalization.
"""

import torch
from torch_geometric.data import Dataset, HeteroData

from dataloaders.sean_graph_dataset_v4 import SEANGraphDatasetV4
from dataloaders.socialnav_hetero_graph_dataset_v1 import SocialNavGraphDatasetV1


SEAN_ID = 0
SNSB_ID = 1
EDGE_FEATURE_DIM = 8
SEAN_TIMESTEPS = 40
SNSB_TIMESTEPS = 20
SEAN_ROBOT_DIM = 592
SNSB_ROBOT_DIM = 60  # T*3 = 20*3
SNSB_BYSTANDER_DIM = 60  # T*3 = 20*3 flattened trajectory (not edge features — those are 8D)

UNIFIED_EDGE_TYPES = [
    ("robot", "to", "bystander"),
    ("bystander", "to", "robot"),
    ("bystander", "to", "bystander"),
]


def _empty_edge_attr(timesteps):
    return torch.zeros(0, timesteps, EDGE_FEATURE_DIM)


def _empty_edge_index():
    return torch.zeros(2, 0, dtype=torch.long)


def _has_edges(data, edge_type):
    return edge_type in getattr(data, "edge_types", []) and hasattr(data[edge_type], "edge_index")


def _infer_timesteps(data, fallback):
    for edge_type in getattr(data, "edge_types", []):
        if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr.numel() > 0:
            return int(data[edge_type].edge_attr.shape[1])
    return fallback


class UnifiedSEANDataset(Dataset):
    """
    Wrap SEANGraphDatasetV4 and fold follower + bystander into unified bystander.

    Native SEAN:
      robot, follower, bystander
      7 edge types

    Unified output:
      robot, bystander
      3 edge types
      follower becomes bystander local index 0
      old bystanders shift by +1
    """

    def __init__(self, h5_path=None, base_dataset=None, transform=None):
        super().__init__(root=None, transform=transform)
        if base_dataset is None:
            if h5_path is None:
                raise ValueError("UnifiedSEANDataset requires h5_path or base_dataset.")
            base_dataset = SEANGraphDatasetV4(h5_path)
        self.base_dataset = base_dataset

    @property
    def keys(self):
        return getattr(self.base_dataset, "keys", None)

    def set_robot_stats(self, train_indices):
        if not hasattr(self.base_dataset, "set_robot_stats"):
            raise AttributeError("Wrapped SEAN dataset does not expose set_robot_stats.")
        self.base_dataset.set_robot_stats(train_indices)

    def len(self):
        return len(self.base_dataset)

    def get(self, idx):
        source = self.base_dataset[idx]
        timesteps = _infer_timesteps(source, fallback=SEAN_TIMESTEPS)
        if timesteps != SEAN_TIMESTEPS:
            raise ValueError(f"UnifiedSEANDataset expected T={SEAN_TIMESTEPS}, got T={timesteps}.")

        data = HeteroData()
        data["robot"].x = source["robot"].x
        if data["robot"].x.shape[-1] != SEAN_ROBOT_DIM:
            raise ValueError(
                f"UnifiedSEANDataset expected robot.x dim={SEAN_ROBOT_DIM}, "
                f"got {data['robot'].x.shape[-1]}."
            )
        data["bystander"].num_nodes = 1 + int(source["bystander"].num_nodes)

        if hasattr(source["robot"], "y"):
            data["robot"].y = source["robot"].y

        srcs = {et: [] for et in UNIFIED_EDGE_TYPES}
        dsts = {et: [] for et in UNIFIED_EDGE_TYPES}
        attrs = {et: [] for et in UNIFIED_EDGE_TYPES}

        def add_edges(native_et, unified_et, src_offset=0, dst_offset=0):
            if not _has_edges(source, native_et):
                return
            edge_index = source[native_et].edge_index
            edge_attr = source[native_et].edge_attr
            for edge_idx in range(edge_index.shape[1]):
                srcs[unified_et].append(int(edge_index[0, edge_idx]) + src_offset)
                dsts[unified_et].append(int(edge_index[1, edge_idx]) + dst_offset)
                attrs[unified_et].append(edge_attr[edge_idx])

        add_edges(
            ("robot", "to", "follower"),
            ("robot", "to", "bystander"),
            src_offset=0,
            dst_offset=0,
        )
        add_edges(
            ("robot", "to", "bystander"),
            ("robot", "to", "bystander"),
            src_offset=0,
            dst_offset=1,
        )
        add_edges(
            ("follower", "to", "robot"),
            ("bystander", "to", "robot"),
            src_offset=0,
            dst_offset=0,
        )
        add_edges(
            ("bystander", "to", "robot"),
            ("bystander", "to", "robot"),
            src_offset=1,
            dst_offset=0,
        )
        add_edges(
            ("follower", "to", "bystander"),
            ("bystander", "to", "bystander"),
            src_offset=0,
            dst_offset=1,
        )
        add_edges(
            ("bystander", "to", "follower"),
            ("bystander", "to", "bystander"),
            src_offset=1,
            dst_offset=0,
        )
        add_edges(
            ("bystander", "to", "bystander"),
            ("bystander", "to", "bystander"),
            src_offset=1,
            dst_offset=1,
        )

        for edge_type in UNIFIED_EDGE_TYPES:
            if attrs[edge_type]:
                data[edge_type].edge_index = torch.tensor(
                    [srcs[edge_type], dsts[edge_type]],
                    dtype=torch.long,
                )
                data[edge_type].edge_attr = torch.stack(attrs[edge_type])
            else:
                data[edge_type].edge_index = _empty_edge_index()
                data[edge_type].edge_attr = _empty_edge_attr(timesteps)

        data.dataset_id = SEAN_ID
        return data


class UnifiedSocialNavDataset(Dataset):
    """
    Wrap SocialNavGraphDatasetV1 and rename human -> bystander.

    Robot input becomes [robot trajectory | goal] so synthetic T=20 produces
    the UnifiedGNNVAE expected 62D robot feature vector.
    """

    def __init__(self, h5_path=None, base_dataset=None, transform=None, preload=True, zero_node_features=False):
        super().__init__(root=None, transform=transform)
        if base_dataset is None:
            if h5_path is None:
                raise ValueError("UnifiedSocialNavDataset requires h5_path or base_dataset.")
            base_dataset = SocialNavGraphDatasetV1(h5_path, preload=preload)
        self.base_dataset = base_dataset
        self.robot_mean = None
        self.robot_std = None
        self.bystander_mean = None
        self.bystander_std = None
        self.zero_node_features = zero_node_features

    @property
    def keys(self):
        return getattr(self.base_dataset, "keys", None)

    def len(self):
        return len(self.base_dataset)

    def set_feature_stats(self, train_indices):
        robot_features = []
        bystander_features = []

        for idx in train_indices:
            source = self.base_dataset[int(idx)]
            timesteps = _infer_timesteps(source, fallback=int(source.num_timesteps))
            if timesteps != SNSB_TIMESTEPS:
                raise ValueError(
                    f"UnifiedSocialNavDataset expected T={SNSB_TIMESTEPS}, got T={timesteps} "
                    f"while computing train stats at index {idx}."
                )

            robot_features.append(source["robot"].x)  # [1, 60]
            if source["human"].x.shape[0] > 0:
                bystander_features.append(source["human"].x)

        robot_features = torch.cat(robot_features, dim=0)
        self.robot_mean = robot_features.mean(dim=0)
        self.robot_std = robot_features.std(dim=0, unbiased=False)

        if bystander_features:
            bystander_features = torch.cat(bystander_features, dim=0)
            self.bystander_mean = bystander_features.mean(dim=0)
            self.bystander_std = bystander_features.std(dim=0, unbiased=False)
        else:
            self.bystander_mean = torch.zeros(SNSB_BYSTANDER_DIM)
            self.bystander_std = torch.ones(SNSB_BYSTANDER_DIM)

        print(
            f"[UnifiedSocialNavDataset] Feature stats from train split (n={len(train_indices)} graphs): "
            f"robot std min={self.robot_std.min():.4f} max={self.robot_std.max():.4f} | "
            f"bystander std min={self.bystander_std.min():.4f} max={self.bystander_std.max():.4f}"
        )

    def _normalize_robot(self, robot_x):
        if self.robot_mean is None or self.robot_std is None:
            raise RuntimeError("SNSB feature stats not set - call set_feature_stats(train_indices).")
        return (robot_x - self.robot_mean) / (self.robot_std + 1e-8)

    def _normalize_bystander(self, bystander_x):
        if self.bystander_mean is None or self.bystander_std is None:
            raise RuntimeError("SNSB feature stats not set - call set_feature_stats(train_indices).")
        return (bystander_x - self.bystander_mean) / (self.bystander_std + 1e-8)

    def get(self, idx):
        source = self.base_dataset[idx]
        timesteps = _infer_timesteps(source, fallback=int(source.num_timesteps))

        data = HeteroData()
        robot_x = source["robot"].x
        expected_robot_dim = timesteps * 3
        if robot_x.shape[-1] != expected_robot_dim:
            raise ValueError(
                f"UnifiedSocialNavDataset expected robot.x dim={expected_robot_dim}, "
                f"got {robot_x.shape[-1]}."
            )
        data["robot"].x = torch.zeros(robot_x.shape[0], expected_robot_dim) if self.zero_node_features else self._normalize_robot(robot_x)
        data["robot"].traj = source["robot"].traj

        if self.zero_node_features:
            expected_bystander_dim = timesteps * 3
            data["bystander"].x = torch.zeros(source["human"].x.shape[0], expected_bystander_dim)
        else:
            data["bystander"].x = self._normalize_bystander(source["human"].x)
        expected_bystander_dim = timesteps * 3
        if data["bystander"].x.shape[-1] != expected_bystander_dim:
            raise ValueError(
                f"UnifiedSocialNavDataset expected bystander.x dim={expected_bystander_dim}, "
                f"got {data['bystander'].x.shape[-1]}."
            )
        data["bystander"].traj = source["human"].traj
        data["bystander"].y = source["human"].y
        if hasattr(source["human"], "action_probs"):
            data["bystander"].action_probs = source["human"].action_probs

        edge_map = {
            ("robot", "to", "human"): ("robot", "to", "bystander"),
            ("human", "to", "robot"): ("bystander", "to", "robot"),
            ("human", "to", "human"): ("bystander", "to", "bystander"),
        }

        for source_et, target_et in edge_map.items():
            if _has_edges(source, source_et):
                data[target_et].edge_index = source[source_et].edge_index
                data[target_et].edge_attr = source[source_et].edge_attr
            else:
                data[target_et].edge_index = _empty_edge_index()
                data[target_et].edge_attr = _empty_edge_attr(timesteps)

        data.dataset_id = SNSB_ID
        data.scene = getattr(source, "scene", "")
        data.episode_key = getattr(source, "episode_key", "")
        data.real = getattr(source, "real", 0)
        data.num_humans = getattr(source, "num_humans", int(source["human"].x.shape[0]))
        data.num_timesteps = timesteps
        return data
