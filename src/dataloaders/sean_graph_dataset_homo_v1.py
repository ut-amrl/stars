"""
v5: Homogeneous Data loader - single node/edge tensors for homogeneous GNN.

Changes from v4:
  Returns torch_geometric.data.Data instead of HeteroData.
  All 18 nodes in one matrix data.x [18, 592]
  data.node_type [18] int64 - 0=robot, 1=follower, 2=bystander (Used for looking up embeddings)
  Single data.edge_index [2, E] and data.edge_attr [E, 40, 8]
"""

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class SEANGraphDatasetV5(Dataset):
    def __init__(self, h5_path, transform=None, robot_mean=None, robot_std=None):
        super().__init__(root=None, transform=transform)
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))

        self.robot_mean = robot_mean
        self.robot_std  = robot_std
        self._print_edge_stats()

    def set_robot_stats(self, train_indices):
        self.robot_mean, self.robot_std = self._compute_robot_stats(train_indices)

    def _compute_robot_stats(self, indices=None):
        keys = [self.keys[i] for i in indices] if indices is not None else self.keys
        all_robot_x = []
        with h5py.File(self.h5_path, 'r') as f:
            for key in keys:
                all_robot_x.append(f[key]['node_features'][0])
        all_robot_x = np.stack(all_robot_x)
        mean = torch.tensor(all_robot_x.mean(axis=0), dtype=torch.float)
        std  = torch.tensor(all_robot_x.std(axis=0),  dtype=torch.float)
        print(f"[SEANGraphDatasetV5] Robot feature stats (n={len(keys)}): mean [{mean.min():.2f}, {mean.max():.2f}]  std [{std.min():.4f}, {std.max():.2f}]")
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
            f"[SEANGraphDatasetV5] {len(self.keys)} graphs | "
            f"edges min={edge_counts.min()} max={edge_counts.max()} mean={edge_counts.mean():.1f} | "
            f"bystanders mean={bystander_counts.mean():.1f}/16"
        )

    def len(self):
        return len(self.keys)

    def get(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            grp            = f[key]
            node_features  = torch.tensor(grp['node_features'][:], dtype=torch.float)  
            node_type      = torch.tensor(grp['node_type'][:],     dtype=torch.long)   
            valid_mask     = torch.tensor(grp['valid_mask'][:],    dtype=torch.float)  
            edge_index_all = torch.tensor(grp['edge_index'][:],    dtype=torch.long)   
            edge_attr_all  = torch.tensor(grp['edge_attr'][:],     dtype=torch.float)  
            y              = torch.tensor(grp['label'][:],         dtype=torch.float)  

        # --- Normalize robot node features ---
        if self.robot_mean is None or self.robot_std is None:
            raise RuntimeError("robot_mean/robot_std not set - call dataset.set_robot_stats(train_indices) after splitting.")
        node_features[0] = (node_features[0] - self.robot_mean) / (self.robot_std + 1e-8)

        # --- Filter edges where either node is never present ---
        node_presence = valid_mask.sum(dim=1) > 0
        keep          = node_presence[edge_index_all[0]] & node_presence[edge_index_all[1]]
        edge_index    = edge_index_all[:, keep]
        edge_attr     = edge_attr_all[keep]

        data          = Data()
        data.x        = node_features
        data.node_type = node_type      
        data.edge_index = edge_index          
        data.edge_attr  = edge_attr            
        data.y          = y.unsqueeze(0)        

        return data
