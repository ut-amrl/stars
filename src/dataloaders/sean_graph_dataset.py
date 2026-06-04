"""
PyTorch Geometric Dataset that reads the SEAN graph HDF5 file
produced by sean_to_hdf5.py.

Each sample is one timestep and contains:
  x           : node features  [N, 5]   (x, y, cos, sin, node_type)
  edge_index  : connectivity   [2, E]
  edge_attr   : edge features  [E, 3]   (distance, rel_dir_x, rel_dir_y)
  occ         : occupancy grid [72, 72] normalised to [0, 1]
  global_attr : behavior label [1]
  y           : true_label     [3]

Usage:
  from dataloaders.sean_graph_dataset import SEANGraphDataset
  dataset = SEANGraphDataset("sean_together.h5")
  data = dataset[0]
  print(data.x.shape, data.edge_index.shape)
"""

import h5py
import torch
from torch_geometric.data import Dataset, Data


class SEANGraphDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        super().__init__(root=None, transform=transform)
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))

    def len(self):
        return len(self.keys)

    def get(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[key]
            x           = torch.tensor(grp["node_features"][:],  dtype=torch.float)
            edge_index  = torch.tensor(grp["edge_index"][:],     dtype=torch.long)
            edge_attr   = torch.tensor(grp["edge_attr"][:],      dtype=torch.float)
            occ         = torch.tensor(grp["occupancy_grid"][:], dtype=torch.float)
            global_attr = torch.tensor(grp["global_attr"][:],    dtype=torch.float)
            y           = torch.tensor(grp["label"][:],          dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            occ=occ,
            global_attr=global_attr,
            y=y
        )
