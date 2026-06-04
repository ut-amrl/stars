import torch
import h5py
from torch_geometric.data import Dataset, Data
import os

class HDF5StackedTemporalDataset(Dataset):
    """
    Reads scene-level graph information from a compact HDF5 format.
    Each group in the HDF5 file represents a single graph/scene.
    """
    def __init__(self, h5_file_path, transform=None, pre_transform=None):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.h5_file_path = h5_file_path
        
        if os.path.exists(self.h5_file_path):
            with h5py.File(str(self.h5_file_path), 'r') as f:
                self.keys = list(f.keys())
        else:
            print(f"Warning: HDF5 file not found at {self.h5_file_path}. Initializing with empty keys if you are just testing architecture.")
            self.keys = []

    def len(self):
        return len(self.keys)

    def get(self, idx):
        key = self.keys[idx]
        with h5py.File(str(self.h5_file_path), 'r') as f:
            group = f[key]
            
            # Node attributes: predicted shape e.g. [num_nodes, num_timesteps, channels, H, W]
            x = torch.tensor(group['node_attributes'][:], dtype=torch.float)
            
            # Edge Index: [2, num_edges] connecting the nodes
            edge_index = torch.tensor(group['edge_index'][:], dtype=torch.long)
            
            # Edge attributes: e.g. metric distance, cosine similarity: [num_edges, num_timesteps, 2]
            edge_attr = torch.tensor(group['edge_attributes'][:], dtype=torch.float)
            
            # Global attribute: text description embedding: [1, text_embedding_dim]
            global_attr = torch.tensor(group['global_attribute'][:], dtype=torch.float)
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_attr=global_attr
            )
            
        return data

def create_mock_hdf5(file_path, num_graphs=5, num_nodes=4, num_timesteps=3, H=32, W=32):
    """Utility to create a mock HDF5 dataset to verify the training loop."""
    import numpy as np
    
    with h5py.File(file_path, 'w') as f:
        for i in range(num_graphs):
            group = f.create_group(f"scene_{i}")
            
            # Create synthetic occupancy grid [num_nodes, num_timesteps, channels, H, W]
            # using 1 channel for occupancy
            node_attr = np.random.rand(num_nodes, num_timesteps, 1, H, W).astype(np.float32)
            group.create_dataset('node_attributes', data=node_attr)
            
            # Create fully connected edge index or random
            # Example: 0->1, 1->2, 2->3
            edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
            group.create_dataset('edge_index', data=edge_index)
            
            num_edges = edge_index.shape[1]
            
            # Edge attributes: distance and cosine similarity [num_edges, num_timesteps, 2]
            edge_attr = np.random.rand(num_edges, num_timesteps, 2).astype(np.float32)
            group.create_dataset('edge_attributes', data=edge_attr)
            
            # Global attribute [1, 512] text embedding
            global_attr = np.random.rand(1, 512).astype(np.float32)
            group.create_dataset('global_attribute', data=global_attr)
    print(f"Created mock HDF5 dataset at {file_path}")
