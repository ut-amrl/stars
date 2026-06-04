import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import argparse
import os
import sys

# Ensure import paths work regardless of where the script is called from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.stacked_temporal_attributes import HDF5StackedTemporalDataset, create_mock_hdf5
from models.mpgnn import GraphAutoencoder

def train():
    parser = argparse.ArgumentParser(description="Train Graph Autoencoder on Scene Graphs")
    parser.add_argument('--hdf5_path', type=str, default='mock_data.h5', help='Path to HDF5 data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Model configuration
    parser.add_argument('--num_timesteps', type=int, default=3, help='Temporal window size')
    parser.add_argument('--node_channels', type=int, default=1, help='Number of channels in occupancy grid')
    parser.add_argument('--edge_features', type=int, default=2, help='Number of edge features (e.g. distance, similarity)')
    parser.add_argument('--global_dim', type=int, default=512, help='Text embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension in GNN')
    parser.add_argument('--H', type=int, default=32, help='Height of occupancy grid')
    parser.add_argument('--W', type=int, default=32, help='Width of occupancy grid')
    
    args = parser.parse_args()
    
    if args.hdf5_path == 'mock_data.h5' and not os.path.exists('mock_data.h5'):
        print("Creating mock HDF5 data for testing...")
        create_mock_hdf5('mock_data.h5', num_graphs=10, num_nodes=4, 
                         num_timesteps=args.num_timesteps, 
                         H=args.H, W=args.W)

    print(f"Loading HDF5 dataset from {args.hdf5_path}...")
    dataset = HDF5StackedTemporalDataset(args.hdf5_path)
    
    # PyG Dataloader automatically handles batching of variable-size graphs
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    node_in_channels = args.num_timesteps * args.node_channels
    edge_in_features = args.num_timesteps * args.edge_features
    
    print("Initializing Model...")
    model = GraphAutoencoder(
        node_in_channels=node_in_channels,
        edge_in_features=edge_in_features,
        global_in_dim=args.global_dim,
        hidden_dim=args.hidden_dim,
        H=args.H,
        W=args.W
    )
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    print("Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_data in loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            # batch_data contains x, edge_index, edge_attr, global_attr, batch
            reconstructed_x = model(
                batch_data.x, 
                batch_data.edge_index, 
                batch_data.edge_attr, 
                batch_data.global_attr, 
                batch_data.batch
            )
            
            # Autoencoder loss: reconstruct the node attributes
            loss = criterion(reconstructed_x, batch_data.x)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_data.num_graphs
            
        mean_loss = total_loss / max(1, len(dataset))
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {mean_loss:.4f}")

    print("Training finished successfully.")

if __name__ == '__main__':
    train()
