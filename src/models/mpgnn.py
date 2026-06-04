import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, H=32, W=32):
        super().__init__()
        # 2D CNN over stacked time and channels. 
        # Output spatial dim will be reduced.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # H/2, W/2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # H/4, W/4
        )
        self.flat_dim = 32 * (H // 4) * (W // 4)
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        # x is [num_nodes, num_timesteps, channels, H, W]
        N, T, C, H_in, W_in = x.shape
        # Merge Time and Channels for 2D Conv
        x = x.view(N, T * C, H_in, W_in)
        x = self.conv(x)
        x = x.view(N, -1)
        return self.mlp(x)

class NodeDecoder(nn.Module):
    def __init__(self, hidden_dim, out_channels, out_H=32, out_W=32):
        super().__init__()
        self.out_channels = out_channels
        self.out_H = out_H
        self.out_W = out_W
        self.flat_dim = 32 * (out_H // 4) * (out_W // 4)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_dim)
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # H/2, W/2
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1) # H, W
        )
        
    def forward(self, x):
        # x is [num_nodes, hidden_dim]
        # output should be [num_nodes, T*C, H, W]
        x = self.mlp(x)
        x = x.view(x.shape[0], 32, self.out_H // 4, self.out_W // 4)
        x = self.deconv(x)
        return x

class MPGNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, node_dim], edge_attr: [E, edge_dim], u: [B, global_dim]
        # Map global feature u to nodes based on batch
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            
        # Select global features for each node and each edge
        u_node = u[batch]
        row, col = edge_index
        u_edge = u[batch[row]]

        # Pass messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, u_edge=u_edge)
        
        # Update nodes
        x_updated = self.node_mlp(torch.cat([x, out, u_node], dim=-1))
        return x_updated

    def message(self, x_i, x_j, edge_attr, u_edge):
        # x_i: target node, x_j: source node
        msg = torch.cat([x_i, x_j, edge_attr, u_edge], dim=-1)
        return self.edge_mlp(msg)


class GraphAutoencoder(nn.Module):
    def __init__(self, 
                 node_in_channels,     # T * C (num_timesteps * feature_channels)
                 edge_in_features,     # T * F (num_timesteps * edge_features)
                 global_in_dim=512, 
                 hidden_dim=64, 
                 num_mp_layers=2,
                 H=32, 
                 W=32):
        super().__init__()
        
        self.node_encoder = NodeEncoder(node_in_channels, hidden_dim, H, W)
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(global_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mp_layers = nn.ModuleList([
            MPGNNLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_mp_layers)
        ])
        
        self.node_decoder = NodeDecoder(hidden_dim, node_in_channels, H, W)

    def forward(self, x, edge_index, edge_attr, u, batch=None):
        # x shape validation: expect [N, T, C, H, W]
        N, T, C, H_in, W_in = x.shape
        
        # Encode inputs
        h_node = self.node_encoder(x)
        
        # edge_attr: [num_edges, T, F] -> flatten to [num_edges, T*F]
        h_edge = edge_attr.view(edge_attr.size(0), -1)
        h_edge = self.edge_encoder(h_edge)
        
        h_global = self.global_encoder(u)
        
        # Message Passing
        for mp_layer in self.mp_layers:
            h_node = mp_layer(h_node, edge_index, h_edge, h_global, batch)
            
        # Decode back to node attributes
        x_reconstructed = self.node_decoder(h_node)
        
        # Output is [num_nodes, T*C, H, W], reshape back to [num_nodes, T, C, H_out, W_out]
        x_reconstructed = x_reconstructed.view(N, T, C, x.shape[3], x.shape[4])
        return x_reconstructed
