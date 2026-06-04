"""
GNN-VAE for social scene encoding.

Architecture:
  Encoder:
    - Per-node MLP: node features [N, node_dim] -> [N, hidden_dim]
    - Per-edge MLP: edge features [E, edge_dim] -> [E, hidden_dim]
    - Message passing layers: aggregate neighbourhood info into each node
    - Global pooling: mean all node embeddings -> single graph vector [hidden_dim]
    - Project to mu [latent_dim] and logvar [latent_dim]
    - Reparameterize -> z [latent_dim]

  Decoder:
    - Broadcast graph-level z to each node, then MLP: [N, latent_dim] -> [N, node_dim]

Loss (computed outside, in training loop):
  recon_loss = MSE(x_recon, x)
  kl_loss    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
  total_loss = recon_loss + beta * kl_loss

Input per sample (from SEANGraphDataset):
  x          : [N, 5]   node features  (x, y, cos, sin, node_type)
  edge_index : [2, E]   graph connectivity
  edge_attr  : [E, 3]   edge features  (distance, rel_dir_x, rel_dir_y)
  global_attr: [1]      behavior label
  batch      : [N]      batch assignment (from PyG DataLoader)

Output:
  x_recon : [N, 5]      reconstructed node features
  mu      : [B, latent_dim]
  logvar  : [B, latent_dim]
  z       : [B, latent_dim]
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


# ---------------------------------------------------------------------------
# Message Passing Layer
# ---------------------------------------------------------------------------

class MPLayer(MessagePassing):
    """Single message passing step.

    For each node: aggregate messages from neighbours, then update node state.
    Messages are computed from (source node, target node, edge features).
    """
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='mean')
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim]
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [N, node_dim]
        return self.update_mlp(torch.cat([x, agg], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, latent_dim, num_mp_layers=3):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mp_layers = nn.ModuleList([
            MPLayer(hidden_dim, hidden_dim) for _ in range(num_mp_layers)
        ])

        # Global pooling -> mu, logvar
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_encoder(x)           # [N, hidden_dim]
        e = self.edge_encoder(edge_attr)   # [E, hidden_dim]

        for mp in self.mp_layers:
            h = h + mp(h, edge_index, e)   # residual connection

        # Global mean pooling over nodes -> one vector per graph in batch
        h_graph = scatter(h, batch, dim=0, reduce='mean')  # [B, hidden_dim]

        mu     = self.fc_mu(h_graph)       # [B, latent_dim]
        logvar = self.fc_logvar(h_graph)   # [B, latent_dim]
        return mu, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class GNNDecoder(nn.Module):
    """Decode z back to node features.

    Since z is a single graph-level vector but we need per-node outputs,
    we broadcast z to all N nodes then use an MLP to reconstruct each node.
    """
    def __init__(self, latent_dim, hidden_dim, node_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, z, batch):
        # Broadcast graph-level z to each node using batch assignment
        z_per_node = z[batch]              # [N, latent_dim]
        return self.mlp(z_per_node)        # [N, node_dim]


# ---------------------------------------------------------------------------
# GNN-VAE
# ---------------------------------------------------------------------------

class GNNVAE(nn.Module):
    def __init__(self,
                 node_dim=5,
                 edge_dim=3,
                 hidden_dim=128,
                 latent_dim=64,
                 num_mp_layers=3):
        """
        Args:
            node_dim     : input node feature size (default 5: x,y,cos,sin,type)
            edge_dim     : input edge feature size (default 3: dist,rel_x,rel_y)
            hidden_dim   : hidden size throughout encoder/decoder
            latent_dim   : size of latent z
            num_mp_layers: number of message passing layers in encoder
        """
        super().__init__()
        self.encoder = GNNEncoder(node_dim, edge_dim, hidden_dim, latent_dim, num_mp_layers)
        self.decoder = GNNDecoder(latent_dim, hidden_dim, node_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # at inference, just use mean

    def forward(self, x, edge_index, edge_attr, batch):
        mu, logvar = self.encoder(x, edge_index, edge_attr, batch)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decoder(z, batch)
        return x_recon, mu, logvar, z

    def encode(self, x, edge_index, edge_attr, batch):
        """Encode only - returns z (mean, no sampling)."""
        mu, _ = self.encoder(x, edge_index, edge_attr, batch)
        return mu

    def decode(self, z, batch):
        """Decode a latent z -> node features."""
        return self.decoder(z, batch)
