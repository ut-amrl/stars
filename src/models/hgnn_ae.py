"""
HGNN-AE: Homogeneous Graph Neural Network Autoencoder.

Robot node (type 0) is initialized from projected 592D features + type embedding.
Follower/bystander nodes (types 1/2) are initialized from type embedding only (no GT features).
Edge features [40, 8] are flattened and encoded with a small MLP.
Message passing uses attention over neighbor messages (HEAT-style, simplified).
Decoder reconstructs robot node features and all edge features.

Loss = recon_robot + recon_edges  (no KL, plain AE)

Node types: 0=robot, 1=follower, 2=bystander
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

N_NODE_TYPES     = 3
N_TIMESTEPS      = 40
EDGE_FEATURE_DIM = 8
TEMPORAL_FLAT    = N_TIMESTEPS * EDGE_FEATURE_DIM   
ROBOT_DIM        = 592


class EdgeEncoder(nn.Module):
    """Flatten [E, 40, 8] -> [E, hidden_dim]."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(TEMPORAL_FLAT, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr.view(edge_attr.shape[0], -1)) 


class MPLayer(nn.Module):
    """
    One round of message passing.
    Message = MLP([h_src | h_dst | edge_enc])
    Attention score from same input, softmax over incoming messages per node.
    Update = LayerNorm(h + MLP([h | agg]))
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_enc):
        src_idx, dst_idx = edge_index[0], edge_index[1]
        inp    = torch.cat([h[src_idx], h[dst_idx], edge_enc], dim=-1)
        msgs   = self.msg_mlp(inp)
        scores = self.attn_mlp(inp).squeeze(-1)

        N     = h.shape[0]
        max_s = scatter(scores, dst_idx, dim=0, reduce='max', dim_size=N)
        exp_s = torch.exp(scores - max_s[dst_idx])
        sum_s = scatter(exp_s, dst_idx, dim=0, reduce='sum', dim_size=N)
        alpha = exp_s / sum_s[dst_idx].clamp(min=1e-8)

        agg = scatter(alpha.unsqueeze(1) * msgs, dst_idx, dim=0,
                      reduce='sum', dim_size=N)

        updated = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + updated)


class HGNNEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_mp_layers):
        super().__init__()
        self.node_proj     = nn.Linear(ROBOT_DIM, hidden_dim)
        self.node_type_emb = nn.Embedding(N_NODE_TYPES, hidden_dim)
        self.edge_enc      = EdgeEncoder(hidden_dim)
        self.mp_layers     = nn.ModuleList([MPLayer(hidden_dim) for _ in range(num_mp_layers)])
        self.fc_z          = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        h = self.node_type_emb(data.node_type)
        robot_mask = data.node_type == 0
        robot_proj = torch.zeros_like(h)
        robot_proj[robot_mask] = self.node_proj(data.x[robot_mask])
        h = h + robot_proj
        edge_enc = self.edge_enc(data.edge_attr)
        for mp in self.mp_layers:
            h = mp(h, data.edge_index, edge_enc)
        return self.fc_z(h)


class HGNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.pre_decode = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.robot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, ROBOT_DIM)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, TEMPORAL_FLAT)
        )

    def forward(self, z, data):
        h           = self.pre_decode(z)
        robot_recon = self.robot_head(h[data.node_type == 0])

        src_idx, dst_idx = data.edge_index[0], data.edge_index[1]
        edge_recon = self.edge_head(
            torch.cat([h[src_idx], h[dst_idx]], dim=-1)
        ).view(-1, N_TIMESTEPS, EDGE_FEATURE_DIM)

        return robot_recon, edge_recon


class HGNNAE(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=32, num_mp_layers=2):
        super().__init__()
        self.encoder = HGNNEncoder(hidden_dim, latent_dim, num_mp_layers)
        self.decoder = HGNNDecoder(latent_dim, hidden_dim)

    def forward(self, data):
        z                       = self.encoder(data)
        robot_recon, edge_recon = self.decoder(z, data)
        return robot_recon, edge_recon

    def encode(self, data):
        """Return [z_robot || z_follower || mean(z_bystanders)] for downstream use. [B, 3*latent_dim]"""
        z     = self.encoder(data)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        B     = int(batch.max()) + 1
        z_robot    = z[data.node_type == 0]
        z_follower = z[data.node_type == 1]
        z_b_mean   = scatter(z[data.node_type == 2], batch[data.node_type == 2],
                             dim=0, reduce='mean', dim_size=B)
        return torch.cat([z_robot, z_follower, z_b_mean], dim=-1)   # [B, 3*latent_dim]
