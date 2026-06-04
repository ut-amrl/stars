"""
Homogeneous GNN Autoencoder for the combined SEAN + SSUB dataset.

Handles two sources of heterogeneous features gracefully:

  data.x          [N, 592]    Robot node features (SEAN only; zeros for SSUB).
                              A bias-free linear projection means zeros contribute
                              nothing — SSUB robot falls back to type-embedding only.

  data.valid_mask [N, 40]     Per-node per-timestep presence (SEAN bystanders can
                              be absent at individual timesteps; SSUB is all-ones).
                              Used to build an edge-level padding mask so the
                              TemporalEdgeTransformer ignores invalid timesteps and
                              mean-pools only over valid ones.

  data.node_type  [N]         0=robot, 1=people  (unified across both datasets)
  data.edge_attr  [E, 40, 8]  SE(2) relative pose + deltas (SSUB upsampled 20→40)
  data.edge_index [2, E]      Fully-connected over ever-present nodes

forward(data)        → (edge_recon [E,40,8], node_feat_recon [N,592], mu [N,L], logvar [N,L])
encode_nodes(data)   → mu          [N, latent_dim]
encode(data)         → graph_vec   [B, 2 * latent_dim]  [z_robot ‖ mean(z_people)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import scatter


N_NODE_TYPES     = 2
UNIFIED_TIMESTEPS = 40
EDGE_FEATURE_DIM  = 8
ROBOT_FEAT_DIM    = 592


class TemporalEdgeTransformer(nn.Module):
    """
    Encode edge_attr [E, T, 8] → [E, H].

    Accepts an optional boolean padding mask [E, T] (True = invalid timestep)
    which is forwarded to the TransformerEncoder and used to exclude invalid
    positions from the mean-pool.  For SSUB (all timesteps valid) the mask is
    None and the behaviour is identical to hgnn_ssub_v2.
    """

    def __init__(self, edge_dim=EDGE_FEATURE_DIM, hidden_dim=64,
                 num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(edge_dim, hidden_dim)
        self.pos_embed  = nn.Parameter(torch.zeros(1, UNIFIED_TIMESTEPS, hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_attr, src_key_padding_mask=None):
        """
        Args:
            edge_attr             [E, T, 8]
            src_key_padding_mask  [E, T] bool or None — True = ignore this timestep
        """
        h = self.input_proj(edge_attr) + self.pos_embed   # [E, T, H]
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Mean-pool over valid timesteps only
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(-1)   # [E, T, 1]
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            h = h.mean(dim=1)

        return self.out_norm(h)   # [E, H]


class CombinedHGNNEncoder(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=32, num_mp_layers=2,
                 edge_transformer_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.node_type_emb = nn.Embedding(N_NODE_TYPES, hidden_dim)

        # Projects SEAN robot features onto node embeddings.
        # bias=False ensures proj(zeros) = 0 so SSUB robots are unaffected.
        self.robot_feat_proj = nn.Linear(ROBOT_FEAT_DIM, hidden_dim, bias=False)

        self.edge_encoder = TemporalEdgeTransformer(
            hidden_dim=hidden_dim,
            num_layers=edge_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.convs = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                beta=True,
                dropout=dropout,
                edge_dim=hidden_dim,
            )
            for _ in range(num_mp_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_mp_layers)])
        self.dropout  = nn.Dropout(dropout)
        self.to_mu     = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def _encode_h(self, data):
        h = self.node_type_emb(data.node_type)

        if data.x is not None and data.x.shape[-1] == ROBOT_FEAT_DIM:
            robot_mask = data.node_type == 0
            h = h.clone()
            h[robot_mask] = h[robot_mask] + self.robot_feat_proj(data.x[robot_mask])

        edge_pad_mask = None
        if hasattr(data, "valid_mask") and data.valid_mask is not None:
            src, dst = data.edge_index[0], data.edge_index[1]
            edge_valid    = (data.valid_mask[src] * data.valid_mask[dst]).bool()
            edge_pad_mask = ~edge_valid

        edge_emb = self.edge_encoder(data.edge_attr, src_key_padding_mask=edge_pad_mask)

        use_first = getattr(self, "use_first_mp_layer", False)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_next = conv(h, data.edge_index, edge_attr=edge_emb)
            h = norm(h + self.dropout(F.gelu(h_next)))
            if use_first and i == 0:
                break

        return h

    def forward(self, data):
        h = self._encode_h(data)
        mu     = self.to_mu(h)
        logvar = self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        z   = mu + std * torch.randn_like(std)
        return z, mu, logvar


class CombinedHGNNDecoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, UNIFIED_TIMESTEPS * EDGE_FEATURE_DIM),
        )
        # Shared node feature reconstruction head — same architecture for all
        # node types.  In practice only SEAN robot nodes have non-zero targets;
        # the training script masks the loss to those nodes.
        self.node_feat_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ROBOT_FEAT_DIM),
        )

    def forward(self, z, edge_index):
        h   = self.node_proj(z)
        src, dst = edge_index[0], edge_index[1]
        edge_recon      = self.edge_head(
            torch.cat([h[src], h[dst]], dim=-1)
        ).view(-1, UNIFIED_TIMESTEPS, EDGE_FEATURE_DIM)   # [E, 40, 8]
        node_feat_recon = self.node_feat_head(z)           # [N, 592]
        return edge_recon, node_feat_recon


class HGNNCombinedVAE(nn.Module):
    """
    Homogeneous graph VAE for the combined SEAN + SSUB dataset.

    forward(data)       → (edge_recon, node_feat_recon, mu, logvar)
    encode_nodes(data)  → mu  [N, latent_dim]
    encode(data)        → [B, 2 * latent_dim]  [z_robot ‖ mean(z_people)]
    """

    def __init__(self, hidden_dim=64, latent_dim=32, num_mp_layers=2,
                 edge_transformer_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = CombinedHGNNEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_mp_layers=num_mp_layers,
            edge_transformer_layers=edge_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = CombinedHGNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, data):
        z, mu, logvar = self.encoder(data)
        edge_recon, node_feat_recon = self.decoder(z, data.edge_index)
        return edge_recon, node_feat_recon, mu, logvar

    def encode_nodes(self, data):
        h = self.encoder._encode_h(data)
        if getattr(self, "use_first_mp_layer", False):
            return h
        return self.encoder.to_mu(h)

    def encode(self, data):
        mu = self.encode_nodes(data)
        batch = getattr(
            data, "batch",
            torch.zeros(mu.shape[0], dtype=torch.long, device=mu.device),
        )
        B = int(batch.max()) + 1

        robot_mask  = data.node_type == 0
        people_mask = data.node_type == 1

        z_robot  = scatter(mu[robot_mask],  batch[robot_mask],  dim=0, reduce="mean", dim_size=B)
        z_people = scatter(mu[people_mask], batch[people_mask], dim=0, reduce="mean", dim_size=B)
        return torch.cat([z_robot, z_people], dim=-1)
