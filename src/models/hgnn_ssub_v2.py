import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import scatter


N_NODE_TYPES = 2
N_TIMESTEPS = 20
EDGE_FEATURE_DIM = 8


class TemporalEdgeTransformer(nn.Module):
    """Encode edge_attr [E, T, F] into one edge vector [E, H]."""

    def __init__(
        self,
        edge_dim=EDGE_FEATURE_DIM,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(edge_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, N_TIMESTEPS, hidden_dim))

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

    def forward(self, edge_attr):
        if edge_attr.shape[1] != N_TIMESTEPS:
            raise ValueError(f"Expected {N_TIMESTEPS} timesteps, got {edge_attr.shape[1]}")

        h = self.input_proj(edge_attr) + self.pos_embed
        h = self.transformer(h)
        return self.out_norm(h.mean(dim=1))


class EdgeOnlyHGNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        latent_dim=32,
        num_mp_layers=2,
        edge_transformer_layers=1,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.node_type_emb = nn.Embedding(N_NODE_TYPES, hidden_dim)
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
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_mp_layers)])
        self.dropout = nn.Dropout(dropout)
        self.to_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        h = self.node_type_emb(data.node_type)
        edge_emb = self.edge_encoder(data.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, data.edge_index, edge_attr=edge_emb)
            h = norm(h + self.dropout(F.gelu(h_next)))

        return self.to_z(h)


class EdgeOnlyHGNNDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=32,
        hidden_dim=64,
        num_timesteps=N_TIMESTEPS,
        edge_dim=EDGE_FEATURE_DIM,
        dropout=0.1,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.edge_dim = edge_dim

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
            nn.Linear(hidden_dim, num_timesteps * edge_dim),
        )

    def forward(self, z, edge_index):
        h = self.node_proj(z)
        src, dst = edge_index[0], edge_index[1]
        edge_recon = self.edge_head(torch.cat([h[src], h[dst]], dim=-1))
        return edge_recon.view(-1, self.num_timesteps, self.edge_dim)


class HGNNSSubAE(nn.Module):
    """
    Plain homogeneous graph autoencoder for variable-size graphs.

    forward(data) returns edge_recon [E, 20, 8].
    encode_nodes(data) returns per-node latents [N, latent_dim].
    encode(data) returns graph latents [B, 2 * latent_dim]:
      [robot_latent || mean_human_latent]
    """

    def __init__(
        self,
        hidden_dim=64,
        latent_dim=32,
        num_mp_layers=2,
        edge_transformer_layers=1,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = EdgeOnlyHGNNEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_mp_layers=num_mp_layers,
            edge_transformer_layers=edge_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = EdgeOnlyHGNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, data):
        z = self.encoder(data)
        return self.decoder(z, data.edge_index)

    def encode_nodes(self, data):
        return self.encoder(data)

    def encode(self, data):
        z = self.encoder(data)
        batch = getattr(
            data,
            "batch",
            torch.zeros(z.shape[0], dtype=torch.long, device=z.device),
        )
        B = int(batch.max()) + 1

        robot_mask = data.node_type == 0
        human_mask = data.node_type == 1

        z_robot = scatter(z[robot_mask], batch[robot_mask], dim=0, reduce="mean", dim_size=B)
        z_human = scatter(z[human_mask], batch[human_mask], dim=0, reduce="mean", dim_size=B)
        return torch.cat([z_robot, z_human], dim=-1)
