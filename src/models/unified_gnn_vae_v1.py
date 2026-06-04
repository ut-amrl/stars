"""
Heterogeneous GNN-VAE pipeline — combined SEAN+SNSB model.

Unified GNN-VAE v1 for SEAN Together + SocialNav-SUB.

This model keeps the v4 SEAN GNN-VAE structure, but collapses all non-robot
human actors into one bystander node type so SEAN and SNSB share the same
heterogeneous message-passing backbone.

Expected input HeteroData contract:
  - data.dataset_id: 0 for SEAN, 1 for SNSB. Batches must be single-source.
  - node types: robot, bystander
  - edge types:
      (robot, to, bystander)
      (bystander, to, robot)
      (bystander, to, bystander)
  - SEAN robot.x: [B, 592], edge_attr: [E, 40, 8], bystander has num_nodes
  - SNSB robot.x: [B, 60], bystander.x: [N, 60], edge_attr: [E, 20, 8]

Forward returns:
  robot_recon, edge_recons, mu_dict, logvar_dict
matching gnn_vae_v4.py.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


SEAN_ID = 0
SNSB_ID = 1
DATASET_ID_TO_KEY = {
    SEAN_ID: "sean",
    SNSB_ID: "snsb",
}

SEAN_TIMESTEPS = 40
SNSB_TIMESTEPS = 20
EDGE_FEATURE_DIM = 8

SEAN_ROBOT_DIM = 592
SNSB_ROBOT_DIM = 60
SNSB_BYSTANDER_DIM = 60
SEAN_BYSTANDER_EMBED_DIM = 8

NODE_TYPES = ["robot", "bystander"]
EDGE_TYPES = [
    ("robot", "to", "bystander"),
    ("bystander", "to", "robot"),
    ("bystander", "to", "bystander"),
]
ET_KEYS = [f"{s}__{d}" for s, _, d in EDGE_TYPES]
ET_KEY2IDX = {k: i for i, k in enumerate(ET_KEYS)}


def _et_key(et):
    return f"{et[0]}__{et[2]}"


def _edge_type_idx(et, device):
    return torch.tensor(ET_KEY2IDX[_et_key(et)], device=device)


def _has_edge_store(data, et):
    return et in getattr(data, "edge_types", []) and hasattr(data[et], "edge_index")


def _dataset_key(data):
    """
    Return 'sean' or 'snsb'. A PyG Batch may collate dataset_id into a tensor;
    this model intentionally rejects mixed-source batches.
    """
    if not hasattr(data, "dataset_id"):
        raise ValueError("UnifiedGNNVAE expects data.dataset_id: 0=SEAN, 1=SNSB.")

    dataset_id = data.dataset_id
    if torch.is_tensor(dataset_id):
        ids = dataset_id.detach().cpu().view(-1).long()
        unique_ids = torch.unique(ids)
        if unique_ids.numel() != 1:
            raise ValueError(
                "UnifiedGNNVAE requires single-source batches; got mixed dataset_id values "
                f"{unique_ids.tolist()}."
            )
        dataset_id = int(unique_ids.item())
    elif isinstance(dataset_id, (list, tuple)):
        ids = [int(x) for x in dataset_id]
        unique_ids = sorted(set(ids))
        if len(unique_ids) != 1:
            raise ValueError(
                "UnifiedGNNVAE requires single-source batches; got mixed dataset_id values "
                f"{unique_ids}."
            )
        dataset_id = unique_ids[0]
    else:
        dataset_id = int(dataset_id)

    if dataset_id not in DATASET_ID_TO_KEY:
        raise ValueError(f"Unknown dataset_id={dataset_id}; expected 0=SEAN or 1=SNSB.")
    return DATASET_ID_TO_KEY[dataset_id]


def _num_nodes(data, node_type):
    if hasattr(data[node_type], "x") and data[node_type].x is not None:
        return int(data[node_type].x.shape[0])
    if hasattr(data[node_type], "num_nodes") and data[node_type].num_nodes is not None:
        return int(data[node_type].num_nodes)
    return 0


class TemporalEdgeEncoder(nn.Module):
    """
    Transformer over temporal edge features.

    Input:  [E, T, 8]
    Output: [E, H]
    """

    def __init__(
        self,
        hidden_dim,
        num_timesteps,
        n_heads=4,
        n_layers=3,
        ff_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 2 * hidden_dim
        self.num_timesteps = num_timesteps
        self.input_proj = nn.Linear(EDGE_FEATURE_DIM, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_timesteps, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, edge_attr):
        if edge_attr.dim() != 3:
            raise ValueError(f"edge_attr must be [E, T, 8], got shape {tuple(edge_attr.shape)}.")
        if edge_attr.shape[1] != self.num_timesteps:
            raise ValueError(
                f"Expected edge_attr T={self.num_timesteps}, got T={edge_attr.shape[1]}."
            )
        if edge_attr.shape[2] != EDGE_FEATURE_DIM:
            raise ValueError(
                f"Expected edge_attr feature dim={EDGE_FEATURE_DIM}, got {edge_attr.shape[2]}."
            )

        x = self.input_proj(edge_attr) + self.pos_embed
        x = self.transformer(x)
        return x.mean(dim=1)


class HeteroMPLayer(nn.Module):
    """
    One HEAT-style hetero message-passing layer with edge updates.

    Per-layer updates (Thompson-style two-block GN):
      e^(l+1) = LayerNorm( e^(l) + msg_mlp([h_src | h_dst | e^(l) | type_emb]) )
      h^(l+1) = LayerNorm( h^(l) + update_mlp([h^(l) | attn_agg({e^(l+1)})]) )

    Returns (h_dict, edge_feat_dict) so layers can be stacked and downstream
    probes can read refined edge embeddings.
    """

    def __init__(self, hidden_dim, n_edge_types=len(EDGE_TYPES)):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_type_embed = nn.Embedding(n_edge_types, hidden_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.update_mlps = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for t in NODE_TYPES
        })
        self.norms = nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in NODE_TYPES})
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_dict, edge_index_dict, edge_feat_dict):
        hidden_dim = next(iter(h_dict.values())).shape[1]
        device = next(iter(h_dict.values())).device

        agg = {t: torch.zeros_like(h_dict[t]) for t in NODE_TYPES}
        msgs_by = {t: [] for t in NODE_TYPES}
        scores_by = {t: [] for t in NODE_TYPES}
        idxs_by = {t: [] for t in NODE_TYPES}
        new_edge_feat_dict = {}

        for et in EDGE_TYPES:
            if et not in edge_index_dict:
                continue
            src_type, _, dst_type = et
            edge_index = edge_index_dict[et]
            edge_feat = edge_feat_dict[et]
            src_idx, dst_idx = edge_index[0], edge_index[1]

            if src_idx.numel() == 0:
                new_edge_feat_dict[et] = edge_feat
                continue

            type_emb = self.edge_type_embed(_edge_type_idx(et, device))
            type_emb = type_emb.unsqueeze(0).expand(src_idx.shape[0], -1)

            inp = torch.cat(
                [
                    h_dict[src_type][src_idx],
                    h_dict[dst_type][dst_idx],
                    edge_feat,
                    type_emb,
                ],
                dim=-1,
            )

            msg = self.msg_mlp(inp)
            new_edge_feat_dict[et] = self.edge_norm(edge_feat + msg)

            msgs_by[dst_type].append(msg)
            scores_by[dst_type].append(self.attn_mlp(inp).squeeze(-1))
            idxs_by[dst_type].append(dst_idx)

        for node_type in NODE_TYPES:
            if not msgs_by[node_type]:
                continue

            all_msgs = torch.cat(msgs_by[node_type], dim=0)
            all_scores = torch.cat(scores_by[node_type], dim=0)
            all_idx = torch.cat(idxs_by[node_type], dim=0)

            max_scores = torch.full(
                (h_dict[node_type].shape[0],),
                -1e9,
                device=device,
                dtype=all_scores.dtype,
            )
            max_scores.scatter_reduce_(0, all_idx, all_scores, reduce="amax", include_self=True)
            exp_scores = torch.exp(all_scores - max_scores[all_idx])
            sum_exp = torch.zeros(
                h_dict[node_type].shape[0],
                device=device,
                dtype=all_scores.dtype,
            )
            sum_exp.scatter_add_(0, all_idx, exp_scores)
            alpha = exp_scores / sum_exp[all_idx].clamp(min=1e-8)

            weighted_msgs = alpha.unsqueeze(1) * all_msgs
            agg[node_type].scatter_add_(
                0,
                all_idx.unsqueeze(1).expand(-1, hidden_dim),
                weighted_msgs,
            )

        out = {}
        for node_type in NODE_TYPES:
            updated = self.update_mlps[node_type](
                torch.cat([h_dict[node_type], agg[node_type]], dim=-1)
            )
            out[node_type] = self.norms[node_type](h_dict[node_type] + updated)
        return out, new_edge_feat_dict


class UnifiedGNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        latent_dim,
        num_mp_layers,
        dropout=0.1,
        sean_robot_dim=SEAN_ROBOT_DIM,
        sean_bystander_embed_dim=SEAN_BYSTANDER_EMBED_DIM,
        snsb_timesteps=SNSB_TIMESTEPS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sean_bystander_embed_dim = sean_bystander_embed_dim

        # SEAN robot: 592D → hidden
        self.sean_robot_stem = nn.Sequential(
            nn.Linear(sean_robot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # SEAN/SNSB bystanders + SNSB robot: 8D type embed + 8D mean edge_attr → hidden
        self.sean_bystander_stem = nn.Sequential(
            nn.Linear(sean_bystander_embed_dim + EDGE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.snsb_robot_stem = nn.Sequential(
            nn.Linear(sean_bystander_embed_dim + EDGE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.snsb_bystander_stem = nn.Sequential(
            nn.Linear(sean_bystander_embed_dim + EDGE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable type tokens (8D, same dim as SEAN bystander embed)
        self.sean_bystander_embed = nn.Parameter(torch.zeros(sean_bystander_embed_dim))
        self.snsb_robot_embed = nn.Parameter(torch.zeros(sean_bystander_embed_dim))
        self.snsb_bystander_embed = nn.Parameter(torch.zeros(sean_bystander_embed_dim))

        self.edge_encoders = nn.ModuleDict({
            "sean": TemporalEdgeEncoder(hidden_dim, SEAN_TIMESTEPS, dropout=dropout),
            "snsb": TemporalEdgeEncoder(hidden_dim, snsb_timesteps, dropout=dropout),
        })
        self.edge_type_embed = nn.Embedding(len(EDGE_TYPES), hidden_dim)

        self.mp_layers = nn.ModuleList([
            HeteroMPLayer(hidden_dim) for _ in range(num_mp_layers)
        ])
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def _encode_sean_bystanders(self, data, device):
        n_bystanders = _num_nodes(data, "bystander")
        if n_bystanders == 0:
            return torch.zeros(0, self.hidden_dim, device=device)

        pos_per_bystander = torch.zeros(n_bystanders, EDGE_FEATURE_DIM, device=device)
        counts = torch.zeros(n_bystanders, 1, device=device)

        et_r2b = ("robot", "to", "bystander")
        if _has_edge_store(data, et_r2b) and hasattr(data[et_r2b], "edge_attr"):
            edge_attr = data[et_r2b].edge_attr
            if edge_attr.numel() > 0:
                pos_feat = edge_attr.mean(dim=1)
                dst_idx = data[et_r2b].edge_index[1]
                pos_per_bystander.scatter_add_(
                    0,
                    dst_idx.unsqueeze(1).expand(-1, EDGE_FEATURE_DIM),
                    pos_feat,
                )
                counts.scatter_add_(
                    0,
                    dst_idx.unsqueeze(1),
                    torch.ones(len(dst_idx), 1, device=device),
                )
                pos_per_bystander = pos_per_bystander / counts.clamp(min=1)

        bystander_embed = self.sean_bystander_embed.unsqueeze(0).expand(n_bystanders, -1)
        bystander_input = torch.cat([bystander_embed, pos_per_bystander], dim=-1)
        return self.sean_bystander_stem(bystander_input)

    def _mean_edge_attr_for_nodes(self, data, specs, num_nodes, device):
        pos = torch.zeros(num_nodes, EDGE_FEATURE_DIM, device=device)
        counts = torch.zeros(num_nodes, 1, device=device)

        for et, node_axis in specs:
            if not _has_edge_store(data, et) or not hasattr(data[et], "edge_attr"):
                continue
            edge_attr = data[et].edge_attr
            if edge_attr.numel() == 0:
                continue
            node_idx = data[et].edge_index[node_axis]
            feat = edge_attr.mean(dim=1)
            pos.scatter_add_(
                0,
                node_idx.unsqueeze(1).expand(-1, EDGE_FEATURE_DIM),
                feat,
            )
            counts.scatter_add_(
                0,
                node_idx.unsqueeze(1),
                torch.ones(len(node_idx), 1, device=device),
            )

        return pos / counts.clamp(min=1)

    def _encode_snsb_type_nodes_from_edges(self, data, device, n_robots, n_bystanders):
        robot_pos = self._mean_edge_attr_for_nodes(
            data,
            [
                (("robot", "to", "bystander"), 0),
                (("bystander", "to", "robot"), 1),
            ],
            n_robots,
            device,
        )
        robot_embed = self.snsb_robot_embed.unsqueeze(0).expand(n_robots, -1)
        h_robot = self.snsb_robot_stem(torch.cat([robot_embed, robot_pos], dim=-1))

        if n_bystanders == 0:
            h_bystander = torch.zeros(0, self.hidden_dim, device=device)
        else:
            bystander_pos = self._mean_edge_attr_for_nodes(
                data,
                [
                    (("robot", "to", "bystander"), 1),
                    (("bystander", "to", "robot"), 0),
                ],
                n_bystanders,
                device,
            )
            bystander_embed = self.snsb_bystander_embed.unsqueeze(0).expand(n_bystanders, -1)
            h_bystander = self.snsb_bystander_stem(
                torch.cat([bystander_embed, bystander_pos], dim=-1)
            )

        return {"robot": h_robot, "bystander": h_bystander}

    def _encode_nodes(self, data, dataset_key):
        if dataset_key == "sean":
            device = data["robot"].x.device
            h = {"robot": self.sean_robot_stem(data["robot"].x)}
            h["bystander"] = self._encode_sean_bystanders(data, device)
            return h

        # SNSB: edge-based init for both robot and bystander (same pattern as SEAN bystanders)
        device = self.snsb_robot_embed.device
        n_robots = _num_nodes(data, "robot")
        n_bystanders = _num_nodes(data, "bystander")
        return self._encode_snsb_type_nodes_from_edges(data, device, n_robots, n_bystanders)

    def _encode_edges(self, data, dataset_key):
        if dataset_key == "sean":
            device = data["robot"].x.device
        else:
            device = self.snsb_robot_embed.device
        edge_index_dict = {}
        edge_feat_dict = {}

        for et in EDGE_TYPES:
            if not _has_edge_store(data, et):
                continue
            if not hasattr(data[et], "edge_attr") or data[et].edge_attr is None:
                raise ValueError(f"Edge type {et} is missing edge_attr.")

            edge_index = data[et].edge_index
            edge_attr = data[et].edge_attr
            if edge_index.shape[1] == 0:
                continue

            type_emb = self.edge_type_embed(_edge_type_idx(et, device))
            edge_index_dict[et] = edge_index
            edge_feat_dict[et] = self.edge_encoders[dataset_key](edge_attr) + type_emb

        return edge_index_dict, edge_feat_dict

    def forward(self, data, return_edges=False):
        dataset_key = _dataset_key(data)
        h = self._encode_nodes(data, dataset_key)
        edge_index_dict, edge_feat_dict = self._encode_edges(data, dataset_key)

        for mp_layer in self.mp_layers:
            h, edge_feat_dict = mp_layer(h, edge_index_dict, edge_feat_dict)

        mu_dict = {}
        logvar_dict = {}
        device = next(v.device for v in h.values() if v.numel() > 0)
        for node_type in NODE_TYPES:
            if h[node_type].shape[0] > 0:
                mu_dict[node_type] = self.fc_mu(h[node_type])
                logvar_dict[node_type] = self.fc_logvar(h[node_type])
            else:
                mu_dict[node_type] = torch.zeros(0, self.latent_dim, device=device)
                logvar_dict[node_type] = torch.zeros(0, self.latent_dim, device=device)

        if return_edges:
            return mu_dict, logvar_dict, edge_feat_dict
        return mu_dict, logvar_dict


class DatasetEdgeDecoder(nn.Module):
    """
    Dataset-specific temporal edge decoder.

    Input:  [h_src | h_dst | edge_type_emb] = [E, 3H]
    Output: [E, T, 8]
    """

    def __init__(self, hidden_dim, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_timesteps * EDGE_FEATURE_DIM),
        )

    def forward(self, h_src, h_dst, type_emb):
        type_emb = type_emb.unsqueeze(0).expand(h_src.shape[0], -1)
        inp = torch.cat([h_src, h_dst, type_emb], dim=-1)
        return self.mlp(inp).view(-1, self.num_timesteps, EDGE_FEATURE_DIM)


class UnifiedGNNDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        sean_robot_dim=SEAN_ROBOT_DIM,
        snsb_timesteps=SNSB_TIMESTEPS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pre_decode = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        # SEAN robot has 592D reconstruction target; SNSB robot has none (edge-only signal)
        self.sean_robot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sean_robot_dim),
        )
        self.edge_decoders = nn.ModuleDict({
            "sean": DatasetEdgeDecoder(hidden_dim, SEAN_TIMESTEPS),
            "snsb": DatasetEdgeDecoder(hidden_dim, snsb_timesteps),
        })
        self.edge_type_embed = nn.Embedding(len(EDGE_TYPES), hidden_dim)

    def forward(self, z_dict, data):
        dataset_key = _dataset_key(data)
        device = next(v.device for v in z_dict.values() if v.shape[0] > 0)

        h = {}
        for node_type in NODE_TYPES:
            if z_dict[node_type].shape[0] > 0:
                h[node_type] = self.pre_decode(z_dict[node_type])
            else:
                h[node_type] = torch.zeros(0, self.hidden_dim, device=device)

        robot_recon = self.sean_robot_head(h["robot"]) if dataset_key == "sean" else None

        edge_recons = {}
        for et in EDGE_TYPES:
            if not _has_edge_store(data, et):
                continue
            edge_index = data[et].edge_index
            if edge_index.shape[1] == 0:
                continue
            src_type, _, dst_type = et
            if h[src_type].shape[0] == 0 or h[dst_type].shape[0] == 0:
                continue

            type_emb = self.edge_type_embed(_edge_type_idx(et, device))
            edge_recons[et] = self.edge_decoders[dataset_key](
                h[src_type][edge_index[0]],
                h[dst_type][edge_index[1]],
                type_emb,
            )

        return robot_recon, edge_recons


class UnifiedGNNVAE(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=64,
        num_mp_layers=2,
        dropout=0.1,
        sean_robot_dim=SEAN_ROBOT_DIM,
        sean_bystander_embed_dim=SEAN_BYSTANDER_EMBED_DIM,
        snsb_timesteps=SNSB_TIMESTEPS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = UnifiedGNNEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_mp_layers=num_mp_layers,
            dropout=dropout,
            sean_robot_dim=sean_robot_dim,
            sean_bystander_embed_dim=sean_bystander_embed_dim,
            snsb_timesteps=snsb_timesteps,
        )
        self.decoder = UnifiedGNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            sean_robot_dim=sean_robot_dim,
            snsb_timesteps=snsb_timesteps,
        )

    def reparameterize(self, mu_dict, logvar_dict):
        z_dict = {}
        for node_type in NODE_TYPES:
            if self.training:
                z_dict[node_type] = (
                    mu_dict[node_type]
                    + torch.randn_like(mu_dict[node_type]) * torch.exp(0.5 * logvar_dict[node_type])
                )
            else:
                z_dict[node_type] = mu_dict[node_type]
        return z_dict

    def forward(self, data):
        mu_dict, logvar_dict = self.encoder(data)
        z_dict = self.reparameterize(mu_dict, logvar_dict)
        robot_recon, edge_recons = self.decoder(z_dict, data)
        return robot_recon, edge_recons, mu_dict, logvar_dict

    def encode_nodes(self, data):
        """Return per-node posterior means for downstream node-level probes."""
        mu_dict, _ = self.encoder(data)
        return mu_dict

    def encode_nodes_and_edges(self, data):
        """
        Return (mu_dict, edge_feat_dict) for downstream edge-level probes
        edge_feat_dict holds the final post-MP edge embeddings.
        """
        mu_dict, _, edge_feat_dict = self.encoder(data, return_edges=True)
        return mu_dict, edge_feat_dict

    def encode_graph(self, data):
        """
        Return [z_robot | mean(z_bystander)] using posterior means.

        Shape: [B, 2 * latent_dim]. This is the intended graph-level probe
        representation for the unified two-node-type taxonomy.
        """
        mu_dict, _ = self.encoder(data)
        return self.pool_robot_bystander(mu_dict, data)

    def encode(self, data):
        """Alias for graph-level encoding, consistent with v4's downstream use."""
        return self.encode_graph(data)

    def pool_robot_bystander(self, z_dict, data):
        device = next(v.device for v in z_dict.values() if v.shape[0] > 0)
        if hasattr(data["robot"], "batch") and data["robot"].batch is not None:
            robot_batch = data["robot"].batch
            batch_size = int(robot_batch.max()) + 1 if robot_batch.numel() > 0 else 0
        else:
            robot_batch = torch.zeros(z_dict["robot"].shape[0], dtype=torch.long, device=device)
            batch_size = z_dict["robot"].shape[0]

        if z_dict["robot"].shape[0] == batch_size:
            z_robot = z_dict["robot"]
        else:
            z_robot = scatter(
                z_dict["robot"],
                robot_batch,
                dim=0,
                reduce="mean",
                dim_size=batch_size,
            )

        if z_dict["bystander"].shape[0] > 0:
            if hasattr(data["bystander"], "batch") and data["bystander"].batch is not None:
                bystander_batch = data["bystander"].batch
            else:
                bystander_batch = torch.zeros(
                    z_dict["bystander"].shape[0],
                    dtype=torch.long,
                    device=device,
                )
            z_bystander = scatter(
                z_dict["bystander"],
                bystander_batch,
                dim=0,
                reduce="mean",
                dim_size=batch_size,
            )
        else:
            z_bystander = torch.zeros(batch_size, self.latent_dim, device=device)

        return torch.cat([z_robot, z_bystander], dim=-1)
