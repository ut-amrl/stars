"""
Heterogeneous GNN-VAE pipeline — SEAN-only v4 model.

GNN-VAE v4: Follower treated like bystander - no GT labels as node features.

Changes from v3:
  - follower_dim=3 removed; follower_embed = nn.Parameter(zeros(follower_embed_dim=8))
  - follower_enc takes 8D learnable embed (same as bystander, but no positional features
    since there is only one follower and no ambiguity)
  - HeteroDecoderV4: no follower_head, no follower reconstruction
  - forward() returns (robot_recon, edge_recons, mu_dict, logvar_dict) - 4 values
  - Fully self-supervised Stage 1: no GT labels anywhere in encoder input

Architecture: HEAT-style MP, shared edge encoder/decoder, pool_attn, bystander — same as unified_gnn_vae_v1.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

N_TIMESTEPS      = 40
EDGE_FEATURE_DIM = 8
TEMPORAL_FLAT    = N_TIMESTEPS * EDGE_FEATURE_DIM   # 320

NODE_TYPES = ['robot', 'follower', 'bystander']
EDGE_TYPES = [
    ('robot',     'to', 'follower'),
    ('follower',  'to', 'robot'),
    ('robot',     'to', 'bystander'),
    ('bystander', 'to', 'robot'),
    ('follower',  'to', 'bystander'),
    ('bystander', 'to', 'follower'),
    ('bystander', 'to', 'bystander'),
]
ET_KEYS    = [f'{s}__{d}' for s, _, d in EDGE_TYPES]
ET_KEY2IDX = {k: i for i, k in enumerate(ET_KEYS)}


def _et_key(et):
    return f'{et[0]}__{et[2]}'


# ---------------------------------------------------------------------------
# Edge encoder / decoder and MP layer
# ---------------------------------------------------------------------------

class SharedEdgeEncoder(nn.Module):
    """
    Small Transformer over 40 timesteps -> [H], shared across all 7 edge types.
    Full receptive field - every timestep attends to every other.
    Edge type injected as learned embedding added to encoded output.
    Input: [E, 40, 8]  Output: [E, H]
    """
    def __init__(self, hidden_dim, n_edge_types=7, n_heads=2, n_layers=2, ff_dim=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(EDGE_FEATURE_DIM, hidden_dim)
        self.pos_embed  = nn.Parameter(torch.randn(1, N_TIMESTEPS, hidden_dim) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim,
                batch_first=True, dropout=dropout
            ),
            num_layers=n_layers
        )
        self.type_embed = nn.Embedding(n_edge_types, hidden_dim)

    def forward(self, edge_attr, et_key):
        x = self.input_proj(edge_attr) + self.pos_embed   # [E, 40, H]
        x = self.transformer(x)                            # [E, 40, H]
        x = x.mean(dim=1)                                  # [E, H]
        x = x + self.type_embed(
            torch.tensor(ET_KEY2IDX[et_key], device=edge_attr.device)
        )
        return x


class HeteroMPLayer(nn.Module):
    """
    One round of hetero message passing with shared message MLP + edge type embedding.
    HEAT-style attention: score per edge from [h_src | h_dst | e | type_emb].
    Scatter weighted messages to dst nodes, then update per node type with residual + LayerNorm.
    """
    def __init__(self, hidden_dim, n_edge_types=7):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_type_embed = nn.Embedding(n_edge_types, hidden_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        self.update_mlps = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim)
            ) for t in NODE_TYPES
        })
        self.norms = nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in NODE_TYPES})

    def forward(self, h_dict, edge_index_dict, edge_feat_dict):
        H      = next(iter(h_dict.values())).shape[1]
        device = next(iter(h_dict.values())).device

        agg     = {t: torch.zeros_like(h_dict[t]) for t in NODE_TYPES}
        attn_ws = {t: [] for t in NODE_TYPES}
        msgs_by = {t: [] for t in NODE_TYPES}
        idxs_by = {t: [] for t in NODE_TYPES}

        for et in EDGE_TYPES:
            if et not in edge_index_dict:
                continue
            s, _, d = et
            ei      = edge_index_dict[et]
            ef      = edge_feat_dict[et]
            src_idx, dst_idx = ei[0], ei[1]

            type_emb = self.edge_type_embed(
                torch.tensor(ET_KEY2IDX[_et_key(et)], device=device)
            ).unsqueeze(0).expand(len(src_idx), -1)

            inp = torch.cat([h_dict[s][src_idx], h_dict[d][dst_idx], ef, type_emb], dim=-1)

            msgs   = self.msg_mlp(inp)
            scores = self.attn_mlp(inp).squeeze(-1)

            msgs_by[d].append(msgs)
            attn_ws[d].append(scores)
            idxs_by[d].append(dst_idx)

        for t in NODE_TYPES:
            if not msgs_by[t]:
                continue
            all_msgs   = torch.cat(msgs_by[t], dim=0)
            all_scores = torch.cat(attn_ws[t], dim=0)
            all_idx    = torch.cat(idxs_by[t], dim=0)

            max_scores = torch.zeros(h_dict[t].shape[0], device=device).fill_(-1e9)
            max_scores.scatter_reduce_(0, all_idx, all_scores, reduce='amax', include_self=True)
            exp_scores = torch.exp(all_scores - max_scores[all_idx])
            sum_exp    = torch.zeros(h_dict[t].shape[0], device=device)
            sum_exp.scatter_add_(0, all_idx, exp_scores)
            alpha      = exp_scores / sum_exp[all_idx].clamp(min=1e-8)

            weighted_msgs = alpha.unsqueeze(1) * all_msgs
            agg[t].scatter_add_(0, all_idx.unsqueeze(1).expand(-1, H), weighted_msgs)

        out = {}
        for t in NODE_TYPES:
            updated = self.update_mlps[t](torch.cat([h_dict[t], agg[t]], dim=-1))
            out[t]  = self.norms[t](h_dict[t] + updated)
        return out


# ---------------------------------------------------------------------------
# Encoder - per-node z's; follower uses learnable embed (no GT input)
# ---------------------------------------------------------------------------

class GNNEncoderV4(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_mp_layers,
                 robot_dim=592, follower_embed_dim=8, bystander_embed_dim=8,
                 edge_transformer_layers=2, edge_transformer_heads=2,
                 dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Node encoders
        self.robot_enc = nn.Sequential(
            nn.Linear(robot_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Follower: shared learnable embed only - no GT features, no positional init
        self.follower_embed = nn.Parameter(torch.zeros(follower_embed_dim))
        self.follower_enc   = nn.Sequential(
            nn.Linear(follower_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Bystander: learnable embed + mean robot->bystander edge attr as positional init
        self.bystander_embed = nn.Parameter(torch.zeros(bystander_embed_dim))
        self.bystander_enc   = nn.Sequential(
            nn.Linear(bystander_embed_dim + EDGE_FEATURE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_enc  = SharedEdgeEncoder(
            hidden_dim,
            n_heads=edge_transformer_heads,
            n_layers=edge_transformer_layers,
            dropout=dropout,
        )
        self.mp_layers = nn.ModuleList([HeteroMPLayer(hidden_dim) for _ in range(num_mp_layers)])

        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        device = data['robot'].x.device

        # --- Encode nodes ---
        h = {}
        h['robot'] = self.robot_enc(data['robot'].x)

        # Follower: expand shared embed to (n_follower, embed_dim)
        n_f = int(data['follower'].num_nodes)
        h['follower'] = self.follower_enc(
            self.follower_embed.unsqueeze(0).expand(n_f, -1)
        )

        # Bystander: embed + positional features from robot->bystander edges
        n_b = int(data['bystander'].num_nodes) if hasattr(data['bystander'], 'num_nodes') else 0
        if n_b > 0:
            et_r2b = ('robot', 'to', 'bystander')
            if hasattr(data[et_r2b], 'edge_attr') and hasattr(data[et_r2b], 'edge_index'):
                ea       = data[et_r2b].edge_attr
                pos_feat = ea.mean(dim=1)
                dst_idx  = data[et_r2b].edge_index[1]
                pos_per_b = torch.zeros(n_b, EDGE_FEATURE_DIM, device=device)
                cnt       = torch.zeros(n_b, 1, device=device)
                pos_per_b.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, EDGE_FEATURE_DIM), pos_feat)
                cnt.scatter_add_(0, dst_idx.unsqueeze(1), torch.ones(len(dst_idx), 1, device=device))
                pos_per_b = pos_per_b / cnt.clamp(min=1)
            else:
                pos_per_b = torch.zeros(n_b, EDGE_FEATURE_DIM, device=device)
            bystander_input = torch.cat([
                self.bystander_embed.unsqueeze(0).expand(n_b, -1),
                pos_per_b
            ], dim=-1)
            h['bystander'] = self.bystander_enc(bystander_input)
        else:
            h['bystander'] = torch.zeros(0, self.hidden_dim, device=device)

        # --- Encode edges ---
        edge_index_dict = {}
        edge_feat_dict  = {}
        for et in EDGE_TYPES:
            if hasattr(data[et], 'edge_index'):
                edge_index_dict[et] = data[et].edge_index
                edge_feat_dict[et]  = self.edge_enc(data[et].edge_attr, _et_key(et))

        # --- Message passing ---
        for mp in self.mp_layers:
            h = mp(h, edge_index_dict, edge_feat_dict)

        # --- Per-node mu/logvar ---
        mu_dict     = {}
        logvar_dict = {}
        for t in NODE_TYPES:
            if h[t].shape[0] > 0:
                mu_dict[t]     = self.fc_mu(h[t])
                logvar_dict[t] = self.fc_logvar(h[t])
            else:
                mu_dict[t]     = torch.zeros(0, self.latent_dim, device=device)
                logvar_dict[t] = torch.zeros(0, self.latent_dim, device=device)

        return mu_dict, logvar_dict


# ---------------------------------------------------------------------------
# Decoder - no follower reconstruction
# ---------------------------------------------------------------------------

class SharedEdgeDecoder(nn.Module):
    """
    Shared edge decoder + type embeddings.
    Input: [h_src | h_dst | type_emb] = [3H] -> [320] -> [40, 8]
    """
    def __init__(self, hidden_dim, n_edge_types=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, TEMPORAL_FLAT)
        )
        self.type_embed = nn.Embedding(n_edge_types, hidden_dim)

    def forward(self, h_src, h_dst, et_key):
        type_emb = self.type_embed(
            torch.tensor(ET_KEY2IDX[et_key], device=h_src.device)
        ).unsqueeze(0).expand(h_src.shape[0], -1)
        inp = torch.cat([h_src, h_dst, type_emb], dim=-1)
        return self.mlp(inp).view(-1, N_TIMESTEPS, EDGE_FEATURE_DIM)


class HeteroDecoderV4(nn.Module):
    """
    Decode per-node z_dict -> robot node features + all edge features.
    No follower reconstruction (no GT target for follower in v4).
    """
    def __init__(self, latent_dim, hidden_dim, robot_dim=592):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pre_decode = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.robot_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, robot_dim)
        )
        self.edge_dec   = SharedEdgeDecoder(hidden_dim)

    def forward(self, z_dict, data):
        device = next(v.device for v in z_dict.values() if v.shape[0] > 0)

        h = {}
        for t in NODE_TYPES:
            if z_dict[t].shape[0] > 0:
                h[t] = self.pre_decode(z_dict[t])
            else:
                h[t] = torch.zeros(0, self.hidden_dim, device=device)

        robot_recon = self.robot_head(h['robot'])

        edge_recons = {}
        for et in EDGE_TYPES:
            if hasattr(data[et], 'edge_index') and h[et[0]].shape[0] > 0 and h[et[2]].shape[0] > 0:
                ei = data[et].edge_index
                edge_recons[et] = self.edge_dec(
                    h[et[0]][ei[0]], h[et[2]][ei[1]], _et_key(et)
                )

        return robot_recon, edge_recons


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class GNNVAEv4(nn.Module):
    def __init__(self,
                 robot_dim=592,
                 follower_embed_dim=8,
                 bystander_embed_dim=8,
                 hidden_dim=128,
                 latent_dim=64,
                 num_mp_layers=2,
                 edge_transformer_layers=2,
                 edge_transformer_heads=2,
                 dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = GNNEncoderV4(hidden_dim, latent_dim, num_mp_layers,
                                        robot_dim, follower_embed_dim, bystander_embed_dim,
                                        edge_transformer_layers=edge_transformer_layers,
                                        edge_transformer_heads=edge_transformer_heads,
                                        dropout=dropout)
        self.decoder    = HeteroDecoderV4(latent_dim, hidden_dim, robot_dim)

        # 3-head attention pool: per-node z's -> z_graph [B, 3 * latent_dim] (used via pool_z)
        self.pool_attn  = nn.Linear(latent_dim, 3)

        # MLP pool: type-sep 192D concat -> 192D (trained via aux task with --incl_mlp_pool)
        self.pool_mlp   = nn.Sequential(
            nn.Linear(3 * latent_dim, 3 * latent_dim), nn.ReLU(),
            nn.Linear(3 * latent_dim, 3 * latent_dim)
        )

    def pool_z(self, z_dict, data):
        """
        3-head attention pool over ALL per-node z's -> [B, 3 * latent_dim].
        pool_attn: Linear(latent_dim, 3) - 3 independent attention heads.
        Each head produces a 64D weighted sum over all nodes; concat -> 192D.
        """
        device = next(v.device for v in z_dict.values() if v.shape[0] > 0)
        B = int(data['robot'].batch.max()) + 1

        all_z = torch.cat(
            [z_dict[t] for t in NODE_TYPES if z_dict[t].shape[0] > 0], dim=0
        )
        all_batch = torch.cat([
            data[t].batch for t in NODE_TYPES
            if z_dict[t].shape[0] > 0 and hasattr(data[t], 'batch')
        ], dim=0)

        scores = self.pool_attn(all_z)  # [N_total, 3]

        heads = []
        for k in range(3):
            s_k   = scores[:, k]
            max_s = scatter(s_k, all_batch, dim=0, reduce='max', dim_size=B)
            exp_s = torch.exp(s_k - max_s[all_batch])
            sum_s = scatter(exp_s, all_batch, dim=0, reduce='sum', dim_size=B)
            alpha = exp_s / sum_s[all_batch].clamp(min=1e-8)
            heads.append(
                scatter(alpha.unsqueeze(1) * all_z, all_batch, dim=0,
                        reduce='sum', dim_size=B)
            )

        return torch.cat(heads, dim=-1)  # [B, 3*latent_dim]

    def pool_z_mlp(self, mu_dict, data):
        """
        MLP pool: type-sep 192D concat [z_robot | z_follower | mean(z_bystanders)] -> pool_mlp -> 192D.
        Input is identical to the default probe embedding; pool_mlp learns a nonlinear compression.
        Trained via aux task (--incl_mlp_pool); frozen at probe time (--mlp_pool).
        """
        from torch_geometric.utils import scatter
        device = next(v.device for v in mu_dict.values() if v.shape[0] > 0)
        B = int(data['robot'].batch.max()) + 1
        z_robot    = mu_dict['robot']
        z_follower = mu_dict['follower']
        if mu_dict['bystander'].shape[0] > 0:
            z_b_mean = scatter(mu_dict['bystander'], data['bystander'].batch,
                               dim=0, reduce='mean', dim_size=B)
        else:
            z_b_mean = torch.zeros(B, self.latent_dim, device=device)
        z_concat = torch.cat([z_robot, z_follower, z_b_mean], dim=-1)  # [B, 3*latent_dim]
        return self.pool_mlp(z_concat)                                  # [B, 3*latent_dim]

    def reparameterize(self, mu_dict, logvar_dict):
        z_dict = {}
        for t in NODE_TYPES:
            if self.training:
                z_dict[t] = mu_dict[t] + torch.randn_like(mu_dict[t]) * torch.exp(0.5 * logvar_dict[t])
            else:
                z_dict[t] = mu_dict[t]
        return z_dict

    def forward(self, data):
        mu_dict, logvar_dict         = self.encoder(data)
        z_dict                       = self.reparameterize(mu_dict, logvar_dict)
        robot_recon, edge_recons     = self.decoder(z_dict, data)
        return robot_recon, edge_recons, mu_dict, logvar_dict

    def encode(self, data):
        """Return pooled z_graph for downstream use."""
        mu_dict, _ = self.encoder(data)
        return self.pool_z(mu_dict, data)
