"""
Social-STGCNN backbone with classification head for SEAN/SNSB supervised baseline.

ST-GCN layers from Social-STGCNN (Mohamed et al., 2020) with the trajectory
prediction head replaced by a classification MLP.

ConvTemporalGraphical and STGCNBlock are faithful copies of the original.
Key invariant: adjacency A must have shape (T, N, N) matching the time dim of x.

Input:  v (1, 8, T, N)  — node features aggregated from incoming edge attrs
        A (T, N, N)      — static adjacency expanded across T timesteps
SEAN:   logits (3,)      — competence / surprise / intention (mean-pool over time+nodes)
SNSB:   logits (N_b, 5) — per-bystander action class     (mean-pool over time)
"""

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    """Exact copy from Social-STGCNN (Mohamed et al. 2020)."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        # A: (T, N, N)  x: (batch, C, T, N)
        assert A.size(0) == self.kernel_size, \
            f"Expected A.size(0)=={self.kernel_size}, got {A.size(0)}"
        x = self.conv(x)
        x = torch.einsum("nctv,tvw->nctw", (x, A))
        return x.contiguous(), A


class STGCNBlock(nn.Module):
    """Exact copy of st_gcn from Social-STGCNN (use_mdn=False path)."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0.0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels,
                      (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.prelu(self.tcn(x) + res)
        return x, A


class SocialSTGCNNClassifier(nn.Module):
    """
    ST-GCN backbone (faithful to Social-STGCNN) + classification heads.

    seq_len must match the T dimension of the input data:
      SEAN: seq_len=40, SNSB: seq_len=20
    """

    def __init__(self, n_stgcnn=2, input_feat=8, hidden_feat=64,
                 seq_len=40, kernel_size=3, dropout=0.1,
                 sean_classes=3, snsb_classes=5):
        super().__init__()
        self.seq_len = seq_len

        self.st_gcns = nn.ModuleList()
        # kernel_size[1] = seq_len so A shape (seq_len, N, N) satisfies the assertion
        self.st_gcns.append(
            STGCNBlock(input_feat, hidden_feat, (kernel_size, seq_len), dropout=dropout))
        for _ in range(1, n_stgcnn):
            self.st_gcns.append(
                STGCNBlock(hidden_feat, hidden_feat, (kernel_size, seq_len), dropout=dropout))

        self.sean_head = nn.Linear(hidden_feat, sean_classes)
        self.snsb_head = nn.Linear(hidden_feat, snsb_classes)

    def _encode(self, v, A):
        """v: (1, C, T, N), A: (T, N, N) → (1, hidden, T, N)."""
        for layer in self.st_gcns:
            v, A = layer(v, A)
        return v

    def forward_sean(self, v, A):
        """→ logits (3,)"""
        h = self._encode(v, A)           # (1, hidden, T, N)
        h = h.mean(dim=(2, 3))           # (1, hidden) — pool time + nodes
        return self.sean_head(h).squeeze(0)   # (3,)

    def forward_snsb(self, v, A, n_bystanders):
        """→ logits (n_bystanders, 5)"""
        h = self._encode(v, A)           # (1, hidden, T, N)
        h = h.mean(dim=2).squeeze(0).T  # (N, hidden) — pool time
        bystander_h = h[1:1 + n_bystanders]  # robot is node 0
        return self.snsb_head(bystander_h)    # (n_bystanders, 5)


def build_stgcnn_input(data, device):
    """
    Convert unified HeteroData to Social-STGCNN inputs.

    Returns:
        v : (1, 8, T, N) — per-node features from mean of incoming edge attrs
        A : (T, N, N)    — static fully-connected normalized adjacency expanded over T
        T : int
        N : int
        n_bystanders : int
    """
    from dataloaders.unified_gnn_vae_datasets_v1 import UNIFIED_EDGE_TYPES

    n_robot = data["robot"].x.shape[0]       # 1
    n_bystander = int(data["bystander"].num_nodes)
    N = n_robot + n_bystander

    # Infer T from any non-empty edge
    T = None
    for et in UNIFIED_EDGE_TYPES:
        if hasattr(data[et], "edge_attr") and data[et].edge_attr.numel() > 0:
            T = int(data[et].edge_attr.shape[1])
            break
    if T is None:
        raise ValueError("Cannot infer T: all edge_attr tensors are empty.")

    # Aggregate incoming edge features per node → (N, T, 8)
    node_feats = torch.zeros(N, T, 8, device=device)
    node_counts = torch.zeros(N, device=device)

    edge_offset_map = {
        ("robot", "to", "bystander"): (0,        n_robot),
        ("bystander", "to", "robot"): (n_robot,  0),
        ("bystander", "to", "bystander"): (n_robot, n_robot),
    }

    for et, (_, dst_off) in edge_offset_map.items():
        if not (hasattr(data[et], "edge_index") and hasattr(data[et], "edge_attr")):
            continue
        ea = data[et].edge_attr
        if ea.numel() == 0:
            continue
        ei = data[et].edge_index.to(device)   # (2, E)
        ea = ea.float().to(device)             # (E, T, 8)
        dst_global = ei[1] + dst_off           # global node indices (E,)
        node_feats.index_add_(0, dst_global, ea)
        node_counts.index_add_(0, dst_global, torch.ones(ei.shape[1], device=device))

    mask = node_counts > 0
    node_feats[mask] /= node_counts[mask].view(-1, 1, 1)

    # (N, T, 8) → (1, 8, T, N)
    v = node_feats.permute(2, 1, 0).unsqueeze(0)  # (1, 8, T, N)

    # Fully-connected normalized adjacency, expanded to (T, N, N)
    A_static = torch.ones(N, N, device=device)
    A_static.fill_diagonal_(0.0)
    row_sum = A_static.sum(dim=1, keepdim=True).clamp(min=1.0)
    A_static = A_static / row_sum               # (N, N)
    A = A_static.unsqueeze(0).expand(T, N, N)   # (T, N, N) — no copy, just view

    return v, A, T, N, n_bystander
