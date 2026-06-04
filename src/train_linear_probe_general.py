"""
Unified Linear Probe General Script

This script unifies the functionality of four separate Stage-2 linear probe evaluation scripts:
1. train_unified_linear_probe_v1.py (SNSB Actions on HGNNCombinedVAE)
2. train_unified_linear_probe_SEAN_v1.py (SEAN Emotions on HGNNCombinedVAE)
3. train_snsb_linear_probe_v3.py (SNSB Actions on UnifiedGNNVAE)
4. train_swm_forecasting_linear_probe_v1.py (Forecasting on HGNNCombinedAE)

It also incorporates the multi-VAE batch evaluation capability from `eval_label_efficiency.py`.

Supported Tasks:
- `snsb_actions`: 5-class action classification (avoid, follow, overtake, yield, not_consider).
- `sean_emotions`: 3-way multi-label binary classification (competence, surprise, intention).
- `forecasting`: 2D continuous regression predicting future pedestrian position (FDE, ADE).

Supported Architectures:
- `UnifiedGNNVAE`: The heterogeneous message-passing backbone used in v3+.
- `HGNNCombinedVAE`: The homogeneous graph VAE used in early models.
- `HGNNCombinedAE`: The homogeneous graph AE used for forecasting.

Execution Modes:
1. Single Checkpoint (Multi-Run Sweep): Set --checkpoint. You can provide multiple --fractions
   and --num_runs to run a label-efficiency sweep with multiple random probe initializations.
   This runs a fully featured training loop (stratified sampling, adaptive reweighting, plots).
2. Multi-VAE Batch: Set --vae_dirs to evaluate multiple independently-trained VAE checkpoints
   in a single invocation. This automatically uses a lightweight, stripped-down training loop
   to quickly sweep over (VAEs x fractions x seeds x tasks), appending to a single raw CSV.

To add new upstream models or downstream tasks, search for TODO in this script
"""

import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

try:
    from torch_geometric.utils import scatter
except ImportError:
    def scatter(src, index, dim=0, reduce="sum", dim_size=None):
        if dim != 0: raise ValueError("Fallback scatter only supports dim=0.")
        if dim_size is None: dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out = src.new_zeros((dim_size,) + src.shape[1:])
        count = src.new_zeros((dim_size,) + (1,) * (src.dim() - 1))
        out.index_add_(0, index, src)
        count.index_add_(0, index, torch.ones_like(src[:, :1]))
        if reduce == "mean": return out / count.clamp(min=1)
        return out

# Append src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

try:
    import wandb
except ImportError:
    wandb = None

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset, DATASET_SEAN, DATASET_SSUB_REAL
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from dataloaders.swm_segment_forecasting_dataset import SWMSegmentForecastingDataset
from train_swm_forecasting_linear_probe_v1 import split_indices_random, split_indices_temporal_gap, split_indices_episode

try:
    from train_snsb_linear_probe_v3 import load_oracle_eval_dataset_v3, _ZeroNodeWrapper
except ImportError:
    load_oracle_eval_dataset_v3 = None
    _ZeroNodeWrapper = None

# Models
from models.unified_gnn_vae_v1 import UnifiedGNNVAE
from models.hgnn_combined_v2 import HGNNCombinedVAE
from models.hgnn_combined_v1 import HGNNCombinedAE

# Constants
SEAN_LABELS = ["competence", "surprise", "intention"]
LABEL_THRESHOLD = 3
SNSB_CLASSES = ["avoid", "follow", "overtake", "yield", "not_consider"]
IGNORE_LABEL = -100

def sigmoid_focal_loss(logits, targets, gamma=2.0, alpha=0.25, reduction="mean"):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == "mean": return focal.mean()
    if reduction == "sum": return focal.sum()
    return focal

def compute_label_loss(logits, targets, use_focal, gamma, alpha, reduction="mean"):
    if use_focal: return sigmoid_focal_loss(logits, targets, gamma, alpha, reduction)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_run_dir(experiments_root: str, run_name: str) -> str:
    os.makedirs(experiments_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(experiments_root, f"{timestamp}_{run_name}" if run_name else timestamp)
    run_dir = base
    suffix = 1
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = f"{base}_{suffix}"
    os.makedirs(run_dir)
    return run_dir

# -----------------------------------------------------------------------------
# PyG Data Wrappers
# -----------------------------------------------------------------------------

def homo_to_hetero(data):
    from torch_geometric.data import HeteroData
    NODE_TYPE_ROBOT = 0
    hdata = HeteroData()
    robot_mask = (data.node_type == NODE_TYPE_ROBOT)
    by_mask    = ~robot_mask
    hdata["robot"].x             = data.x[robot_mask]
    hdata["bystander"].x         = data.x[by_mask]
    hdata["robot"].num_nodes     = int(robot_mask.sum())
    hdata["bystander"].num_nodes = int(by_mask.sum())

    old2new_r = {int(i): j for j, i in enumerate(robot_mask.nonzero(as_tuple=True)[0].tolist())}
    old2new_b = {int(i): j for j, i in enumerate(by_mask.nonzero(as_tuple=True)[0].tolist())}

    for et in [("robot","to","bystander"), ("bystander","to","robot"), ("bystander","to","bystander")]:
        src_type, _, dst_type = et
        sm = old2new_r if src_type == "robot" else old2new_b
        dm = old2new_r if dst_type == "robot" else old2new_b
        src_old, dst_old = data.edge_index[0], data.edge_index[1]
        mask = torch.tensor([int(s.item()) in sm and int(d.item()) in dm
                             for s, d in zip(src_old, dst_old)])
        ei   = data.edge_index[:, mask]
        ea   = data.edge_attr[mask] if data.edge_attr is not None else None
        if ei.shape[1] == 0:
            hdata[et].edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            hdata[et].edge_index = torch.stack([
                torch.tensor([sm[int(s)] for s in ei[0]], dtype=torch.long),
                torch.tensor([dm[int(d)] for d in ei[1]], dtype=torch.long),
            ])
        if ea is not None:
            hdata[et].edge_attr = ea

    hdata.emotion_label = getattr(data, "emotion_label", None)
    hdata.dataset_id    = torch.tensor([0], dtype=torch.long)
    hdata.scene         = getattr(data, "scene", "")
    return hdata

class SEANWrapper(torch.utils.data.Dataset):
    def __init__(self, subset): self.subset = subset
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx): return homo_to_hetero(self.subset[idx])

class SNSBWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, scenes):
        self.subset = subset
        self.scenes = scenes
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        data = self.subset[idx]
        robot_x = getattr(data["robot"], "x", None)
        bystander_x = getattr(data["bystander"], "x", None)
        if robot_x is not None:
            data["robot"].num_nodes = int(robot_x.shape[0])
            data["robot"].x = None
        elif not hasattr(data["robot"], "num_nodes"):
            data["robot"].num_nodes = 1
        if bystander_x is not None:
            data["bystander"].num_nodes = int(bystander_x.shape[0])
            data["bystander"].x = None
        elif not hasattr(data["bystander"], "num_nodes"):
            data["bystander"].num_nodes = 0
        data.scene = self.scenes[idx]
        return data

def _interp_1d(arr, target_T):
    """Linearly interpolate a numpy array along axis -2 (time) to target_T."""
    T_raw = arr.shape[-2]
    if T_raw == target_T:
        return arr
    x_old = np.linspace(0, 1, T_raw)
    x_new = np.linspace(0, 1, target_T)
    leading = arr.shape[:-2]
    F = arr.shape[-1]
    out = np.zeros(leading + (target_T, F), dtype=arr.dtype)
    for idx in np.ndindex(*leading):
        for f in range(F):
            out[idx + (slice(None), f)] = np.interp(x_new, x_old, arr[idx + (slice(None), f)])
    return out

class InterpGraphDataset(torch.utils.data.Dataset):
    """Linearly interpolates edge_attr from T_raw to target_T for homogeneous PyG Data batches."""
    def __init__(self, base_dataset, target_T):
        self.base_dataset = base_dataset
        self.target_T = target_T
        if hasattr(base_dataset, "scenes"):
            self.scenes = base_dataset.scenes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx].clone()
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            return data
        
        T = data.edge_attr.shape[1]
        if T == self.target_T:
            return data
            
        # Interpolate edge_attr [E, T, 8]
        ea = data.edge_attr.numpy()
        if ea.shape[0] > 0:
            ea_interp = _interp_1d(ea, self.target_T)
            data.edge_attr = torch.from_numpy(ea_interp)
        else:
            data.edge_attr = torch.zeros(0, self.target_T, ea.shape[2])
            
        # Interpolate valid_mask if present
        if hasattr(data, "valid_mask") and data.valid_mask is not None:
            vm = data.valid_mask.numpy()
            if vm.shape[0] > 0:
                vm_3d = vm[..., np.newaxis]
                vm_interp = _interp_1d(vm_3d, self.target_T)
                vm_interp = (vm_interp[..., 0] > 0.5).astype(vm.dtype)
                data.valid_mask = torch.from_numpy(vm_interp)

        return data

# -----------------------------------------------------------------------------
# Encoder Loading Factory
# -----------------------------------------------------------------------------

def load_frozen_encoder(model_type: str, ckpt_path: str, device: torch.device, fallback_args: argparse.Namespace):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    
    latent_dim = int(saved_args.get("latent_dim", getattr(fallback_args, "latent_dim", 64)))
    hidden_dim = int(saved_args.get("hidden_dim", getattr(fallback_args, "hidden_dim", 128)))
    num_mp_layers = int(saved_args.get("num_mp_layers", getattr(fallback_args, "num_mp_layers", 2)))
    
    if model_type == "UnifiedGNNVAE":
        snsb_timesteps = int(saved_args.get("snsb_timesteps", getattr(fallback_args, "snsb_timesteps", 20)))
        model = UnifiedGNNVAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim, num_mp_layers=num_mp_layers,
            snsb_timesteps=snsb_timesteps,
        ).to(device)
    elif model_type == "HGNNCombinedVAE":
        edge_transformer_layers = int(saved_args.get("edge_transformer_layers", getattr(fallback_args, "edge_transformer_layers", 1)))
        num_heads = int(saved_args.get("num_heads", getattr(fallback_args, "num_heads", 4)))
        model = HGNNCombinedVAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim, num_mp_layers=num_mp_layers,
            edge_transformer_layers=edge_transformer_layers, num_heads=num_heads,
        ).to(device)
    elif model_type == "HGNNCombinedAE":
        edge_transformer_layers = int(saved_args.get("edge_transformer_layers", getattr(fallback_args, "edge_transformer_layers", 2)))
        num_heads = int(saved_args.get("num_heads", getattr(fallback_args, "num_heads", 4)))
        model = HGNNCombinedAE(
            hidden_dim=hidden_dim, latent_dim=latent_dim, num_mp_layers=num_mp_layers,
            edge_transformer_layers=edge_transformer_layers, num_heads=num_heads,
        ).to(device)
    else: # TODO: add new model here
        raise ValueError(f"Unknown model_type: {model_type}")
    
    skipped, missing, unexpected = [], [], []
    model_state = model.state_dict()
    compatible = {}
    for key, value in ckpt["model_state_dict"].items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    return model, ckpt, latent_dim

# -----------------------------------------------------------------------------
# Embedding Extraction logic
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings_snsb(model, loader, device, model_type, embedding_mode="concat"):
    """Returns (z [N_by, D_out], y [N_by], graph_scenes [N_by], action_probs [N_by])."""
    zs, ys, scenes_out, probs_out = [], [], [], []
    has_probs = True
    
    for batch in tqdm(loader, desc="Extracting SNSB features"):
        batch_scenes = getattr(batch, "scene", None)
        if batch_scenes is None:
            # Fallback for datasets that don't pass scenes through
            batch_scenes = ["unknown_scene"] * (int(batch.batch.max().item()) + 1 if hasattr(batch, "batch") else 1)
            
        batch = batch.to(device)
        
        if model_type == "UnifiedGNNVAE":
            mu_dict, _ = model.encoder(batch)
            z_robot = mu_dict["robot"]
            z_by = mu_dict["bystander"]
            by_batch = batch["bystander"].batch
            robot_exp = z_robot[by_batch]
            
            if embedding_mode == "concat": z = torch.cat([robot_exp, z_by], dim=-1)
            elif embedding_mode == "bystander_only": z = z_by
            elif embedding_mode == "robot_only": z = robot_exp
            else: z = torch.cat([robot_exp, z_by], dim=-1)
            
            labels = batch["bystander"].y
            valid = labels != IGNORE_LABEL
            if hasattr(batch["bystander"], "action_probs"):
                probs = batch["bystander"].action_probs
            else:
                has_probs = False; probs = None
                
            by_batch_cpu = by_batch.cpu()
            
        else: # HGNN models
            z_nodes = model.encode_nodes(batch)
            labels = batch.action_label
            valid = labels != IGNORE_LABEL
            if hasattr(batch, "action_probs"):
                probs = batch.action_probs
            else:
                has_probs = False; probs = None
                
            z = z_nodes
            by_batch_cpu = getattr(batch, "batch", torch.zeros(len(labels), dtype=torch.long, device=device)).cpu()
            
        if valid.sum() == 0: continue
        
        sc_per_by = [batch_scenes[int(by_batch_cpu[i])] for i in range(len(by_batch_cpu)) if valid[i]]
        zs.append(z[valid].cpu())
        ys.append(labels[valid].cpu())
        scenes_out.extend(sc_per_by)
        if has_probs: probs_out.append(probs[valid].cpu())
        
    probs_cat = torch.cat(probs_out, dim=0) if has_probs and probs_out else None
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0), scenes_out, probs_cat

@torch.no_grad()
def extract_embeddings_sean(model, loader, device, model_type, embedding_mode="graph_level"):
    zs, ys, scenes_out = [], [], []
    for batch in tqdm(loader, desc="Extracting SEAN features"):
        batch_scenes = getattr(batch, "scene", None)
        if batch_scenes is None:
            batch_scenes = ["unknown_scene"] * (int(batch.batch.max().item()) + 1 if hasattr(batch, "batch") else 1)
            
        batch = batch.to(device)
        
        if model_type == "UnifiedGNNVAE":
            mu_dict, _ = model.encoder(batch)
            z_robot = mu_dict["robot"]
            if mu_dict["bystander"].shape[0] > 0:
                z_b_mean = scatter(mu_dict["bystander"], batch["bystander"].batch, dim=0, reduce="mean", dim_size=z_robot.shape[0])
            else:
                z_b_mean = torch.zeros_like(z_robot)
                
            if embedding_mode == "graph_level" or embedding_mode == "concat":
                z = torch.cat([z_robot, z_b_mean], dim=-1)
            elif embedding_mode == "robot_only":
                z = z_robot
            elif embedding_mode == "bystander_only":
                z = z_b_mean
            else:
                z = torch.cat([z_robot, z_b_mean], dim=-1)
        else: # HGNN models
            z_nodes = model.encode_nodes(batch)
            node_batch = getattr(batch, "batch", torch.zeros(z_nodes.shape[0], dtype=torch.long, device=device))
            B = int(node_batch.max()) + 1
            robot_mask = batch.node_type == 0
            people_mask = batch.node_type == 1
            
            z_robot = scatter(z_nodes[robot_mask], node_batch[robot_mask], dim=0, reduce="mean", dim_size=B)
            z_people = scatter(z_nodes[people_mask], node_batch[people_mask], dim=0, reduce="mean", dim_size=B)
            z = torch.cat([z_robot, z_people], dim=-1)

        raw = batch.emotion_label
        if raw is None: continue
        raw = raw.view(-1, len(SEAN_LABELS))
        y_bin = (raw >= LABEL_THRESHOLD).float()
        y_bin[:, 1] = 1.0 - y_bin[:, 1] # flip_surprise
        
        zs.append(z.cpu())
        ys.append(y_bin.cpu())
        scenes_out.extend(batch_scenes)
        
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0), scenes_out

@torch.no_grad()
def extract_embeddings_forecasting(model, loader, device, model_type):
    x_list, target_xy_list, baseline_xy_list = [], [], []
    obs_last_xy_list, future_xy_list, future_valid_list = [], [], []
    scenes_out = []
    
    for batch in tqdm(loader, desc="Extracting forecasting features"):
        batch_scenes = getattr(batch, "episode_key", None)
        if batch_scenes is None:
            batch_scenes = ["unknown_scene"] * (int(batch.batch.max().item()) + 1 if hasattr(batch, "batch") else 1)
        
        batch = batch.to(device)
        
        z_nodes = model.encode_nodes(batch)
        node_batch = getattr(batch, "batch", torch.zeros(z_nodes.shape[0], dtype=torch.long, device=device))
        n_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1
        
        node_type = batch.node_type.long()
        robot_mask = node_type == 0
        human_mask = (node_type == 1) & batch.target_valid.bool()
        
        if int(human_mask.sum().item()) == 0: continue
        
        robot_z_by_graph = scatter(z_nodes[robot_mask], node_batch[robot_mask], dim=0, reduce="mean", dim_size=n_graphs)
        human_z = z_nodes[human_mask]
        human_graph = node_batch[human_mask]
        features = torch.cat([human_z, robot_z_by_graph[human_graph]], dim=-1)
        
        x_list.append(features.cpu())
        target_xy_list.append(batch.target_xy[human_mask].cpu())
        baseline_xy_list.append(batch.baseline_xy[human_mask].cpu())
        obs_last_xy_list.append(batch.obs_xy[human_mask, -1, :].cpu())
        future_xy_list.append(batch.future_xy[human_mask].cpu())
        future_valid_list.append(batch.future_valid[human_mask].float().cpu())
        
        human_graph_cpu = human_graph.cpu()
        scenes_out.extend([batch_scenes[int(human_graph_cpu[i])] for i in range(len(human_graph_cpu))])

    if not x_list: raise RuntimeError("No supervised human nodes found in forecasting extraction.")
    
    return {
        "x": torch.cat(x_list, dim=0),
        "target_xy": torch.cat(target_xy_list, dim=0),
        "baseline_xy": torch.cat(baseline_xy_list, dim=0),
        "obs_last_xy": torch.cat(obs_last_xy_list, dim=0),
        "future_xy": torch.cat(future_xy_list, dim=0),
        "future_valid": torch.cat(future_valid_list, dim=0),
    }, scenes_out

# -----------------------------------------------------------------------------
# Head Training loops
# -----------------------------------------------------------------------------

def _train_head_lightweight(head, loss_fn, tr_z, tr_y, va_z, va_y, lr, epochs, patience, device, task="default"):
    """Minimal loop for multi-VAE batch sweeps to minimize overhead."""
    opt = AdamW(head.parameters(), lr=lr)
    best_loss = float('inf')
    best_state = None
    strikes = 0
    head.to(device)
    tr_z, tr_y = tr_z.to(device), tr_y.to(device)
    va_z, va_y = va_z.to(device), va_y.to(device)
    
    for epoch in range(epochs):
        head.train()
        opt.zero_grad()
        if task == "sean_emotions":
            loss = torch.stack([
                F.binary_cross_entropy_with_logits(head[d](tr_z).squeeze(-1), tr_y[:, d].float())
                for d in range(3)
            ]).mean()
        else:
            loss = loss_fn(head(tr_z), tr_y)
        loss.backward()
        opt.step()
        
        head.eval()
        with torch.no_grad():
            if task == "sean_emotions":
                v_loss = torch.stack([
                    F.binary_cross_entropy_with_logits(head[d](va_z).squeeze(-1), va_y[:, d].float())
                    for d in range(3)
                ]).mean().item()
            else:
                v_loss = loss_fn(head(va_z), va_y).item()
            
        if v_loss < best_loss - 1e-5:
            best_loss = v_loss
            strikes = 0
            best_state = {k: v.cpu() for k, v in head.state_dict().items()}
        else:
            strikes += 1
            if strikes >= patience: break
            
    head.load_state_dict(best_state)
    return head

def _train_head_full(head, loss_fn, tr_z, tr_y, va_z, va_y, args, device, 
                     sample_weights=None, ce_weights=None, is_regression=False, 
                     horizon_alphas=None, extra_tensors=None, task="default"):
    """Full training loop with all the bells and whistles for single/sweep mode."""
    opt = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=args.lr_patience)
    
    best_val_loss = float("inf")
    es_counter = 0
    best_state = None
    
    head.to(device)
    n_train = tr_z.shape[0]
    
    # Pre-move validation to device
    va_z = va_z.to(device)
    if is_regression and extra_tensors is not None:
        val_target = extra_tensors['val_target_xy'].to(device)
    else:
        va_y = va_y.to(device)
    
    for epoch in range(1, args.epochs + 1):
        head.train()
        
        # Adaptive sampling
        if sample_weights is not None:
            if getattr(args, 'adaptive_warmup_epochs', 0) > 0 and epoch > args.adaptive_warmup_epochs:
                progress = min(1.0, (epoch - args.adaptive_warmup_epochs) / max(1, args.epochs - args.adaptive_warmup_epochs))
                alpha = getattr(args, 'adaptive_max_alpha', 0.3) * progress
                head.eval()
                with torch.no_grad():
                    tr_z_dev = tr_z.to(device)
                    tr_y_dev = tr_y.to(device)
                    # We only support CrossEntropy for adaptive loss reweighting
                    per_sample_loss = F.cross_entropy(head(tr_z_dev), tr_y_dev, reduction="none")
                head.train()
                loss_weights = per_sample_loss.cpu() / per_sample_loss.cpu().mean().clamp(min=1e-8)
                blended = (1.0 - alpha) * sample_weights.cpu() + alpha * loss_weights
                epoch_weights = blended
            else:
                epoch_weights = sample_weights
            idx_order = torch.multinomial(epoch_weights.float(), n_train, replacement=True)
        else:
            idx_order = torch.randperm(n_train)
            
        t_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            idx = idx_order[i : i + args.batch_size]
            z_b = tr_z[idx].to(device)
            if is_regression and extra_tensors is not None:
                y_b = extra_tensors['train_target_xy'][idx].to(device)
            else:
                y_b = tr_y[idx].to(device)
                
            opt.zero_grad()
            if task == "sean_emotions":
                use_focal = getattr(args, "use_focal_loss", False)
                focal_gamma = getattr(args, "focal_gamma", 2.0)
                focal_alpha = getattr(args, "focal_alpha", 0.25)
                loss = torch.stack([
                    compute_label_loss(head[d](z_b).squeeze(-1), y_b[:, d],
                                       use_focal, focal_gamma, focal_alpha)
                    for d in range(3)
                ]).mean()
            elif ce_weights is not None and not is_regression:
                loss = loss_fn(head(z_b), y_b, weight=ce_weights.to(device))
            else:
                loss = loss_fn(head(z_b), y_b)
            loss.backward()
            opt.step()
            t_loss += loss.item()
            n_batches += 1
        t_loss /= max(1, n_batches)
        
        head.eval()
        with torch.no_grad():
            if task == "sean_emotions":
                use_focal = getattr(args, "use_focal_loss", False)
                focal_gamma = getattr(args, "focal_gamma", 2.0)
                focal_alpha = getattr(args, "focal_alpha", 0.25)
                lg = torch.stack([head[d](va_z).squeeze(-1) for d in range(3)], dim=1)
                v_loss = torch.stack([
                    compute_label_loss(lg[:, d], va_y[:, d], use_focal, focal_gamma, focal_alpha)
                    for d in range(3)
                ]).mean().item()
            elif is_regression and extra_tensors is not None:
                v_loss = loss_fn(head(va_z), val_target).item()
            else:
                if ce_weights is not None and not is_regression:
                    v_loss = loss_fn(head(va_z), va_y, weight=ce_weights.to(device)).item()
                else:
                    v_loss = loss_fn(head(va_z), va_y).item()
        
        scheduler.step(v_loss)
        
        if args.wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                f"loss/{task}/train": t_loss,
                f"loss/{task}/val": v_loss,
                f"loss/{task}/lr": opt.param_groups[0]["lr"],
            })
        
        if v_loss < best_val_loss - getattr(args, 'early_stopping_min_delta', 1e-4):
            best_val_loss = v_loss
            es_counter = 0
            best_state = {k: v.cpu() for k, v in head.state_dict().items()}
        else:
            es_counter += 1
            if es_counter >= args.early_stopping_patience:
                break
                
    head.load_state_dict(best_state)
    return head

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_snsb_metrics_dict(logits_np, labels_np, prefix):
    preds_np = logits_np.argmax(axis=1)
    metrics = {f"{prefix}/acc": float(accuracy_score(labels_np, preds_np))}
    f1_per_class = f1_score(labels_np, preds_np, average=None, labels=list(range(len(SNSB_CLASSES))), zero_division=0)
    for i, name in enumerate(SNSB_CLASSES):
        metrics[f"{prefix}/f1_{name}"] = float(f1_per_class[i])
    metrics[f"{prefix}/f1_macro"] = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
    metrics[f"{prefix}/f1_weighted"] = float(f1_score(labels_np, preds_np, average="weighted", zero_division=0))
    return metrics

def compute_sean_metrics_dict(logits_np, labels_np, prefix):
    metrics = {}
    f1_list = []
    auc_list = []
    for i, name in enumerate(SEAN_LABELS):
        prob = torch.sigmoid(torch.tensor(logits_np[:, i])).numpy()
        pred = (prob > 0.5).astype(float)
        gt = labels_np[:, i]
        
        f1 = f1_score(gt, pred, zero_division=0)
        try:
            auc = roc_auc_score(gt, prob)
        except ValueError:
            auc = 0.5
            
        metrics[f"{prefix}/f1_{name}"] = float(f1)
        metrics[f"{prefix}/auc_{name}"] = float(auc)
        f1_list.append(f1)
        auc_list.append(auc)
        
    metrics[f"{prefix}/f1_macro"] = float(np.mean(f1_list))
    metrics[f"{prefix}/auc_macro"] = float(np.mean(auc_list))
    return metrics

def compute_displacement_metrics(pred_xy, target_xy, baseline_xy, obs_last_xy, future_xy, future_valid, horizon_alphas):
    fde_pred = torch.linalg.norm(pred_xy - target_xy, dim=1)
    fde_base = torch.linalg.norm(baseline_xy - target_xy, dim=1)
    pred_path = obs_last_xy.unsqueeze(1) + horizon_alphas * (pred_xy.unsqueeze(1) - obs_last_xy.unsqueeze(1))
    base_path = obs_last_xy.unsqueeze(1) + horizon_alphas * (baseline_xy.unsqueeze(1) - obs_last_xy.unsqueeze(1))
    err_pred = torch.linalg.norm(pred_path - future_xy, dim=-1)
    err_base = torch.linalg.norm(base_path - future_xy, dim=-1)
    mask = future_valid
    ade_pred = (err_pred * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    ade_base = (err_base * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return {"fde_pred": fde_pred, "fde_base": fde_base, "ade_pred": ade_pred, "ade_base": ade_base}

# -----------------------------------------------------------------------------
# Orchestrators
# -----------------------------------------------------------------------------

def run_multi_vae_batch(args, device):
    """
    Lightweight, minimal overhead sweep across multiple VAEs, fractions, and seeds.
    Writes results to a single CSV file.
    """
    print(f"Starting Multi-VAE Batch over {len(args.vae_dirs)} directories.")
    csv_out = args.output
    if not csv_out:
        csv_out = os.path.join(args.experiments_root, "multi_vae_results.csv")
    
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    write_header = not os.path.exists(csv_out)
    
    for vae_dir in args.vae_dirs:
        ckpt_path = os.path.join(vae_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(vae_dir, "best_unified_gnn_vae_v3_seednone.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {vae_dir}: no checkpoint found.")
            continue
            
        print(f"--- Loading {vae_dir} ---")
        model, ckpt, latent_dim = load_frozen_encoder(args.model_type, ckpt_path, device, args)
        
        # Determine paths
        split_path = ckpt.get("split_path")
        if split_path and not os.path.isabs(split_path):
            split_path = os.path.join(vae_dir, os.path.basename(split_path))
        if not split_path or not os.path.exists(split_path):
            split_path = os.path.join(vae_dir, "split_used_v3_seednone.pt")
        
        if not os.path.exists(split_path):
            print(f"Skipping {vae_dir}: split file not found.")
            continue
            
        split = torch.load(split_path, map_location="cpu")
        
        for task in args.tasks:
            print(f"  Extracting embeddings for {task}...")
            # Setup datasets
            if task == "snsb_actions":
                h5_path = getattr(args, "h5_path", None) or ckpt.get("snsb_h5_path", "/workspace/lidar-detect/social_world_model/social_world_model/src/dataloaders/social_nav_v3.h5")
                if args.model_type == "UnifiedGNNVAE":
                    ds = UnifiedSocialNavDataset(h5_path, zero_node_features=ckpt.get("zero_node_features", True))
                    if _ZeroNodeWrapper: ds = _ZeroNodeWrapper(ds)
                    scenes = [f"scene_{i}" for i in range(len(ds))] # Fallback
                    ds = SNSBWrapper(ds, scenes)
                else:
                    robot_mean = ckpt.get("robot_mean", None)
                    robot_std = ckpt.get("robot_std", None)
                    ds = UnifiedGraphDataset(h5_path, ssub=True, robot_mean=robot_mean, robot_std=robot_std)
                    if hasattr(model, "encoder") and hasattr(model.encoder, "edge_encoder") and hasattr(model.encoder.edge_encoder, "pos_embed"):
                        target_T = model.encoder.edge_encoder.pos_embed.shape[1]
                        ds = InterpGraphDataset(ds, target_T)

                tr_idx, va_idx, te_idx = split["snsb_train"], split["snsb_val"], split["snsb_test"] if "snsb_test" in split else split["snsb_val"]
                loader_fn = PyGDataLoader
                extract_fn = extract_embeddings_snsb
            elif task == "sean_emotions":
                h5_path = getattr(args, "h5_path", None) or ckpt.get("sean_h5_path", "/workspace/lidar-detect/social_world_model/social_world_model/src/dataloaders/combined_v1.h5")
                robot_mean = ckpt.get("robot_mean", None)
                robot_std = ckpt.get("robot_std", None)
                ds = UnifiedGraphDataset(h5_path, ssub=False, robot_mean=robot_mean, robot_std=robot_std)
                if args.model_type == "UnifiedGNNVAE":
                    ds = SEANWrapper(ds)
                else:
                    if hasattr(model, "encoder") and hasattr(model.encoder, "edge_encoder") and hasattr(model.encoder.edge_encoder, "pos_embed"):
                        target_T = model.encoder.edge_encoder.pos_embed.shape[1]
                        ds = InterpGraphDataset(ds, target_T)

                tr_idx, va_idx, te_idx = split["sean_train"], split["sean_val"], split["sean_test"]
                loader_fn = PyGDataLoader
                extract_fn = extract_embeddings_sean
            elif task == "forecasting":
                print("Skipping forecasting in Multi-VAE Batch (regression).")
                continue
                
            # Full loaders
            tr_full = loader_fn(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=False)
            va_full = loader_fn(Subset(ds, va_idx), batch_size=args.batch_size, shuffle=False)
            te_full = loader_fn(Subset(ds, te_idx), batch_size=args.batch_size, shuffle=False)
            
            # Extract
            z_tr, y_tr, sc_tr, *_ = extract_fn(model, tr_full, device, args.model_type, args.embedding_mode)
            z_va, y_va, _, *_ = extract_fn(model, va_full, device, args.model_type, args.embedding_mode)
            z_te, y_te, _, *_ = extract_fn(model, te_full, device, args.model_type, args.embedding_mode)
            
            sc_tr_np = np.array(sc_tr)
            unique_scenes = np.unique(sc_tr_np)
            
            for frac in args.fractions:
                for seed in range(args.n_subset_seeds):
                    # Subsample
                    rng = np.random.default_rng(seed)
                    n_keep = max(1, int(len(unique_scenes) * frac))
                    keep_scenes = rng.choice(unique_scenes, size=n_keep, replace=False)
                    keep_mask = np.isin(sc_tr_np, keep_scenes)
                    sub_z = z_tr[keep_mask]
                    sub_y = y_tr[keep_mask]
                    
                    if len(sub_y) == 0: continue
                    
                    # Train Lightweight
                    if task == "snsb_actions":
                        head = nn.Linear(sub_z.shape[1], len(SNSB_CLASSES))
                        loss_fn = lambda p, t: F.cross_entropy(p, t.long())
                    else:
                        head = nn.ModuleList([nn.Linear(sub_z.shape[1], 1) for _ in range(len(SEAN_LABELS))])
                        loss_fn = None
                        
                    head = _train_head_lightweight(head, loss_fn, sub_z, sub_y, z_va, y_va, args.lr, args.epochs, getattr(args, 'early_stopping_patience', 10), device, task=task)
                    
                    # Evaluate
                    head.to(device)
                    head.eval()
                    with torch.no_grad():
                        if task == "snsb_actions":
                            logits = head(z_te.to(device)).cpu().numpy()
                            metrics = compute_snsb_metrics_dict(logits, y_te.numpy(), "test")
                        else:
                            logits = torch.cat([h(z_te.to(device)) for h in head], dim=-1).cpu().numpy()
                            metrics = compute_sean_metrics_dict(logits, y_te.numpy(), "test")
                    
                    # Write to CSV
                    row = {
                        "vae_dir": vae_dir,
                        "task": task,
                        "fraction": frac,
                        "seed": seed,
                    }
                    row.update(metrics)
                    
                    fieldnames = ["vae_dir", "task", "fraction", "seed"] + list(metrics.keys())
                    with open(csv_out, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                            write_header = False
                        writer.writerow(row)
                        
    print(f"Multi-VAE Batch completed. Results saved to {csv_out}")

def run_single_sweep(args, device):
    """
    Full training logic for a single checkpoint. Supports sweeping over fractions.
    """
    print(f"Starting Single/Sweep Mode using checkpoint {args.checkpoint}")
    run_dir = create_run_dir(args.experiments_root, args.run_name)
    
    if args.wandb and wandb is not None:
        wandb.init(project=getattr(args, 'wandb_project', 'social_world_model'), config=vars(args), name=os.path.basename(run_dir))
    
    model, ckpt, latent_dim = load_frozen_encoder(args.model_type, args.checkpoint, device, args)
    split_path = args.split_file or ckpt.get("split_path")
    split = torch.load(split_path, map_location="cpu") if split_path and os.path.exists(split_path) else None
    
    for task in args.tasks:
        print(f"\n=== Processing Task: {task} ===")
        # Build Dataset
        if task == "snsb_actions":
            h5_path = getattr(args, "h5_path", None) or ckpt.get("snsb_h5_path", "/workspace/lidar-detect/social_world_model/social_world_model/src/dataloaders/social_nav_v3.h5")
            if args.model_type == "UnifiedGNNVAE":
                ds = UnifiedSocialNavDataset(h5_path, zero_node_features=ckpt.get("zero_node_features", True))
                if _ZeroNodeWrapper: ds = _ZeroNodeWrapper(ds)
                scenes = [f"scene_{i}" for i in range(len(ds))]
                ds = SNSBWrapper(ds, scenes)
            else:
                robot_mean = ckpt.get("robot_mean", None)
                robot_std = ckpt.get("robot_std", None)
                ds = UnifiedGraphDataset(h5_path, ssub=True, robot_mean=robot_mean, robot_std=robot_std)
                if hasattr(model, "encoder") and hasattr(model.encoder, "edge_encoder") and hasattr(model.encoder.edge_encoder, "pos_embed"):
                    target_T = model.encoder.edge_encoder.pos_embed.shape[1]
                    ds = InterpGraphDataset(ds, target_T)

            if split:
                tr_idx, va_idx, te_idx = split["snsb_train"], split["snsb_val"], split["snsb_test"] if "snsb_test" in split else split["snsb_val"]
            else:
                tr_idx, va_idx, te_idx = list(range(int(len(ds)*0.8))), list(range(int(len(ds)*0.8), int(len(ds)*0.9))), list(range(int(len(ds)*0.9), len(ds)))
            loader_fn = PyGDataLoader
            extract_fn = extract_embeddings_snsb
            is_regression = False
            
        elif task == "sean_emotions":
            h5_path = getattr(args, "h5_path", None) or ckpt.get("sean_h5_path", "/workspace/lidar-detect/social_world_model/social_world_model/src/dataloaders/combined_v1.h5")
            robot_mean = ckpt.get("robot_mean", None)
            robot_std = ckpt.get("robot_std", None)
            ds = UnifiedGraphDataset(h5_path, ssub=False, robot_mean=robot_mean, robot_std=robot_std)
            if args.model_type == "UnifiedGNNVAE":
                ds = SEANWrapper(ds)
            else:
                if hasattr(model, "encoder") and hasattr(model.encoder, "edge_encoder") and hasattr(model.encoder.edge_encoder, "pos_embed"):
                    target_T = model.encoder.edge_encoder.pos_embed.shape[1]
                    ds = InterpGraphDataset(ds, target_T)

            if split:
                tr_idx, va_idx, te_idx = split["sean_train"], split["sean_val"], split["sean_test"]
            else:
                tr_idx, va_idx, te_idx = list(range(int(len(ds)*0.8))), list(range(int(len(ds)*0.8), int(len(ds)*0.9))), list(range(int(len(ds)*0.9), len(ds)))
            loader_fn = PyGDataLoader
            extract_fn = extract_embeddings_sean
            is_regression = False
            
        elif task == "forecasting":
            segments_root = getattr(args, "segments_root", "dataloaders/swm_data_dense_bag_200s/segments")
            saved_args = ckpt.get("args", {}) if ckpt else {}
            clip_end = saved_args.get("clip_end_frame", args.clip_end_frame)
            target_offset = saved_args.get("target_offset_frame", args.target_offset_frame)
            require_future = saved_args.get("require_future_labels", True)
            
            # Update args so downstream evaluation metrics use correct horizon
            args.clip_end_frame = clip_end
            args.target_offset_frame = target_offset

            ds = SWMSegmentForecastingDataset(
                segments_root=segments_root,
                clip_end_frame=clip_end,
                target_offset_frame=target_offset,
                require_future_labels=require_future,
            )
            if split and "train" in split:
                tr_idx, va_idx, te_idx = split["train"], split["val"], split["test"]
            else:
                tr_idx, va_idx, te_idx = split_indices_random(len(ds), args.train_ratio, args.val_ratio, args.seed)
            loader_fn = PyGDataLoader
            extract_fn = extract_embeddings_forecasting
            is_regression = True

        # TODO: add new task here
            
        # Loaders
        tr_full = loader_fn(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=False)
        va_full = loader_fn(Subset(ds, va_idx), batch_size=args.batch_size, shuffle=False)
        te_full = loader_fn(Subset(ds, te_idx), batch_size=args.batch_size, shuffle=False)
        
        print("Extracting features...")
        if is_regression:
            tr_tensors, sc_tr = extract_fn(model, tr_full, device, args.model_type)
            va_tensors, _ = extract_fn(model, va_full, device, args.model_type)
            te_tensors, _ = extract_fn(model, te_full, device, args.model_type)
            z_tr = tr_tensors["x"]
            y_tr = None
            z_va = va_tensors["x"]
            y_va = None
            z_te = te_tensors["x"]
            y_te = None
        else:
            z_tr, y_tr, sc_tr, *_ = extract_fn(model, tr_full, device, args.model_type, args.embedding_mode)
            z_va, y_va, _, *_ = extract_fn(model, va_full, device, args.model_type, args.embedding_mode)
            z_te, y_te, _, *_ = extract_fn(model, te_full, device, args.model_type, args.embedding_mode)
        
        sc_tr_np = np.array(sc_tr)
        unique_scenes = np.unique(sc_tr_np)
        
        csv_out = os.path.join(run_dir, f"summary_{task}.csv")
        write_header = True
        
        for frac in args.fractions:
            for run_idx in range(args.num_runs):
                seed = args.seed + run_idx
                seed_everything(seed)
                
                rng = np.random.default_rng(seed)
                n_keep = max(1, int(len(unique_scenes) * frac))
                keep_scenes = rng.choice(unique_scenes, size=n_keep, replace=False)
                keep_mask = np.isin(sc_tr_np, keep_scenes)
                
                sub_z = z_tr[keep_mask]
                
                # Head + Loss setup
                sample_weights = None
                ce_weights = None
                extra_tensors = None
                
                if task == "snsb_actions":
                    sub_y = y_tr[keep_mask]
                    head = nn.Linear(sub_z.shape[1], len(SNSB_CLASSES))
                    loss_fn = F.cross_entropy
                    if not getattr(args, 'no_class_weights', True):
                        counts = torch.bincount(sub_y.cpu(), minlength=len(SNSB_CLASSES)).float()
                        ce_weights = (sub_y.shape[0] / (len(SNSB_CLASSES) * counts.clamp(min=1)))
                    if not getattr(args, 'no_stratified', True):
                        counts = torch.bincount(sub_y.cpu(), minlength=len(SNSB_CLASSES)).float()
                        inv_sqrt = 1.0 / torch.sqrt(counts.clamp(min=1))
                        sample_weights = inv_sqrt[sub_y.cpu()]
                        
                elif task == "sean_emotions":
                    sub_y = y_tr[keep_mask]
                    head = nn.ModuleList([nn.Linear(sub_z.shape[1], 1) for _ in range(len(SEAN_LABELS))])
                    loss_fn = None
                    
                elif task == "forecasting":
                    sub_y = None
                    extra_tensors = {
                        "train_target_xy": tr_tensors["target_xy"][keep_mask],
                        "val_target_xy": va_tensors["target_xy"]
                    }
                    head = nn.Linear(sub_z.shape[1], 2)
                    loss_fn = F.mse_loss

                # TODO: add new task config here

                print(f"Training [{task}] frac={frac} run={run_idx} (N={sub_z.shape[0]})")
                head = _train_head_full(head, loss_fn, sub_z, sub_y, z_va, y_va, args, device,
                                        sample_weights=sample_weights, ce_weights=ce_weights,
                                        is_regression=is_regression, extra_tensors=extra_tensors,
                                        task=task)
                
                # Evaluation
                head.eval()
                head.to(device)
                with torch.no_grad():
                    if task == "snsb_actions":
                        logits = head(z_te.to(device)).cpu().numpy()
                        metrics = compute_snsb_metrics_dict(logits, y_te.numpy(), "test")
                    elif task == "sean_emotions":
                        logits = torch.cat([h(z_te.to(device)) for h in head], dim=-1).cpu().numpy()
                        metrics = compute_sean_metrics_dict(logits, y_te.numpy(), "test")
                    elif task == "forecasting":
                        preds = head(z_te.to(device)).cpu()
                        horizon = args.target_offset_frame - args.clip_end_frame
                        alphas = torch.linspace(1.0/horizon, 1.0, steps=horizon, device='cpu').view(1, horizon, 1)
                        disp = compute_displacement_metrics(preds, te_tensors["target_xy"], te_tensors["baseline_xy"],
                                                            te_tensors["obs_last_xy"], te_tensors["future_xy"],
                                                            te_tensors["future_valid"], alphas)
                        metrics = {"test/fde": disp["fde_pred"].mean().item(), "test/ade": disp["ade_pred"].mean().item()}
                    # TODO: add new task metric here

                row = {"task": task, "fraction": frac, "run_idx": run_idx}
                row.update(metrics)
                
                if args.wandb and wandb is not None:
                    wb_log = {
                        "task": task,
                        "fraction": frac,
                        "run_idx": run_idx,
                    }
                    wb_log.update({f"final/{task}/{k}": v for k, v in metrics.items()})
                    wandb.log(wb_log)
                
                fieldnames = ["task", "fraction", "run_idx"] + list(metrics.keys())
                with open(csv_out, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(row)
                    
    if args.wandb and wandb is not None:
        wandb.finish()
        
    print(f"Run complete. Artifacts saved in {run_dir}")

# -----------------------------------------------------------------------------
# Main & CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Linear Probe General Script")
    
    # Task and Model
    parser.add_argument("--tasks", nargs="+", choices=["snsb_actions", "sean_emotions", "forecasting"], required=True)
    parser.add_argument("--model_type", choices=["UnifiedGNNVAE", "HGNNCombinedVAE", "HGNNCombinedAE"], required=True)
    
    # Execution Modes
    parser.add_argument("--checkpoint", type=str, default=None, help="Path for single/sweep mode")
    parser.add_argument("--vae_dirs", nargs="+", type=str, default=None, help="Paths for multi-VAE batch mode")
    parser.add_argument("--lightweight", action="store_true", help="Force lightweight minimal training loop")
    
    # Feature extraction
    parser.add_argument("--embedding_mode", choices=["concat", "node_only", "robot_only", "bystander_only", "graph_level"], default="concat")
    
    # Sweep Config
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0])
    parser.add_argument("--n_subset_seeds", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    # Data overrides
    parser.add_argument("--h5_path", type=str, default=None)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--segments_root", type=str, default="dataloaders/swm_data_dense_bag_200s/segments")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    
    # Forecasting specifics
    parser.add_argument("--clip_end_frame", type=int, default=75)
    parser.add_argument("--target_offset_frame", type=int, default=105)
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_patience", type=int, default=8)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=0)
    
    # Options
    parser.add_argument("--no_stratified", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--experiments_root", type=str, default="experiments/linear_probes")
    parser.add_argument("--run_name", type=str, default="general_lp")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--key_file", type=str, default=None, help="Path to keys.json file containing WANDB key")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load W&B key if key_file is provided
    if args.key_file:
        if os.path.exists(args.key_file):
            try:
                import json
                with open(args.key_file, "r") as f:
                    keys = json.load(f)
                if "WANDB" in keys:
                    wandb_key = keys["WANDB"]
                    os.environ["WANDB_API_KEY"] = wandb_key
                    if wandb is not None:
                        wandb.login(key=wandb_key)
                        args.wandb = True
                        print("Successfully logged into W&B using API key from key file.")
                    else:
                        print("Warning: WANDB key found, but wandb package is not installed.")
                else:
                    print(f"Warning: 'WANDB' key not found in {args.key_file}")
            except Exception as e:
                print(f"Warning: Failed to load W&B key from {args.key_file}: {e}")
        else:
            print(f"Warning: Key file not found: {args.key_file}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    if args.vae_dirs is not None and len(args.vae_dirs) > 0:
        # Multi-VAE Batch mode
        run_multi_vae_batch(args, device)
    elif args.checkpoint is not None:
        # Single Sweep mode
        run_single_sweep(args, device)
    else:
        print("Error: Must specify either --checkpoint or --vae_dirs")
        sys.exit(1)

if __name__ == "__main__":
    main()
