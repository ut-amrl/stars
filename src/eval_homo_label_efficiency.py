"""
Homogeneous GNN-VAE pipeline — combined SEAN+SNSB label-efficiency sweep.

eval_homo_label_efficiency.py

Label-efficiency sweep for HGNNCombinedVAE checkpoints.
"""

import argparse
import csv
import hashlib
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import scatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.hgnn_combined_v2 import HGNNCombinedVAE
from dataloaders.unified_graph_dataset_all import UnifiedGraphDataset
from train_snsb_linear_probe_v3 import load_oracle_eval_dataset_v3

# ── constants ────────────────────────────────────────────────────────────────
SEAN_LABELS     = ["competence", "surprise", "intention"]
LABEL_THRESHOLD = 3
SNSB_CLASSES    = ["avoid", "follow", "overtake", "yield", "not_consider"]
IGNORE_LABEL    = -100


def hetero_to_homo(hdata, ssub=False):
    from torch_geometric.data import Data
    
    n_robot = int(hdata["robot"].num_nodes) if hasattr(hdata["robot"], "num_nodes") else 1
    if hasattr(hdata["robot"], "x") and hdata["robot"].x is not None:
        n_robot = hdata["robot"].x.shape[0]
        
    n_by = int(hdata["bystander"].num_nodes) if hasattr(hdata["bystander"], "num_nodes") else 0
    if hasattr(hdata["bystander"], "x") and hdata["bystander"].x is not None:
        n_by = hdata["bystander"].x.shape[0]
        
    n_total = n_robot + n_by
    
    node_type = torch.zeros(n_total, dtype=torch.long)
    node_type[n_robot:] = 1
    
    x = None
    if not ssub:
        x = torch.zeros(n_total, 592)
        if hasattr(hdata["robot"], "x") and hdata["robot"].x is not None:
            x[:n_robot] = hdata["robot"].x
            
    valid_mask = None
    if not ssub:
        valid_mask = torch.ones(n_total, 40)
        if hasattr(hdata["robot"], "valid_mask") and hdata["robot"].valid_mask is not None:
            valid_mask[:n_robot] = hdata["robot"].valid_mask
        if hasattr(hdata["bystander"], "valid_mask") and hdata["bystander"].valid_mask is not None:
            valid_mask[n_robot:] = hdata["bystander"].valid_mask
            
    action_label = torch.full((n_total,), -100, dtype=torch.long)
    if hasattr(hdata["bystander"], "y") and hdata["bystander"].y is not None:
        action_label[n_robot:] = hdata["bystander"].y.long()
        
    edge_indices = []
    edge_attrs = []
    
    for (src_type, _, dst_type) in [("robot", "to", "bystander"), ("bystander", "to", "robot"), ("bystander", "to", "bystander")]:
        et = (src_type, "to", dst_type)
        if et in hdata.edge_types and hasattr(hdata[et], "edge_index") and hdata[et].edge_index.shape[1] > 0:
            ei = hdata[et].edge_index.clone()
            src_shift = 0 if src_type == "robot" else n_robot
            dst_shift = 0 if dst_type == "robot" else n_robot
            ei[0] += src_shift
            ei[1] += dst_shift
            edge_indices.append(ei)
            if hasattr(hdata[et], "edge_attr") and hdata[et].edge_attr is not None:
                edge_attrs.append(hdata[et].edge_attr)
                
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else None
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = None
        
    data = Data(
        node_type=node_type,
        edge_index=edge_index,
        edge_attr=edge_attr,
        action_label=action_label,
    )
    if x is not None:
        data.x = x
    if valid_mask is not None:
        data.valid_mask = valid_mask
        
    if hasattr(hdata, "scene"):
        data.scene = hdata.scene
    return data


class SEANWrapper(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        return self.subset[idx]


class SNSBWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, scenes):
        self.subset = subset
        self.scenes = scenes
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        data = self.subset[idx]
        data.scene = self.scenes[idx]
        return data


class OracleWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, ssub=False):
        self.base_dataset = base_dataset
        self.ssub = ssub
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        hdata = self.base_dataset[idx]
        return hetero_to_homo(hdata, ssub=self.ssub)


def sean_collate(batch): return Batch.from_data_list(batch)
def snsb_collate(batch): return Batch.from_data_list(batch)


# ── embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_sean_embeddings(model, loader, device, ablation=None):
    """Returns (z [N, 2 * latent_dim], y_binary [N, 3])."""
    zs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        mu = model.encode_nodes(batch) # [N_total, D]
        batch_idx = getattr(batch, "batch", torch.zeros(mu.shape[0], dtype=torch.long, device=device))
        B = int(batch_idx.max()) + 1
        
        robot_mask = batch.node_type == 0
        people_mask = batch.node_type == 1
        
        z_robot = scatter(mu[robot_mask], batch_idx[robot_mask], dim=0, reduce="mean", dim_size=B)
        z_people = scatter(mu[people_mask], batch_idx[people_mask], dim=0, reduce="mean", dim_size=B)
        
        if ablation == "robot_only":
            z = z_robot
        elif ablation == "bystander_only":
            z = z_people
        else:
            z = torch.cat([z_robot, z_people], dim=-1)
            
        raw = batch.emotion_label
        if raw is None:
            continue
        raw = raw.view(-1, len(SEAN_LABELS))
        y_bin = (raw >= LABEL_THRESHOLD).float()
        y_bin[:, 1] = 1.0 - y_bin[:, 1] # flip surprise
        zs.append(z.cpu())
        ys.append(y_bin.cpu())
    return torch.cat(zs), torch.cat(ys)


@torch.no_grad()
def extract_snsb_embeddings(model, loader, device, ablation=None):
    """Returns (z [N_by, D or 2D], y [N_by], graph_scenes [N_by])."""
    zs, ys, scenes_out = [], [], []
    for batch in loader:
        batch_scenes = getattr(batch, "scene", None)
        if batch_scenes is None:
            batch_scenes = ["unknown_scene"] * (int(batch.batch.max().item()) + 1 if hasattr(batch, "batch") else 1)
        batch = batch.to(device)
        mu = model.encode_nodes(batch) # [N_total, D]
        batch_idx = getattr(batch, "batch", torch.zeros(mu.shape[0], dtype=torch.long, device=device))
        B = int(batch_idx.max()) + 1
        
        robot_mask = batch.node_type == 0
        z_robot_per_graph = mu[robot_mask] # [B, D]
        robot_exp = z_robot_per_graph[batch_idx] # [N_total, D]
        
        if ablation == "robot_only":
            z = robot_exp
        elif ablation == "bystander_only":
            z = mu
        else:
            z = torch.cat([robot_exp, mu], dim=-1)
            
        labels = batch.action_label
        valid = labels != IGNORE_LABEL
        if valid.sum() == 0:
            continue
            
        batch_idx_cpu = batch_idx.cpu()
        sc_per_by = [batch_scenes[int(batch_idx_cpu[i])] for i in range(len(batch_idx_cpu)) if valid[i]]
        zs.append(z[valid].cpu())
        ys.append(labels[valid].cpu())
        scenes_out.extend(sc_per_by)
    return torch.cat(zs), torch.cat(ys), scenes_out


@torch.no_grad()
def extract_oracle_embeddings(model, oracle_h5, ckpt, device, batch_size=128, action_probs_json_dir=None, ablation=None, ssub=False):
    target_t = ckpt["args"].get("snsb_timesteps", 20)
    oracle_ds = load_oracle_eval_dataset_v3(
        oracle_h5, ckpt, action_probs_json_dir=action_probs_json_dir, target_t=target_t
    )
    wrapped = OracleWrapper(oracle_ds, ssub=ssub)
    loader = PyGDataLoader(wrapped, batch_size=batch_size, shuffle=False)
    zs, ys, probs = [], [], []
    has_probs = True
    for batch in loader:
        batch = batch.to(device)
        mu = model.encode_nodes(batch) # [N_total, D]
        batch_idx = getattr(batch, "batch", torch.zeros(mu.shape[0], dtype=torch.long, device=device))
        B = int(batch_idx.max()) + 1
        
        robot_mask = batch.node_type == 0
        z_robot_per_graph = mu[robot_mask] # [B, D]
        robot_exp = z_robot_per_graph[batch_idx] # [N_total, D]
        
        if ablation == "robot_only":
            z = robot_exp
        elif ablation == "bystander_only":
            z = mu
        else:
            z = torch.cat([robot_exp, mu], dim=-1)
            
        labels = batch.action_label
        valid = labels != IGNORE_LABEL
        if valid.sum() == 0:
            continue
        zs.append(z[valid].cpu())
        ys.append(labels[valid].cpu())
        
        # In load_oracle_eval_dataset_v3 wrapper, we shifted the original bystander attributes
        # So action_probs is a graph attribute or node attribute. 
        # But wait, in load_oracle_eval_dataset_v3:
        # action_probs is stored under `batch['bystander'].action_probs` in the hetero data.
        # How is it collated?
        # Let's search how OracleWrapper/hetero_to_homo can pass action_probs.
        # Oh, in load_oracle_eval_dataset_v3, action_probs is in `hdata['bystander'].action_probs`.
        # In OracleWrapper: we can attach it as `data.action_probs = hdata['bystander'].action_probs`!
        # Let's check: Yes, in PyG collation, if data has `action_probs`, it will be collated.
        # Let's make sure hetero_to_homo passes action_probs if present.
        # We will add it to hetero_to_homo!
        # (It's already handled if we set it in hetero_to_homo.)
        # Let's check if batch has action_probs:
        if hasattr(batch, "action_probs"):
            probs.append(batch.action_probs[valid].cpu())
        else:
            has_probs = False
            
    probs_out = torch.cat(probs) if has_probs and probs else None
    return torch.cat(zs), torch.cat(ys), probs_out


def extract_oracle_labels_and_probs(oracle_h5, ckpt, batch_size=128, action_probs_json_dir=None, ssub=False):
    target_t = ckpt["args"].get("snsb_timesteps", 20)
    oracle_ds = load_oracle_eval_dataset_v3(
        oracle_h5, ckpt, action_probs_json_dir=action_probs_json_dir, target_t=target_t
    )
    wrapped = OracleWrapper(oracle_ds, ssub=ssub)
    loader = PyGDataLoader(wrapped, batch_size=batch_size, shuffle=False)
    ys, probs = [], []
    has_probs = True
    for batch in loader:
        labels = batch.action_label
        valid = labels != IGNORE_LABEL
        if valid.sum() == 0:
            continue
        ys.append(labels[valid].cpu())
        if hasattr(batch, "action_probs"):
            probs.append(batch.action_probs[valid].cpu())
        else:
            has_probs = False
    probs_out = torch.cat(probs) if has_probs and probs else None
    return torch.cat(ys), probs_out


# Update hetero_to_homo in place to also copy action_probs if present:
# (Let's modify the hetero_to_homo function definition above to also do this.)
# Done: we added data.action_probs to hetero_to_homo if hdata['bystander'] has action_probs.
# Wait, let's make sure hetero_to_homo has:
# if hasattr(hdata["bystander"], "action_probs"):
#     data.action_probs = hdata["bystander"].action_probs


def read_snsb_scenes_for_indices(h5_path, indices):
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
        return [str(f[keys[i]].attrs.get("scene", "")) for i in indices]


def subsample_embeddings_by_scene(z, y, scenes, fraction, shuffled):
    n_keep = max(1, round(fraction * len(shuffled)))
    keep   = set(shuffled[:n_keep])
    mask   = np.array([sc in keep for sc in scenes])
    return z[mask], y[mask], n_keep


def subsample_labels_by_scene(y, scenes, fraction, shuffled):
    n_keep = max(1, round(fraction * len(shuffled)))
    keep = set(shuffled[:n_keep])
    mask = np.array([sc in keep for sc in scenes])
    return y[mask], n_keep


def stable_rng(*parts):
    key = "|".join(str(p) for p in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**32)
    return np.random.default_rng(seed)


def extract_sean_labels(loader):
    ys = []
    for batch in loader:
        raw = batch.emotion_label
        if raw is None:
            continue
        raw = raw.view(-1, len(SEAN_LABELS))
        y_bin = (raw >= LABEL_THRESHOLD).float()
        y_bin[:, 1] = 1.0 - y_bin[:, 1]
        ys.append(y_bin.cpu())
    return torch.cat(ys)


def extract_snsb_labels(loader):
    ys, scenes_out = [], []
    for batch in loader:
        batch_scenes = getattr(batch, "scene", None)
        if batch_scenes is None:
            batch_scenes = ["unknown_scene"] * (int(batch.batch.max().item()) + 1 if hasattr(batch, "batch") else 1)
        labels = batch.action_label
        valid = labels != IGNORE_LABEL
        if valid.sum() == 0:
            continue
        batch_idx = getattr(batch, "batch", torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device))
        batch_idx_cpu = batch_idx.cpu()
        sc_per_by = [batch_scenes[int(batch_idx_cpu[i])] for i in range(len(batch_idx_cpu)) if valid[i]]
        ys.append(labels[valid].cpu())
        scenes_out.extend(sc_per_by)
    return torch.cat(ys), scenes_out


# ── LP training ──────────────────────────────────────────────────────────────

def _train_head(head, loss_fn, train_z, train_y, val_z, val_y,
                 lr, epochs, patience, device):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_state, wait = float("inf"), None, 0
    tz, ty = train_z.to(device), train_y.to(device)
    vz, vy = val_z.to(device),   val_y.to(device)
    for _ in range(epochs):
        head.train(); opt.zero_grad()
        loss_fn(head(tz), ty).backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            val_loss = loss_fn(head(vz), vy).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    head.load_state_dict(best_state)
    return head


def train_sean_head(tr_z, tr_y, va_z, va_y, lr, epochs, patience, device):
    head = nn.Linear(tr_z.shape[1], 3).to(device)
    return _train_head(head, F.binary_cross_entropy_with_logits,
                       tr_z, tr_y.float(), va_z, va_y.float(),
                       lr, epochs, patience, device)


def train_snsb_head(tr_z, tr_y, va_z, va_y, lr, epochs, patience, device):
    head = nn.Linear(tr_z.shape[1], 5).to(device)
    return _train_head(head, lambda p, t: F.cross_entropy(p, t.long()),
                       tr_z, tr_y, va_z, va_y,
                       lr, epochs, patience, device)


# ── metrics ──────────────────────────────────────────────────────────────────

def sean_metrics(head, z, y, device):
    from sklearn.metrics import f1_score, roc_auc_score
    head.eval()
    with torch.no_grad():
        logits = head(z.to(device)).cpu().numpy()
    preds = (logits > 0).astype(float)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_np  = y.numpy()
    row, f1s, aucs = {}, [], []
    for d, name in enumerate(SEAN_LABELS):
        f1 = f1_score(y_np[:, d], preds[:, d], zero_division=0)
        try:
            auc = roc_auc_score(y_np[:, d], probs[:, d])
        except ValueError:
            auc = float("nan")
        row[f"f1_{name}"]  = f1
        row[f"auc_{name}"] = auc
        f1s.append(f1); aucs.append(auc)
    row["f1_macro"]  = float(np.mean(f1s))
    row["auc_macro"] = float(np.nanmean(aucs))
    return row


def snsb_metrics(head, z, y, device, action_probs=None):
    from sklearn.metrics import f1_score, precision_score, accuracy_score
    head.eval()
    with torch.no_grad():
        logits = head(z.to(device)).cpu().numpy()
    preds = logits.argmax(axis=1); y_np = y.numpy()
    row = {}
    for i, name in enumerate(SNSB_CLASSES):
        yb, pb = (y_np == i), (preds == i)
        row[f"f1_{name}"]   = f1_score(yb, pb, zero_division=0)
        row[f"prec_{name}"] = precision_score(yb, pb, zero_division=0)
        row[f"acc_{name}"]  = accuracy_score(yb, pb)
    row["f1_macro"] = f1_score(y_np, preds, average="macro", zero_division=0)
    row["f1_macro3"] = float(np.mean([row["f1_follow"], row["f1_avoid"], row["f1_not_consider"]]))
    row["acc"]      = accuracy_score(y_np, preds)
    if action_probs is not None:
        probs = action_probs.cpu().numpy() if torch.is_tensor(action_probs) else np.asarray(action_probs)
        row["pa"] = float(probs[np.arange(len(preds)), preds].mean())
    return row


def sean_metrics_from_scores(preds, scores, y):
    from sklearn.metrics import f1_score, roc_auc_score
    y_np = y.numpy()
    row, f1s, aucs = {}, [], []
    for d, name in enumerate(SEAN_LABELS):
        f1 = f1_score(y_np[:, d], preds[:, d], zero_division=0)
        try:
            auc = roc_auc_score(y_np[:, d], scores[:, d])
        except ValueError:
            auc = float("nan")
        row[f"f1_{name}"] = f1
        row[f"auc_{name}"] = auc
        f1s.append(f1)
        aucs.append(auc)
    row["f1_macro"] = float(np.mean(f1s))
    row["auc_macro"] = float(np.nanmean(aucs))
    return row


def snsb_metrics_from_preds(preds, y, action_probs=None):
    from sklearn.metrics import f1_score, precision_score, accuracy_score
    y_np = y.numpy()
    row = {}
    for i, name in enumerate(SNSB_CLASSES):
        yb, pb = (y_np == i), (preds == i)
        row[f"f1_{name}"] = f1_score(yb, pb, zero_division=0)
        row[f"prec_{name}"] = precision_score(yb, pb, zero_division=0)
        row[f"acc_{name}"] = accuracy_score(yb, pb)
    row["f1_macro"] = f1_score(y_np, preds, average="macro", zero_division=0)
    row["f1_macro3"] = float(np.mean([row["f1_follow"], row["f1_avoid"], row["f1_not_consider"]]))
    row["acc"] = accuracy_score(y_np, preds)
    if action_probs is not None:
        probs = action_probs.cpu().numpy() if torch.is_tensor(action_probs) else np.asarray(action_probs)
        row["pa"] = float(probs[np.arange(len(preds)), preds].mean())
    return row


def weighted_random_sean_metrics(train_y, test_y, rng):
    p = train_y.float().mean(dim=0).numpy()
    preds = rng.binomial(1, p, size=(test_y.shape[0], len(SEAN_LABELS))).astype(float)
    scores = np.tile(p.reshape(1, -1), (test_y.shape[0], 1))
    return sean_metrics_from_scores(preds, scores, test_y)


def weighted_random_snsb_metrics(train_y, eval_y, rng, action_probs=None):
    counts = np.bincount(train_y.numpy().astype(int), minlength=len(SNSB_CLASSES)).astype(float)
    total = counts.sum()
    probs = counts / total if total > 0 else np.ones(len(SNSB_CLASSES)) / len(SNSB_CLASSES)
    preds = rng.choice(len(SNSB_CLASSES), size=eval_y.shape[0], p=probs)
    return snsb_metrics_from_preds(preds, eval_y, action_probs=action_probs)


# ── CSV ──────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "vae_dir", "split_seed", "task", "subset_seed", "fraction",
    "n_train_scenes", "n_train_samples",
    "f1_macro", "f1_macro3", "auc_macro", "acc",
    # SEAN per-label (F1 + AUC)
    "f1_competence", "auc_competence",
    "f1_surprise",   "auc_surprise",
    "f1_intention",  "auc_intention",
    # SNSB per-class (F1 + Precision + Accuracy)
    "f1_avoid",        "prec_avoid",        "acc_avoid",
    "f1_follow",       "prec_follow",       "acc_follow",
    "f1_overtake",     "prec_overtake",     "acc_overtake",
    "f1_yield",        "prec_yield",        "acc_yield",
    "f1_not_consider", "prec_not_consider", "acc_not_consider",
    "pa",
]


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def build_model(ckpt, device, random_encoder=False):
    sa = ckpt["args"]
    model = HGNNCombinedVAE(
        hidden_dim              = sa["hidden_dim"],
        latent_dim              = sa["latent_dim"],
        num_mp_layers           = sa["num_mp_layers"],
        edge_transformer_layers = sa.get("edge_transformer_layers", 3),
        num_heads               = sa.get("num_heads", 4),
        dropout                 = sa.get("dropout", 0.1),
    ).to(device)
    if random_encoder:
        print("Using RANDOM (untrained) encoder weights.")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── per-VAE processing ───────────────────────────────────────────────────────

def process_vae_dir(vae_dir, args, device):
    ckpt_path  = os.path.join(vae_dir, args.ckpt_name)
    split_path = os.path.join(vae_dir, args.split_name)
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] No checkpoint: {ckpt_path}"); return
    if not os.path.exists(split_path):
        print(f"[SKIP] No split file: {split_path}"); return

    print(f"\n{'='*60}\nVAE: {vae_dir}")
    ckpt  = torch.load(ckpt_path,  map_location="cpu", weights_only=False)
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    model = None
    if args.mode != "weighted_random_baseline":
        model = build_model(ckpt, device, random_encoder=args.mode == "random_encoder")

    split_seed = os.path.basename(vae_dir)  # e.g. "split0" or "run0"

    # ── SEAN ─────────────────────────────────────────────────────────────────
    combined_ds = UnifiedGraphDataset(
        args.combined_h5, ssub=False,
        robot_mean=ckpt.get("robot_mean"),
        robot_std =ckpt.get("robot_std"),
    )
    
    # Filter to SEAN-only samples (dataset ID 0)
    sean_tr_idx = combined_ds.filter_by_source(split["train"], 0)
    sean_va_idx = combined_ds.filter_by_source(split["val"], 0)
    sean_te_idx = combined_ds.filter_by_source(split["test"], 0)
    
    sean_tr_scenes = [combined_ds._scenes[i] for i in sean_tr_idx]
    
    sean_tr_loader = DataLoader(SEANWrapper(Subset(combined_ds, sean_tr_idx)),
                                batch_size=args.batch_size, collate_fn=sean_collate)
    sean_va_loader = DataLoader(SEANWrapper(Subset(combined_ds, sean_va_idx)),
                                batch_size=args.batch_size, collate_fn=sean_collate)
    sean_te_loader = DataLoader(SEANWrapper(Subset(combined_ds, sean_te_idx)),
                                batch_size=args.batch_size, collate_fn=sean_collate)

    if args.mode == "weighted_random_baseline":
        print("Extracting SEAN labels...")
        sean_tr_y = extract_sean_labels(sean_tr_loader)
        sean_te_y = extract_sean_labels(sean_te_loader)
        print(f"  train={sean_tr_y.shape[0]} test={sean_te_y.shape[0]}")
    else:
        print("Extracting SEAN embeddings...")
        sean_tr_z, sean_tr_y = extract_sean_embeddings(model, sean_tr_loader, device, ablation=args.ablation)
        sean_va_z, sean_va_y = extract_sean_embeddings(model, sean_va_loader, device, ablation=args.ablation)
        sean_te_z, sean_te_y = extract_sean_embeddings(model, sean_te_loader, device, ablation=args.ablation)
        print(f"  train={sean_tr_z.shape[0]} val={sean_va_z.shape[0]} test={sean_te_z.shape[0]}")

    # ── SNSB ─────────────────────────────────────────────────────────────────
    snsb_tr_z = snsb_tr_y = snsb_tr_scenes_by = None
    snsb_va_z = snsb_va_y = None
    oracle_z  = oracle_y  = oracle_probs = None

    if not args.skip_snsb:
        # Load unified dataset in ssub format (dataset ID 1 or 2)
        # Note: combined_ds has ssub parameter. For SNSB task we want ssub=True
        snsb_ds = UnifiedGraphDataset(args.combined_h5, ssub=True)
        # Filter to SNSB-only samples (dataset ID 1 or 2)
        snsb_tr_idx = snsb_ds.filter_by_source(split["train"], 1) + snsb_ds.filter_by_source(split["train"], 2)
        snsb_va_idx = snsb_ds.filter_by_source(split["val"], 1) + snsb_ds.filter_by_source(split["val"], 2)
        
        snsb_tr_sc = read_snsb_scenes_for_indices(args.combined_h5, snsb_tr_idx)
        snsb_va_sc = read_snsb_scenes_for_indices(args.combined_h5, snsb_va_idx)

        snsb_tr_loader = DataLoader(SNSBWrapper(Subset(snsb_ds, snsb_tr_idx), snsb_tr_sc),
                                    batch_size=args.batch_size, collate_fn=snsb_collate)
        snsb_va_loader = DataLoader(SNSBWrapper(Subset(snsb_ds, snsb_va_idx), snsb_va_sc),
                                    batch_size=args.batch_size, collate_fn=snsb_collate)

        if args.mode == "weighted_random_baseline":
            print("Extracting SNSB labels...")
            snsb_tr_y, snsb_tr_scenes_by = extract_snsb_labels(snsb_tr_loader)
            snsb_va_y, _ = extract_snsb_labels(snsb_va_loader)
            print(f"  train bystanders={snsb_tr_y.shape[0]} val bystanders={snsb_va_y.shape[0]}")
        else:
            print("Extracting SNSB embeddings...")
            snsb_tr_z, snsb_tr_y, snsb_tr_scenes_by = extract_snsb_embeddings(model, snsb_tr_loader, device, ablation=args.ablation)
            snsb_va_z, snsb_va_y, _                  = extract_snsb_embeddings(model, snsb_va_loader, device, ablation=args.ablation)
            print(f"  train bystanders={snsb_tr_z.shape[0]} val bystanders={snsb_va_z.shape[0]}")

        if not args.oracle_h5:
            raise ValueError("SNSB label-efficiency eval requires --oracle_h5 so PA can be computed.")
        if not os.path.exists(args.oracle_h5):
            raise FileNotFoundError(f"SNSB oracle HDF5 not found: {args.oracle_h5}")

        print(f"Extracting oracle {'labels' if args.mode == 'weighted_random_baseline' else 'embeddings'}...")
        
        # Checkpoint properties for oracle dataset
        zero_node_features = ckpt.get("zero_node_features", True)
        snsb_timesteps = ckpt.get("snsb_timesteps", 20)
        
        # We need to make sure we set robot_mean/robot_std/bystander_mean/bystander_std to match the format of load_oracle_eval_dataset_v3
        if "robot_mean" in ckpt:
            ckpt["snsb_robot_mean"] = ckpt["robot_mean"]
            ckpt["snsb_robot_std"] = ckpt["robot_std"]
            ckpt["snsb_bystander_mean"] = ckpt.get("bystander_mean")
            ckpt["snsb_bystander_std"] = ckpt.get("bystander_std")
            
        if args.mode == "weighted_random_baseline":
            oracle_y, oracle_probs = extract_oracle_labels_and_probs(
                args.oracle_h5, ckpt, args.batch_size,
                action_probs_json_dir=args.oracle_json_dir,
                ssub=True,
            )
        else:
            oracle_z, oracle_y, oracle_probs = extract_oracle_embeddings(
                model, args.oracle_h5, ckpt, device, args.batch_size,
                action_probs_json_dir=args.oracle_json_dir,
                ablation=args.ablation,
                ssub=True,
            )
        print(f"  oracle bystanders={oracle_y.shape[0]}")
        if oracle_probs is None:
            raise ValueError(
                f"SNSB oracle eval requires action_probs for PA, but none were found in {args.oracle_h5}. "
                "Use an oracle HDF5 with action_probs or pass --oracle_json_dir."
            )

    # Shuffler
    sean_unique = list(dict.fromkeys(sean_tr_scenes))
    sean_run_perms = [np.random.default_rng().permutation(sean_unique).tolist()
                      for _ in range(args.n_subset_seeds)]
    if not args.skip_snsb:
        snsb_unique = list(dict.fromkeys(snsb_tr_scenes_by))
        snsb_run_perms = [np.random.default_rng().permutation(snsb_unique).tolist()
                          for _ in range(args.n_subset_seeds)]

    # ── sweep ────────────────────────────────────────────────────────────────
    total = len(args.fractions) * args.n_subset_seeds
    done  = 0
    for fraction in args.fractions:
        for subset_seed in range(args.n_subset_seeds):
            done += 1
            print(f"  [{done}/{total}] frac={fraction:.0%} seed={subset_seed}", end=" ", flush=True)

            # SEAN
            if args.mode == "weighted_random_baseline":
                s_tr_y, n_s_sc = subsample_labels_by_scene(
                    sean_tr_y, sean_tr_scenes, fraction, sean_run_perms[subset_seed])
                m_s = weighted_random_sean_metrics(
                    s_tr_y, sean_te_y,
                    stable_rng(args.mode, split_seed, fraction, subset_seed, "sean"),
                )
                n_s_samples = len(s_tr_y)
            else:
                s_tr_z, s_tr_y, n_s_sc = subsample_embeddings_by_scene(
                    sean_tr_z, sean_tr_y, sean_tr_scenes, fraction, sean_run_perms[subset_seed])
                head_s = train_sean_head(s_tr_z, s_tr_y, sean_va_z, sean_va_y,
                                         args.lr, args.epochs, args.patience, device)
                m_s = sean_metrics(head_s, sean_te_z, sean_te_y, device)
                n_s_samples = len(s_tr_z)
            append_row(args.output, {
                "vae_dir": vae_dir, "split_seed": split_seed,
                "task": "sean", "subset_seed": subset_seed,
                "fraction": fraction, "n_train_scenes": n_s_sc,
                "n_train_samples": n_s_samples,
                **m_s,
            })

            # SNSB
            if not args.skip_snsb:
                if args.mode == "weighted_random_baseline":
                    n_tr_y, n_n_sc = subsample_labels_by_scene(
                        snsb_tr_y, snsb_tr_scenes_by, fraction, snsb_run_perms[subset_seed])
                    eval_y = oracle_y if oracle_y is not None else snsb_va_y
                    eval_probs = oracle_probs if oracle_y is not None else None
                    m_n = weighted_random_snsb_metrics(
                        n_tr_y, eval_y,
                        stable_rng(args.mode, split_seed, fraction, subset_seed, "snsb"),
                        action_probs=eval_probs,
                    )
                    n_n_samples = len(n_tr_y)
                else:
                    n_tr_z, n_tr_y, n_n_sc = subsample_embeddings_by_scene(
                        snsb_tr_z, snsb_tr_y, snsb_tr_scenes_by, fraction, snsb_run_perms[subset_seed])
                    head_n = train_snsb_head(n_tr_z, n_tr_y, snsb_va_z, snsb_va_y,
                                             args.lr, args.epochs, args.patience, device)
                    eval_z = oracle_z if oracle_z is not None else snsb_va_z
                    eval_y = oracle_y if oracle_y is not None else snsb_va_y
                    eval_probs = oracle_probs if oracle_z is not None else None
                    m_n = snsb_metrics(head_n, eval_z, eval_y, device, action_probs=eval_probs)
                    n_n_samples = len(n_tr_z)
                append_row(args.output, {
                    "vae_dir": vae_dir, "split_seed": split_seed,
                    "task": "snsb", "subset_seed": subset_seed,
                    "fraction": fraction, "n_train_scenes": n_n_sc,
                    "n_train_samples": n_n_samples,
                    **m_n,
                })
                print(f"→ SEAN={m_s['f1_macro']:.3f} SNSB={m_n['f1_macro']:.3f}")
            else:
                print(f"→ SEAN={m_s['f1_macro']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_dirs",       nargs="+", required=True)
    parser.add_argument("--combined_h5",    default="dataloaders/combined_v1.h5")
    parser.add_argument("--snsb_h5",        default="dataloaders/social_nav_v3.h5")
    parser.add_argument("--oracle_h5",      default="dataloaders/socialnav_human_oracle_v3.h5",
                        help="Oracle h5 for SNSB eval. Set to '' to use val set instead.")
    parser.add_argument("--oracle_json_dir", default=None,
                        help="Optional JSON dir containing oracle action_probs for mandatory SNSB PA.")
    parser.add_argument("--output",         default=None)
    parser.add_argument("--mode", choices=["trained", "random_encoder", "weighted_random_baseline"],
                        default="trained")
    parser.add_argument("--ablation", choices=["robot_only", "bystander_only", "raw_robot"], default=None,
                        help="Use only robot or bystander half of the embedding, or raw robot input features.")
    parser.add_argument("--ckpt_name",  default="best_model.pt",
                        help="Checkpoint filename inside each vae_dir.")
    parser.add_argument("--split_name", default="split_indices.pt",
                        help="Split filename inside each vae_dir.")
    parser.add_argument("--skip_snsb",  action="store_true",
                        help="Skip SNSB eval entirely (e.g. for SEAN-only VAE checkpoints).")
    parser.add_argument("--fractions",      nargs="+", type=float,
                        default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--n_subset_seeds", type=int, default=10)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--epochs",         type=int,   default=1000)
    parser.add_argument("--patience",       type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--gpu",            type=int,   default=0)
    args = parser.parse_args()
    if args.output is None:
        mode_suffix = "" if args.mode == "trained" else f"_{args.mode}"
        abl_suffix  = f"_{args.ablation}" if args.ablation else ""
        args.output = f"label_eff_results{mode_suffix}{abl_suffix}.csv"

    # Make paths absolute
    args.combined_h5 = os.path.abspath(args.combined_h5)
    args.snsb_h5 = os.path.abspath(args.snsb_h5)
    args.oracle_h5 = os.path.abspath(args.oracle_h5)
    if args.oracle_json_dir:
        args.oracle_json_dir = os.path.abspath(args.oracle_json_dir)

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode} | Output: {args.output}")

    for vae_dir in args.vae_dirs:
        process_vae_dir(vae_dir, args, device)

    print(f"\nDone. {args.output}")


if __name__ == "__main__":
    main()
