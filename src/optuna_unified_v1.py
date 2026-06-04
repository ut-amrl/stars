"""
Heterogeneous GNN-VAE pipeline — combined SEAN+SNSB Optuna search.

Optuna hyperparameter search for UnifiedGNNVAE v3 (final split, adaptive SNSB sampling).

Each trial:
  Stage 1 — train UnifiedGNNVAE (truncated + early stopping)
  Objective — best combined VAE validation loss

Fixed per study: hidden_dim, latent_dim, num_mp_layers=2, batch_size=32,
       VAE weight_decay=1e-4
Tuned: vae_lr, beta_max {0.1,0.5,0.9}, warmup_epochs, dropout,
       adaptive_warmup_epochs, adaptive_max_alpha
Fixed: adaptive_ramp_epochs=100

Run parallel workers (one per GPU):
    GPUS=(1 2 4 5 6)
    for i in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} python optuna_unified_v1.py \\
            --study_name unified_real_adaptive_v2 \\
            --storage sqlite:///optuna_unified_real_adaptive_v2.db &
    done
    wait

Report best:
    python optuna_unified_v1.py --mode report \\
        --study_name unified_real_adaptive_v2 \\
        --storage sqlite:///optuna_unified_real_adaptive_v2.db
"""

import os
import sys
import json
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloaders.unified_graph_dataset_v1 import DATASET_SEAN, DATASET_SSUB_REAL, UnifiedGraphDataset
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.unified_gnn_vae_v1 import EDGE_TYPES as UNIFIED_EDGE_TYPES, NODE_TYPES, UnifiedGNNVAE

# ── fixed architecture defaults ───────────────────────────────────────────────
NUM_MP_LAYERS = 2
VAE_WEIGHT_DECAY = 1e-4
PROBE_WEIGHT_DECAY = 1e-4
ADAPTIVE_RAMP_EPOCHS = 100
SEAN_ID = DATASET_SEAN
SNSB_ID = DATASET_SSUB_REAL

EDGE_LOSS_WEIGHTS = {
    "sean": {
        ("robot", "to", "bystander"): 3.5,
        ("bystander", "to", "robot"): 3.5,
        ("bystander", "to", "bystander"): 0.1,
    },
    "snsb": {
        ("robot", "to", "bystander"): 3.0,
        ("bystander", "to", "robot"): 3.0,
        ("bystander", "to", "bystander"): 0.5,
    },
}

ACTION_CLASS_NAMES = ["avoid", "follow", "overtake", "yield", "not_consider"]
NUM_CLASSES        = len(ACTION_CLASS_NAMES)
VAE_ES_PATIENCE    = 20
PROBE_ES_PATIENCE  = 20
N_REAL             = 3052  # indices 0..3051 = real in social_nav_v3.h5


# ── helpers ───────────────────────────────────────────────────────────────────

def dataset_key(batch):
    did = batch.dataset_id
    if torch.is_tensor(did):
        did = int(torch.unique(did.view(-1).long()).item())
    return "sean" if int(did) == SEAN_ID else "snsb"


def homo_to_hetero(data):
    """Convert final combined_v1.h5 homogeneous SEAN graph to UnifiedGNNVAE HeteroData."""
    node_type = data.node_type
    robot_mask = node_type == 0
    bystander_mask = node_type == 1

    n_total = node_type.shape[0]
    n_robots = int(robot_mask.sum())
    n_bystanders = int(bystander_mask.sum())

    local_idx = torch.full((n_total,), -1, dtype=torch.long)
    local_idx[robot_mask.nonzero(as_tuple=True)[0]] = torch.arange(n_robots, dtype=torch.long)
    local_idx[bystander_mask.nonzero(as_tuple=True)[0]] = torch.arange(n_bystanders, dtype=torch.long)

    hdata = HeteroData()
    hdata.dataset_id = torch.tensor([SEAN_ID], dtype=torch.long)
    hdata["robot"].x = data.x[robot_mask]
    hdata["bystander"].num_nodes = n_bystanders
    if hasattr(data, "action_label"):
        hdata["bystander"].y = data.action_label[bystander_mask].long()

    src_nodes = data.edge_index[0]
    dst_nodes = data.edge_index[1]
    src_types = node_type[src_nodes]
    dst_types = node_type[dst_nodes]

    for (st, dt), et in [
        ((0, 1), ("robot", "to", "bystander")),
        ((1, 0), ("bystander", "to", "robot")),
        ((1, 1), ("bystander", "to", "bystander")),
    ]:
        mask = (src_types == st) & (dst_types == dt)
        if mask.any():
            hdata[et].edge_index = torch.stack(
                [local_idx[src_nodes[mask]], local_idx[dst_nodes[mask]]], dim=0
            )
            hdata[et].edge_attr = data.edge_attr[mask]
    return hdata


class SEANHeteroWrapper(Dataset):
    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def len(self):
        return len(self.subset)

    def get(self, idx):
        return homo_to_hetero(self.subset[idx])


class DropSNSBNodeFeatures(Dataset):
    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def len(self):
        return len(self.subset)

    def get(self, idx):
        data = self.subset[idx]
        data["robot"].num_nodes = int(data["robot"].x.shape[0])
        data["bystander"].num_nodes = int(data["bystander"].x.shape[0])
        data["robot"].x = None
        data["bystander"].x = None
        return data


def get_snsb_split_indices(combined_ds, split, snsb_ds):
    """Map final split SNSB scenes in combined_v1.h5 onto social_nav_v3.h5 real samples."""
    snsb_train_combined = [i for i in split["train"] if combined_ds._dataset_ids[i] == SNSB_ID]
    snsb_val_combined = [i for i in split["val"] if combined_ds._dataset_ids[i] == SNSB_ID]

    train_scenes = set(combined_ds._scenes[i] for i in snsb_train_combined)
    val_scenes = set(combined_ds._scenes[i] for i in snsb_val_combined)
    assert not (train_scenes & val_scenes), "Scene overlap between SNSB train and val scenes."

    snsb_train_idx, snsb_val_idx, unmatched = [], [], []
    for i in range(N_REAL):
        sample = snsb_ds.base_dataset[i]
        scene = getattr(sample, "scene", None) or ""
        if scene in train_scenes:
            snsb_train_idx.append(i)
        elif scene in val_scenes:
            snsb_val_idx.append(i)
        else:
            unmatched.append(i)

    assert len(snsb_train_idx) == len(snsb_train_combined), (
        f"SNSB train count mismatch: combined={len(snsb_train_combined)} social_nav_v3={len(snsb_train_idx)}"
    )
    assert len(snsb_val_idx) == len(snsb_val_combined), (
        f"SNSB val count mismatch: combined={len(snsb_val_combined)} social_nav_v3={len(snsb_val_idx)}"
    )
    print(
        f"SNSB scene split verified: train={len(snsb_train_idx)} val={len(snsb_val_idx)} "
        f"unmatched={len(unmatched)} (oracle/test, expected)"
    )
    return snsb_train_idx, snsb_val_idx


def compute_vae_loss(robot_recon, edge_recons, batch, mu_dict, logvar_dict, beta):
    key    = dataset_key(batch)
    if robot_recon is not None:
        device = robot_recon.device
        recon_robot = F.mse_loss(robot_recon, batch["robot"].x)
    else:
        device = next(v.device for v in mu_dict.values() if v.numel() > 0)
        recon_robot = torch.tensor(0.0, device=device)

    edge_losses, edge_weights = [], []
    for et in UNIFIED_EDGE_TYPES:
        if et in edge_recons and hasattr(batch[et], "edge_attr"):
            edge_losses.append(F.mse_loss(edge_recons[et], batch[et].edge_attr))
            edge_weights.append(EDGE_LOSS_WEIGHTS[key][et])
    if edge_losses:
        w           = torch.tensor(edge_weights, device=device)
        recon_edges = (w * torch.stack(edge_losses)).sum() / w.sum()
    else:
        recon_edges = torch.tensor(0.0, device=device)

    all_mu     = torch.cat([mu_dict[t] for t in NODE_TYPES if mu_dict[t].shape[0] > 0])
    all_logvar = torch.cat([logvar_dict[t] for t in NODE_TYPES if logvar_dict[t].shape[0] > 0])
    kl         = -0.5 * torch.mean(1 + all_logvar - all_mu.pow(2) - all_logvar.exp())

    return recon_robot + recon_edges + beta * kl



def extract_bystander_z(model, loader, device):
    all_z, all_y = [], []
    model.eval()
    print(f"  [extract_bystander_z] loader batches={len(loader)} dataset_size={len(loader.dataset)}")
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            try:
                batch = batch.to(device)
                mu_dict, _ = model.encoder(batch)
                bst_idx    = batch["bystander"].batch
                z = torch.cat([mu_dict["robot"][bst_idx], mu_dict["bystander"]], dim=1)
                all_z.append(z.cpu())
                all_y.append(batch["bystander"].y.long().cpu())
            except Exception as _e:
                import traceback as _tb
                print(f"  [extract_bystander_z] batch {batch_i} FAILED: {type(_e).__name__}: {_e}")
                _tb.print_exc()
                raise
    if not all_z:
        raise RuntimeError(
            f"extract_bystander_z: 0 samples collected from loader "
            f"(batches={len(loader)}, dataset_size={len(loader.dataset)})"
        )
    return torch.cat(all_z), torch.cat(all_y)


ACTIVE_CLASSES = [0, 1, 4]  # avoid, follow, not_consider — overtake/yield always 0

def val_f1_macro(head, val_z, val_y):
    head.eval()
    with torch.no_grad():
        preds = head(val_z).argmax(dim=1).cpu().numpy()
    return float(f1_score(val_y.cpu().numpy(), preds, labels=ACTIVE_CLASSES, average="macro", zero_division=0))


def val_ce_loss(head, val_z, val_y):
    head.eval()
    with torch.no_grad():
        return float(F.cross_entropy(head(val_z), val_y).item())


def build_stratified_weights(snsb_train):
    """Per-graph inverse-sqrt rarest-class weights, normalized for WeightedRandomSampler."""
    from collections import Counter

    global_counts = Counter()
    per_graph_labels = []
    for i in range(len(snsb_train)):
        sample = snsb_train[i]
        labels = sample["bystander"].y.long().tolist()
        per_graph_labels.append(labels)
        for c in labels:
            global_counts[c] += 1

    weights = []
    for labels in per_graph_labels:
        if not labels:
            weights.append(1.0)
        else:
            rarest = min(labels, key=lambda c: global_counts.get(c, 1))
            weights.append(1.0 / (global_counts[rarest] ** 0.5))
    weights_t = torch.tensor(weights, dtype=torch.float64)
    total = weights_t.sum()
    if total <= 0 or not torch.isfinite(total):
        return torch.full((len(snsb_train),), 1.0 / max(1, len(snsb_train)), dtype=torch.float64)
    print(f"  Stratified weights: {len(snsb_train)} graphs, class counts={dict(global_counts)}")
    return weights_t / total


def compute_snsb_per_sample_losses(model, snsb_train, device, batch_size):
    """Compute per-graph SNSB reconstruction losses for adaptive sampler blending."""
    loader = DataLoader(snsb_train, batch_size=batch_size, shuffle=False, num_workers=4)
    was_training = model.training
    model.eval()
    all_losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            robot_recon, edge_recons, mu_dict, logvar_dict = model(batch)

            bsz = int(batch["robot"].num_nodes)
            per_graph_robot = torch.zeros(bsz, device=device)
            if robot_recon is not None:
                robot_node_loss = F.mse_loss(robot_recon, batch["robot"].x, reduction="none").mean(dim=-1)
                robot_batch = batch["robot"].batch if hasattr(batch["robot"], "batch") else torch.arange(bsz, device=device)
                per_graph_robot.scatter_add_(0, robot_batch, robot_node_loss)

            edge_type_sum = torch.zeros(bsz, device=device)
            edge_type_weight_sum = torch.zeros(bsz, device=device)
            for et in UNIFIED_EDGE_TYPES:
                if et not in edge_recons or not hasattr(batch[et], "edge_attr"):
                    continue
                edge_index = batch[et].edge_index
                if edge_index.shape[1] == 0:
                    continue
                per_edge = F.mse_loss(edge_recons[et], batch[et].edge_attr, reduction="none").mean(dim=(-2, -1))
                src_type = et[0]
                edge_batch = batch[src_type].batch[edge_index[0]]
                edge_weight = float(EDGE_LOSS_WEIGHTS["snsb"].get(et, 1.0))

                per_type_sum = torch.zeros(bsz, device=device)
                per_type_count = torch.zeros(bsz, device=device)
                per_type_sum.scatter_add_(0, edge_batch, per_edge)
                per_type_count.scatter_add_(0, edge_batch, torch.ones_like(per_edge))
                has_type = per_type_count > 0
                per_type_mean = per_type_sum / per_type_count.clamp(min=1.0)

                edge_type_sum[has_type] += per_type_mean[has_type] * edge_weight
                edge_type_weight_sum[has_type] += edge_weight

            per_graph_edge = edge_type_sum / edge_type_weight_sum.clamp(min=1e-8)
            all_losses.extend((per_graph_robot + per_graph_edge).detach().cpu().tolist())

    model.train(was_training)
    losses_t = torch.tensor(all_losses, dtype=torch.float64)
    if losses_t.numel() != len(snsb_train):
        raise RuntimeError(f"Adaptive SNSB loss count mismatch: got {losses_t.numel()} for {len(snsb_train)} graphs.")
    total = losses_t.sum()
    if total <= 0 or not torch.isfinite(total):
        return torch.full((len(snsb_train),), 1.0 / max(1, len(snsb_train)), dtype=torch.float64)
    return losses_t / total


# ── Stage 1: VAE ─────────────────────────────────────────────────────────────

def train_vae(trial, args, sean_dataset, snsb_dataset, snsb_train_idx, snsb_val_idx,
              sean_train_idx, sean_val_idx, device):
    vae_lr = trial.suggest_float("vae_lr", 1e-4, 1e-2, log=True)
    beta_max = trial.suggest_categorical("beta_max", [0.1, 0.5, 0.9])
    warmup_epochs = trial.suggest_categorical("warmup_epochs", [10, 25, 50])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    adaptive_warmup_epochs = trial.suggest_categorical("adaptive_warmup_epochs", [10, 20])
    adaptive_max_alpha = trial.suggest_categorical("adaptive_max_alpha", [0.25, 0.5, 0.75])
    adaptive_ramp_epochs = ADAPTIVE_RAMP_EPOCHS  # fixed at 100
    batch_size = 32  # fixed

    snsb_train = DropSNSBNodeFeatures(torch.utils.data.Subset(snsb_dataset, snsb_train_idx))
    snsb_val = DropSNSBNodeFeatures(torch.utils.data.Subset(snsb_dataset, snsb_val_idx))
    sean_train = SEANHeteroWrapper(torch.utils.data.Subset(sean_dataset, sean_train_idx))
    sean_val = SEANHeteroWrapper(torch.utils.data.Subset(sean_dataset, sean_val_idx))

    stratified_weights = build_stratified_weights(snsb_train)
    snsb_sampler = WeightedRandomSampler(stratified_weights, num_samples=len(snsb_train), replacement=True)

    snsb_train_loader = DataLoader(snsb_train, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, sampler=snsb_sampler)
    snsb_val_loader   = DataLoader(snsb_val,   batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    sean_train_loader = DataLoader(sean_train, batch_size=batch_size, shuffle=True,  num_workers=4, persistent_workers=True)
    sean_val_loader   = DataLoader(sean_val,   batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    step_count = max(len(sean_train_loader), len(snsb_train_loader))

    model = UnifiedGNNVAE(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_mp_layers=NUM_MP_LAYERS,
        dropout=dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=vae_lr, weight_decay=VAE_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val  = float("inf")
    best_state= None
    es_counter= 0

    def _next(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    for epoch in range(1, args.vae_epochs + 1):
        beta = min(beta_max, beta_max * epoch / max(1, warmup_epochs))

        if epoch > adaptive_warmup_epochs:
            loss_weights = compute_snsb_per_sample_losses(model, snsb_train, device, batch_size)
            epochs_since = epoch - adaptive_warmup_epochs
            alpha = min(adaptive_max_alpha, adaptive_max_alpha * epochs_since / max(adaptive_ramp_epochs, 1))
            combined = (1.0 - alpha) * stratified_weights + alpha * loss_weights
            snsb_sampler.weights = (combined / combined.sum()).double()

        model.train()
        snsb_it = iter(snsb_train_loader)
        sean_it = iter(sean_train_loader)
        for _ in range(step_count):
            snsb_b, snsb_it = _next(snsb_it, snsb_train_loader)
            sean_b, sean_it = _next(sean_it, sean_train_loader)
            snsb_b, sean_b  = snsb_b.to(device), sean_b.to(device)
            optimizer.zero_grad()
            rr_s, er_s, mu_s, lv_s = model(snsb_b)
            rr_e, er_e, mu_e, lv_e = model(sean_b)
            loss = (compute_vae_loss(rr_s, er_s, snsb_b, mu_s, lv_s, beta) +
                    compute_vae_loss(rr_e, er_e, sean_b, mu_e, lv_e, beta)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            snsb_v_losses = []
            for b in snsb_val_loader:
                b = b.to(device)
                rr, er, mu, lv = model(b)
                snsb_v_losses.append(compute_vae_loss(rr, er, b, mu, lv, beta).item())
            snsb_v = float(np.mean(snsb_v_losses))

            sean_v_losses = []
            for b in sean_val_loader:
                b = b.to(device)
                rr, er, mu, lv = model(b)
                sean_v_losses.append(compute_vae_loss(rr, er, b, mu, lv, beta).item())
            sean_v = float(np.mean(sean_v_losses))
        val_total = (snsb_v + sean_v) / 2
        scheduler.step(val_total)

        if epoch % 20 == 0:
            print(f"  [Trial {trial.number}] VAE epoch {epoch:04d} | val={val_total:.4f} (snsb={snsb_v:.4f} sean={sean_v:.4f}) | es={es_counter}/{VAE_ES_PATIENCE}")
        trial.set_user_attr("vae_epoch", epoch)
        trial.set_user_attr("vae_val", round(val_total, 5))
        trial.report(val_total, epoch)
        if trial.should_prune():
            trial.set_user_attr("pruned_with_vae_val", round(best_val, 5))
            raise optuna.exceptions.TrialPruned()

        if val_total < best_val:
            best_val   = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= VAE_ES_PATIENCE:
                trial.set_user_attr("vae_stop_epoch", epoch)
                break

    model.load_state_dict(best_state)
    trial.set_user_attr("best_vae_val", round(best_val, 5))
    return model, best_val


# ── Stage 2: SNSB probe ───────────────────────────────────────────────────────

def train_probe(trial, args, model, snsb_dataset, snsb_train_idx, snsb_val_idx, device):
    probe_lr   = trial.suggest_float("probe_lr", 1e-4, 1e-2, log=True)
    batch_size = getattr(args, "batch_size", 32)
    latent_dim = model.latent_dim

    snsb_train = torch.utils.data.Subset(snsb_dataset, snsb_train_idx)
    snsb_val   = torch.utils.data.Subset(snsb_dataset, snsb_val_idx)
    train_loader = DataLoader(snsb_train, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    val_loader   = DataLoader(snsb_val,   batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    train_z, train_y = extract_bystander_z(model, train_loader, device)
    val_z,   val_y   = extract_bystander_z(model, val_loader,   device)

    train_z, train_y = train_z.to(device), train_y.to(device)
    val_z, val_y     = val_z.to(device), val_y.to(device)

    head      = torch.nn.Linear(2 * latent_dim, NUM_CLASSES).to(device)
    optimizer = AdamW(head.parameters(), lr=probe_lr, weight_decay=PROBE_WEIGHT_DECAY)

    best_loss = float("inf")
    best_f1 = -1.0
    best_state = None
    es_counter = 0

    for epoch in range(1, args.probe_epochs + 1):
        head.train()
        perm = torch.randperm(train_z.shape[0], device=device)
        for i in range(0, train_z.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            F.cross_entropy(head(train_z[idx]), train_y[idx]).backward()
            optimizer.step()

        loss = val_ce_loss(head, val_z, val_y)
        f1 = val_f1_macro(head, val_z, val_y)

        trial.report(loss, args.vae_epochs + epoch)
        if trial.should_prune():
            if best_loss < float("inf"):
                trial.set_user_attr("pruned_with_best_val", round(best_loss, 5))
                return best_loss
            raise optuna.exceptions.TrialPruned()

        if loss < best_loss:
            best_loss = loss
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= PROBE_ES_PATIENCE:
                trial.set_user_attr("probe_stop_epoch", epoch)
                break

    trial.set_user_attr("best_val_f1_macro", best_f1)
    if best_state is not None:
        head.load_state_dict(best_state)
    return best_loss


# ── objective ─────────────────────────────────────────────────────────────────

def objective(trial, args, sean_dataset, snsb_dataset,
              snsb_train_idx, snsb_val_idx, sean_train_idx, sean_val_idx, device):
    import traceback as _tb
    try:
        print(f"\n[Trial {trial.number}] VAE ... params={trial.params}")
        _, vae_val = train_vae(trial, args, sean_dataset, snsb_dataset,
                               snsb_train_idx, snsb_val_idx, sean_train_idx, sean_val_idx, device)
        print(f"[Trial {trial.number}] best_vae_val={vae_val:.4f}")
        return vae_val
    except optuna.exceptions.TrialPruned:
        raise
    except RuntimeError as e:
        tb_str = _tb.format_exc()
        print(tb_str)
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Trial {trial.number}] CUDA OOM -> pruned")
            trial.set_user_attr("crash_reason", "CUDA OOM")
        else:
            trial.set_user_attr("crash_reason", f"RuntimeError: {str(e)[:300]}")
            trial.set_user_attr("crash_tb", tb_str[-800:])
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        tb_str = _tb.format_exc()
        print(tb_str)
        trial.set_user_attr("crash_reason", f"{type(e).__name__}: {str(e)[:300]}")
        trial.set_user_attr("crash_tb", tb_str[-800:])
        raise optuna.exceptions.TrialPruned()


# ── main ──────────────────────────────────────────────────────────────────────

def resolve_storage(storage):
    if not storage.startswith("journal://"):
        return storage

    journal_path = storage[len("journal://"):]
    if not journal_path:
        raise ValueError("journal storage requires a path, e.g. journal://optuna_run.log")

    parent = os.path.dirname(journal_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    backend_cls = getattr(optuna.storages.journal, "JournalFileBackend", None)
    if backend_cls is not None:
        backend = backend_cls(journal_path)
    else:
        backend = optuna.storages.JournalFileStorage(journal_path)
    return optuna.storages.JournalStorage(backend)


def trials_csv_path(storage, gpu_id):
    if storage.startswith("sqlite:///"):
        path = storage[len("sqlite:///"):].split("?", 1)[0]
        base = path[:-3] if path.endswith(".db") else path
    elif storage.startswith("journal://"):
        path = storage[len("journal://"):]
        base, _ = os.path.splitext(path)
    else:
        base = storage.replace("/", "_").replace(":", "_")
    return f"{base}_trials_gpu{gpu_id}.csv"


def report_best(args):
    study = optuna.load_study(study_name=args.study_name, storage=resolve_storage(args.storage))
    best  = study.best_trial
    print(f"\nBest trial #{best.number}  |  VAE val loss = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    with open(args.out_json, "w") as f:
        json.dump({
            "best_trial": best.number,
            "best_vae_val_loss": best.value,
            "params": best.params,
        }, f, indent=2)
    print(f"Saved -> {args.out_json}")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Completed: {len(completed)} / {len(study.trials)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         default="search", choices=["search", "report"])
    parser.add_argument("--study_name",   default="unified_real_adaptive_v2")
    parser.add_argument("--storage",      default="sqlite:///optuna_unified_real_adaptive_v2.db")
    parser.add_argument("--n_trials",     type=int, default=50)
    parser.add_argument("--vae_epochs",   type=int, default=1000)
    parser.add_argument("--probe_epochs", type=int, default=1000)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--hidden_dim",    type=int, default=128)
    parser.add_argument("--latent_dim",    type=int, default=64)
    parser.add_argument("--h5_path",      default="dataloaders/combined_v1.h5")
    parser.add_argument("--snsb_h5_path", default="dataloaders/social_nav_v3.h5")
    parser.add_argument("--split_file",   default="checkpoints/split_indices_final.pt")
    parser.add_argument("--out_json",     default="best_hyperparams_unified.json")
    args = parser.parse_args()

    if args.mode == "report":
        report_best(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    print(f"Worker | device={device} | GPU={gpu_id} | study={args.study_name}")

    # ── load datasets ─────────────────────────────────────────────────────────
    print("Loading final combined split source ...")
    combined_dataset = UnifiedGraphDataset(args.h5_path, ssub=False)
    split = torch.load(args.split_file, map_location="cpu", weights_only=False)

    sean_train_idx = [i for i in split["train"] if combined_dataset._dataset_ids[i] == SEAN_ID]
    sean_val_idx = [i for i in split["val"] if combined_dataset._dataset_ids[i] == SEAN_ID]
    combined_dataset.set_robot_stats(sean_train_idx)
    print(f"SEAN final split: train={len(sean_train_idx)} val={len(sean_val_idx)}")

    print(f"Loading SNSB from {args.snsb_h5_path} ...")
    snsb_dataset = UnifiedSocialNavDataset(args.snsb_h5_path, zero_node_features=True)
    snsb_train_idx, snsb_val_idx = get_snsb_split_indices(combined_dataset, split, snsb_dataset)

    # ── study ─────────────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=args.study_name,
        storage=resolve_storage(args.storage),
        direction="minimize",
        sampler=TPESampler(seed=None, n_startup_trials=5),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
        load_if_exists=True,
    )

    csv_path = trials_csv_path(args.storage, gpu_id)
    PARAM_COLS = [
        "vae_lr", "beta_max", "warmup_epochs", "dropout",
        "adaptive_warmup_epochs", "adaptive_max_alpha",
    ]
    FIELDNAMES = ["trial", "vae_val_loss", "best_so_far", "is_best"] + PARAM_COLS

    def csv_callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        best = study.best_trial
        row  = {
            "trial": trial.number,
            "vae_val_loss": round(trial.value, 4),
            "best_so_far":  round(best.value, 4),
            "is_best":      "***BEST***" if trial.number == best.number else "",
            **{k: trial.params.get(k, "") for k in PARAM_COLS},
        }
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        flag = "  *** NEW BEST ***" if trial.number == best.number else ""
        parts = "  ".join(f"{k}={v}" for k, v in trial.params.items())
        print(
            f"[Trial {trial.number:03d}] vae_val_loss={trial.value:.4f}  "
            f"best={best.value:.4f}{flag}"
        )
        print(f"           {parts}")

    def _obj(trial):
        return objective(trial, args, combined_dataset, snsb_dataset,
                         snsb_train_idx, snsb_val_idx, sean_train_idx, sean_val_idx, device)

    study.optimize(_obj, n_trials=args.n_trials, callbacks=[csv_callback])

    best = study.best_trial
    print(f"\nSearch done. Best trial #{best.number}  vae_val_loss={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"Run --mode report to save JSON.")


if __name__ == "__main__":
    main()
