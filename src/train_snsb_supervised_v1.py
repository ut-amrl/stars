"""
Supervised HeteroGNN for SNSB action classification (ablation vs. VAE+probe).

End-to-end training of UnifiedGNNEncoder + linear head on SNSB real data.

Split protocol identical to train_unified_gnn_vae_v3.py:
  - Load combined_v1.h5 + split_indices_final.pt to recover the fixed scene split
  - Map scenes to social_nav_v3.h5 real samples via get_snsb_split_indices
  - Supervised and VAE+probe see the exact same train/val episodes

Adaptive sampling identical to train_unified_gnn_vae_v3.py (SNSB path only):
  - Stratified init: per-graph weight = 1/sqrt(count of rarest class in graph)
  - After adaptive_warmup_epochs: blend with per-graph CE loss weights
  - alpha ramps linearly to adaptive_max_alpha over adaptive_ramp_epochs

Test: socialnav_human_oracle_v3.h5 (60-episode human oracle, T=10 -> interp T=20)

Usage:
  python train_snsb_supervised_v1.py \
      --snsb_h5     dataloaders/social_nav_v3.h5 \
      --combined_h5 dataloaders/combined_v1.h5 \
      --split_file  checkpoints/split_indices_final.pt \
      --oracle_h5   dataloaders/socialnav_human_oracle_v3.h5 \
      --out_dir     experiments/snsb_supervised_v1/run1
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.socialnav_hetero_graph_dataset_v1 import SocialNavGraphDatasetV1
from dataloaders.unified_graph_dataset_v1 import DATASET_SSUB_REAL, UnifiedGraphDataset
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.unified_gnn_vae_v1 import SNSB_TIMESTEPS, UnifiedGNNEncoder

try:
    import wandb
except Exception:
    wandb = None

ACTION_CLASS_NAMES = ["avoid", "follow", "overtake", "yield", "not_consider"]
NUM_CLASSES        = len(ACTION_CLASS_NAMES)
SNSB_ID            = DATASET_SSUB_REAL
N_REAL             = 3052


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SupervisedSNSBGNN(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=128, num_mp_layers=2,
                 num_classes=NUM_CLASSES, snsb_timesteps=SNSB_TIMESTEPS, dropout=0.2):
        super().__init__()
        self.encoder = UnifiedGNNEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_mp_layers=num_mp_layers,
            dropout=dropout,
            snsb_timesteps=snsb_timesteps,
        )
        self.head = nn.Linear(2 * latent_dim, num_classes)

    def forward(self, data):
        mu_dict, _ = self.encoder(data)
        bystander_batch = data["bystander"].batch
        robot_z = mu_dict["robot"][bystander_batch]
        z = torch.cat([robot_z, mu_dict["bystander"]], dim=1)
        return self.head(z)


# ---------------------------------------------------------------------------
# Scene split — identical to train_unified_gnn_vae_v3.py
# ---------------------------------------------------------------------------

def get_snsb_split_indices(combined_ds, split, snsb_ds):
    snsb_train_combined = [i for i in split["train"] if combined_ds._dataset_ids[i] == SNSB_ID]
    snsb_val_combined   = [i for i in split["val"]   if combined_ds._dataset_ids[i] == SNSB_ID]

    train_scenes = set(combined_ds._scenes[i] for i in snsb_train_combined)
    val_scenes   = set(combined_ds._scenes[i] for i in snsb_val_combined)
    assert not (train_scenes & val_scenes), "Scene overlap between train and val!"

    snsb_train_idx, snsb_val_idx, unmatched = [], [], []
    for i in range(N_REAL):
        scene = getattr(snsb_ds.base_dataset[i], "scene", None) or ""
        if scene in train_scenes:
            snsb_train_idx.append(i)
        elif scene in val_scenes:
            snsb_val_idx.append(i)
        else:
            unmatched.append(i)

    assert len(snsb_train_idx) == len(snsb_train_combined), (
        f"SNSB train count mismatch: combined={len(snsb_train_combined)} "
        f"social_nav_v3={len(snsb_train_idx)}"
    )
    assert len(snsb_val_idx) == len(snsb_val_combined), (
        f"SNSB val count mismatch: combined={len(snsb_val_combined)} "
        f"social_nav_v3={len(snsb_val_idx)}"
    )

    print(f"SNSB scene split: train={len(snsb_train_idx)}  val={len(snsb_val_idx)}  "
          f"unmatched={len(unmatched)} (oracle/held-out scenes, expected)")
    return snsb_train_idx, snsb_val_idx


# ---------------------------------------------------------------------------
# Adaptive stratified sampling — identical to train_unified_gnn_vae_v3.py
# ---------------------------------------------------------------------------

def build_stratified_weights(train_subset):
    """Per-graph weight = 1/sqrt(count of rarest class in graph). Identical to VAE trainer."""
    global_counts = Counter()
    per_graph_labels = []
    for i in range(len(train_subset)):
        labels = train_subset[i]["bystander"].y.long().tolist()
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
    wt = torch.tensor(weights, dtype=torch.float64)
    print(f"  Stratified weights: {len(weights)} graphs, class counts={dict(global_counts)}")
    return wt / wt.sum()


def compute_per_sample_ce_losses(model, dataset, device, batch_size, num_workers):
    """Per-graph mean CE loss (unnormalized), used for adaptive sampler blending."""
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        persistent_workers=num_workers > 0)
    all_losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            labels = batch["bystander"].y.long()
            per_node_loss = F.cross_entropy(logits, labels, reduction="none")
            bystander_batch = batch["bystander"].batch
            n_graphs = int(bystander_batch.max().item()) + 1
            per_graph = torch.zeros(n_graphs, device=device)
            counts    = torch.zeros(n_graphs, device=device)
            per_graph.scatter_add_(0, bystander_batch, per_node_loss)
            counts.scatter_add_(0, bystander_batch, torch.ones_like(per_node_loss))
            per_graph = per_graph / counts.clamp(min=1)
            all_losses.extend(per_graph.detach().cpu().tolist())
    model.train(was_training)
    losses_t = torch.tensor(all_losses, dtype=torch.float64)
    total = losses_t.sum()
    if total <= 0 or not torch.isfinite(total):
        return torch.full((len(dataset),), 1.0 / max(1, len(dataset)), dtype=torch.float64)
    return losses_t / total


def make_loader(dataset, batch_size, shuffle, num_workers, sampler=None):
    persistent = num_workers > 0
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers, persistent_workers=persistent,
        sampler=sampler,
    )


# ---------------------------------------------------------------------------
# Oracle interpolation (T=10 -> T=20, same as train_snsb_linear_probe_v3.py)
# ---------------------------------------------------------------------------

def _interp_1d(arr, target_T):
    T_raw = arr.shape[-2]
    if T_raw == target_T:
        return arr
    x_old = np.linspace(0, 1, T_raw)
    x_new = np.linspace(0, 1, target_T)
    leading  = arr.shape[:-2]
    feat_dim = arr.shape[-1]
    out = np.zeros(leading + (target_T, feat_dim), dtype=arr.dtype)
    for idx in np.ndindex(*leading):
        for f in range(feat_dim):
            out[idx + (slice(None), f)] = np.interp(x_new, x_old, arr[idx + (slice(None), f)])
    return out


class InterpSocialNavDataset(Dataset):
    def __init__(self, base_dataset, target_t):
        super().__init__(root=None)
        self.base_dataset = base_dataset
        self.target_t = int(target_t)

    def len(self):
        return len(self.base_dataset)

    def get(self, idx):
        data = self.base_dataset[idx]
        T = int(data.num_timesteps)
        if T == self.target_t:
            return data

        robot_traj = data["robot"].traj.numpy()
        data["robot"].traj = torch.from_numpy(_interp_1d(robot_traj, self.target_t))
        data["robot"].x    = torch.from_numpy(
            _interp_1d(robot_traj, self.target_t).reshape(1, self.target_t * 3)
        )

        if data["human"].x.shape[0] > 0:
            human_traj        = data["human"].traj.numpy()
            human_traj_interp = _interp_1d(human_traj, self.target_t)
            data["human"].traj = torch.from_numpy(human_traj_interp)
            P = human_traj_interp.shape[0]
            data["human"].x   = torch.from_numpy(human_traj_interp.reshape(P, self.target_t * 3))

        for et in [("robot","to","human"), ("human","to","robot"), ("human","to","human")]:
            if hasattr(data[et], "edge_attr"):
                ea = data[et].edge_attr.numpy()
                if ea.shape[0] > 0:
                    data[et].edge_attr = torch.from_numpy(_interp_1d(ea, self.target_t))
                else:
                    data[et].edge_attr = torch.zeros(0, self.target_t, ea.shape[2])

        data.num_timesteps = self.target_t
        return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits_np, labels_np, prefix):
    preds = logits_np.argmax(axis=1)
    m = {f"{prefix}/acc": float(accuracy_score(labels_np, preds))}
    f1_per = f1_score(labels_np, preds, average=None,
                      labels=list(range(NUM_CLASSES)), zero_division=0)
    for i, name in enumerate(ACTION_CLASS_NAMES):
        m[f"{prefix}/f1_{name}"] = float(f1_per[i])
    m[f"{prefix}/f1_macro"]    = float(f1_score(labels_np, preds, average="macro",    zero_division=0))
    m[f"{prefix}/f1_weighted"] = float(f1_score(labels_np, preds, average="weighted", zero_division=0))
    # f1_macro3: mean of the 3 primary classes, matches eval_label_efficiency.py
    m[f"{prefix}/f1_macro3"]   = float(np.mean([m[f"{prefix}/f1_follow"],
                                                 m[f"{prefix}/f1_avoid"],
                                                 m[f"{prefix}/f1_not_consider"]]))
    return m


def prediction_agreement(logits_np, probs_np):
    preds = logits_np.argmax(axis=1)
    return float(probs_np[np.arange(len(preds)), preds].mean())


def save_confusion_matrix(gt, logits, out_dir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds   = logits.argmax(axis=1)
    cm      = confusion_matrix(gt, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ACTION_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(ACTION_CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({tag})")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm_norm[i,j] > 0.5 else "black")
    fig.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {path}")


def append_csv_and_print(metrics, csv_path, run_id, prefix):
    row = {"run_id": run_id, **{k: float(v) for k, v in metrics.items()}}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    rows = []
    with open(csv_path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k != "run_id" else v) for k, v in r.items()})

    print(f"\n=== CUMULATIVE ({prefix}) over {len(rows)} runs ===")
    for key in [f"{prefix}/acc", f"{prefix}/f1_macro", f"{prefix}/f1_macro3", f"{prefix}/f1_weighted"]:
        if key not in rows[0]:
            continue
        vals = [r[key] for r in rows]
        print(f"  {key}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")
    pa_key = f"{prefix}/pa"
    if pa_key in rows[0]:
        vals = [r[pa_key] for r in rows]
        print(f"  {pa_key}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def run_eval(model, loader, device, ce_weights=None):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches  = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch)
            labels = batch["bystander"].y.long()
            loss   = F.cross_entropy(logits, labels, weight=ce_weights)
            total_loss += loss.item(); n_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    return (torch.cat(all_logits).numpy(),
            torch.cat(all_labels).numpy(),
            total_loss / max(1, n_batches))


def run_oracle_forward(model, loader, device):
    model.eval()
    all_logits, all_labels, all_probs = [], [], []
    has_probs = True
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_logits.append(model(batch).cpu())
            all_labels.append(batch["bystander"].y.long().cpu())
            if has_probs:
                if hasattr(batch["bystander"], "action_probs"):
                    all_probs.append(batch["bystander"].action_probs.cpu())
                else:
                    has_probs = False
    probs = torch.cat(all_probs).numpy() if has_probs and all_probs else None
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy(), probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed: {args.seed}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Scene split — identical to VAE trainer ----
    print(f"Loading combined_v1.h5 split from {args.split_file} ...")
    combined_ds = UnifiedGraphDataset(args.combined_h5, ssub=False)
    split       = torch.load(args.split_file, map_location="cpu", weights_only=False)

    print(f"Loading SNSB dataset from {args.snsb_h5} ...")
    snsb_ds = UnifiedSocialNavDataset(args.snsb_h5, zero_node_features=True)

    train_idx, val_idx = get_snsb_split_indices(combined_ds, split, snsb_ds)

    # Scene-based label fraction subsampling (identical logic to eval_label_efficiency.py)
    if args.label_fraction < 1.0:
        train_scenes   = [getattr(snsb_ds.base_dataset[i], "scene", "") for i in train_idx]
        unique_scenes  = sorted(set(train_scenes))
        rng            = np.random.default_rng(args.subsample_seed)
        shuffled       = rng.permutation(unique_scenes).tolist()
        n_keep         = max(1, round(args.label_fraction * len(unique_scenes)))
        keep_scenes    = set(shuffled[:n_keep])
        train_idx      = [i for i, sc in zip(train_idx, train_scenes) if sc in keep_scenes]
        print(f"Label fraction {args.label_fraction:.0%}: {len(train_idx)} samples "
              f"({n_keep}/{len(unique_scenes)} scenes)  subsample_seed={args.subsample_seed}")

    torch.save({"train": train_idx, "val": val_idx, "split_file": args.split_file,
                "label_fraction": args.label_fraction, "subsample_seed": args.subsample_seed},
               os.path.join(args.out_dir, "split_indices.pt"))

    train_sub = torch.utils.data.Subset(snsb_ds, train_idx)
    val_sub   = torch.utils.data.Subset(snsb_ds, val_idx)

    # ---- Adaptive stratified sampler — identical to VAE trainer ----
    sampler            = None
    stratified_weights = None
    if args.weighted_sampling:
        print("Computing stratified sampling weights (SNSB)...")
        stratified_weights = build_stratified_weights(train_sub)
        sampler = WeightedRandomSampler(
            stratified_weights, num_samples=len(train_sub), replacement=True
        )

    train_loader = make_loader(train_sub, args.batch_size, True,  args.num_workers, sampler=sampler)
    val_loader   = make_loader(val_sub,   args.batch_size, False, args.num_workers)

    # ---- CE class weights from train labels ----
    train_labels = torch.cat([snsb_ds[i]["bystander"].y for i in train_idx])
    counts     = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    ce_weights = (train_labels.shape[0] / (NUM_CLASSES * counts.clamp(min=1))).to(device)
    print("Train class distribution:")
    for i, name in enumerate(ACTION_CLASS_NAMES):
        c = int((train_labels == i).sum())
        print(f"  {name}: {c} ({100. * c / len(train_labels):.1f}%)")
    print(f"CE class weights: {dict(zip(ACTION_CLASS_NAMES, ce_weights.cpu().tolist()))}")

    # ---- Model ----
    model = SupervisedSNSBGNN(
        hidden_dim     = args.hidden_dim,
        latent_dim     = args.latent_dim,
        num_mp_layers  = args.num_mp_layers,
        dropout        = args.dropout,
        snsb_timesteps = SNSB_TIMESTEPS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}  hidden={args.hidden_dim}  latent={args.latent_dim}  "
          f"mp_layers={args.num_mp_layers}  dropout={args.dropout}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    if wandb is not None and args.wandb:
        wandb.init(project="social_world_model",
                   name=f"snsb_supervised_v1_seed{args.seed}",
                   config=vars(args))

    # ---- Training loop ----
    best_val_loss = float("inf")
    es_counter    = 0
    best_path     = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):

        # Adaptive sampler update — identical to VAE trainer (SNSB path)
        if args.weighted_sampling and sampler is not None and epoch > args.adaptive_warmup_epochs:
            epochs_since = epoch - args.adaptive_warmup_epochs
            alpha = min(args.adaptive_max_alpha,
                        args.adaptive_max_alpha * epochs_since / max(args.adaptive_ramp_epochs, 1))
            loss_w   = compute_per_sample_ce_losses(model, train_sub, device,
                                                     args.batch_size, args.num_workers)
            combined = (1.0 - alpha) * stratified_weights + alpha * loss_w
            sampler.weights = (combined / combined.sum()).double()

        # Train
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch  = batch.to(device)
            logits = model(batch)
            labels = batch["bystander"].y.long()
            loss   = F.cross_entropy(logits, labels, weight=ce_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        val_logits, val_labels, val_loss = run_eval(model, val_loader, device,
                                                     ce_weights=ce_weights)
        val_m = compute_metrics(val_logits, val_labels, "val")
        scheduler.step(val_loss)

        print(f"Epoch {epoch:04d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_f1_macro={val_m['val/f1_macro']:.4f}")

        if wandb is not None and args.wandb:
            wandb.log({"epoch": epoch, "train/loss": train_loss,
                       "val/loss": val_loss, **val_m})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            es_counter    = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "args": vars(args), "val_loss": val_loss}, best_path)
            print(f"  Saved best (val_loss={val_loss:.4f})")
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.es_patience})")
                break

    # ---- Val report on best checkpoint ----
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_logits, val_labels, _ = run_eval(model, val_loader, device)
    val_m = compute_metrics(val_logits, val_labels, "val")
    print(f"\n=== VAL (best ckpt epoch {ckpt['epoch']}) ===")
    print(classification_report(val_labels, val_logits.argmax(axis=1),
                                 labels=list(range(NUM_CLASSES)),
                                 target_names=ACTION_CLASS_NAMES, zero_division=0))
    save_confusion_matrix(val_labels, val_logits, args.out_dir, "val")

    run_id  = os.path.basename(args.out_dir)
    val_csv = os.path.join(os.path.dirname(args.out_dir), "val_results.csv")
    append_csv_and_print(val_m, val_csv, run_id, "val")

    # ---- Oracle eval ----
    if args.oracle_h5:
        print(f"\n{'=' * 60}")
        print(f"ORACLE EVAL: {args.oracle_h5}")
        print(f"{'=' * 60}")

        oracle_base   = SocialNavGraphDatasetV1(args.oracle_h5)
        oracle_interp = InterpSocialNavDataset(oracle_base, target_t=SNSB_TIMESTEPS)
        oracle_ds     = UnifiedSocialNavDataset(base_dataset=oracle_interp, zero_node_features=True)
        oracle_loader = DataLoader(oracle_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)

        oracle_logits, oracle_labels, oracle_probs = run_oracle_forward(model, oracle_loader, device)
        oracle_m = compute_metrics(oracle_logits, oracle_labels, "oracle")

        if oracle_probs is not None:
            oracle_m["oracle/pa"] = prediction_agreement(oracle_logits, oracle_probs)
        else:
            print("WARNING: no action_probs in oracle set — PA not computed.")

        print(classification_report(oracle_labels, oracle_logits.argmax(axis=1),
                                     labels=list(range(NUM_CLASSES)),
                                     target_names=ACTION_CLASS_NAMES, zero_division=0))
        print(f"  f1_macro={oracle_m['oracle/f1_macro']:.3f}  "
              f"f1_weighted={oracle_m['oracle/f1_weighted']:.3f}"
              + (f"  pa={oracle_m['oracle/pa']:.3f}" if "oracle/pa" in oracle_m else ""))

        save_confusion_matrix(oracle_labels, oracle_logits, args.out_dir, "oracle")
        np.savez_compressed(
            os.path.join(args.out_dir, "oracle_outputs.npz"),
            gt=oracle_labels, logits=oracle_logits,
            **({"action_probs": oracle_probs} if oracle_probs is not None else {}),
            class_names=np.array(ACTION_CLASS_NAMES),
        )

        if wandb is not None and args.wandb:
            wandb.log(oracle_m)

        oracle_csv = os.path.join(os.path.dirname(args.out_dir), "oracle_results.csv")
        append_csv_and_print(oracle_m, oracle_csv, run_id, "oracle")

        # Shared label-eff CSV (one row per run, includes fraction + subsample_seed)
        if args.results_csv:
            row = {
                "run_id": run_id,
                "label_fraction": args.label_fraction,
                "subsample_seed": args.subsample_seed,
                "seed": args.seed,
                **{k: float(v) for k, v in oracle_m.items()},
            }
            write_header = not os.path.exists(args.results_csv)
            with open(args.results_csv, "a", newline="") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)

    if wandb is not None and args.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snsb_h5",     default="dataloaders/social_nav_v3.h5")
    parser.add_argument("--combined_h5", default="dataloaders/combined_v1.h5")
    parser.add_argument("--split_file",  default="checkpoints/split_indices_final.pt")
    parser.add_argument("--oracle_h5",   default="dataloaders/socialnav_human_oracle_v3.h5")
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--seed",        type=int,   default=1)
    parser.add_argument("--epochs",      type=int,   default=1000)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=0.00011564)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--es_patience", type=int,   default=20)
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--latent_dim",  type=int,   default=128)
    parser.add_argument("--num_mp_layers",type=int,  default=2)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--device",      default="cuda:0")
    # Adaptive sampling — matches run_train_vae.sh defaults
    parser.add_argument("--weighted_sampling",      action="store_true", default=True)
    parser.add_argument("--no_weighted_sampling",   dest="weighted_sampling", action="store_false")
    parser.add_argument("--adaptive_warmup_epochs", type=int,   default=20)
    parser.add_argument("--adaptive_max_alpha",     type=float, default=0.75)
    parser.add_argument("--adaptive_ramp_epochs",   type=int,   default=100)
    parser.add_argument("--label_fraction",  type=float, default=1.0,
                        help="Fraction of train scenes to use (scene-based subsample)")
    parser.add_argument("--subsample_seed",  type=int,   default=0,
                        help="Seed for scene shuffle when label_fraction < 1.0")
    parser.add_argument("--results_csv",     default=None,
                        help="Shared CSV to append oracle results row (for label-eff sweep)")
    parser.add_argument("--wandb",       action="store_true")
    args = parser.parse_args()
    train(args)
