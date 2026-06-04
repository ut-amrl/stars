"""
SNS (Social Navigation Synthetic/Bystander) linear probe: 5-class action classification.

Trains a single linear head on per-human-node embeddings from
HGNNCombinedVAE.encode_nodes() → [N_human, latent_dim].

Classes: avoid, follow, overtake, yield, not_consider
"""

import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

import sys
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SRC_DIR)

from probes.common import flatten_wandb_metrics
from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset

ACTION_CLASSES = ["avoid", "follow", "overtake", "yield", "not_consider"]
NUM_CLASSES = len(ACTION_CLASSES)
IGNORE_LABEL = -100


def extract_node_embeddings(model, loader, device):
    """Extract per-human-node embeddings, filtering out IGNORE_LABEL nodes."""
    all_z, all_y = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            num_nodes = batch.num_nodes
            num_edges = batch.edge_index.shape[1]
            print(f"  [SNS Extraction] Batch {i}: num_graphs={batch.num_graphs}, num_nodes={num_nodes}, num_edges={num_edges}")
            z = model.encode_nodes(batch)           # [N_total, latent_dim]
            keep = batch.action_label != IGNORE_LABEL
            if keep.any():
                all_z.append(z[keep].cpu())
                all_y.append(batch.action_label[keep].cpu())
    if not all_z:
        return None, None
    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0).long()


def class_counts(y):
    counts = torch.bincount(y.cpu(), minlength=NUM_CLASSES)
    return {ACTION_CLASSES[i]: int(counts[i]) for i in range(NUM_CLASSES)}


def weighted_nanmean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = ~np.isnan(values)
    if keep.sum() == 0:
        return float("nan")
    denom = weights[keep].sum()
    if denom <= 0:
        return float(np.nanmean(values[keep]))
    return float((values[keep] * weights[keep]).sum() / denom)


def compute_metrics(logits, labels, pauc_max_fpr=0.1):
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    y_pred = probs.argmax(axis=1)
    support = np.bincount(y_true, minlength=NUM_CLASSES)

    per_class_f1 = f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES),
                            average=None, zero_division=0)
    per_class_auc = []
    for c in range(NUM_CLASSES):
        y_bin = (y_true == c).astype(np.int64)
        if np.unique(y_bin).size < 2:
            per_class_auc.append(float("nan"))
            continue
        per_class_auc.append(float(roc_auc_score(y_bin, probs[:, c])))

    metrics = {
        "loss":         float(F.cross_entropy(logits, labels).detach().cpu()),
        "weighted_f1":  float(f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES),
                                       average="weighted", zero_division=0)),
        "weighted_auc": weighted_nanmean(per_class_auc, support),
    }
    for c, name in enumerate(ACTION_CLASSES):
        metrics[f"f1_{name}"] = float(per_class_f1[c])
    return metrics


def eval_probe(probe, z, y, device, batch_size):
    probe.eval()
    logits_list = []
    loss_total = 0.0
    n_seen = 0
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start:start + batch_size].to(device)
            y_b = y[start:start + batch_size].to(device)
            lg = probe(z_b)
            logits_list.append(lg.cpu())
            loss_total += float(F.cross_entropy(lg, y_b).cpu()) * y_b.numel()
            n_seen += y_b.numel()
    logits = torch.cat(logits_list, dim=0)
    metrics = compute_metrics(logits, y)
    metrics["loss"] = loss_total / max(1, n_seen)
    return metrics, logits


def compute_stratified_weights(y):
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    inv_freq = (counts.sum() / counts.clamp(min=1)).sqrt()
    weights = inv_freq[y]
    return weights / weights.sum()


def train_one_epoch(probe, z, y, device, optimizer, batch_size, sample_weights=None):
    probe.train()
    perm = (torch.multinomial(sample_weights.double(), num_samples=z.shape[0], replacement=True)
            if sample_weights is not None else torch.randperm(z.shape[0]))
    total_loss = 0.0
    n_seen = 0
    for start in range(0, z.shape[0], batch_size):
        idx = perm[start:start + batch_size]
        z_b = z[idx].to(device)
        y_b = y[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(z_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * y_b.numel()
        n_seen += y_b.numel()
    return total_loss / max(1, n_seen)


def train_sns_probe(encoder, dataset, splits, args, device, run_dir,
                    wb_run=None):
    """Train SNS action classification linear probe. Returns final metrics dict."""
    print("\n" + "=" * 60)
    print("SNS LINEAR PROBE (action classification)")
    print("=" * 60)
    if getattr(args, "unfreeze_encoder", False):
        print("  Notice: SNS is evaluated as a linear probe and will continue to use frozen features.")

    persistent = int(args.num_workers) > 0
    batch_size = min(16, int(args.batch_size))
    loaders = {
        split: DataLoader(
            torch.utils.data.Subset(dataset, splits[split]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(args.num_workers),
            persistent_workers=persistent,
        )
        for split in ["train", "val"]
    }

    # If a dedicated eval h5 is provided, use it as the test split
    # (matches homogeneous eval protocol from eval_ssub.py)
    eval_h5_path = getattr(args, "sns_eval_h5_path", None)
    if eval_h5_path and os.path.isfile(eval_h5_path):
        print(f"Loading SNS test dataset from: {eval_h5_path}")
        eval_dataset = UnifiedGraphDataset(
            h5_path=eval_h5_path,
            ssub=True,
            robot_mean=dataset.robot_mean,
            robot_std=dataset.robot_std,
        )
        loaders["test"] = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(args.num_workers),
            persistent_workers=persistent,
        )
        print(f"  Eval dataset: {len(eval_dataset)} episodes")
    else:
        loaders["test"] = DataLoader(
            torch.utils.data.Subset(dataset, splits["test"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(args.num_workers),
            persistent_workers=persistent,
        )
        if eval_h5_path:
            print(f"WARNING: sns_eval_h5_path={eval_h5_path!r} not found; "
                  "falling back to in-distribution test split.")

    print(f"Split: train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={'eval_h5' if eval_h5_path and os.path.isfile(eval_h5_path) else len(splits['test'])}")

    embeddings, labels = {}, {}
    for split in ["train", "val", "test"]:
        z, y = extract_node_embeddings(encoder, loaders[split], device)
        embeddings[split] = z
        labels[split] = y
        if z is not None:
            print(f"  {split}: {z.shape[0]} SSUB human nodes  classes={class_counts(y)}")
        else:
            print(f"  {split}: 0 valid SSUB human nodes (skipping)")

    probe = nn.Linear(embeddings["train"].shape[1], NUM_CLASSES).to(device)
    print(f"Probe params: {sum(p.numel() for p in probe.parameters()):,}")

    optimizer = torch.optim.AdamW(probe.parameters(),
                                  lr=float(args.lr),
                                  weight_decay=float(args.weight_decay))

    use_weighted = getattr(args, "sns_use_weighted_sampling", True)
    stratified_weights = None
    current_weights = None
    if use_weighted:
        stratified_weights = compute_stratified_weights(labels["train"])
        current_weights = stratified_weights.clone()
        ess = float(1.0 / (stratified_weights ** 2).sum())
        print(f"Stratified sampling: weight range=[{stratified_weights.min():.6f}, "
              f"{stratified_weights.max():.6f}]  ESS={ess:.0f}/{len(labels['train'])}")

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    patience = int(args.early_stopping_patience)
    min_delta = float(args.early_stopping_min_delta)
    best_path = os.path.join(run_dir, "best_linear_probe_sns.pt")
    eval_every = max(1, int(args.eval_metrics_every))

    for epoch in range(1, int(args.epochs) + 1):
        train_one_epoch(probe, embeddings["train"], labels["train"], device,
                        optimizer, int(args.batch_size), sample_weights=current_weights)

        train_metrics, _ = eval_probe(probe, embeddings["train"], labels["train"],
                                      device, int(args.batch_size))
        val_metrics, _ = eval_probe(probe, embeddings["val"], labels["val"],
                                    device, int(args.batch_size))

        if wb_run is not None:
            log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            log.update(flatten_wandb_metrics(train_metrics, "sns_train"))
            log.update(flatten_wandb_metrics(val_metrics, "sns_val"))
            wb_run.log(log, step=epoch)

        if epoch == 1 or epoch % eval_every == 0:
            print(
                f"[SNS] Epoch {epoch:04d}/{args.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"train_wf1={train_metrics['weighted_f1']:.3f} "
                f"val_wf1={val_metrics['weighted_f1']:.3f} | "
                f"val_auc={val_metrics['weighted_auc']:.3f}"
            )

        if val_metrics["loss"] < best_val - min_delta:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "probe_state_dict": probe.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
            }, best_path)
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[SNS] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    # Load best and evaluate
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    probe.load_state_dict(ckpt["probe_state_dict"])

    final_metrics = {}
    for split in ["train", "val", "test"]:
        if embeddings[split] is not None:
            m, _ = eval_probe(probe, embeddings[split], labels[split],
                              device, int(args.batch_size))
            final_metrics[split] = m
        else:
            final_metrics[split] = None

    print(f"\n[SNS] Best epoch: {best_epoch} | best val loss: {best_val:.4f}")
    for split in ["train", "val", "test"]:
        m = final_metrics[split]
        if m is not None:
            print(f"  {split}: loss={m['loss']:.4f}  weighted_f1={m['weighted_f1']:.3f}  "
                  f"weighted_auc={m['weighted_auc']:.3f}")
        else:
            print(f"  {split}: skipped (no valid SSUB human nodes)")

    return final_metrics
