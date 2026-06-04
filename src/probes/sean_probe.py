"""
SEAN-T linear probe: emotion classification (competence, surprise, intention).

Trains 3 independent binary linear heads on graph-level embeddings from
HGNNCombinedVAE.encode() → [B, 2 * latent_dim].
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

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset, DATASET_SEAN
from probes.common import flatten_wandb_metrics

LABEL_NAMES = ["competence", "surprise", "intention"]
LABEL_THRESHOLD = 3


def binarize_labels(emotion_label):
    """Convert 3-element emotion label to binary targets."""
    el = emotion_label.float()
    y = torch.zeros(3, dtype=torch.float32)
    y[0] = 1.0 if el[0].item() >= LABEL_THRESHOLD else 0.0
    y[1] = 1.0 if el[1].item() < LABEL_THRESHOLD else 0.0   # surprise inverted
    y[2] = 1.0 if el[2].item() >= LABEL_THRESHOLD else 0.0
    return y


def extract_embeddings(model, loader, device, latent_dim, embedding_mode="full"):
    """Extract graph-level embeddings for SEAN samples."""
    all_z, all_y = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z_full = model.encode(batch)   # [B, 2 * latent_dim] or [B, 2 * hidden_dim]
            D = z_full.shape[1] // 2
            if embedding_mode == "full":
                z = z_full
            else:
                z = z_full[:, :D]
            y_raw = batch.emotion_label.view(-1, 3)
            B = z.shape[0]
            for i in range(B):
                all_z.append(z[i].cpu())
                all_y.append(binarize_labels(y_raw[i].cpu()))
    if not all_z:
        raise RuntimeError("No SEAN samples found in this split.")
    return torch.stack(all_z, dim=0), torch.stack(all_y, dim=0)


def compute_metrics(logits_np, gt_np):
    metrics = {}
    preds = (logits_np > 0).astype(float)
    for d, name in enumerate(LABEL_NAMES):
        f1 = f1_score(gt_np[:, d], preds[:, d], zero_division=0)
        try:
            auc = roc_auc_score(gt_np[:, d], logits_np[:, d])
        except ValueError:
            auc = float("nan")
        metrics[f"f1_{name}"] = float(f1)
        metrics[f"auc_{name}"] = float(auc)
    metrics["mean_f1"] = float(np.mean([metrics[f"f1_{n}"] for n in LABEL_NAMES]))
    metrics["mean_auc"] = float(np.nanmean([metrics[f"auc_{n}"] for n in LABEL_NAMES]))
    return metrics


def eval_probe(heads, z, y, device, batch_size):
    for h in heads:
        h.eval()
    logits_list = []
    loss_total = 0.0
    n_seen = 0
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start:start + batch_size].to(device)
            y_b = y[start:start + batch_size].to(device)
            lg = torch.stack([heads[d](z_b).squeeze(-1) for d in range(3)], dim=1)
            logits_list.append(lg.cpu())
            loss = torch.stack([
                F.binary_cross_entropy_with_logits(lg[:, d], y_b[:, d])
                for d in range(3)
            ]).mean()
            loss_total += loss.item() * y_b.shape[0]
            n_seen += y_b.shape[0]
    logits = torch.cat(logits_list, dim=0)
    metrics = compute_metrics(logits.numpy(), y.numpy())
    metrics["loss"] = loss_total / max(1, n_seen)
    return metrics, logits


def train_sean_probe(encoder, dataset, splits, args, device, run_dir,
                     wb_run=None):
    """Train SEAN-T linear probe. Returns final metrics dict."""
    print("\n" + "=" * 60)
    print("SEAN-T LINEAR PROBE")
    print("=" * 60)
    if getattr(args, "unfreeze_encoder", False):
        print("  Notice: SEAN-T is evaluated as a linear probe and will continue to use frozen features.")

    # Filter to SEAN-only samples
    sean_splits = {}
    for split_name in ["train", "val", "test"]:
        sean_splits[split_name] = dataset.filter_by_source(splits[split_name], DATASET_SEAN)
        print(f"  {split_name}: {len(sean_splits[split_name])} SEAN samples "
              f"(of {len(splits[split_name])} total)")

    persistent = int(args.num_workers) > 0
    loaders = {
        split: DataLoader(
            torch.utils.data.Subset(dataset, sean_splits[split]),
            batch_size=min(16, int(args.batch_size)),
            shuffle=False,
            num_workers=int(args.num_workers),
            persistent_workers=persistent,
        )
        for split in ["train", "val", "test"]
    }

    latent_dim = int(args.latent_dim)
    embedding_mode = str(getattr(args, "sean_embedding_mode", "full"))
    embeddings, labels = {}, {}
    for split in ["train", "val", "test"]:
        z, y = extract_embeddings(encoder, loaders[split], device, latent_dim, embedding_mode)
        embeddings[split] = z
        labels[split] = y
        pos_counts = y.sum(dim=0).int().tolist()
        print(f"  {split}: {z.shape[0]} graphs, embed_dim={z.shape[1]}, "
              f"pos=[comp={pos_counts[0]}, surp={pos_counts[1]}, int={pos_counts[2]}]")

    input_dim = embeddings["train"].shape[1]
    print(f"Embedding mode: {embedding_mode} → input_dim={input_dim}")
    heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(3)]).to(device)
    print(f"Probe params: {sum(p.numel() for p in heads.parameters()):,}")

    optimizer = torch.optim.AdamW(heads.parameters(),
                                  lr=float(args.lr),
                                  weight_decay=float(args.weight_decay))

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    patience = int(args.early_stopping_patience)
    min_delta = float(args.early_stopping_min_delta)
    best_path = os.path.join(run_dir, "best_linear_probe_sean.pt")
    eval_every = max(1, int(args.eval_metrics_every))

    for epoch in range(1, int(args.epochs) + 1):
        heads.train()
        N = embeddings["train"].shape[0]
        perm = torch.randperm(N)
        t_loss = 0.0
        n_batches = 0
        for start in range(0, N, int(args.batch_size)):
            idx = perm[start:start + int(args.batch_size)]
            z_b = embeddings["train"][idx].to(device)
            y_b = labels["train"][idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([
                F.binary_cross_entropy_with_logits(
                    heads[d](z_b).squeeze(-1), y_b[:, d])
                for d in range(3)
            ]).mean()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n_batches += 1
        t_loss /= max(1, n_batches)

        train_metrics, _ = eval_probe(heads, embeddings["train"], labels["train"],
                                      device, int(args.batch_size))
        val_metrics, _ = eval_probe(heads, embeddings["val"], labels["val"],
                                    device, int(args.batch_size))

        if wb_run is not None:
            log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            log.update(flatten_wandb_metrics(train_metrics, "sean_train"))
            log.update(flatten_wandb_metrics(val_metrics, "sean_val"))
            wb_run.log(log, step=epoch)

        if epoch == 1 or epoch % eval_every == 0:
            print(
                f"[SEAN] Epoch {epoch:04d}/{args.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_f1=[{val_metrics['f1_competence']:.3f}, "
                f"{val_metrics['f1_surprise']:.3f}, "
                f"{val_metrics['f1_intention']:.3f}] "
                f"val_auc=[{val_metrics['auc_competence']:.3f}, "
                f"{val_metrics['auc_surprise']:.3f}, "
                f"{val_metrics['auc_intention']:.3f}]"
            )

        if val_metrics["loss"] < best_val - min_delta:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "heads_state_dict": heads.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
            }, best_path)
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[SEAN] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    # Load best and evaluate
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    heads.load_state_dict(ckpt["heads_state_dict"])

    final_metrics = {}
    for split in ["train", "val", "test"]:
        m, _ = eval_probe(heads, embeddings[split], labels[split],
                          device, int(args.batch_size))
        final_metrics[split] = m

    print(f"\n[SEAN] Best epoch: {best_epoch} | best val loss: {best_val:.4f}")
    for split in ["train", "val", "test"]:
        m = final_metrics[split]
        print(f"  {split}: loss={m['loss']:.4f}  "
              f"f1=[{m['f1_competence']:.3f}, {m['f1_surprise']:.3f}, {m['f1_intention']:.3f}]  "
              f"auc=[{m['auc_competence']:.3f}, {m['auc_surprise']:.3f}, {m['auc_intention']:.3f}]")

    return final_metrics
