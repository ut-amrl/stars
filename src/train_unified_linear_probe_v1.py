import argparse
import csv
import math
import os
import random
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from dataloaders.unified_graph_dataset_all import (
    UnifiedGraphDataset, DATASET_SSUB_REAL, DATASET_SSUB_SYNTH,
)
from models.hgnn_combined_v2 import HGNNCombinedVAE

ACTION_CLASSES = ["avoid", "follow", "overtake", "yield", "not_consider"]
NUM_CLASSES    = len(ACTION_CLASSES)
IGNORE_LABEL   = -100


def load_config(config_path):
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    section = "linear_probe_hgnn_params"
    if section not in cfg:
        raise ValueError(f"Config missing section '{section}'")
    args = SimpleNamespace(**cfg[section])
    args.config_path        = config_path
    args.checkpoint         = os.path.abspath(args.checkpoint)
    args.split_indices_path = os.path.abspath(args.split_indices_path)
    args.h5_path            = os.path.abspath(args.h5_path)
    args.experiments_root   = os.path.abspath(args.experiments_root)
    if hasattr(args, "eval_h5_path"):
        args.eval_h5_path = os.path.abspath(args.eval_h5_path)
    return args


def create_run_dir(experiments_root, run_name):
    os.makedirs(experiments_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name      = str(run_name).strip()
    run_dir   = os.path.join(experiments_root, f"{timestamp}_{name}" if name else timestamp)
    suffix = 1
    base   = run_dir
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = f"{base}_{suffix}"
    os.makedirs(run_dir)
    return run_dir


def resolve_device(device_cfg):
    s = str(device_cfg).lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_plot_formats(plot_formats):
    if isinstance(plot_formats, str):
        tokens = [p.strip().lower() for p in plot_formats.replace(",", " ").split() if p.strip()]
    elif isinstance(plot_formats, (list, tuple)):
        tokens = [str(p).strip().lower() for p in plot_formats if str(p).strip()]
    else:
        tokens = ["png"]
    return list(dict.fromkeys(tokens)) or ["png"]


def load_encoder(args, device):
    ckpt       = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    robot_mean = ckpt.get("robot_mean", None)
    robot_std  = ckpt.get("robot_std",  None)

    model = HGNNCombinedVAE(
        hidden_dim              = int(args.hidden_dim),
        latent_dim              = int(args.latent_dim),
        num_mp_layers           = int(args.num_mp_layers),
        edge_transformer_layers = int(args.edge_transformer_layers),
        num_heads               = int(args.num_heads),
        dropout                 = float(args.dropout),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"Missing checkpoint keys: {list(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {list(unexpected)}")

    saved_args = ckpt.get("args", {})
    for key in ["hidden_dim", "latent_dim", "num_mp_layers", "edge_transformer_layers", "num_heads"]:
        if isinstance(saved_args, dict) and key in saved_args:
            if int(saved_args[key]) != int(getattr(args, key)):
                print(f"WARNING: config {key}={getattr(args, key)} differs from "
                      f"checkpoint {key}={saved_args[key]}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, robot_mean, robot_std


def extract_embeddings(model, loader, device):
    all_z, all_y = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z     = model.encode_nodes(batch)
            keep  = batch.action_label != IGNORE_LABEL
            if keep.any():
                all_z.append(z[keep].cpu())
                all_y.append(batch.action_label[keep].cpu())
    if not all_z:
        raise RuntimeError("No valid SSUB human nodes found in this split.")
    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0).long()


def extract_test_embeddings(model, loader, device):
    all_z, all_y, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z     = model.encode_nodes(batch)
            keep  = batch.action_label != IGNORE_LABEL
            if keep.any():
                all_z.append(z[keep].cpu())
                all_y.append(batch.action_label[keep].cpu())
                all_probs.append(batch.action_probs[keep].cpu())
    if not all_z:
        raise RuntimeError("No valid human nodes found in eval dataset.")
    return (torch.cat(all_z, dim=0),
            torch.cat(all_y, dim=0).long(),
            torch.cat(all_probs, dim=0))


def compute_pa(logits, action_probs):
    y_pred = logits.argmax(dim=1).numpy()
    gt_probs = action_probs.numpy()
    return float(gt_probs[np.arange(len(y_pred)), y_pred].mean())


def class_counts(y):
    counts = torch.bincount(y.cpu(), minlength=NUM_CLASSES)
    return {ACTION_CLASSES[i]: int(counts[i]) for i in range(NUM_CLASSES)}


def weighted_nanmean(values, weights):
    values  = np.asarray(values,  dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep    = ~np.isnan(values)
    if keep.sum() == 0:
        return float("nan")
    denom = weights[keep].sum()
    if denom <= 0:
        return float(np.nanmean(values[keep]))
    return float((values[keep] * weights[keep]).sum() / denom)


def compute_metrics(logits, labels, pauc_max_fpr):
    probs   = F.softmax(logits, dim=-1).detach().cpu().numpy()
    y_true  = labels.detach().cpu().numpy()
    y_pred  = probs.argmax(axis=1)
    support = np.bincount(y_true, minlength=NUM_CLASSES)

    per_class_f1 = f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES),
                             average=None, zero_division=0)
    per_class_auc, per_class_pauc = [], []
    for c in range(NUM_CLASSES):
        y_bin = (y_true == c).astype(np.int64)
        if np.unique(y_bin).size < 2:
            per_class_auc.append(float("nan"))
            per_class_pauc.append(float("nan"))
            continue
        per_class_auc.append(float(roc_auc_score(y_bin, probs[:, c])))
        per_class_pauc.append(float(roc_auc_score(y_bin, probs[:, c], max_fpr=pauc_max_fpr)))

    metrics = {
        "loss":          float(F.cross_entropy(logits, labels).detach().cpu()),
        "weighted_f1":   float(f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES),
                                        average="weighted", zero_division=0)),
        "weighted_auc":  weighted_nanmean(per_class_auc,  support),
        "weighted_pauc": weighted_nanmean(per_class_pauc, support),
    }
    for c, name in enumerate(ACTION_CLASSES):
        metrics[f"f1_{name}"] = float(per_class_f1[c])
    return metrics


def eval_probe(probe, z, y, device, batch_size, pauc_max_fpr):
    probe.eval()
    logits_list = []
    loss_total  = 0.0
    n_seen      = 0
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start:start+batch_size].to(device)
            y_b = y[start:start+batch_size].to(device)
            lg  = probe(z_b)
            logits_list.append(lg.cpu())
            loss_total += float(F.cross_entropy(lg, y_b).cpu()) * y_b.numel()
            n_seen     += y_b.numel()
    logits  = torch.cat(logits_list, dim=0)
    metrics = compute_metrics(logits, y, pauc_max_fpr)
    metrics["loss"] = loss_total / max(1, n_seen)
    return metrics, logits


# ── Sampling ──────────────────────────────────────────────────────────────────
def compute_stratified_weights(y):
    counts   = torch.bincount(y, minlength=NUM_CLASSES).float()
    inv_freq = (counts.sum() / counts.clamp(min=1)).sqrt()
    weights  = inv_freq[y]
    return weights / weights.sum()


def compute_per_sample_losses(probe, z, y, device, batch_size):
    probe.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start:start+batch_size].to(device)
            y_b = y[start:start+batch_size].to(device)
            losses.append(F.cross_entropy(probe(z_b), y_b, reduction="none").cpu())
    return torch.cat(losses, dim=0)


def train_one_epoch(probe, z, y, device, optimizer, batch_size, sample_weights=None):
    probe.train()
    perm = (torch.multinomial(sample_weights.double(), num_samples=z.shape[0], replacement=True)
            if sample_weights is not None else torch.randperm(z.shape[0]))
    total_loss = 0.0
    n_seen     = 0
    for start in range(0, z.shape[0], batch_size):
        idx = perm[start:start+batch_size]
        z_b = z[idx].to(device)
        y_b = y[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(z_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * y_b.numel()
        n_seen     += y_b.numel()
    return total_loss / max(1, n_seen)


def flatten_wandb_metrics(metrics, prefix):
    out = {f"{prefix}/{k}": metrics[k]
           for k in ["loss", "weighted_f1", "weighted_auc", "weighted_pauc"]}
    for name in ACTION_CLASSES:
        out[f"{prefix}/f1_{name}"] = metrics[f"f1_{name}"]
    if "pa" in metrics:
        out[f"{prefix}/pa"] = metrics["pa"]
    return out


def history_fields():
    base = ["loss", "weighted_f1", "weighted_auc", "weighted_pauc"] + \
           [f"f1_{c}" for c in ACTION_CLASSES]
    fields = ["epoch", "lr"]
    for split in ["train", "val"]:
        fields.extend(f"{split}_{m}" for m in base)
    return fields


def save_history_csv(history, path):
    fields = history_fields()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_final_metrics_csv(final_metrics, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        for split, m in final_metrics.items():
            for k, v in m.items():
                writer.writerow({"split": split, "metric": k, "value": v})


def save_history_plots(history, run_dir, plot_formats):
    epochs = [r["epoch"] for r in history]
    panels = [("loss", "Cross Entropy"), ("weighted_f1", "Weighted F1"),
              ("weighted_auc", "Weighted OvR AUC"), ("weighted_pauc", "Weighted OvR pAUC")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (metric, title) in zip(axes.flatten(), panels):
        for split in ["train", "val"]:
            ax.plot(epochs, [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    paths = []
    for fmt in plot_formats:
        p = os.path.join(run_dir, f"probe_train_val_curves.{fmt}")
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)

    f1_fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ["train", "val"]):
        for name in ACTION_CLASSES:
            ax.plot(epochs, [r[f"{split}_f1_{name}"] for r in history], label=name)
        ax.set_title(f"{split.title()} Per-Class F1")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    f1_fig.tight_layout()
    for fmt in plot_formats:
        p = os.path.join(run_dir, f"per_class_f1_curves.{fmt}")
        f1_fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(f1_fig)
    return paths


def plot_confusion(y_true, logits, split, run_dir, plot_formats):
    y_np = y_true.cpu().numpy()
    pred = logits.softmax(dim=-1).argmax(dim=1).cpu().numpy()
    cm   = confusion_matrix(y_np, pred, labels=np.arange(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{split} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(ACTION_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(ACTION_CLASSES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    paths = []
    for fmt in plot_formats:
        p = os.path.join(run_dir, f"{split}_confusion_matrix.{fmt}")
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_roc(y_true, logits, split, run_dir, plot_formats, pauc_max_fpr):
    y_np  = y_true.cpu().numpy()
    probs = logits.softmax(dim=-1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    for c, name in enumerate(ACTION_CLASSES):
        y_bin = (y_np == c).astype(np.int64)
        if np.unique(y_bin).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        auc  = roc_auc_score(y_bin, probs[:, c])
        pauc = roc_auc_score(y_bin, probs[:, c], max_fpr=pauc_max_fpr)
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f} pAUC={pauc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_title(f"{split} One-vs-Rest ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = []
    for fmt in plot_formats:
        p = os.path.join(run_dir, f"{split}_roc_curves.{fmt}")
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────
def train(args):
    seed_everything(int(args.train_seed))
    device = resolve_device(args.device)
    print(f"Device: {device}  |  Train seed: {args.train_seed}")

    encoder, robot_mean, robot_std = load_encoder(args, device)
    print("Frozen HGNNCombinedVAE encoder loaded.")

    dataset = UnifiedGraphDataset(
        args.h5_path,
        ssub=args.ssub,
        robot_mean=robot_mean,
        robot_std=robot_std,
    )
    splits = torch.load(args.split_indices_path, weights_only=False)

    ssub_splits = {}
    for split_name in ["train", "val"]:
        ssub_real  = dataset.filter_by_source(splits[split_name], DATASET_SSUB_REAL)
        ssub_synth = dataset.filter_by_source(splits[split_name], DATASET_SSUB_SYNTH)
        ssub_splits[split_name] = ssub_real + ssub_synth
        print(f"  {split_name}: {len(ssub_splits[split_name])} SSUB samples "
              f"({len(ssub_real)} real + {len(ssub_synth)} synth) "
              f"of {len(splits[split_name])} total")

    train_pct = float(getattr(args, "train_pct", 1.0))
    if train_pct < 1.0:
        train_indices = ssub_splits["train"]
        scene_to_indices = {}
        for idx in train_indices:
            scene = dataset.scenes[idx]
            scene_to_indices.setdefault(scene, []).append(idx)
        all_scenes = sorted(scene_to_indices.keys())
        n_total = len(all_scenes)
        n_use = max(1, math.ceil(n_total * train_pct))
        selected_scenes = sorted(random.sample(all_scenes, n_use))
        ssub_splits["train"] = [idx for s in selected_scenes for idx in scene_to_indices[s]]
        print(f"  train_pct={train_pct:.2f}: using {n_use}/{n_total} SSUB scenes "
              f"→ {len(ssub_splits['train'])} samples")

    persistent = False
    loaders = {
        split: DataLoader(
            torch.utils.data.Subset(dataset, ssub_splits[split]),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            persistent_workers=persistent,
        )
        for split in ["train", "val"]
    }
    print(f"SSUB split: train={len(ssub_splits['train'])}  val={len(ssub_splits['val'])}")

    embeddings, labels = {}, {}
    for split in ["train", "val"]:
        z, y = extract_embeddings(encoder, loaders[split], device)
        embeddings[split] = z
        labels[split]     = y
        print(f"  {split}: {z.shape[0]} SSUB human nodes  classes={class_counts(y)}")
    del loaders

    probe = nn.Linear(int(args.latent_dim), NUM_CLASSES).to(device)
    print(f"Probe params: {sum(p.numel() for p in probe.parameters()):,}")

    optimizer = torch.optim.AdamW(probe.parameters(),
                                  lr=float(args.lr),
                                  weight_decay=float(args.weight_decay))

    use_weighted  = getattr(args, "use_weighted_sampling", False)
    use_adaptive  = getattr(args, "adaptive_reweight", False)
    warmup_epochs = int(getattr(args, "adaptive_warmup_epochs", 10))
    ramp_epochs   = int(getattr(args, "adaptive_ramp_epochs", 200))
    max_alpha     = float(getattr(args, "adaptive_max_alpha", 0.5))

    stratified_weights = None
    current_weights    = None
    if use_weighted:
        stratified_weights = compute_stratified_weights(labels["train"])
        current_weights    = stratified_weights.clone()
        ess = float(1.0 / (stratified_weights ** 2).sum())
        print(f"Stratified sampling: weight range=[{stratified_weights.min():.6f}, "
              f"{stratified_weights.max():.6f}]  ESS={ess:.0f}/{len(labels['train'])}")

    use_wandb = bool(args.wandb)
    if use_wandb and wandb is None:
        print("wandb not installed — continuing without W&B.")
        use_wandb = False
    wb_run = None
    if use_wandb:
        wb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or os.path.basename(args.run_dir),
            config=vars(args),
            dir=args.run_dir,
        )
    else:
        print("wandb logging disabled.")

    best_val   = float("inf")
    best_epoch = 0
    no_improve = 0
    patience   = int(args.early_stopping_patience)
    min_delta  = float(args.early_stopping_min_delta)
    best_path  = os.path.join(args.run_dir, "best_linear_probe.pt")
    history    = []
    pauc_max_fpr = float(args.pauc_max_fpr)
    eval_every   = max(1, int(args.eval_metrics_every))

    for epoch in range(1, int(args.epochs) + 1):
        if use_weighted and use_adaptive and stratified_weights is not None and epoch > warmup_epochs:
            loss_weights = compute_per_sample_losses(
                probe, embeddings["train"], labels["train"], device, int(args.batch_size)
            )
            loss_weights    = loss_weights / loss_weights.sum()
            epochs_since    = epoch - warmup_epochs
            alpha           = min(max_alpha, max_alpha * epochs_since / max(ramp_epochs, 1))
            current_weights = (1.0 - alpha) * stratified_weights + alpha * loss_weights
            current_weights = current_weights / current_weights.sum()
            ess = float(1.0 / (current_weights ** 2).sum())
            print(f"  Adaptive reweight: alpha={alpha:.3f}  ESS={ess:.0f}/{len(labels['train'])}")

        train_one_epoch(probe, embeddings["train"], labels["train"], device,
                        optimizer, int(args.batch_size), sample_weights=current_weights)

        train_metrics, _ = eval_probe(probe, embeddings["train"], labels["train"],
                                      device, int(args.batch_size), pauc_max_fpr)
        val_metrics,   _ = eval_probe(probe, embeddings["val"],   labels["val"],
                                      device, int(args.batch_size), pauc_max_fpr)

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for split, m in [("train", train_metrics), ("val", val_metrics)]:
            for k, v in m.items():
                row[f"{split}_{k}"] = v
        history.append(row)

        if wb_run is not None:
            log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            log.update(flatten_wandb_metrics(train_metrics, "train"))
            log.update(flatten_wandb_metrics(val_metrics,   "val"))
            wb_run.log(log, step=epoch)

        if epoch == 1 or epoch % eval_every == 0:
            print(
                f"Epoch {epoch:04d}/{args.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"train_wf1={train_metrics['weighted_f1']:.3f} "
                f"val_wf1={val_metrics['weighted_f1']:.3f} | "
                f"val_auc={val_metrics['weighted_auc']:.3f}  "
                f"val_pauc={val_metrics['weighted_pauc']:.3f}"
            )

        if val_metrics["loss"] < best_val - min_delta:
            best_val   = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch":              epoch,
                "probe_state_dict":   probe.state_dict(),
                "ae_checkpoint":      args.checkpoint,
                "split_indices_path": args.split_indices_path,
                "ssub":               args.ssub,
                "args":               vars(args),
                "val_metrics":        val_metrics,
            }, best_path)
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    probe.load_state_dict(ckpt["probe_state_dict"])
    probe.eval()

    final_metrics, final_logits = {}, {}
    final_labels = {}
    for split in ["train", "val"]:
        m, lg = eval_probe(probe, embeddings[split], labels[split],
                           device, int(args.batch_size), pauc_max_fpr)
        final_metrics[split] = m
        final_logits[split]  = lg
        final_labels[split]  = labels[split]

    if hasattr(args, "eval_h5_path") and args.eval_h5_path:
        print(f"\nLoading eval dataset: {args.eval_h5_path}")
        eval_dataset = UnifiedGraphDataset(
            args.eval_h5_path,
            ssub=args.ssub,
            robot_mean=robot_mean,
            robot_std=robot_std,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
        test_z, test_y, test_probs = extract_test_embeddings(encoder, eval_loader, device)
        test_metrics, test_logits = eval_probe(probe, test_z, test_y,
                                               device, int(args.batch_size), pauc_max_fpr)
        test_metrics["pa"] = compute_pa(test_logits, test_probs)
        final_metrics["test"] = test_metrics
        final_logits["test"]  = test_logits
        final_labels["test"]  = test_y
        print(f"  test: {test_z.shape[0]} nodes  classes={class_counts(test_y)}  "
              f"PA={test_metrics['pa']:.3f}")
        del eval_loader

    save_history_csv(history, os.path.join(args.run_dir, args.history_csv_filename))
    save_final_metrics_csv(final_metrics, os.path.join(args.run_dir, "final_metrics.csv"))

    if bool(args.save_plots):
        fmts       = normalize_plot_formats(args.plot_formats)
        plot_paths = save_history_plots(history, args.run_dir, fmts)
        for split in final_labels:
            plot_paths += plot_confusion(final_labels[split], final_logits[split],
                                         split, args.run_dir, fmts)
            plot_paths += plot_roc(final_labels[split], final_logits[split],
                                   split, args.run_dir, fmts, pauc_max_fpr)
        if wb_run is not None:
            for p in plot_paths:
                if os.path.splitext(p)[1].lower() in {".png", ".jpg", ".jpeg"}:
                    wb_run.log({f"plots/{os.path.splitext(os.path.basename(p))[0]}":
                                wandb.Image(p)})

    if wb_run is not None:
        final_log = {}
        for split, m in final_metrics.items():
            final_log.update(flatten_wandb_metrics(m, f"final/{split}"))
        wb_run.log(final_log)
        for k, v in final_log.items():
            wb_run.summary[k] = v
        wb_run.finish()

    print(f"\nBest epoch: {best_epoch} | best val loss: {best_val:.4f}")
    print(f"Checkpoint: {best_path}")
    for split in final_metrics:
        m = final_metrics[split]
        line = (f"  {split}: loss={m['loss']:.4f}  weighted_f1={m['weighted_f1']:.3f}  "
                f"weighted_auc={m['weighted_auc']:.3f}  weighted_pauc={m['weighted_pauc']:.3f}")
        if "pa" in m:
            line += f"  pa={m['pa']:.3f}"
        print(line)

    return final_metrics, best_epoch


def summary_csv_fields():
    fields = ["run", "train_seed", "train_pct", "best_epoch"]
    for split in ["train", "val"]:
        for m in ["loss", "weighted_f1", "weighted_auc", "weighted_pauc"] + \
                 [f"f1_{c}" for c in ACTION_CLASSES]:
            fields.append(f"{split}_{m}")
    for c in ACTION_CLASSES:
        fields.append(f"test_f1_{c}")
    fields.append("test_pa")
    return fields


def save_summary_csv(rows, path):
    fields = summary_csv_fields()
    numeric_fields = [f for f in fields if f not in ("run", "train_seed", "train_pct", "best_epoch")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
        vals = {}
        for k in numeric_fields:
            v = [r[k] for r in rows if k in r and r[k] != ""]
            if v:
                vals[k] = v
        mean_row = {"run": "mean", "train_seed": "", "train_pct": "", "best_epoch": ""}
        std_row  = {"run": "std",  "train_seed": "", "train_pct": "", "best_epoch": ""}
        for k, v in vals.items():
            arr = np.array(v)
            mean_row[k] = f"{arr.mean():.4f}"
            std_row[k]  = f"{arr.std():.4f}"
        writer.writerow(mean_row)
        writer.writerow(std_row)


TRAIN_PCTS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params/config.yaml")
    parser.add_argument("--ssub", action="store_true",
                        help="SSUB-only mode (no valid_mask, no node features)")
    parser.add_argument("--train_seed", type=int, default=None,
                        help="Seed for probe init/training (random if omitted)")
    parser.add_argument("--train_pct", type=float, default=None,
                        help="Single fraction to run (omit to sweep all: 0.01,0.05,0.1,0.25,0.5,1.0)")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of independent probe runs per percentile")
    cli_args = parser.parse_args()

    args     = load_config(cli_args.config)
    args.ssub = cli_args.ssub
    num_runs = cli_args.num_runs
    pcts     = [cli_args.train_pct] if cli_args.train_pct is not None else TRAIN_PCTS

    if num_runs <= 1 and len(pcts) == 1:
        args.train_pct  = pcts[0]
        args.train_seed = cli_args.train_seed if cli_args.train_seed is not None else random.randint(0, 2**31 - 1)
        args.run_dir = create_run_dir(args.experiments_root, args.run_name)
        shutil.copy2(args.config_path, os.path.join(args.run_dir, "config.yaml"))
        with open(os.path.join(args.run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)
        train(args)
    else:
        parent_dir = create_run_dir(args.experiments_root,
                                    f"{args.run_name}_{num_runs}runs")
        shutil.copy2(args.config_path, os.path.join(parent_dir, "config.yaml"))
        all_rows = []

        for pct in pcts:
            args.train_pct = pct
            pct_dir = os.path.join(parent_dir, f"pct_{int(pct*100):03d}")
            os.makedirs(pct_dir)

            for run_idx in range(1, num_runs + 1):
                args.train_seed = random.randint(0, 2**31 - 1)
                args.run_dir = os.path.join(pct_dir, f"run_{run_idx:02d}")
                os.makedirs(args.run_dir)
                with open(os.path.join(args.run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
                    yaml.safe_dump(vars(args), f, sort_keys=False)

                print(f"\n{'='*60}")
                print(f"pct={pct:.2f}  |  Run {run_idx}/{num_runs}  |  train_seed={args.train_seed}")
                print(f"{'='*60}")

                final_metrics, best_epoch = train(args)

                row = {"run": run_idx, "train_seed": args.train_seed,
                       "train_pct": pct, "best_epoch": best_epoch}
                for split in ["train", "val"]:
                    if split in final_metrics:
                        for k, v in final_metrics[split].items():
                            row[f"{split}_{k}"] = v
                if "test" in final_metrics:
                    test_m = final_metrics["test"]
                    for c in ACTION_CLASSES:
                        row[f"test_f1_{c}"] = test_m.get(f"f1_{c}", "")
                    row["test_pa"] = test_m.get("pa", "")
                all_rows.append(row)

        summary_path = os.path.join(parent_dir, "summary.csv")
        save_summary_csv(all_rows, summary_path)
        print(f"\n{'='*60}")
        print(f"All done. {len(pcts)} percentiles × {num_runs} runs = {len(all_rows)} total")
        print(f"Summary → {summary_path}")
        print(f"{'='*60}")
