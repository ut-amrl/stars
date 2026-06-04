"""
Stage 2: supervised linear probe on frozen SocialNav-SUB HGNN autoencoder.

The probe is trained on per-human node embeddings from HGNNSSubAE.encode_nodes.
For v2 graphs, all nodes are valid; the probe keeps human nodes and ignores
the robot node because the action labels are per-human.
"""

import argparse
import csv
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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import DataLoader

try:
    import wandb
except ImportError:
    wandb = None


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from dataloaders.socialnav_homo_dataset_v2 import NODE_TYPE_HUMAN, SocialNavHGraphDatasetHomoV2
from models.hgnn_ssub_v2 import HGNNSSubAE


ACTION_CLASSES = ["avoid", "follow", "overtake", "yield", "not_consider"]
NUM_CLASSES = len(ACTION_CLASSES)


def resolve_path(path, base_dir=SRC_DIR):
    path = str(path)
    if os.path.isabs(path):
        return path
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(os.path.join(base_dir, path))


def load_config(config_path):
    config_path = resolve_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = cfg["linear_probe_hgnn_ssub_v1"]
    args = SimpleNamespace(**cfg)
    args.config_path = config_path
    args.checkpoint = resolve_path(args.checkpoint)
    args.h5_path = resolve_path(args.h5_path)
    args.split_indices_path = resolve_path(args.split_indices_path)
    args.experiments_root = resolve_path(args.experiments_root)
    return args


def create_run_dir(experiments_root, run_name):
    os.makedirs(experiments_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = str(run_name).strip()
    dirname = f"{timestamp}_{name}" if name else timestamp
    run_dir = os.path.join(experiments_root, dirname)

    suffix = 1
    base = run_dir
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = f"{base}_{suffix}"

    os.makedirs(run_dir)
    return run_dir


def resolve_device(device_cfg):
    device_cfg = str(device_cfg).lower()
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


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


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.classifier(z)


def compute_stratified_weights_from_labels(y):
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    total = counts.sum()
    inv_freq = (total / counts.clamp(min=1)).sqrt()
    weights = inv_freq[y]
    return weights / weights.sum()


def compute_per_sample_ce_losses(probe, z, y, device, batch_size):
    probe.eval()
    all_losses = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start : start + batch_size].to(device)
            y_b = y[start : start + batch_size].to(device)
            logits = probe(z_b)
            per_sample = F.cross_entropy(logits, y_b, reduction="none")
            all_losses.append(per_sample.cpu())
    return torch.cat(all_losses, dim=0)


def load_encoder(args, device):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})

    model = HGNNSSubAE(
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        num_mp_layers=int(args.num_mp_layers),
        edge_transformer_layers=int(args.edge_transformer_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"Missing checkpoint keys left at init: {list(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys ignored: {list(unexpected)}")

    for key in ["hidden_dim", "latent_dim", "num_mp_layers", "edge_transformer_layers", "num_heads"]:
        if isinstance(saved_args, dict) and key in saved_args and int(saved_args[key]) != int(getattr(args, key)):
            print(
                f"WARNING: config {key}={getattr(args, key)} differs from "
                f"checkpoint args {key}={saved_args[key]}."
            )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, ckpt


def extract_human_embeddings(model, loader, device):
    all_z = []
    all_y = []
    model.eval()

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            z = model.encode_nodes(batch)

        y = batch.action_label
        keep = batch.node_type == NODE_TYPE_HUMAN
        all_z.append(z[keep].cpu())
        all_y.append(y[keep].cpu())

    if not all_z:
        raise RuntimeError("No batches were loaded while extracting embeddings.")

    z = torch.cat(all_z, dim=0)
    y = torch.cat(all_y, dim=0).long()
    return z, y


def make_loaders(dataset, splits, batch_size, num_workers):
    persistent = num_workers > 0
    loaders = {}
    for split in ["train", "val", "test"]:
        subset = torch.utils.data.Subset(dataset, splits[split])
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent,
        )
    return loaders


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


def compute_metrics(logits, labels, pauc_max_fpr):
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    y_pred = probs.argmax(axis=1)
    support = np.bincount(y_true, minlength=NUM_CLASSES)
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average=None,
        zero_division=0,
    )

    per_class_auc = []
    per_class_pauc = []
    for class_idx in range(NUM_CLASSES):
        binary_true = (y_true == class_idx).astype(np.int64)
        if np.unique(binary_true).size < 2:
            per_class_auc.append(float("nan"))
            per_class_pauc.append(float("nan"))
            continue
        per_class_auc.append(float(roc_auc_score(binary_true, probs[:, class_idx])))
        per_class_pauc.append(
            float(roc_auc_score(binary_true, probs[:, class_idx], max_fpr=pauc_max_fpr))
        )

    metrics = {
        "loss": float(F.cross_entropy(logits, labels).detach().cpu()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES), average="weighted", zero_division=0)),
        "macro_auc": float(np.nanmean(per_class_auc)) if not np.all(np.isnan(per_class_auc)) else float("nan"),
        "weighted_auc": weighted_nanmean(per_class_auc, support),
        "macro_pauc": float(np.nanmean(per_class_pauc)) if not np.all(np.isnan(per_class_pauc)) else float("nan"),
        "weighted_pauc": weighted_nanmean(per_class_pauc, support),
    }
    for class_idx, class_name in enumerate(ACTION_CLASSES):
        metrics[f"f1_{class_name}"] = float(per_class_f1[class_idx])
    return metrics


def flatten_wandb_metrics(metrics, prefix):
    out = {
        f"{prefix}/loss": metrics["loss"],
        f"{prefix}/weighted_f1": metrics["weighted_f1"],
        f"{prefix}/macro_f1": metrics["macro_f1"],
        f"{prefix}/weighted_auc": metrics["weighted_auc"],
        f"{prefix}/macro_auc": metrics["macro_auc"],
        f"{prefix}/weighted_pauc": metrics["weighted_pauc"],
        f"{prefix}/macro_pauc": metrics["macro_pauc"],
    }
    for class_name in ACTION_CLASSES:
        out[f"{prefix}/f1_{class_name}"] = metrics[f"f1_{class_name}"]
    return out


def eval_probe(probe, z, y, device, batch_size, pauc_max_fpr):
    probe.eval()
    logits_chunks = []
    loss_total = 0.0
    n_seen = 0

    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start : start + batch_size].to(device)
            y_b = y[start : start + batch_size].to(device)
            logits = probe(z_b)
            loss = F.cross_entropy(logits, y_b)
            logits_chunks.append(logits.cpu())
            loss_total += float(loss.cpu()) * y_b.numel()
            n_seen += y_b.numel()

    logits = torch.cat(logits_chunks, dim=0)
    metrics = compute_metrics(logits, y, pauc_max_fpr)
    metrics["loss"] = loss_total / max(1, n_seen)
    return metrics, logits


def focal_loss(logits, targets, gamma=2.0, class_weights=None):
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    p_correct = torch.exp(-ce)
    return ((1 - p_correct) ** gamma * ce).mean()


def train_one_epoch(probe, z, y, device, optimizer, batch_size, sample_weights=None, class_weights=None, focal_gamma=None):
    probe.train()
    if sample_weights is not None:
        perm = torch.multinomial(sample_weights.double(), num_samples=z.shape[0], replacement=True)
    else:
        perm = torch.randperm(z.shape[0])
    total_loss = 0.0
    n_seen = 0

    for start in range(0, z.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        z_b = z[idx].to(device)
        y_b = y[idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = probe(z_b)
        if focal_gamma is not None:
            loss = focal_loss(logits, y_b, gamma=focal_gamma, class_weights=class_weights)
        else:
            loss = F.cross_entropy(logits, y_b, weight=class_weights)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * y_b.numel()
        n_seen += y_b.numel()

    return total_loss / max(1, n_seen)


def history_fields():
    fields = ["epoch", "lr"]
    base_metrics = [
        "loss",
        "macro_f1",
        "weighted_f1",
        "macro_auc",
        "weighted_auc",
        "macro_pauc",
        "weighted_pauc",
    ]
    base_metrics.extend(f"f1_{class_name}" for class_name in ACTION_CLASSES)
    for split in ["train", "val"]:
        fields.extend(f"{split}_{metric}" for metric in base_metrics)
    return fields


def history_row(epoch, lr, train_metrics, val_metrics):
    row = {"epoch": epoch, "lr": lr}
    for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
        for key, value in metrics.items():
            row[f"{split}_{key}"] = value
    return row


def save_history_csv(history, path):
    fields = history_fields()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_final_metrics_csv(final_metrics, path):
    fields = ["split", "metric", "value"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, metrics in final_metrics.items():
            for key, value in metrics.items():
                writer.writerow({"split": split, "metric": key, "value": value})


def save_history_plots(history, run_dir, plot_formats):
    epochs = [row["epoch"] for row in history]
    panels = [
        ("loss", "Cross Entropy"),
        ("weighted_f1", "Weighted F1"),
        ("macro_f1", "Macro F1"),
        ("weighted_auc", "Weighted OvR AUC"),
        ("weighted_pauc", "Weighted OvR pAUC"),
        ("macro_pauc", "Macro OvR pAUC"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, panels):
        for split in ["train", "val"]:
            values = [row[f"{split}_{metric}"] for row in history]
            ax.plot(epochs, values, label=split)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()

    paths = []
    for fmt in plot_formats:
        path = os.path.join(run_dir, f"probe_train_val_curves.{fmt}")
        fig.savefig(path, format=fmt)
        paths.append(path)
    plt.close(fig)

    f1_fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ["train", "val"]):
        for class_name in ACTION_CLASSES:
            values = [row[f"{split}_f1_{class_name}"] for row in history]
            ax.plot(epochs, values, label=class_name)
        ax.set_title(f"{split.title()} Per-Class F1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    f1_fig.tight_layout()

    for fmt in plot_formats:
        path = os.path.join(run_dir, f"per_class_f1_curves.{fmt}")
        f1_fig.savefig(path, format=fmt)
        paths.append(path)
    plt.close(f1_fig)
    return paths


def plot_confusion(y_true, logits, split, run_dir, plot_formats):
    y_np = y_true.cpu().numpy()
    pred = logits.softmax(dim=-1).argmax(dim=1).cpu().numpy()
    cm = confusion_matrix(y_np, pred, labels=np.arange(NUM_CLASSES))

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
        path = os.path.join(run_dir, f"{split}_confusion_matrix.{fmt}")
        fig.savefig(path, format=fmt)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_roc(y_true, logits, split, run_dir, plot_formats, pauc_max_fpr):
    y_np = y_true.cpu().numpy()
    probs = logits.softmax(dim=-1).cpu().numpy()

    roc_fig, roc_ax = plt.subplots(figsize=(8, 7))
    plotted_roc = False

    for class_idx, class_name in enumerate(ACTION_CLASSES):
        y_bin = (y_np == class_idx).astype(np.int64)
        if np.unique(y_bin).size < 2:
            continue

        fpr, tpr, _ = roc_curve(y_bin, probs[:, class_idx])
        roc_auc = roc_auc_score(y_bin, probs[:, class_idx])
        roc_pauc = roc_auc_score(y_bin, probs[:, class_idx], max_fpr=pauc_max_fpr)
        roc_ax.plot(fpr, tpr, label=f"{class_name} AUC={roc_auc:.3f} pAUC={roc_pauc:.3f}")
        plotted_roc = True

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    roc_ax.set_title(f"{split} One-vs-Rest ROC")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.grid(True, alpha=0.3)
    if plotted_roc:
        roc_ax.legend(fontsize=8)
    roc_fig.tight_layout()

    paths = []
    for fmt in plot_formats:
        roc_path = os.path.join(run_dir, f"{split}_roc_curves.{fmt}")
        roc_fig.savefig(roc_path, format=fmt)
        paths.append(roc_path)
    plt.close(roc_fig)
    return paths


def save_all_plots(history, final_logits, labels_by_split, run_dir, plot_formats, pauc_max_fpr):
    paths = save_history_plots(history, run_dir, plot_formats)
    for split in ["train", "val", "test"]:
        paths.extend(plot_confusion(labels_by_split[split], final_logits[split], split, run_dir, plot_formats))
        paths.extend(plot_roc(labels_by_split[split], final_logits[split], split, run_dir, plot_formats, pauc_max_fpr))
    return paths


def train(args):
    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    print(f"Device: {device}")

    dataset = SocialNavHGraphDatasetHomoV2(args.h5_path)
    splits = torch.load(args.split_indices_path, weights_only=False)

    loaders = make_loaders(
        dataset,
        splits,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    encoder, _ = load_encoder(args, device)
    print("Frozen SocialNav HGNN encoder loaded.")
    print(
        f"Using fixed split: train={len(splits['train'])} graphs | "
        f"val={len(splits['val'])} graphs | test={len(splits['test'])} graphs"
    )

    embeddings = {}
    labels = {}
    for split in ["train", "val", "test"]:
        z, y = extract_human_embeddings(encoder, loaders[split], device)
        embeddings[split] = z
        labels[split] = y
        print(f"{split}: z={tuple(z.shape)} labels={class_counts(y)}")

    use_class_weighted_loss = getattr(args, "use_class_weighted_loss", False)
    class_weights = None
    if use_class_weighted_loss:
        counts = torch.bincount(labels["train"], minlength=NUM_CLASSES).float()
        class_weights = (counts.sum() / (NUM_CLASSES * counts.clamp(min=1))).to(device)
        print(
            "Class-weighted CE enabled: "
            + " ".join(f"{ACTION_CLASSES[i]}={class_weights[i]:.2f}" for i in range(NUM_CLASSES))
        )

    focal_gamma = None
    if getattr(args, "use_focal_loss", False):
        focal_gamma = float(getattr(args, "focal_gamma", 2.0))
        print(f"Focal loss enabled: gamma={focal_gamma}")

    use_weighted = getattr(args, "use_weighted_sampling", False)
    use_adaptive = getattr(args, "adaptive_reweight", False)
    warmup_epochs = getattr(args, "adaptive_warmup_epochs", 10)
    max_alpha = getattr(args, "adaptive_max_alpha", 0.5)

    stratified_weights = None
    current_weights = None
    if use_weighted:
        stratified_weights = compute_stratified_weights_from_labels(labels["train"])
        current_weights = stratified_weights.clone()
        ess = float(1.0 / (stratified_weights**2).sum())
        print(
            f"Weighted sampling enabled: "
            f"weight range=[{stratified_weights.min():.6f}, {stratified_weights.max():.6f}] "
            f"ESS={ess:.0f}/{len(labels['train'])}"
        )

    probe_hidden = int(getattr(args, "probe_hidden_dim", 128))
    probe_drop = float(getattr(args, "probe_dropout", 0.1))
    probe = MLPProbe(
        input_dim=int(args.latent_dim),
        hidden_dim=probe_hidden,
        num_classes=NUM_CLASSES,
        dropout=probe_drop,
    ).to(device)
    print(f"Probe params: {sum(p.numel() for p in probe.parameters()):,}")

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    use_wandb = bool(args.wandb)
    if use_wandb and wandb is None:
        print("wandb requested in config, but package is not installed; continuing without W&B.")
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

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = os.path.join(args.run_dir, "best_linear_probe_hgnn_ssub_v1.pt")
    history = []
    pauc_max_fpr = float(args.pauc_max_fpr)

    for epoch in range(1, int(args.epochs) + 1):
        if (
            use_weighted
            and use_adaptive
            and stratified_weights is not None
            and epoch > warmup_epochs
        ):
            loss_weights = compute_per_sample_ce_losses(
                probe, embeddings["train"], labels["train"], device, int(args.batch_size)
            )
            loss_weights = loss_weights / loss_weights.sum()

            epochs_since_warmup = epoch - warmup_epochs
            max_adaptive_epochs = max(int(args.epochs) - warmup_epochs, 1)
            alpha = min(max_alpha, max_alpha * epochs_since_warmup / max_adaptive_epochs)

            current_weights = (1.0 - alpha) * stratified_weights + alpha * loss_weights
            current_weights = current_weights / current_weights.sum()

            ess = float(1.0 / (current_weights**2).sum())
            print(f"  Adaptive reweight: alpha={alpha:.3f}, ESS={ess:.0f}/{len(labels['train'])}")

        train_one_epoch(
            probe,
            embeddings["train"],
            labels["train"],
            device,
            optimizer,
            int(args.batch_size),
            sample_weights=current_weights,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )

        train_metrics, _ = eval_probe(
            probe,
            embeddings["train"],
            labels["train"],
            device,
            int(args.batch_size),
            pauc_max_fpr,
        )
        val_metrics, _ = eval_probe(
            probe,
            embeddings["val"],
            labels["val"],
            device,
            int(args.batch_size),
            pauc_max_fpr,
        )

        scheduler.step(val_metrics["loss"])

        row = history_row(epoch, optimizer.param_groups[0]["lr"], train_metrics, val_metrics)
        history.append(row)

        log_dict = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_dict.update(flatten_wandb_metrics(train_metrics, "train"))
        log_dict.update(flatten_wandb_metrics(val_metrics, "val"))
        if use_weighted and use_adaptive and epoch > warmup_epochs:
            log_dict["sampling/alpha"] = alpha
            log_dict["sampling/effective_sample_size"] = ess
        if wb_run is not None:
            wb_run.log(log_dict, step=epoch)

        should_print = (
            epoch == 1
            or epoch == int(args.epochs)
            or epoch % max(1, int(args.eval_metrics_every)) == 0
        )
        if should_print:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"train_wf1={train_metrics['weighted_f1']:.3f} "
                f"val_wf1={val_metrics['weighted_f1']:.3f} | "
                f"val_auc={val_metrics['weighted_auc']:.3f} "
                f"val_pauc={val_metrics['weighted_pauc']:.3f}"
            )

        improved = val_metrics["loss"] < (best_val - float(args.early_stopping_min_delta))
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "probe_state_dict": probe.state_dict(),
                    "encoder_checkpoint": args.checkpoint,
                    "split_indices_path": args.split_indices_path,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            int(args.early_stopping_patience) > 0
            and epochs_without_improvement >= int(args.early_stopping_patience)
        ):
            print(
                "Early stopping triggered: "
                f"no val loss improvement for {epochs_without_improvement} epochs."
            )
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    probe.load_state_dict(ckpt["probe_state_dict"])
    probe.eval()

    final_metrics = {}
    final_logits = {}
    for split in ["train", "val", "test"]:
        metrics, logits = eval_probe(
            probe,
            embeddings[split],
            labels[split],
            device,
            int(args.batch_size),
            pauc_max_fpr,
        )
        final_metrics[split] = metrics
        final_logits[split] = logits

    save_history_csv(history, os.path.join(args.run_dir, args.history_csv_filename))
    save_final_metrics_csv(final_metrics, os.path.join(args.run_dir, "final_metrics.csv"))

    plot_paths = []
    if bool(args.save_plots):
        plot_paths = save_all_plots(
            history,
            final_logits,
            labels,
            args.run_dir,
            normalize_plot_formats(args.plot_formats),
            pauc_max_fpr,
        )

    if wb_run is not None:
        final_log = {}
        for split, metrics in final_metrics.items():
            final_log.update(flatten_wandb_metrics(metrics, f"final/{split}"))
        wb_run.log(final_log)
        for key, value in final_log.items():
            wb_run.summary[key] = value

        for path in plot_paths:
            stem, ext = os.path.splitext(os.path.basename(path))
            if ext.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            wb_run.log({f"plots/{stem}": wandb.Image(path)})

        wb_run.finish()

    print(f"Best epoch: {best_epoch} | best val loss: {best_val:.4f}")
    print(f"Checkpoint: {best_path}")
    for split in ["train", "val", "test"]:
        metrics = final_metrics[split]
        print(
            f"{split}: loss={metrics['loss']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"weighted_auc={metrics['weighted_auc']:.3f} "
            f"weighted_pauc={metrics['weighted_pauc']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="params/config.yaml",
        help="Path to YAML config file. Training parameters come from linear_probe_hgnn_ssub_v1.",
    )
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    args.run_dir = create_run_dir(args.experiments_root, args.run_name)

    shutil.copy2(args.config_path, os.path.join(args.run_dir, "config.yaml"))
    with open(os.path.join(args.run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    train(args)