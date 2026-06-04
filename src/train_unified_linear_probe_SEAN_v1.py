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
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset, DATASET_SEAN
from models.hgnn_combined_v2 import HGNNCombinedVAE

LABEL_NAMES = ["competence", "surprise", "intention"]
LABEL_THRESHOLD = 3


def load_config(config_path):
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    section = "linear_probe_sean_params"
    if section not in cfg:
        raise ValueError(f"Config missing section '{section}'")
    args = SimpleNamespace(**cfg[section])
    args.config_path        = config_path
    args.checkpoint         = os.path.abspath(args.checkpoint)
    args.split_indices_path = os.path.abspath(args.split_indices_path)
    args.h5_path            = os.path.abspath(args.h5_path)
    args.experiments_root   = os.path.abspath(args.experiments_root)
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

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, robot_mean, robot_std


def binarize_labels(emotion_label):
    el = emotion_label.float()
    y = torch.zeros(3, dtype=torch.float32)
    y[0] = 1.0 if el[0].item() >= LABEL_THRESHOLD else 0.0
    y[1] = 1.0 if el[1].item() < LABEL_THRESHOLD else 0.0  # surprise inverted
    y[2] = 1.0 if el[2].item() >= LABEL_THRESHOLD else 0.0
    return y


def sigmoid_focal_loss(logits, targets, gamma=2.0, alpha=0.25, reduction="mean"):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == "mean":
        return focal.mean()
    if reduction == "sum":
        return focal.sum()
    return focal


def compute_label_loss(logits, targets, use_focal, gamma, alpha, reduction="mean"):
    if use_focal:
        return sigmoid_focal_loss(logits, targets, gamma, alpha, reduction)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)


def compute_stratified_weights(y):
    combo = (y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]).long()
    counts = torch.bincount(combo, minlength=8).float()
    inv_freq = (counts.sum() / counts.clamp(min=1)).sqrt()
    weights = inv_freq[combo]
    return weights / weights.sum()


def compute_per_sample_losses(heads, z, y, device, batch_size,
                              use_focal=False, gamma=2.0, alpha=0.25):
    for h in heads:
        h.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start:start+batch_size].to(device)
            y_b = y[start:start+batch_size].to(device)
            per_sample = torch.stack([
                compute_label_loss(heads[d](z_b).squeeze(-1), y_b[:, d],
                                   use_focal, gamma, alpha, reduction="none")
                for d in range(3)
            ], dim=1).mean(dim=1)
            losses.append(per_sample.cpu())
    return torch.cat(losses, dim=0)


def extract_embeddings(model, loader, device, latent_dim, embedding_mode="robot_only"):
    all_z, all_y = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z_full = model.encode(batch)
            if embedding_mode == "full":
                z = z_full
            else:
                z = z_full[:, :latent_dim]
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
    metrics["mean_f1"]  = float(np.mean([metrics[f"f1_{n}"] for n in LABEL_NAMES]))
    metrics["mean_auc"] = float(np.nanmean([metrics[f"auc_{n}"] for n in LABEL_NAMES]))
    return metrics


def eval_probe(heads, z, y, device, batch_size,
               use_focal=False, gamma=2.0, alpha=0.25):
    for h in heads:
        h.eval()
    logits_list = []
    loss_total  = 0.0
    n_seen      = 0
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start:start+batch_size].to(device)
            y_b = y[start:start+batch_size].to(device)
            lg  = torch.stack([heads[d](z_b).squeeze(-1) for d in range(3)], dim=1)
            logits_list.append(lg.cpu())
            loss = torch.stack([
                compute_label_loss(lg[:, d], y_b[:, d], use_focal, gamma, alpha)
                for d in range(3)
            ]).mean()
            loss_total += loss.item() * y_b.shape[0]
            n_seen     += y_b.shape[0]
    logits  = torch.cat(logits_list, dim=0)
    metrics = compute_metrics(logits.numpy(), y.numpy())
    metrics["loss"] = loss_total / max(1, n_seen)
    return metrics, logits


def flatten_wandb_metrics(metrics, prefix):
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def history_fields():
    base = ["loss", "mean_f1", "mean_auc"]
    for n in LABEL_NAMES:
        base += [f"f1_{n}", f"auc_{n}"]
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


def save_history_plots(history, run_dir, plot_formats):
    epochs = [r["epoch"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set_title("Loss (BCE)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for name in LABEL_NAMES:
        axes[1].plot(epochs, [r[f"val_f1_{name}"] for r in history], label=name)
    axes[1].set_title("Val F1")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for name in LABEL_NAMES:
        axes[2].plot(epochs, [r[f"val_auc_{name}"] for r in history], label=name)
    axes[2].set_title("Val AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    for fmt in plot_formats:
        fig.savefig(os.path.join(run_dir, f"sean_probe_curves.{fmt}"), format=fmt)
    plt.close(fig)


def plot_roc(y_true, logits, split, run_dir, plot_formats):
    y_np  = y_true.numpy()
    probs = torch.sigmoid(logits).numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    for d, name in enumerate(LABEL_NAMES):
        if np.unique(y_np[:, d]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_np[:, d], probs[:, d])
        auc = roc_auc_score(y_np[:, d], probs[:, d])
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_title(f"{split} ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for fmt in plot_formats:
        fig.savefig(os.path.join(run_dir, f"{split}_roc_curves.{fmt}"), format=fmt)
    plt.close(fig)


def train(args):
    seed_everything(int(args.train_seed))
    device = resolve_device(args.device)
    print(f"Device: {device}  |  Train seed: {args.train_seed}")

    encoder, robot_mean, robot_std = load_encoder(args, device)
    print("Frozen HGNNCombinedVAE encoder loaded.")

    dataset = UnifiedGraphDataset(
        args.h5_path,
        ssub=False,
        robot_mean=robot_mean,
        robot_std=robot_std,
    )
    splits = torch.load(args.split_indices_path, weights_only=False)

    sean_splits = {}
    for split_name in ["train", "val", "test"]:
        sean_splits[split_name] = dataset.filter_by_source(splits[split_name], DATASET_SEAN)
        print(f"  {split_name}: {len(sean_splits[split_name])} SEAN samples "
              f"(of {len(splits[split_name])} total)")

    train_pct = float(getattr(args, "train_pct", 1.0))
    if train_pct < 1.0:
        train_indices = sean_splits["train"]
        scene_to_indices = {}
        for idx in train_indices:
            scene = dataset.scenes[idx]
            scene_to_indices.setdefault(scene, []).append(idx)
        all_scenes = sorted(scene_to_indices.keys())
        n_total = len(all_scenes)
        n_use = max(1, math.ceil(n_total * train_pct))
        selected_scenes = sorted(random.sample(all_scenes, n_use))
        sean_splits["train"] = [idx for s in selected_scenes for idx in scene_to_indices[s]]
        print(f"  train_pct={train_pct:.2f}: using {n_use}/{n_total} scenes "
              f"→ {len(sean_splits['train'])} samples")

    persistent = False
    loaders = {
        split: DataLoader(
            torch.utils.data.Subset(dataset, sean_splits[split]),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            persistent_workers=persistent,
        )
        for split in ["train", "val", "test"]
    }

    latent_dim = int(args.latent_dim)
    embedding_mode = str(getattr(args, "embedding_mode", "robot_only"))
    embeddings, labels = {}, {}
    for split in ["train", "val", "test"]:
        z, y = extract_embeddings(encoder, loaders[split], device, latent_dim, embedding_mode)
        embeddings[split] = z
        labels[split]     = y
        pos_counts = y.sum(dim=0).int().tolist()
        print(f"  {split}: {z.shape[0]} graphs, embed_dim={z.shape[1]}, "
              f"pos_counts=[comp={pos_counts[0]}, surp={pos_counts[1]}, int={pos_counts[2]}]")

    del loaders

    input_dim = latent_dim * 2 if embedding_mode == "full" else latent_dim
    print(f"Embedding mode: {embedding_mode} → input_dim={input_dim}")
    heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(3)]).to(device)
    print(f"Probe params: {sum(p.numel() for p in heads.parameters()):,}")

    optimizer = torch.optim.AdamW(heads.parameters(),
                                  lr=float(args.lr),
                                  weight_decay=float(args.weight_decay))

    use_focal   = bool(getattr(args, "use_focal_loss", False))
    focal_gamma = float(getattr(args, "focal_gamma", 2.0))
    focal_alpha = float(getattr(args, "focal_alpha", 0.25))
    if use_focal:
        print(f"Focal loss enabled: gamma={focal_gamma}, alpha={focal_alpha}")

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
    best_path  = os.path.join(args.run_dir, "best_linear_probe_sean.pt")
    history    = []
    eval_every = max(1, int(args.eval_metrics_every))

    for epoch in range(1, int(args.epochs) + 1):
        if use_weighted and use_adaptive and stratified_weights is not None and epoch > warmup_epochs:
            loss_weights = compute_per_sample_losses(
                heads, embeddings["train"], labels["train"], device, int(args.batch_size),
                use_focal, focal_gamma, focal_alpha
            )
            loss_weights    = loss_weights / loss_weights.sum()
            epochs_since    = epoch - warmup_epochs
            alpha           = min(max_alpha, max_alpha * epochs_since / max(ramp_epochs, 1))
            current_weights = (1.0 - alpha) * stratified_weights + alpha * loss_weights
            current_weights = current_weights / current_weights.sum()
            ess = float(1.0 / (current_weights ** 2).sum())
            print(f"  Adaptive reweight: alpha={alpha:.3f}  ESS={ess:.0f}/{len(labels['train'])}")

        heads.train()
        N = embeddings["train"].shape[0]
        if current_weights is not None:
            perm = torch.multinomial(current_weights.double(), num_samples=N, replacement=True)
        else:
            perm = torch.randperm(N)
        t_loss = 0.0
        n_batches = 0
        for start in range(0, N, int(args.batch_size)):
            idx = perm[start:start+int(args.batch_size)]
            z_b = embeddings["train"][idx].to(device)
            y_b = labels["train"][idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([
                compute_label_loss(heads[d](z_b).squeeze(-1), y_b[:, d],
                                   use_focal, focal_gamma, focal_alpha)
                for d in range(3)
            ]).mean()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n_batches += 1
        t_loss /= max(1, n_batches)

        train_metrics, _ = eval_probe(heads, embeddings["train"], labels["train"],
                                      device, int(args.batch_size),
                                      use_focal, focal_gamma, focal_alpha)
        val_metrics,   _ = eval_probe(heads, embeddings["val"],   labels["val"],
                                      device, int(args.batch_size),
                                      use_focal, focal_gamma, focal_alpha)

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
                f"val_f1=[{val_metrics['f1_competence']:.3f}, "
                f"{val_metrics['f1_surprise']:.3f}, "
                f"{val_metrics['f1_intention']:.3f}] "
                f"val_auc=[{val_metrics['auc_competence']:.3f}, "
                f"{val_metrics['auc_surprise']:.3f}, "
                f"{val_metrics['auc_intention']:.3f}]"
            )

        if val_metrics["loss"] < best_val - min_delta:
            best_val   = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch":            epoch,
                "heads_state_dict": heads.state_dict(),
                "ae_checkpoint":    args.checkpoint,
                "split_indices_path": args.split_indices_path,
                "args":             vars(args),
                "val_metrics":      val_metrics,
            }, best_path)
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    heads.load_state_dict(ckpt["heads_state_dict"])

    final_metrics, final_logits = {}, {}
    for split in ["train", "val", "test"]:
        m, lg = eval_probe(heads, embeddings[split], labels[split],
                           device, int(args.batch_size),
                           use_focal, focal_gamma, focal_alpha)
        final_metrics[split] = m
        final_logits[split]  = lg

    save_history_csv(history, os.path.join(args.run_dir, args.history_csv_filename))

    final_csv_path = os.path.join(args.run_dir, "final_metrics.csv")
    with open(final_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        for split, m in final_metrics.items():
            for k, v in m.items():
                writer.writerow({"split": split, "metric": k, "value": v})
    print(f"Final metrics saved → {final_csv_path}")

    if bool(args.save_plots):
        fmts = normalize_plot_formats(args.plot_formats)
        save_history_plots(history, args.run_dir, fmts)
        for split in ["train", "val", "test"]:
            plot_roc(labels[split], final_logits[split], split, args.run_dir, fmts)

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
    for split in ["train", "val", "test"]:
        m = final_metrics[split]
        print(f"  {split}: loss={m['loss']:.4f}  "
              f"f1=[{m['f1_competence']:.3f}, {m['f1_surprise']:.3f}, {m['f1_intention']:.3f}]  "
              f"auc=[{m['auc_competence']:.3f}, {m['auc_surprise']:.3f}, {m['auc_intention']:.3f}]")

    return final_metrics, best_epoch


def summary_csv_fields():
    fields = ["run", "train_seed", "train_pct", "best_epoch"]
    for split in ["train", "val", "test"]:
        for m in ["loss", "mean_f1", "mean_auc",
                   "f1_competence", "f1_surprise", "f1_intention",
                   "auc_competence", "auc_surprise", "auc_intention"]:
            fields.append(f"{split}_{m}")
    return fields


def save_summary_csv(rows, path):
    fields = summary_csv_fields()
    numeric_fields = [f for f in fields if f not in ("run", "train_seed", "train_pct", "best_epoch")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
        vals = {k: [r[k] for r in rows] for k in numeric_fields}
        mean_row = {"run": "mean", "train_seed": "", "train_pct": "", "best_epoch": ""}
        std_row  = {"run": "std",  "train_seed": "", "train_pct": "", "best_epoch": ""}
        for k in numeric_fields:
            arr = np.array(vals[k])
            mean_row[k] = f"{arr.mean():.4f}"
            std_row[k]  = f"{arr.std():.4f}"
        writer.writerow(mean_row)
        writer.writerow(std_row)


TRAIN_PCTS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params/config.yaml")
    parser.add_argument("--train_seed", type=int, default=None,
                        help="Seed for probe init/training (random if omitted)")
    parser.add_argument("--train_pct", type=float, default=None,
                        help="Single fraction to run (omit to sweep all: 0.01,0.05,0.1,0.25,0.5,1.0)")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of independent probe runs per percentile")
    cli_args = parser.parse_args()

    args     = load_config(cli_args.config)
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
                for split, m in final_metrics.items():
                    for k, v in m.items():
                        row[f"{split}_{k}"] = v
                all_rows.append(row)

        summary_path = os.path.join(parent_dir, "summary.csv")
        save_summary_csv(all_rows, summary_path)
        print(f"\n{'='*60}")
        print(f"All done. {len(pcts)} percentiles × {num_runs} runs = {len(all_rows)} total")
        print(f"Summary → {summary_path}")
        print(f"{'='*60}")
