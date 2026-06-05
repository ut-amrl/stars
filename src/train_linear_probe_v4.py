"""
Heterogeneous GNN-VAE pipeline — SEAN-only v4 linear probe.

Stage 2: Supervised linear probe on frozen v4 representations.

Loads the best checkpoint from Stage 1 (train_gnn_vae_v4.py) for the split and robot
normalization stats. For GNN-based modes it also freezes the encoder. Then it trains
3 separate linear heads to predict binarized labels:
  competence  : >=3 -> 1, else 0
  surprise    : >=3 -> 1, else 0 by default; if --flip_surprise, <3 -> 1
  intention   : >=3 -> 1, else 0

Default input to each head: [z_robot || z_follower || mean(z_bystanders)] [B, 3*latent_dim = 192D].
Other baseline modes swap in raw robot features or learned robot-only MLP latents.
Loss: BCE. Reports accuracy, F1, and AUC on val and test sets.

Split indices are loaded from the checkpoint saved during Stage 1.

No --mask_follower needed: v4 encoder has no GT features as input.

Ablation modes:
  --raw_robot     : skip GNN entirely, probe directly on raw robot features [B, 592]
  --robot_mlp_encoder
                  : train a robot-only MLP autoencoder on 592D robot features,
                    freeze its encoder, then probe on that learned latent
  --weighted_random_baseline
                  : skip training entirely, sample test predictions from
                    Bernoulli(p_train) for 5 runs, where p_train is the train
                    positive rate for each label
  --random_encoder: use randomly initialized (untrained) encoder weights
  --mean_pool     : uniform mean over all per-node z's -> [B, latent_dim=64]
  --attn_pool     : model.pool_z() - 3-head attention over all nodes -> [B, 3 * latent_dim]
                    Only meaningful if VAE was trained with --incl_pool_attn
  --mlp_pool      : model.pool_z_mlp() - frozen 2-layer MLP on 192D concat -> [B, 3 * latent_dim]
                    Only meaningful if VAE was trained with --incl_mlp_pool

Usage:
  cd src/
  python train_linear_probe_v4.py
  python train_linear_probe_v4.py --raw_robot
  python train_linear_probe_v4.py --robot_mlp_encoder
  python train_linear_probe_v4.py --weighted_random_baseline
  python train_linear_probe_v4.py --random_encoder
  python train_linear_probe_v4.py --mlp_pool
"""

import os
import sys
import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloaders.sean_graph_dataset_v4 import SEANGraphDatasetV4
from models.gnn_vae_v4 import GNNVAEv4, NODE_TYPES
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSEANDataset
from models.unified_gnn_vae_v1 import UnifiedGNNVAE

LABEL_NAMES     = ['competence', 'surprise', 'intention']
LABEL_THRESHOLD = 3
RANDOM_BASELINE_RUNS = 5
PROBE_BASELINE_RUNS = 5


def binarize_labels(y, flip_surprise=False):
    """Convert raw Likert labels to binary targets, optionally flipping surprise only."""
    y_bin = (y >= LABEL_THRESHOLD).float()
    if flip_surprise:
        y_bin[:, 1] = 1.0 - y_bin[:, 1]
    return y_bin


def compute_metrics_dict(logits_np, labels_np, prefix):
    """Compute per-label and macro accuracy/F1/AUC from full-split logits and labels."""
    preds_np = (logits_np > 0).astype(float)
    metrics = {}
    acc_vals = []
    f1_vals = []
    auc_vals = []

    for d, name in enumerate(LABEL_NAMES):
        acc = accuracy_score(labels_np[:, d], preds_np[:, d])
        f1  = f1_score(labels_np[:, d], preds_np[:, d], zero_division=0)
        try:
            auc = roc_auc_score(labels_np[:, d], logits_np[:, d])
        except ValueError:
            auc = float("nan")

        metrics[f"{prefix}/acc_{name}"] = acc
        metrics[f"{prefix}/f1_{name}"]  = f1
        metrics[f"{prefix}/auc_{name}"] = auc
        acc_vals.append(acc)
        f1_vals.append(f1)
        auc_vals.append(auc)

    metrics[f"{prefix}/acc_macro"] = float(np.mean(acc_vals))
    metrics[f"{prefix}/f1_macro"]  = float(np.mean(f1_vals))
    metrics[f"{prefix}/auc_macro"] = float(np.nanmean(auc_vals))
    return metrics


def get_mode_tag(args):
    """Compact filename-safe tag for the current probe mode."""
    if args.raw_robot:
        tag = "raw_robot"
    elif args.robot_mlp_encoder:
        tag = f"robot_mlp_encoder_{args.robot_mlp_latent_dim}D"
    elif args.weighted_random_baseline:
        tag = f"weighted_random_baseline_{RANDOM_BASELINE_RUNS}runs"
    elif args.mean_pool:
        tag = "mean_pool_64D"
    elif args.attn_pool:
        tag = "attn_pool_192D"
    elif args.mlp_pool:
        tag = "mlp_pool_192D"
    elif getattr(args, "robot_only", False):
        tag = "robot_only_64D"
    elif getattr(args, "bystander_only", False):
        tag = "bystander_only_64D"
    elif getattr(args, "unified_follower", False):
        tag = "192D_unified_follower"
    elif getattr(args, "unified_checkpoint", False):
        tag = "128D_unified"
    elif getattr(args, "no_follower", False):
        tag = "concat_128D_no_follower"
    elif getattr(args, "zero_follower", False):
        tag = "concat_192D_zero_follower"
    else:
        tag = "concat_192D"
    if args.random_encoder:
        tag += "_random_encoder"
    if args.flip_surprise:
        tag += "_flip_surprise"
    return tag


def get_default_stage1_checkpoint(args):
    """Pick the matching Stage 1 checkpoint for the current probe mode."""
    if args.attn_pool and args.mlp_pool:
        return "checkpoints/best_gnn_vae_v4_pool_attn_mlp_pool.pt"
    if args.attn_pool:
        return "checkpoints/best_gnn_vae_v4_pool_attn.pt"
    if args.mlp_pool:
        return "checkpoints/best_gnn_vae_v4_mlp_pool.pt"
    return "checkpoints/best_gnn_vae_v4.pt"


class RobotMLPAutoencoder(nn.Module):
    """Simple robot-only autoencoder baseline used to learn a frozen 592D -> latent encoder."""
    def __init__(self, input_dim=592, hidden_dim=384, latent_dim=192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def save_test_roc_artifacts(gt_test, logits_test, out_dir, mode_tag):
    """
    Save test-set logits/labels and a 3-panel ROC curve figure.
    One subplot per label; curves are computed from test logits only.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f"roc_test_outputs_{mode_tag}.npz")
    np.savez_compressed(npz_path, gt_test=gt_test, logits_test=logits_test, label_names=np.array(LABEL_NAMES))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for d, name in enumerate(LABEL_NAMES):
        fpr, tpr, _ = roc_curve(gt_test[:, d], logits_test[:, d])
        auc = roc_auc_score(gt_test[:, d], logits_test[:, d])
        ax = axes[d]
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="random")
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right")

    fig.suptitle(f"Test ROC Curves - {mode_tag}")
    fig.tight_layout()

    png_path = os.path.join(out_dir, f"roc_test_{mode_tag}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC figure -> {png_path}")
    print(f"Saved ROC arrays  -> {npz_path}")


def load_compatible_state_dict(model, state_dict):
    """
    Load only checkpoint tensors whose names and shapes still match the current model.
    This keeps older checkpoints usable after non-critical architecture changes.
    """
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return skipped, missing, unexpected


def get_z_concat(model, batch, device, mean_pool=False, attn_pool=False, mlp_pool=False,
                 no_follower=False, zero_follower=False, robot_only=False, bystander_only=False):
    """
    Run frozen encoder, return embeddings.
    default:              [z_robot || z_follower || mean(z_bystanders)] [B, 3*latent_dim]
    no_follower=True:     [z_robot || mean(z_bystanders)] [B, 2*latent_dim]
    zero_follower=True:   [z_robot || zeros || mean(z_bystanders)] [B, 3*latent_dim]
    robot_only=True:      z_robot [B, latent_dim]
    bystander_only=True:  mean(z_follower + z_bystanders) [B, latent_dim]  (all non-robot nodes)
    mean_pool=True:       uniform mean over all per-node z's -> [B, latent_dim]
    attn_pool=True:       model.pool_z() -> [B, 3*latent_dim]
    mlp_pool=True:        model.pool_z_mlp() -> [B, 3*latent_dim]
    """
    from torch_geometric.utils import scatter
    batch = batch.to(device)
    with torch.no_grad():
        mu_dict, _ = model.encoder(batch)
        if attn_pool:
            return model.pool_z(mu_dict, batch)
        if mlp_pool:
            return model.pool_z_mlp(mu_dict, batch)
    B = int(batch['robot'].batch.max()) + 1
    if robot_only:
        return mu_dict['robot']  # [B, latent_dim]
    if mean_pool:
        all_z     = torch.cat([mu_dict[t] for t in NODE_TYPES if mu_dict[t].shape[0] > 0], dim=0)
        all_batch = torch.cat([batch[t].batch for t in NODE_TYPES if mu_dict[t].shape[0] > 0], dim=0)
        return scatter(all_z, all_batch, dim=0, reduce='mean', dim_size=B)
    z_robot    = mu_dict['robot']
    # collect all non-robot node types present
    non_robot_types = [t for t in NODE_TYPES if t != 'robot' and mu_dict[t].shape[0] > 0]
    if non_robot_types:
        nr_z     = torch.cat([mu_dict[t] for t in non_robot_types], dim=0)
        nr_batch = torch.cat([batch[t].batch for t in non_robot_types], dim=0)
        z_nr_mean = scatter(nr_z, nr_batch, dim=0, reduce='mean', dim_size=B)
    else:
        z_nr_mean = torch.zeros(B, model.latent_dim, device=device)
    if bystander_only:
        return z_nr_mean  # [B, latent_dim]
    z_follower = mu_dict['follower']
    if mu_dict['bystander'].shape[0] > 0:
        z_b_mean = scatter(mu_dict['bystander'], batch['bystander'].batch,
                           dim=0, reduce='mean', dim_size=B)
    else:
        z_b_mean = torch.zeros(B, model.latent_dim, device=device)
    if no_follower:
        return torch.cat([z_robot, z_b_mean], dim=-1)  # [B, 2*latent_dim]
    if zero_follower:
        return torch.cat([z_robot, torch.zeros_like(z_follower), z_b_mean], dim=-1)  # [B, 3*latent_dim]
    return torch.cat([z_robot, z_follower, z_b_mean], dim=-1)  # [B, 3*latent_dim]


def extract_embeddings(model, loader, device, raw_robot=False, mean_pool=False, attn_pool=False,
                       mlp_pool=False, unified=False, no_follower=False, zero_follower=False,
                       robot_only=False, bystander_only=False, unified_follower=False):
    """Extract embeddings and raw labels for all samples in loader."""
    all_z, all_y = [], []
    if model is not None:
        model.eval()
    for batch in loader:
        batch = batch.to(device)
        if raw_robot:
            z = batch['robot'].x   # [B, 592]
        elif unified:
            with torch.no_grad():
                from torch_geometric.utils import scatter as pyg_scatter
                mu_dict, _ = model.encoder(batch)
                B = int(batch['robot'].batch.max()) + 1
                if robot_only:
                    z = mu_dict['robot']  # [B, 64D]
                elif bystander_only:
                    if mu_dict['bystander'].shape[0] > 0:
                        z = pyg_scatter(mu_dict['bystander'], batch['bystander'].batch,
                                        dim=0, reduce='mean', dim_size=B)
                    else:
                        z = torch.zeros(B, model.latent_dim, device=device)
                elif unified_follower:
                    # Peel out bystander[0] (the remapped follower) per graph,
                    # mean-pool the rest, then concat -> [B, 3*latent_dim].
                    # UnifiedSEANDataset guarantees follower is local index 0.
                    z_b_all = mu_dict['bystander']       # [N_b_total, 64]
                    b_batch = batch['bystander'].batch    # [N_b_total]
                    z_follower_unified = torch.zeros(B, model.latent_dim, device=device)
                    z_others_sum = torch.zeros(B, model.latent_dim, device=device)
                    others_count = torch.zeros(B, 1, device=device)
                    # first occurrence of each graph index = bystander[0] (the follower)
                    seen = torch.zeros(B, dtype=torch.bool, device=device)
                    for node_i in range(z_b_all.shape[0]):
                        g = int(b_batch[node_i])
                        if not seen[g]:
                            z_follower_unified[g] = z_b_all[node_i]
                            seen[g] = True
                        else:
                            z_others_sum[g] += z_b_all[node_i]
                            others_count[g] += 1
                    z_others_mean = z_others_sum / others_count.clamp(min=1)
                    z = torch.cat([mu_dict['robot'], z_follower_unified, z_others_mean], dim=-1)
                else:
                    z = model.encode_graph(batch)  # [B, 128D]
        else:
            z = get_z_concat(model, batch, device, mean_pool=mean_pool, attn_pool=attn_pool,
                             mlp_pool=mlp_pool, no_follower=no_follower, zero_follower=zero_follower,
                             robot_only=robot_only, bystander_only=bystander_only)
        y = batch['robot'].y   # [B, 3] raw Likert
        all_z.append(z.cpu())
        all_y.append(y.cpu())
    all_z = torch.cat(all_z, dim=0)
    all_y = torch.cat(all_y, dim=0)  # [N, 3] raw Likert
    return all_z, all_y


def extract_labels(loader, device):
    """Extract only raw labels for all samples in a loader."""
    all_y = []
    for batch in loader:
        batch = batch.to(device)
        all_y.append(batch['robot'].y.cpu())
    return torch.cat(all_y, dim=0)


def train_robot_mlp_autoencoder(args, train_x, val_x, test_x, device, log_wandb=True):
    """
    Train a robot-only MLP autoencoder on normalized robot features, then return
    frozen latent embeddings for train/val/test.
    """
    model = RobotMLPAutoencoder(
        input_dim=train_x.shape[1],
        hidden_dim=args.robot_mlp_hidden_dim,
        latent_dim=args.robot_mlp_latent_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.robot_mlp_lr, weight_decay=args.weight_decay)
    best_val = float('inf')
    best_state = None

    print(
        f"Training robot-only MLP encoder: "
        f"{train_x.shape[1]}D -> {args.robot_mlp_hidden_dim}D -> {args.robot_mlp_latent_dim}D"
    )

    for epoch in range(1, args.robot_mlp_epochs + 1):
        model.train()
        perm = torch.randperm(train_x.shape[0], device=device)
        t_loss = 0.0
        n_batches = 0
        for i in range(0, train_x.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            x_b = train_x[idx]
            optimizer.zero_grad()
            recon_b, _ = model(x_b)
            loss = F.mse_loss(recon_b, x_b)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n_batches += 1
        t_loss /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            v_loss = F.mse_loss(model(val_x)[0], val_x).item()

        if wandb is not None and log_wandb:
            wandb.log({
                "mlp_pretrain/train_recon": t_loss,
                "mlp_pretrain/val_recon": v_loss,
            }, step=epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  MLP pretrain {epoch:03d}/{args.robot_mlp_epochs} | "
                f"train_recon={t_loss:.4f} | val_recon={v_loss:.4f}"
            )

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Robot MLP autoencoder failed to produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        train_z = model.encode(train_x)
        val_z = model.encode(val_x)
        test_z = model.encode(test_x)

    print(f"Best robot-only MLP val recon loss: {best_val:.4f}")
    return train_z, val_z, test_z


def summarize_repeated_test_metrics(all_test_metrics, summary_name):
    """Print and return mean/std test metrics across repeated runs."""
    print(f"\n=== {summary_name} SUMMARY OVER {len(all_test_metrics)} RUNS ===")
    summary_log = {}
    for name in LABEL_NAMES:
        acc_vals = [m[f"test/acc_{name}"] for m in all_test_metrics]
        f1_vals  = [m[f"test/f1_{name}"] for m in all_test_metrics]
        auc_vals = [m[f"test/auc_{name}"] for m in all_test_metrics]
        acc_mean, acc_std = float(np.mean(acc_vals)), float(np.std(acc_vals))
        f1_mean, f1_std   = float(np.mean(f1_vals)), float(np.std(f1_vals))
        auc_mean, auc_std = float(np.mean(auc_vals)), float(np.std(auc_vals))
        print(
            f"  {name}: "
            f"acc={acc_mean:.3f} +- {acc_std:.3f} "
            f"f1={f1_mean:.3f} +- {f1_std:.3f} "
            f"auc={auc_mean:.3f} +- {auc_std:.3f}"
        )
        summary_log[f"summary/acc_{name}_mean"] = acc_mean
        summary_log[f"summary/acc_{name}_std"] = acc_std
        summary_log[f"summary/f1_{name}_mean"] = f1_mean
        summary_log[f"summary/f1_{name}_std"] = f1_std
        summary_log[f"summary/auc_{name}_mean"] = auc_mean
        summary_log[f"summary/auc_{name}_std"] = auc_std

    macro_f1_vals = [m["test/f1_macro"] for m in all_test_metrics]
    macro_auc_vals = [m["test/auc_macro"] for m in all_test_metrics]
    print(
        f"  macro: "
        f"f1={float(np.mean(macro_f1_vals)):.3f} +- {float(np.std(macro_f1_vals)):.3f} "
        f"auc={float(np.mean(macro_auc_vals)):.3f} +- {float(np.std(macro_auc_vals)):.3f}"
    )
    summary_log["summary/f1_macro_mean"] = float(np.mean(macro_f1_vals))
    summary_log["summary/f1_macro_std"] = float(np.std(macro_f1_vals))
    summary_log["summary/auc_macro_mean"] = float(np.mean(macro_auc_vals))
    summary_log["summary/auc_macro_std"] = float(np.std(macro_auc_vals))
    return summary_log


def append_probe_result_and_print_cumulative(test_metrics, ckpt_dir, mode_tag, run_id=None):
    """Append one external probe run to a mode-specific CSV and print cumulative mean/std."""
    csv_dir = os.path.dirname(ckpt_dir) or ckpt_dir  # shared parent across run1/run2/... dirs
    suffix   = f"_run{run_id}" if run_id is not None else ""
    csv_path = os.path.join(csv_dir, f"probe_results_{mode_tag}{suffix}.csv")
    row = {"mode_tag": mode_tag}
    for key, value in test_metrics.items():
        row[key] = float(value)

    fieldnames = ["mode_tag"] + list(test_metrics.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (float(v) if k != "mode_tag" else v) for k, v in r.items()})

    print(f"\n=== CUMULATIVE SUMMARY ({mode_tag}) OVER {len(rows)} EXTERNAL RUNS ===")
    for name in LABEL_NAMES:
        acc_vals = [r[f"test/acc_{name}"] for r in rows]
        f1_vals  = [r[f"test/f1_{name}"] for r in rows]
        auc_vals = [r[f"test/auc_{name}"] for r in rows]
        print(
            f"  {name}: "
            f"acc={float(np.mean(acc_vals)):.3f} +- {float(np.std(acc_vals)):.3f} "
            f"f1={float(np.mean(f1_vals)):.3f} +- {float(np.std(f1_vals)):.3f} "
            f"auc={float(np.mean(auc_vals)):.3f} +- {float(np.std(auc_vals)):.3f}"
        )
    macro_f1_vals = [r["test/f1_macro"] for r in rows]
    macro_auc_vals = [r["test/auc_macro"] for r in rows]
    print(
        f"  macro: "
        f"f1={float(np.mean(macro_f1_vals)):.3f} +- {float(np.std(macro_f1_vals)):.3f} "
        f"auc={float(np.mean(macro_auc_vals)):.3f} +- {float(np.std(macro_auc_vals)):.3f}"
    )
    print(f"  results_file: {csv_path}")


def run_linear_probe_once(args, train_z, train_y, val_z, val_y, test_z, test_y,
                          input_dim, ckpt_dir, mode_tag, device, run_idx=None, save_roc=True, log_wandb=True):
    """Train one set of linear heads and return final test metrics."""
    run_tag = mode_tag if run_idx is None else f"{mode_tag}_run{run_idx}"

    heads     = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(3)]).to(device)
    optimizer = AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    best_path     = os.path.join(ckpt_dir, f"best_linear_probe_v4_{run_tag}.pt")
    es_counter    = 0

    if run_idx is not None:
        print(f"\n=== PROBE RUN {run_idx}/{PROBE_BASELINE_RUNS} ({mode_tag}) ===")

    for epoch in range(1, args.epochs + 1):
        heads.train()
        perm  = torch.randperm(train_z.shape[0], device=device)
        t_loss = 0.0
        n_batches = 0
        for i in range(0, train_z.shape[0], args.batch_size):
            idx   = perm[i:i + args.batch_size]
            z_b   = train_z[idx]
            y_b   = train_y[idx]
            optimizer.zero_grad()
            loss  = sum(
                F.binary_cross_entropy_with_logits(heads[d](z_b).squeeze(-1), y_b[:, d])
                for d in range(3)
            ) / 3
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            n_batches += 1
        t_loss /= n_batches

        heads.eval()
        with torch.no_grad():
            v_loss = sum(
                F.binary_cross_entropy_with_logits(heads[d](val_z).squeeze(-1), val_y[:, d])
                for d in range(3)
            ).item() / 3

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f}")

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                logits_train = torch.stack([
                    heads[d](train_z).squeeze(-1) for d in range(3)
                ], dim=1).cpu().numpy()
                logits_val = torch.stack([
                    heads[d](val_z).squeeze(-1) for d in range(3)
                ], dim=1).cpu().numpy()
            gt_train = train_y.cpu().numpy()
            gt_val   = val_y.cpu().numpy()
            preds_val = (logits_val > 0).astype(float)
            log_dict = {}
            if run_idx is not None:
                log_dict["probe_run"] = run_idx
            for d, name in enumerate(LABEL_NAMES):
                f1_t  = f1_score(gt_train[:, d], (logits_train[:, d] > 0).astype(float), zero_division=0)
                auc_t = roc_auc_score(gt_train[:, d], logits_train[:, d])
                f1_v  = f1_score(gt_val[:, d], preds_val[:, d], zero_division=0)
                auc_v = roc_auc_score(gt_val[:, d], logits_val[:, d])
                print(f"  Epoch {epoch:03d} val {name}: f1={f1_v:.3f} auc={auc_v:.3f}")
                log_dict[f"train/f1_{name}"] = f1_t
                log_dict[f"train/auc_{name}"] = auc_t
                log_dict[f"val/f1_{name}"]   = f1_v
                log_dict[f"val/auc_{name}"]  = auc_v
            if wandb is not None and log_wandb:
                wandb.log(log_dict, step=epoch)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            es_counter    = 0
            torch.save({'heads_state_dict': heads.state_dict(), 'epoch': epoch, 'args': vars(args)}, best_path)
            print(f"  Saved best probe (val_loss={v_loss:.4f})")
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.es_patience})")
                break

    print("\n=== TEST SET EVALUATION ===")
    ckpt_probe = torch.load(best_path, weights_only=False)
    heads.load_state_dict(ckpt_probe['heads_state_dict'])
    heads.eval()
    with torch.no_grad():
        logits_test = torch.stack([
            heads[d](test_z).squeeze(-1) for d in range(3)
        ], dim=1).cpu().numpy()
    preds_test = (logits_test > 0).astype(float)
    gt_test = test_y.cpu().numpy()
    test_metrics = {}
    for d, name in enumerate(LABEL_NAMES):
        acc = accuracy_score(gt_test[:, d], preds_test[:, d])
        f1  = f1_score(gt_test[:, d], preds_test[:, d], zero_division=0)
        auc = roc_auc_score(gt_test[:, d], logits_test[:, d])
        print(f"  {name}: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")
        test_metrics[f"test/acc_{name}"] = acc
        test_metrics[f"test/f1_{name}"]  = f1
        test_metrics[f"test/auc_{name}"] = auc

    test_metrics["test/f1_macro"] = float(np.mean([test_metrics[f"test/f1_{n}"] for n in LABEL_NAMES]))
    test_metrics["test/auc_macro"] = float(np.mean([test_metrics[f"test/auc_{n}"] for n in LABEL_NAMES]))
    print(f"  macro: f1={test_metrics['test/f1_macro']:.3f} auc={test_metrics['test/auc_macro']:.3f}")

    if save_roc:
        save_test_roc_artifacts(gt_test, logits_test, ckpt_dir, run_tag)
    if run_idx is None:
        append_probe_result_and_print_cumulative(test_metrics, ckpt_dir, mode_tag, run_id=args.run_id if hasattr(args, 'run_id') else None)
    return test_metrics


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    mode_tag = get_mode_tag(args)

    mode_flags = [
        args.raw_robot,
        args.robot_mlp_encoder,
        args.weighted_random_baseline,
        args.mean_pool,
        args.attn_pool,
        args.mlp_pool,
    ]
    if sum(bool(flag) for flag in mode_flags) > 1:
        raise ValueError("Choose at most one of --raw_robot, --robot_mlp_encoder, --weighted_random_baseline, --mean_pool, --attn_pool, --mlp_pool.")
    if args.random_encoder and (args.raw_robot or args.robot_mlp_encoder or args.weighted_random_baseline):
        raise ValueError("--random_encoder is only meaningful for GNN-based probe modes.")
    if getattr(args, "no_follower", False) and getattr(args, "zero_follower", False):
        raise ValueError("--no_follower and --zero_follower are mutually exclusive.")
    if (getattr(args, "no_follower", False) or getattr(args, "zero_follower", False)) and args.unified_checkpoint:
        raise ValueError("--no_follower/--zero_follower are only for SEAN-only v4 checkpoints, not --unified_checkpoint.")
    exclusive_64d = [getattr(args, "robot_only", False), getattr(args, "bystander_only", False)]
    if sum(exclusive_64d) > 1:
        raise ValueError("--robot_only and --bystander_only are mutually exclusive.")
    if args.unified_checkpoint and (args.mean_pool or args.attn_pool or args.mlp_pool):
        raise ValueError("--mean_pool/--attn_pool/--mlp_pool are not supported with --unified_checkpoint.")
    if getattr(args, "unified_follower", False) and not args.unified_checkpoint:
        raise ValueError("--unified_follower requires --unified_checkpoint.")
    if getattr(args, "unified_follower", False) and (getattr(args, "robot_only", False) or getattr(args, "bystander_only", False)):
        raise ValueError("--unified_follower is mutually exclusive with --robot_only/--bystander_only.")

    if wandb is not None:
        wandb.init(
            project="social_world_model",
            name=f"linear_probe_v4_{mode_tag}_e{args.epochs}",
            config=vars(args),
        )
    else:
        print("wandb not installed; continuing without probe logging.")

    # ---- Load Stage 1 checkpoint ----
    if args.checkpoint is None:
        args.checkpoint = get_default_stage1_checkpoint(args)
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved_args = ckpt['args']

    if args.unified_checkpoint:
        dataset = UnifiedSEANDataset(args.h5_path)
        dataset.base_dataset.robot_mean = ckpt['sean_robot_mean']
        dataset.base_dataset.robot_std  = ckpt['sean_robot_std']
    else:
        dataset = SEANGraphDatasetV4(
            args.h5_path,
            robot_mean=ckpt['robot_mean'],
            robot_std=ckpt['robot_std'],
        )

    latent_dim = saved_args.get('latent_dim', 64)
    if args.raw_robot:
        input_dim = 592
    elif args.robot_mlp_encoder:
        input_dim = args.robot_mlp_latent_dim
    elif args.mean_pool:
        input_dim = latent_dim
    elif getattr(args, "robot_only", False) or getattr(args, "bystander_only", False):
        input_dim = latent_dim       # 64D
    elif getattr(args, "unified_follower", False):
        input_dim = 3 * latent_dim  # [z_robot || z_follower_unified || mean(z_others)] = 192D
    elif args.unified_checkpoint:
        input_dim = 2 * latent_dim  # [z_robot || mean(z_bystanders)] = 128D
    elif getattr(args, "no_follower", False):
        input_dim = 2 * latent_dim  # [z_robot || mean(z_bystanders)] = 128D
    else:
        input_dim = 3 * latent_dim  # covers 192D concat, zero_follower, attn_pool, mlp_pool

    # ---- Load split indices from Stage 1 ----
    if args.unified_checkpoint:
        is_v3 = 'snsb_timesteps' in ckpt  # v3 checkpoint (train_unified_gnn_vae_v3.py)
        if is_v3:
            # v3 saves SEAN split indices into sean_together_v2.h5 space via split_indices_final_sean_only.pt
            split_path = args.sean_split_file or os.path.join(
                os.path.dirname(args.checkpoint), '..', '..', 'split_indices_final_sean_only.pt'
            )
            if not os.path.exists(split_path):
                split_path = 'checkpoints/split_indices_final_sean_only.pt'
            splits = torch.load(split_path, map_location='cpu', weights_only=False)
            train_set = torch.utils.data.Subset(dataset, splits['train'])
            val_set   = torch.utils.data.Subset(dataset, splits['val'])
            test_set  = torch.utils.data.Subset(dataset, splits['test'])
            print(f"[v3 unified] SEAN split from {split_path}")
        else:
            split_path = os.path.join(os.path.dirname(args.checkpoint), 'split_indices_unified_gnn_vae_v2.pt')
            splits_all = torch.load(split_path, weights_only=False)
            sean_splits = splits_all['sean']
            train_set = torch.utils.data.Subset(dataset, sean_splits['train'])
            val_set   = torch.utils.data.Subset(dataset, sean_splits['val'])
            test_set  = torch.utils.data.Subset(dataset, sean_splits['test'])
    else:
        split_path = ckpt.get('split_path', os.path.join(os.path.dirname(args.checkpoint), 'split_indices_v4.pt'))
        if not os.path.isabs(split_path) and not os.path.exists(split_path):
            split_path = os.path.join(os.path.dirname(args.checkpoint), os.path.basename(split_path))
        splits    = torch.load(split_path, weights_only=False)
        train_set = torch.utils.data.Subset(dataset, splits['train'])
        val_set   = torch.utils.data.Subset(dataset, splits['val'])
        test_set  = torch.utils.data.Subset(dataset, splits['test'])
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    persistent = args.num_workers > 0
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=persistent)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=persistent)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=persistent)

    ckpt_dir = os.path.dirname(args.checkpoint) or "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.weighted_random_baseline:
        print("Running weighted random baseline on the fixed v4 split...")
        train_y = binarize_labels(
            extract_labels(train_loader, device),
            flip_surprise=args.flip_surprise
        ).numpy()
        test_y = binarize_labels(
            extract_labels(test_loader, device),
            flip_surprise=args.flip_surprise
        ).numpy()
        p_train = train_y.mean(axis=0)
        print(
            "  Train positive rates: "
            + ", ".join(f"{name}={p_train[d]:.4f}" for d, name in enumerate(LABEL_NAMES))
        )

        run_metrics = {name: {"acc": [], "f1": [], "auc": []} for name in LABEL_NAMES}
        for run_idx in range(RANDOM_BASELINE_RUNS):
            preds_test = np.stack([
                np.random.binomial(1, p_train[d], size=test_y.shape[0])
                for d in range(len(LABEL_NAMES))
            ], axis=1).astype(float)
            logits_test = np.tile(p_train.reshape(1, -1), (test_y.shape[0], 1))

            print(f"  Random baseline run {run_idx + 1}/{RANDOM_BASELINE_RUNS}")
            for d, name in enumerate(LABEL_NAMES):
                acc = accuracy_score(test_y[:, d], preds_test[:, d])
                f1 = f1_score(test_y[:, d], preds_test[:, d], zero_division=0)
                auc = roc_auc_score(test_y[:, d], logits_test[:, d])
                print(f"    {name}: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")
                run_metrics[name]["acc"].append(acc)
                run_metrics[name]["f1"].append(f1)
                run_metrics[name]["auc"].append(auc)

        print("\n=== WEIGHTED RANDOM BASELINE SUMMARY ===")
        summary_log = {}
        for name in LABEL_NAMES:
            acc_mean = float(np.mean(run_metrics[name]["acc"]))
            acc_std = float(np.std(run_metrics[name]["acc"]))
            f1_mean = float(np.mean(run_metrics[name]["f1"]))
            f1_std = float(np.std(run_metrics[name]["f1"]))
            auc_mean = float(np.mean(run_metrics[name]["auc"]))
            auc_std = float(np.std(run_metrics[name]["auc"]))
            print(
                f"  {name}: "
                f"acc={acc_mean:.3f} +- {acc_std:.3f} "
                f"f1={f1_mean:.3f} +- {f1_std:.3f} "
                f"auc={auc_mean:.3f} +- {auc_std:.3f}"
            )
            summary_log[f"random_baseline/acc_{name}_mean"] = acc_mean
            summary_log[f"random_baseline/acc_{name}_std"] = acc_std
            summary_log[f"random_baseline/f1_{name}_mean"] = f1_mean
            summary_log[f"random_baseline/f1_{name}_std"] = f1_std
            summary_log[f"random_baseline/auc_{name}_mean"] = auc_mean
            summary_log[f"random_baseline/auc_{name}_std"] = auc_std

        if wandb is not None:
            wandb.log(summary_log)

        roc_logits = np.tile(p_train.reshape(1, -1), (test_y.shape[0], 1))
        save_test_roc_artifacts(test_y, roc_logits, ckpt_dir, mode_tag)
        return

    model = None
    if not args.raw_robot and not args.robot_mlp_encoder:
        if args.unified_checkpoint:
            model = UnifiedGNNVAE(
                hidden_dim=saved_args.get('hidden_dim', 128),
                latent_dim=saved_args.get('latent_dim', 64),
                num_mp_layers=saved_args.get('num_mp_layers', 2),
                edge_transformer_layers=saved_args.get('edge_transformer_layers', 2),
                edge_transformer_heads=saved_args.get('edge_transformer_heads', 2),
            ).to(device)
        else:
            model = GNNVAEv4(
                robot_dim=592,
                follower_embed_dim=8,
                bystander_embed_dim=8,
                hidden_dim=saved_args.get('hidden_dim', 128),
                latent_dim=saved_args.get('latent_dim', 64),
                num_mp_layers=saved_args.get('num_mp_layers', 2),
            ).to(device)

        if args.random_encoder:
            print("Using RANDOM (untrained) encoder weights.")
        else:
            skipped, missing, unexpected = load_compatible_state_dict(model, ckpt['model_state_dict'])
            if skipped:
                print(f"Skipped incompatible checkpoint keys: {skipped}")
            if unexpected:
                print(f"Unexpected checkpoint keys ignored: {list(unexpected)}")
            if missing:
                print(f"Missing model keys left at init: {list(missing)}")
            if args.attn_pool and any(k.startswith("pool_attn.") for k in skipped):
                print("Warning: checkpoint pool_attn weights were incompatible; --attn_pool is not meaningful for this checkpoint.")
            if args.mlp_pool and any(k.startswith("pool_mlp.") for k in skipped):
                print("Warning: checkpoint pool_mlp weights were incompatible; --mlp_pool is not meaningful for this checkpoint.")

        # Freeze entire VAE - only linear heads will be trained
        for param in model.parameters():
            param.requires_grad = False
        print("VAE frozen (encoder + decoder + pool_attn).")

    # ---- Pre-extract all embeddings ----
    if args.raw_robot:    mode = "raw_robot"
    elif args.robot_mlp_encoder:
        mode = f"robot_mlp_encoder_{args.robot_mlp_latent_dim}D"
    elif args.mean_pool:  mode = "mean_pool_64D"
    elif args.attn_pool:  mode = "attn_pool_192D"
    elif args.mlp_pool:   mode = "mlp_pool_192D"
    elif getattr(args, "robot_only", False):   mode = "robot_only_64D"
    elif getattr(args, "bystander_only", False): mode = "bystander_only_64D"
    elif getattr(args, "unified_follower", False): mode = "192D_unified_follower"
    elif args.unified_checkpoint: mode = "128D_unified"
    else:                 mode = "192D_concat"
    if args.random_encoder:
        mode += " [RANDOM ENCODER]"
    print(f"Extracting embeddings ({mode})...")
    base_raw = args.raw_robot or args.robot_mlp_encoder
    nf = getattr(args, "no_follower", False)
    zf = getattr(args, "zero_follower", False)
    ro = getattr(args, "robot_only", False)
    bo = getattr(args, "bystander_only", False)
    uf = getattr(args, "unified_follower", False)
    train_z, train_y = extract_embeddings(model, train_loader, device, raw_robot=base_raw, mean_pool=args.mean_pool, attn_pool=args.attn_pool, mlp_pool=args.mlp_pool, unified=args.unified_checkpoint, no_follower=nf, zero_follower=zf, robot_only=ro, bystander_only=bo, unified_follower=uf)
    val_z,   val_y   = extract_embeddings(model, val_loader,   device, raw_robot=base_raw, mean_pool=args.mean_pool, attn_pool=args.attn_pool, mlp_pool=args.mlp_pool, unified=args.unified_checkpoint, no_follower=nf, zero_follower=zf, robot_only=ro, bystander_only=bo, unified_follower=uf)
    test_z,  test_y  = extract_embeddings(model, test_loader,  device, raw_robot=base_raw, mean_pool=args.mean_pool, attn_pool=args.attn_pool, mlp_pool=args.mlp_pool, unified=args.unified_checkpoint, no_follower=nf, zero_follower=zf, robot_only=ro, bystander_only=bo, unified_follower=uf)

    print(f"  train_z: {train_z.shape}  val_z: {val_z.shape}  test_z: {test_z.shape}")

    train_y = binarize_labels(train_y, flip_surprise=args.flip_surprise)
    val_y   = binarize_labels(val_y, flip_surprise=args.flip_surprise)
    test_y  = binarize_labels(test_y, flip_surprise=args.flip_surprise)

    if args.raw_robot or args.robot_mlp_encoder:
        all_test_metrics = []
        raw_train_z = train_z
        raw_val_z = val_z
        raw_test_z = test_z
        for run_idx in range(1, PROBE_BASELINE_RUNS + 1):
            if args.robot_mlp_encoder:
                run_train_z, run_val_z, run_test_z = train_robot_mlp_autoencoder(
                    args,
                    raw_train_z.to(device),
                    raw_val_z.to(device),
                    raw_test_z.to(device),
                    device,
                    log_wandb=(run_idx == PROBE_BASELINE_RUNS),
                )
            else:
                run_train_z = raw_train_z.to(device)
                run_val_z = raw_val_z.to(device)
                run_test_z = raw_test_z.to(device)

            test_metrics = run_linear_probe_once(
                args,
                run_train_z,
                train_y.to(device),
                run_val_z,
                val_y.to(device),
                run_test_z,
                test_y.to(device),
                input_dim,
                ckpt_dir,
                mode_tag,
                device,
                run_idx=run_idx,
                save_roc=(run_idx == PROBE_BASELINE_RUNS),
                log_wandb=(run_idx == PROBE_BASELINE_RUNS),
            )
            all_test_metrics.append(test_metrics)

        summary_log = summarize_repeated_test_metrics(all_test_metrics, mode_tag)
        if wandb is not None:
            wandb.log(summary_log)
        return

    run_linear_probe_once(
        args,
        train_z.to(device),
        train_y.to(device),
        val_z.to(device),
        val_y.to(device),
        test_z.to(device),
        test_y.to(device),
        input_dim,
        ckpt_dir,
        mode_tag,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default=None,
                        help="Stage 1 checkpoint to probe. If omitted, auto-selects a checkpoint matching the probe mode.")
    parser.add_argument("--unified_checkpoint", action="store_true",
                        help="Load a UnifiedGNNVAE checkpoint (SNSB+SEAN). Uses 128D [z_robot||mean(z_bystanders)]. "
                             "v2: split from split_indices_unified_gnn_vae_v2.pt. "
                             "v3: split from split_indices_final_sean_only.pt (auto-detected).")
    parser.add_argument("--sean_split_file", default=None,
                        help="Override SEAN split .pt for --unified_checkpoint v3 path.")
    parser.add_argument("--h5_path",        default="/data/sean_together_v2.h5")
    parser.add_argument("--epochs",         type=int,   default=1000)
    parser.add_argument("--es_patience",    type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--raw_robot",      action="store_true",
                        help="Probe directly on raw robot features [592D], skip GNN")
    parser.add_argument("--robot_mlp_encoder", action="store_true",
                        help="Train a robot-only MLP autoencoder on 592D robot features, freeze its encoder, then probe that learned latent")
    parser.add_argument("--robot_mlp_latent_dim", type=int, default=192,
                        help="Latent dimension for the robot-only MLP encoder baseline")
    parser.add_argument("--robot_mlp_hidden_dim", type=int, default=384,
                        help="Hidden dimension for the robot-only MLP encoder baseline")
    parser.add_argument("--robot_mlp_epochs", type=int, default=100,
                        help="Number of pretraining epochs for the robot-only MLP autoencoder baseline")
    parser.add_argument("--robot_mlp_lr", type=float, default=1e-3,
                        help="Learning rate for the robot-only MLP autoencoder baseline")
    parser.add_argument("--weighted_random_baseline", action="store_true",
                        help="Skip training and sample Bernoulli test predictions using train positive rates")
    parser.add_argument("--run_id",         default=None,
                        help="Tag for per-seed CSV output, e.g. --run_id 0 writes probe_results_{mode}_run0.csv")
    parser.add_argument("--flip_surprise", action="store_true",
                        help="Flip surprise only so that <3 is the positive class")
    parser.add_argument("--random_encoder", action="store_true",
                        help="Use randomly initialized encoder (untrained baseline)")
    parser.add_argument("--mean_pool",      action="store_true",
                        help="Uniform mean over all per-node z's -> 64D")
    parser.add_argument("--attn_pool",      action="store_true",
                        help="Use trained pool_z (3-head attention over all nodes -> 192D). Only meaningful if VAE was trained with --incl_pool_attn")
    parser.add_argument("--mlp_pool",       action="store_true",
                        help="Use trained pool_z_mlp (2-layer MLP on 192D concat -> 192D). Only meaningful if VAE was trained with --incl_mlp_pool")
    parser.add_argument("--no_follower",    action="store_true",
                        help="Drop follower slot entirely: probe on [z_robot || mean(z_bystanders)] [128D]")
    parser.add_argument("--zero_follower",  action="store_true",
                        help="Zero follower slot: probe on [z_robot || zeros || mean(z_bystanders)] [192D]")
    parser.add_argument("--robot_only",     action="store_true",
                        help="Probe on z_robot only [64D]. Works with v4 and --unified_checkpoint.")
    parser.add_argument("--bystander_only", action="store_true",
                        help="Probe on mean of all non-robot z's [64D]. Works with v4 and --unified_checkpoint.")
    parser.add_argument("--unified_follower", action="store_true",
                        help="Unified checkpoint only: extract bystander[0] (remapped follower) per graph, "
                             "mean-pool remaining bystanders, concat -> [z_robot||z_follower||mean(z_others)] 192D.")
    args = parser.parse_args()
    train(args)
