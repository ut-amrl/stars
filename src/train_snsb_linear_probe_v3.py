"""
Heterogeneous GNN-VAE pipeline — combined SNSB linear probe.

Stage 2: Supervised linear probe on frozen UnifiedGNNVAE v3 — SNSB action classification.

v3 changes over v2:
  - Loads SNSB train/val from social_nav_v3.h5 using split indices saved by the
    v3 checkpoint (split_used_v3_seed*.pt).
  - Model uses the checkpoint's native SNSB timestep setting (20 for current v3).
  - Oracle eval interpolates to the checkpoint's SNSB timestep count.

All LP logic (training loop, metrics, oracle eval, WRB/majority baselines) identical to v2.

Usage:
  python train_snsb_linear_probe_v3.py --checkpoint checkpoints/best_unified_gnn_vae_v3_seed1.pt \
      --eval_h5_paths dataloaders/socialnav_human_oracle_v3.h5
"""

import os
import sys
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.unified_graph_dataset_v1 import (
    DATASET_SEAN,
    DATASET_SSUB_REAL,
    UnifiedGraphDataset,
)
from dataloaders.socialnav_hetero_graph_dataset_v1 import SocialNavGraphDatasetV1
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.unified_gnn_vae_v1 import (
    SNSB_BYSTANDER_DIM,
    SNSB_ROBOT_DIM,
    UnifiedGNNVAE,
)

try:
    import wandb
except Exception:
    wandb = None

SEAN_ID = DATASET_SEAN
SNSB_ID = DATASET_SSUB_REAL
ACTION_CLASS_NAMES = ["avoid", "follow", "overtake", "yield", "not_consider"]
NUM_CLASSES = len(ACTION_CLASS_NAMES)
PROBE_BASELINE_RUNS = 5


def _interp_1d(arr, target_T):
    """Linearly interpolate a numpy array along axis -2 (time) to target_T."""
    T_raw = arr.shape[-2]
    if T_raw == target_T:
        return arr
    x_old = np.linspace(0, 1, T_raw)
    x_new = np.linspace(0, 1, target_T)
    leading = arr.shape[:-2]
    feat_dim = arr.shape[-1]
    out = np.zeros(leading + (target_T, feat_dim), dtype=arr.dtype)
    for idx in np.ndindex(*leading):
        for f in range(feat_dim):
            out[idx + (slice(None), f)] = np.interp(x_new, x_old, arr[idx + (slice(None), f)])
    return out


def load_compatible_state_dict(model, state_dict):
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


def extract_oracle_labels_and_probs(loader):
    all_y, all_probs = [], []
    has_probs = True
    for batch in loader:
        all_y.append(batch["bystander"].y.long().cpu())
        if hasattr(batch["bystander"], "action_probs"):
            all_probs.append(batch["bystander"].action_probs.cpu())
        else:
            has_probs = False
    probs = torch.cat(all_probs, dim=0).numpy() if has_probs and all_probs else None
    return torch.cat(all_y, dim=0).numpy(), probs


def extract_bystander_embeddings(model, loader, device, bystander_only=False, robot_only=False):
    all_z, all_y = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            mu_dict, _ = model.encoder(batch)

        bystander_batch_idx = batch["bystander"].batch
        robot_z_exp = mu_dict["robot"][bystander_batch_idx]
        if bystander_only:
            z = mu_dict["bystander"]
        elif robot_only:
            z = robot_z_exp
        else:
            z = torch.cat([robot_z_exp, mu_dict["bystander"]], dim=1)
        y = batch["bystander"].y.long()

        all_z.append(z.cpu())
        all_y.append(y.cpu())

    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0)


def extract_bystander_embeddings_with_probs(model, loader, device, bystander_only=False, robot_only=False):
    all_z, all_y, all_probs = [], [], []
    has_probs = True
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            mu_dict, _ = model.encoder(batch)

        bystander_batch_idx = batch["bystander"].batch
        robot_z_exp = mu_dict["robot"][bystander_batch_idx]
        if bystander_only:
            z = mu_dict["bystander"]
        elif robot_only:
            z = robot_z_exp
        else:
            z = torch.cat([robot_z_exp, mu_dict["bystander"]], dim=1)
        y = batch["bystander"].y.long()

        all_z.append(z.cpu())
        all_y.append(y.cpu())
        if hasattr(batch["bystander"], "action_probs"):
            all_probs.append(batch["bystander"].action_probs.cpu())
        else:
            has_probs = False

    probs = torch.cat(all_probs, dim=0) if has_probs and all_probs else None
    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0), probs


def compute_metrics_dict(logits_np, labels_np, prefix):
    preds_np = logits_np.argmax(axis=1)
    metrics = {f"{prefix}/acc": float(accuracy_score(labels_np, preds_np))}

    f1_per_class = f1_score(
        labels_np, preds_np, average=None,
        labels=list(range(NUM_CLASSES)), zero_division=0,
    )
    for i, name in enumerate(ACTION_CLASS_NAMES):
        metrics[f"{prefix}/f1_{name}"] = float(f1_per_class[i])

    metrics[f"{prefix}/f1_macro"] = float(
        f1_score(labels_np, preds_np, average="macro", zero_division=0)
    )
    metrics[f"{prefix}/f1_weighted"] = float(
        f1_score(labels_np, preds_np, average="weighted", zero_division=0)
    )
    return metrics


def compute_prediction_agreement(logits_np, action_probs_np):
    preds_np = logits_np.argmax(axis=1)
    return float(action_probs_np[np.arange(len(preds_np)), preds_np].mean())


def save_test_artifacts(gt_test, logits_test, out_dir, mode_tag, action_probs=None):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f"test_outputs_{mode_tag}.npz")
    arrays = {
        "gt_test": gt_test,
        "logits_test": logits_test,
        "class_names": np.array(ACTION_CLASS_NAMES),
    }
    if action_probs is not None:
        arrays["action_probs"] = action_probs
    np.savez_compressed(npz_path, **arrays)

    preds = logits_test.argmax(axis=1)
    cm = confusion_matrix(gt_test, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ACTION_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(ACTION_CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test Confusion Matrix (normalized) - {mode_tag}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.tight_layout()

    png_path = os.path.join(out_dir, f"confusion_matrix_{mode_tag}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {png_path}")
    print(f"Saved test arrays      -> {npz_path}")


def append_probe_result_and_print_cumulative(test_metrics, ckpt_dir, mode_tag, prefix="test", results_dir=None):
    csv_dir = results_dir or os.path.dirname(ckpt_dir) or ckpt_dir
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"snsb_{prefix}_results_{mode_tag}.csv")
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

    print(f"\n=== CUMULATIVE SUMMARY ({prefix}: {mode_tag}) OVER {len(rows)} EXTERNAL RUNS ===")
    acc_vals = [r[f"{prefix}/acc"] for r in rows]
    print(f"  acc: {float(np.mean(acc_vals)):.3f} +- {float(np.std(acc_vals)):.3f}")
    for name in ACTION_CLASS_NAMES:
        f1_vals = [r[f"{prefix}/f1_{name}"] for r in rows]
        print(f"  f1_{name}: {float(np.mean(f1_vals)):.3f} +- {float(np.std(f1_vals)):.3f}")
    macro_vals = [r[f"{prefix}/f1_macro"] for r in rows]
    print(f"  f1_macro: {float(np.mean(macro_vals)):.3f} +- {float(np.std(macro_vals)):.3f}")
    weighted_vals = [r[f"{prefix}/f1_weighted"] for r in rows]
    print(f"  f1_weighted: {float(np.mean(weighted_vals)):.3f} +- {float(np.std(weighted_vals)):.3f}")
    pa_key = f"{prefix}/pa"
    if pa_key in rows[0]:
        pa_vals = [r[pa_key] for r in rows]
        print(f"  pa: {float(np.mean(pa_vals)):.3f} +- {float(np.std(pa_vals)):.3f}")
    print(f"  results_file: {csv_path}")


def run_linear_probe_once(
    args, train_z, train_y, val_z, val_y, test_z, test_y,
    input_dim, ckpt_dir, mode_tag, device,
    run_idx=None, save_artifacts=True, log_wandb=True,
):
    run_tag = mode_tag if run_idx is None else f"{mode_tag}_run{run_idx}"

    head = nn.Linear(input_dim, NUM_CLASSES).to(device)
    optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce_weights = None
    if not args.no_class_weights:
        counts = torch.bincount(train_y.cpu(), minlength=NUM_CLASSES).float()
        ce_weights = (train_y.shape[0] / (NUM_CLASSES * counts.clamp(min=1))).to(device)
        print(f"  CE class weights: {dict(zip(ACTION_CLASS_NAMES, ce_weights.cpu().tolist()))}")
    else:
        print("  CE class weights disabled.")

    sample_weights = None
    if not args.no_stratified:
        counts = torch.bincount(train_y.cpu(), minlength=NUM_CLASSES).float()
        inv_sqrt = 1.0 / torch.sqrt(counts.clamp(min=1))
        sample_weights = inv_sqrt[train_y.cpu()]
        print(f"  Stratified weights per class: {dict(zip(ACTION_CLASS_NAMES, inv_sqrt.tolist()))}")
    else:
        print("  Stratified sampling disabled.")

    scheduler = None  # LR scheduler disabled; use plain optimizer

    best_val_loss = float("inf")
    es_counter = 0
    best_path = os.path.join(ckpt_dir, f"best_snsb_probe_{run_tag}.pt")

    if run_idx is not None:
        print(f"\n=== PROBE RUN {run_idx}/{PROBE_BASELINE_RUNS} ({mode_tag}) ===")

    n_train = train_z.shape[0]

    for epoch in range(1, args.epochs + 1):
        head.train()

        if sample_weights is not None:
            if args.adaptive_warmup_epochs > 0 and epoch > args.adaptive_warmup_epochs:
                progress = min(1.0, (epoch - args.adaptive_warmup_epochs) /
                               max(1, args.epochs - args.adaptive_warmup_epochs))
                alpha = args.adaptive_max_alpha * progress

                head.eval()
                with torch.no_grad():
                    per_sample_loss = F.cross_entropy(
                        head(train_z), train_y, reduction="none",
                    )
                head.train()
                loss_weights = per_sample_loss.cpu() / per_sample_loss.cpu().mean().clamp(min=1e-8)
                blended = (1.0 - alpha) * sample_weights + alpha * loss_weights
                epoch_weights = blended
            else:
                epoch_weights = sample_weights

            idx_order = torch.multinomial(
                epoch_weights.float(), n_train, replacement=True,
            ).to(device)
        else:
            idx_order = torch.randperm(n_train, device=device)

        t_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            idx = idx_order[i : i + args.batch_size]
            z_b, y_b = train_z[idx], train_y[idx]
            optimizer.zero_grad()
            loss = F.cross_entropy(head(z_b), y_b, weight=ce_weights)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n_batches += 1
        t_loss /= max(1, n_batches)

        head.eval()
        with torch.no_grad():
            v_loss = F.cross_entropy(head(val_z), val_y, weight=ce_weights).item()
            logits_val = head(val_z).cpu().numpy()
        val_metrics = compute_metrics_dict(logits_val, val_y.cpu().numpy(), "val")
        val_f1 = val_metrics["val/f1_macro"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f} "
            f"| val_f1_macro={val_f1:.4f}"
        )

        log_dict = {"epoch": epoch}
        if run_idx is not None:
            log_dict["probe_run"] = run_idx
        log_dict.update(val_metrics)
        if wandb is not None and log_wandb:
            wandb.log(log_dict, step=epoch)

        if v_loss < best_val_loss - 1e-6:
            best_val_loss = v_loss
            es_counter = 0
            torch.save(
                {"head_state_dict": head.state_dict(), "epoch": epoch, "args": vars(args)},
                best_path,
            )
            print(f"  Saved best probe (val_loss={v_loss:.4f})")
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                print(f"  Early stopping probe at epoch {epoch} (patience={args.es_patience})")
                break

    print("\n=== TEST SET EVALUATION ===")
    ckpt_probe = torch.load(best_path, weights_only=False)
    head.load_state_dict(ckpt_probe["head_state_dict"])
    head.eval()
    with torch.no_grad():
        logits_test = head(test_z).cpu().numpy()
    gt_test = test_y.cpu().numpy()

    test_metrics = compute_metrics_dict(logits_test, gt_test, "test")
    print(classification_report(
        gt_test, logits_test.argmax(axis=1),
        labels=list(range(NUM_CLASSES)),
        target_names=ACTION_CLASS_NAMES,
        zero_division=0,
    ))
    print(
        f"  f1_macro={test_metrics['test/f1_macro']:.3f}  "
        f"f1_weighted={test_metrics['test/f1_weighted']:.3f}"
    )

    if save_artifacts:
        save_test_artifacts(gt_test, logits_test, ckpt_dir, run_tag)
    if run_idx is None:
        append_probe_result_and_print_cumulative(
            test_metrics, ckpt_dir, mode_tag, results_dir=args.results_dir,
        )
    return test_metrics


def summarize_repeated_test_metrics(all_test_metrics, summary_name):
    print(f"\n=== {summary_name} SUMMARY OVER {len(all_test_metrics)} RUNS ===")
    summary_log = {}
    acc_vals = [m["test/acc"] for m in all_test_metrics]
    print(f"  acc: {float(np.mean(acc_vals)):.3f} +- {float(np.std(acc_vals)):.3f}")
    summary_log["summary/acc_mean"] = float(np.mean(acc_vals))
    summary_log["summary/acc_std"] = float(np.std(acc_vals))
    for name in ACTION_CLASS_NAMES:
        f1_vals = [m[f"test/f1_{name}"] for m in all_test_metrics]
        print(f"  f1_{name}: {float(np.mean(f1_vals)):.3f} +- {float(np.std(f1_vals)):.3f}")
        summary_log[f"summary/f1_{name}_mean"] = float(np.mean(f1_vals))
        summary_log[f"summary/f1_{name}_std"] = float(np.std(f1_vals))
    macro_vals = [m["test/f1_macro"] for m in all_test_metrics]
    weighted_vals = [m["test/f1_weighted"] for m in all_test_metrics]
    print(f"  f1_macro: {float(np.mean(macro_vals)):.3f} +- {float(np.std(macro_vals)):.3f}")
    print(f"  f1_weighted: {float(np.mean(weighted_vals)):.3f} +- {float(np.std(weighted_vals)):.3f}")
    summary_log["summary/f1_macro_mean"] = float(np.mean(macro_vals))
    summary_log["summary/f1_macro_std"] = float(np.std(macro_vals))
    summary_log["summary/f1_weighted_mean"] = float(np.mean(weighted_vals))
    summary_log["summary/f1_weighted_std"] = float(np.std(weighted_vals))
    return summary_log

# ---------------------------------------------------------------------------
# homo_to_hetero (same as v3 training)
# ---------------------------------------------------------------------------

from torch_geometric.data import HeteroData

def homo_to_hetero(data, zero_node_features=True):
    node_type = data.node_type
    robot_mask = (node_type == 0)
    bystander_mask = (node_type == 1)

    n_total = node_type.shape[0]
    n_robots = int(robot_mask.sum())
    n_bystanders = int(bystander_mask.sum())

    local_idx = torch.full((n_total,), -1, dtype=torch.long)
    local_idx[robot_mask.nonzero(as_tuple=True)[0]] = torch.arange(n_robots, dtype=torch.long)
    local_idx[bystander_mask.nonzero(as_tuple=True)[0]] = torch.arange(n_bystanders, dtype=torch.long)

    did = int(data.dataset_id.item()) if torch.is_tensor(data.dataset_id) else int(data.dataset_id)

    hdata = HeteroData()
    hdata.dataset_id = torch.tensor([did], dtype=torch.long)

    if did == SEAN_ID:
        hdata["robot"].x = data.x[robot_mask]
        hdata["bystander"].num_nodes = n_bystanders
    else:
        if zero_node_features:
            hdata["robot"].num_nodes = n_robots
            hdata["bystander"].num_nodes = n_bystanders
        else:
            hdata["robot"].x = torch.zeros(n_robots, SNSB_ROBOT_DIM)
            hdata["bystander"].x = torch.zeros(n_bystanders, SNSB_BYSTANDER_DIM)

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


class HeteroWrapper(Dataset):
    def __init__(self, subset, zero_node_features=True):
        super().__init__()
        self.subset = subset
        self.zero_node_features = zero_node_features

    def len(self):
        return len(self.subset)

    def get(self, idx):
        return homo_to_hetero(self.subset[idx], zero_node_features=self.zero_node_features)


# ---------------------------------------------------------------------------
# Oracle eval dataset (T=10 → checkpoint SNSB timestep count, drop x when zero_node_features)
# ---------------------------------------------------------------------------

class InterpSocialNavDataset(Dataset):
    """Interpolate oracle data from T_raw to the checkpoint's SNSB timestep count."""

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

        def _t(x):
            return torch.from_numpy(_interp_1d(x.numpy(), self.target_t))

        robot_traj = data["robot"].traj.numpy()
        data["robot"].traj = torch.from_numpy(_interp_1d(robot_traj, self.target_t))
        data["robot"].x = torch.from_numpy(
            _interp_1d(robot_traj, self.target_t).reshape(1, self.target_t * 3)
        )

        if data["human"].x.shape[0] > 0:
            human_traj = data["human"].traj.numpy()
            human_traj_interp = _interp_1d(human_traj, self.target_t)
            data["human"].traj = torch.from_numpy(human_traj_interp)
            P = human_traj_interp.shape[0]
            data["human"].x = torch.from_numpy(human_traj_interp.reshape(P, self.target_t * 3))

        for et in [("robot", "to", "human"), ("human", "to", "robot"), ("human", "to", "human")]:
            if hasattr(data[et], "edge_attr"):
                ea = data[et].edge_attr.numpy()
                if ea.shape[0] > 0:
                    data[et].edge_attr = torch.from_numpy(_interp_1d(ea, self.target_t))
                else:
                    data[et].edge_attr = torch.zeros(0, self.target_t, ea.shape[2])

        data.num_timesteps = self.target_t
        return data


def load_oracle_eval_dataset_v3(h5_path, ckpt, action_probs_json_dir=None,
                                zero_node_features=True, target_t=20):
    """Load oracle dataset interpolated to the checkpoint's SNSB timestep count."""
    base = SocialNavGraphDatasetV1(h5_path, action_probs_json_dir=action_probs_json_dir)
    interped = InterpSocialNavDataset(base, target_t)
    dataset = UnifiedSocialNavDataset(base_dataset=interped, zero_node_features=zero_node_features)
    # Normalization stats only matter if x is actually used; keep them for completeness
    if "snsb_robot_mean" in ckpt:
        dataset.robot_mean = ckpt["snsb_robot_mean"]
        dataset.robot_std = ckpt["snsb_robot_std"]
        dataset.bystander_mean = ckpt["snsb_bystander_mean"]
        dataset.bystander_std = ckpt["snsb_bystander_std"]

    if zero_node_features:
        # Null out node features so encoder uses snsb_*_type_emb, matching training
        dataset = _ZeroNodeWrapper(dataset)

    return dataset


class _ZeroNodeWrapper(Dataset):
    """Nulls out robot.x and bystander.x so the encoder falls back to type embeddings."""
    def __init__(self, base):
        super().__init__(root=None)
        self.base = base

    def len(self):
        return len(self.base)

    def get(self, idx):
        data = self.base[idx]
        data["robot"].num_nodes = int(data["robot"].x.shape[0])
        data["bystander"].num_nodes = int(data["bystander"].x.shape[0])
        data["robot"].x = None
        data["bystander"].x = None
        return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if getattr(args, "bystander_only", False):
        mode_tag = "bystander_only_64D"
    elif getattr(args, "robot_only", False):
        mode_tag = "robot_only_64D"
    elif args.random_encoder:
        mode_tag = "concat_128D_random_encoder"
    else:
        mode_tag = "concat_128D"
    if args.tag:
        mode_tag = f"{mode_tag}_{args.tag}"

    if wandb is not None and args.wandb:
        wandb.init(project="social_world_model",
                   name=f"snsb_lp_v3_{mode_tag}", config=vars(args))
    elif args.wandb:
        print("wandb not installed; continuing without W&B.")

    # ---- Load checkpoint ----
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(ckpt_dir, exist_ok=True)

    zero_node_features = ckpt.get("zero_node_features", True)
    snsb_timesteps = ckpt.get("snsb_timesteps", 20)
    snsb_edge_node_init = ckpt.get("snsb_edge_node_init", False)
    h5_path = args.h5_path or ckpt.get("snsb_h5_path", "/data/social_nav_v3.h5")

    # ---- Load dataset + split ----
    dataset = UnifiedSocialNavDataset(h5_path, zero_node_features=zero_node_features)
    if "snsb_robot_mean" in ckpt:
        dataset.robot_mean = ckpt["snsb_robot_mean"]
        dataset.robot_std = ckpt["snsb_robot_std"]
        dataset.bystander_mean = ckpt["snsb_bystander_mean"]
        dataset.bystander_std = ckpt["snsb_bystander_std"]

    split_path = ckpt.get("split_path")
    if split_path and not os.path.isabs(split_path) and not os.path.exists(split_path):
        split_path = os.path.join(ckpt_dir, os.path.basename(split_path))
    split = torch.load(split_path, map_location="cpu", weights_only=False)

    snsb_train_idx = split["snsb_train"]
    snsb_val_idx   = split["snsb_val"]

    train_wrap = torch.utils.data.Subset(dataset, snsb_train_idx)
    val_wrap   = torch.utils.data.Subset(dataset, snsb_val_idx)

    if zero_node_features:
        train_wrap = _ZeroNodeWrapper(train_wrap)
        val_wrap = _ZeroNodeWrapper(val_wrap)

    print(f"SNSB  train={len(train_wrap)}  val={len(val_wrap)}  zero_node_features={zero_node_features}")

    persistent = args.num_workers > 0
    train_loader = DataLoader(train_wrap, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=persistent)
    val_loader   = DataLoader(val_wrap,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=persistent)

    # ---- Build model ----
    latent_dim = saved_args.get("latent_dim", 64)
    input_dim = latent_dim if (getattr(args, "bystander_only", False) or getattr(args, "robot_only", False)) else 2 * latent_dim

    model = UnifiedGNNVAE(
        hidden_dim=saved_args.get("hidden_dim", 128),
        latent_dim=latent_dim,
        num_mp_layers=saved_args.get("num_mp_layers", 2),
        snsb_timesteps=snsb_timesteps,
    ).to(device)
    print(
        f"input_dim={input_dim}  latent_dim={latent_dim}  "
        f"snsb_timesteps={snsb_timesteps}  snsb_edge_node_init={snsb_edge_node_init}"
    )

    if args.random_encoder:
        print("Using RANDOM (untrained) encoder weights.")
    else:
        skipped, missing, unexpected = load_compatible_state_dict(model, ckpt["model_state_dict"])
        if skipped:    print(f"Skipped incompatible keys: {skipped}")
        if unexpected: print(f"Unexpected keys ignored: {list(unexpected)}")
        if missing:    print(f"Missing keys at init: {list(missing)}")

    for p in model.parameters():
        p.requires_grad = False
    print("UnifiedGNNVAE frozen.")

    nf = getattr(args, "bystander_only", False)
    nr = getattr(args, "robot_only", False)

    # ---- WRB / majority baselines ----
    if args.weighted_random_baseline:
        train_y_np = torch.cat([b["bystander"].y for b in train_loader]).numpy()
        p_train = np.array([(train_y_np == i).mean() for i in range(NUM_CLASSES)])
        majority_class = int(p_train.argmax())
        print(f"\nTrain class probs: { {n: f'{p:.3f}' for n, p in zip(ACTION_CLASS_NAMES, p_train)} }")
        print(f"Majority class: {ACTION_CLASS_NAMES[majority_class]} ({p_train[majority_class]:.3f})")

        if args.eval_h5_paths:
            for eval_h5 in args.eval_h5_paths:
                oracle_dataset = load_oracle_eval_dataset_v3(
                    eval_h5, ckpt, action_probs_json_dir=args.oracle_json_dir,
                    zero_node_features=zero_node_features,
                    target_t=snsb_timesteps,
                )
                oracle_loader = DataLoader(oracle_dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=0)
                oracle_y_np, oracle_probs_np = extract_oracle_labels_and_probs(oracle_loader)
                if oracle_probs_np is None:
                    print("  No action_probs found, skipping PA.")
                    continue
                maj_preds = np.full(len(oracle_y_np), majority_class)
                maj_pa = float(oracle_probs_np[np.arange(len(maj_preds)), maj_preds].mean())
                maj_m = compute_metrics_dict(np.eye(NUM_CLASSES)[maj_preds], oracle_y_np, "maj_oracle")
                print(f"  Majority oracle PA={maj_pa:.3f}  f1_macro={maj_m['maj_oracle/f1_macro']:.3f}")
                pa_vals, macro_vals = [], []
                for _ in range(PROBE_BASELINE_RUNS):
                    preds = np.array([np.random.choice(NUM_CLASSES, p=p_train) for _ in range(len(oracle_y_np))])
                    m = compute_metrics_dict(np.eye(NUM_CLASSES)[preds], oracle_y_np, "wrb_oracle")
                    pa_vals.append(float(oracle_probs_np[np.arange(len(preds)), preds].mean()))
                    macro_vals.append(m["wrb_oracle/f1_macro"])
                print(f"  WRB oracle PA={float(np.mean(pa_vals)):.3f}±{float(np.std(pa_vals)):.3f}  "
                      f"f1_macro={float(np.mean(macro_vals)):.3f}±{float(np.std(macro_vals)):.3f}")
        return

    # ---- Extract embeddings ----
    print(f"Extracting per-bystander embeddings ({input_dim}D = {mode_tag})...")
    train_z, train_y = extract_bystander_embeddings(model, train_loader, device, bystander_only=nf, robot_only=nr)
    val_z,   val_y   = extract_bystander_embeddings(model, val_loader,   device, bystander_only=nf, robot_only=nr)
    print(f"  train_z: {train_z.shape}  val_z: {val_z.shape}")

    print("Train class distribution:")
    for i, name in enumerate(ACTION_CLASS_NAMES):
        count = (train_y == i).sum().item()
        print(f"  {name}: {count} ({100. * count / max(1, len(train_y)):.1f}%)")

    train_z, train_y = train_z.to(device), train_y.to(device)
    val_z,   val_y   = val_z.to(device),   val_y.to(device)

    # ---- Train linear probe ----
    # val embeddings serve as test set (no held-out test in combined_v1.h5 for SNSB)
    run_linear_probe_once(
        args, train_z, train_y, val_z, val_y, val_z, val_y,
        input_dim, ckpt_dir, mode_tag, device,
        log_wandb=args.wandb,
    )

    # ---- Oracle eval ----
    if args.eval_h5_paths:
        best_path = os.path.join(ckpt_dir, f"best_snsb_probe_{mode_tag}.pt")
        ckpt_probe = torch.load(best_path, weights_only=False)
        head = nn.Linear(input_dim, NUM_CLASSES).to(device)
        head.load_state_dict(ckpt_probe["head_state_dict"])
        head.eval()

        for eval_h5 in args.eval_h5_paths:
            labeler_tag = os.path.splitext(os.path.basename(eval_h5))[0]
            print(f"\n{'=' * 60}")
            print(f"ORACLE EVAL: {eval_h5}")
            print(f"{'=' * 60}")
            oracle_dataset = load_oracle_eval_dataset_v3(
                eval_h5, ckpt, action_probs_json_dir=args.oracle_json_dir,
                zero_node_features=zero_node_features,
                target_t=snsb_timesteps,
            )
            oracle_loader = DataLoader(oracle_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=0)
            oracle_z, oracle_y, oracle_probs = extract_bystander_embeddings_with_probs(
                model, oracle_loader, device, bystander_only=nf, robot_only=nr,
            )
            print(f"Oracle bystander examples: {oracle_z.shape[0]}")
            for i, name in enumerate(ACTION_CLASS_NAMES):
                count = (oracle_y == i).sum().item()
                print(f"  {name}: {count} ({100. * count / max(1, len(oracle_y)):.1f}%)")

            with torch.no_grad():
                logits_oracle = head(oracle_z.to(device)).cpu().numpy()
            gt_oracle = oracle_y.numpy()
            oracle_metrics = compute_metrics_dict(logits_oracle, gt_oracle, "oracle")

            oracle_probs_np = oracle_probs.numpy() if oracle_probs is not None else None
            if oracle_probs_np is None:
                raise RuntimeError(
                    f"Oracle eval requires action_probs for PA, but none found in {eval_h5}."
                )
            oracle_metrics["oracle/pa"] = compute_prediction_agreement(logits_oracle, oracle_probs_np)

            from sklearn.metrics import classification_report
            print(classification_report(gt_oracle, logits_oracle.argmax(axis=1),
                                        labels=list(range(NUM_CLASSES)),
                                        target_names=ACTION_CLASS_NAMES, zero_division=0))
            print(f"  f1_macro={oracle_metrics['oracle/f1_macro']:.3f}  "
                  f"f1_weighted={oracle_metrics['oracle/f1_weighted']:.3f}  "
                  f"pa={oracle_metrics['oracle/pa']:.3f}")

            save_test_artifacts(gt_oracle, logits_oracle, ckpt_dir,
                                f"{mode_tag}_{labeler_tag}", action_probs=oracle_probs_np)
            append_probe_result_and_print_cumulative(
                oracle_metrics, ckpt_dir, f"{mode_tag}_{labeler_tag}",
                prefix="oracle", results_dir=args.results_dir,
            )

    if wandb is not None and args.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_unified_gnn_vae_v3_seed*.pt")
    parser.add_argument("--h5_path", default=None,
                        help="Override h5 path (default: from checkpoint)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--es_patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_encoder", action="store_true")
    parser.add_argument("--weighted_random_baseline", action="store_true")
    parser.add_argument("--bystander_only", action="store_true")
    parser.add_argument("--robot_only", action="store_true")

    parser.add_argument("--no_stratified", dest="no_stratified", action="store_true", default=True)
    parser.add_argument("--stratified", dest="no_stratified", action="store_false")
    parser.add_argument("--no_class_weights", dest="no_class_weights", action="store_true", default=True)
    parser.add_argument("--class_weights", dest="no_class_weights", action="store_false")
    parser.add_argument("--no_scheduler", action="store_true")
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=0)
    parser.add_argument("--adaptive_max_alpha", type=float, default=0.3)

    parser.add_argument("--eval_h5_paths", nargs="+", default=None,
                        help="Oracle eval HDF5 paths (T=10 real data, tiled to T=40).")
    parser.add_argument("--oracle_json_dir",
                        default="dataloaders/socialnavsub_model_labels_v2/human_oracle")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()
    train(args)
