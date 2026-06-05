"""
Heterogeneous GNN-VAE pipeline — label-efficiency eval for supervised SNSB ablation.

Mirrors eval_label_efficiency.py (SNSB path only) but loads a supervised
SupervisedSNSBGNN checkpoint instead of a VAE checkpoint.

Protocol:
  1. Load supervised checkpoint, freeze encoder.
  2. Extract train/val/oracle bystander embeddings (identical to VAE probe pipeline).
  3. For each fraction × n_subset_seeds: subsample train embeddings by scene,
     train a fresh linear head, evaluate on oracle.
  4. Append one CSV row per (ckpt_dir, fraction, subset_seed).

Scene subsampling is unseeded per run-unit (fractions nest within one shuffle),
identical to eval_label_efficiency.py.

Usage:
  python eval_snsb_supervised_label_eff.py \\
      --ckpt_dirs experiments/snsb_supervised_v1_optuned/run{1..10} \\
      --output label_eff_supervised_snsb.csv
"""

import argparse
import csv
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.unified_graph_dataset_v1 import DATASET_SSUB_REAL, UnifiedGraphDataset
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.unified_gnn_vae_v1 import SNSB_TIMESTEPS
from train_snsb_supervised_v1 import (
    InterpSocialNavDataset,
    SupervisedSNSBGNN,
    get_snsb_split_indices,
)
from dataloaders.socialnav_hetero_graph_dataset_v1 import SocialNavGraphDatasetV1

SNSB_CLASSES  = ["avoid", "follow", "overtake", "yield", "not_consider"]
NUM_CLASSES   = len(SNSB_CLASSES)
IGNORE_LABEL  = -1
SNSB_ID       = DATASET_SSUB_REAL


# ── embedding extraction ──────────────────────────────────────────────────────

def extract_bystander_embeddings(model, loader, device):
    """Returns (z, y, scenes_per_bystander)."""
    model.eval()
    all_z, all_y, all_scenes = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu_dict, _ = model.encoder(batch)
            by_batch = batch["bystander"].batch
            robot_z  = mu_dict["robot"][by_batch]
            z = torch.cat([robot_z, mu_dict["bystander"]], dim=1)
            labels = batch["bystander"].y.long()
            valid  = labels != IGNORE_LABEL
            if valid.sum() == 0:
                continue
            all_z.append(z[valid].cpu())
            all_y.append(labels[valid].cpu())
            batch_scenes = batch.scene
            all_scenes.extend(
                batch_scenes[int(by_batch[i])] for i in range(len(by_batch)) if valid[i]
            )
    return torch.cat(all_z), torch.cat(all_y), all_scenes


def extract_oracle_embeddings(model, oracle_h5, device, batch_size):
    """Returns (z, y, action_probs_or_None)."""
    oracle_base   = SocialNavGraphDatasetV1(oracle_h5)
    oracle_interp = InterpSocialNavDataset(oracle_base, target_t=SNSB_TIMESTEPS)
    oracle_ds     = UnifiedSocialNavDataset(base_dataset=oracle_interp, zero_node_features=True)
    loader = DataLoader(oracle_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_z, all_y, all_probs = [], [], []
    has_probs = True
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu_dict, _ = model.encoder(batch)
            by_batch = batch["bystander"].batch
            robot_z  = mu_dict["robot"][by_batch]
            z = torch.cat([robot_z, mu_dict["bystander"]], dim=1)
            labels = batch["bystander"].y.long()
            valid  = labels != IGNORE_LABEL
            if valid.sum() == 0:
                continue
            all_z.append(z[valid].cpu())
            all_y.append(labels[valid].cpu())
            if has_probs:
                if hasattr(batch["bystander"], "action_probs"):
                    all_probs.append(batch["bystander"].action_probs[valid].cpu())
                else:
                    has_probs = False
    probs = torch.cat(all_probs) if has_probs and all_probs else None
    return torch.cat(all_z), torch.cat(all_y), probs


# ── subsampling — identical to eval_label_efficiency.py ──────────────────────

def subsample_embeddings_by_scene(z, y, scenes, fraction, shuffled):
    n_keep = max(1, round(fraction * len(shuffled)))
    keep   = set(shuffled[:n_keep])
    mask   = np.array([sc in keep for sc in scenes])
    return z[mask], y[mask], n_keep


# ── LP training — identical to eval_label_efficiency.py ──────────────────────

def _train_head(head, loss_fn, tr_z, tr_y, va_z, va_y, lr, epochs, patience, device):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_state, wait = float("inf"), None, 0
    tz, ty = tr_z.to(device), tr_y.to(device)
    vz, vy = va_z.to(device), va_y.to(device)
    for _ in range(epochs):
        head.train(); opt.zero_grad()
        loss_fn(head(tz), ty).backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            val_loss = loss_fn(head(vz), vy).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss; best_state = {k: v.clone() for k, v in head.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    head.load_state_dict(best_state)
    return head


def train_snsb_head(tr_z, tr_y, va_z, va_y, lr, epochs, patience, device):
    head = nn.Linear(tr_z.shape[1], NUM_CLASSES).to(device)
    return _train_head(head, lambda p, t: F.cross_entropy(p, t.long()),
                       tr_z, tr_y, va_z, va_y, lr, epochs, patience, device)


# ── metrics — identical to eval_label_efficiency.py ──────────────────────────

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
    row["f1_macro"]  = f1_score(y_np, preds, average="macro", zero_division=0)
    row["f1_macro3"] = float(np.mean([row["f1_follow"], row["f1_avoid"], row["f1_not_consider"]]))
    row["acc"]       = accuracy_score(y_np, preds)
    if action_probs is not None:
        probs = action_probs.cpu().numpy() if torch.is_tensor(action_probs) else np.asarray(action_probs)
        row["pa"] = float(probs[np.arange(len(preds)), preds].mean())
    return row


# ── CSV output ────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "ckpt_dir", "fraction", "subset_seed",
    "n_train_scenes", "n_train_samples",
    "f1_macro", "f1_macro3", "acc", "pa",
    "f1_avoid", "f1_follow", "f1_overtake", "f1_yield", "f1_not_consider",
    "prec_avoid", "prec_follow", "prec_overtake", "prec_yield", "prec_not_consider",
    "acc_avoid", "acc_follow", "acc_overtake", "acc_yield", "acc_not_consider",
]

def append_row(output, row):
    write_header = not os.path.exists(output)
    with open(output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dirs",    nargs="+", required=True,
                        help="Supervised checkpoint dirs (each has best_model.pt)")
    parser.add_argument("--snsb_h5",     default="/data/social_nav_v3.h5")
    parser.add_argument("--combined_h5", default="/data/combined_v1.h5")
    parser.add_argument("--split_file",  default="checkpoints/split_indices_final.pt")
    parser.add_argument("--oracle_h5",   default="/data/socialnav_human_oracle_v3.h5")
    parser.add_argument("--output",      default="label_eff_supervised_snsb.csv")
    parser.add_argument("--fractions",   nargs="+", type=float,
                        default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--n_subset_seeds", type=int, default=10)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=1000)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--gpu",         type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Fixed split — loaded once, shared across all ckpts
    combined_ds = UnifiedGraphDataset(args.combined_h5, ssub=False)
    split       = torch.load(args.split_file, map_location="cpu", weights_only=False)
    snsb_ds     = UnifiedSocialNavDataset(args.snsb_h5, zero_node_features=True)
    train_idx, val_idx = get_snsb_split_indices(combined_ds, split, snsb_ds)

    train_loader = DataLoader(torch.utils.data.Subset(snsb_ds, train_idx),
                              batch_size=args.batch_size, shuffle=False, num_workers=4,
                              persistent_workers=True)
    val_loader   = DataLoader(torch.utils.data.Subset(snsb_ds, val_idx),
                              batch_size=args.batch_size, shuffle=False, num_workers=4,
                              persistent_workers=True)

    for ckpt_dir in args.ckpt_dirs:
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_path}")

        ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved     = ckpt.get("args", {})
        model     = SupervisedSNSBGNN(
            hidden_dim    = saved.get("hidden_dim",   256),
            latent_dim    = saved.get("latent_dim",   128),
            num_mp_layers = saved.get("num_mp_layers", 2),
            dropout       = saved.get("dropout",      0.2),
            snsb_timesteps= SNSB_TIMESTEPS,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        for p in model.parameters():
            p.requires_grad = False
        print(f"Loaded & frozen (epoch {ckpt.get('epoch', '?')})")

        # Extract embeddings once per checkpoint
        print("Extracting train embeddings...")
        tr_z, tr_y, tr_scenes = extract_bystander_embeddings(model, train_loader, device)
        print("Extracting val embeddings...")
        va_z, va_y, _         = extract_bystander_embeddings(model, val_loader, device)
        print("Extracting oracle embeddings...")
        or_z, or_y, or_probs  = extract_oracle_embeddings(model, args.oracle_h5, device, args.batch_size)
        print(f"  train={tr_z.shape[0]}  val={va_z.shape[0]}  oracle={or_z.shape[0]}")

        # Unseeded scene permutations — fractions nest within each shuffle
        unique_scenes = list(dict.fromkeys(tr_scenes))
        run_perms = [np.random.default_rng().permutation(unique_scenes).tolist()
                     for _ in range(args.n_subset_seeds)]

        total = len(args.fractions) * args.n_subset_seeds
        done  = 0
        for fraction in args.fractions:
            for subset_seed in range(args.n_subset_seeds):
                done += 1
                print(f"  [{done}/{total}] frac={fraction:.0%} seed={subset_seed}", end=" ", flush=True)

                sub_z, sub_y, n_scenes = subsample_embeddings_by_scene(
                    tr_z, tr_y, tr_scenes, fraction, run_perms[subset_seed])

                head = train_snsb_head(sub_z, sub_y, va_z, va_y,
                                       args.lr, args.epochs, args.patience, device)

                m = snsb_metrics(head, or_z, or_y, device, action_probs=or_probs)
                print(f"→ f1_macro={m['f1_macro']:.3f}  f1_macro3={m['f1_macro3']:.3f}"
                      + (f"  pa={m['pa']:.3f}" if "pa" in m else ""))

                append_row(args.output, {
                    "ckpt_dir": ckpt_dir, "fraction": fraction,
                    "subset_seed": subset_seed, "n_train_scenes": n_scenes,
                    "n_train_samples": len(sub_z), **m,
                })

    print(f"\nDone. Results → {args.output}")


if __name__ == "__main__":
    main()
