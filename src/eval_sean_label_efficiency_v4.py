"""
Heterogeneous GNN-VAE pipeline — SEAN-only v4 label-efficiency sweep.

eval_sean_label_efficiency_v4.py

Label-efficiency sweep for SEAN-only GNN-VAE v4 checkpoints.

For each v4 checkpoint dir:
  1. Load best_gnn_vae_v4.pt + split_indices_v4.pt
  2. Extract frozen SEAN train/val/test embeddings once
  3. For each (fraction, subset_seed): subsample train scenes, train LP heads, eval
  4. Append raw rows compatible with summarize_label_efficiency.py --task sean

Default representation matches train_linear_probe_v4.py:
  [z_robot || z_follower || mean(z_bystanders)]

Probe-only ablations:
  raw_robot: probe directly on normalized robot input features [B, 592]
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloaders.sean_graph_dataset_v4 import SEANGraphDatasetV4
from models.gnn_vae_v4 import GNNVAEv4
from train_linear_probe_v4 import (
    LABEL_NAMES,
    binarize_labels,
    extract_embeddings,
    load_compatible_state_dict,
)


FIELDNAMES = [
    "vae_dir", "split_seed", "task", "subset_seed", "fraction",
    "n_train_scenes", "n_train_samples",
    "f1_macro", "auc_macro", "acc_macro",
    "f1_competence", "auc_competence", "acc_competence",
    "f1_surprise", "auc_surprise", "acc_surprise",
    "f1_intention", "auc_intention", "acc_intention",
]


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_scene_labels(sean_h5, combined_h5):
    """Map sean_together_v2 sample keys to scene IDs via combined_v1.h5 attrs."""
    with h5py.File(sean_h5, "r") as fs:
        keys = sorted(fs.keys(), key=lambda k: (
            _parse_key_index(k) is None,
            _parse_key_index(k) or 0,
            k,
        ))

    scenes = []
    if combined_h5 and os.path.exists(combined_h5):
        with h5py.File(combined_h5, "r") as fc:
            for key in keys:
                if key in fc and "scene" in fc[key].attrs:
                    scenes.append(str(fc[key].attrs["scene"]))
                else:
                    scenes.append(key)
    else:
        scenes = list(keys)
    return scenes


def _parse_key_index(key):
    tail = key.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def subsample_embeddings_by_scene(z, y, scenes, fraction, shuffled):
    """shuffled: precomputed (unseeded) permutation of unique scene names for
    this run-unit. Fraction prefixes nest within the same shuffle."""
    n_keep = max(1, round(fraction * len(shuffled)))
    keep = set(shuffled[:n_keep])
    mask = np.array([scene in keep for scene in scenes])
    return z[mask], y[mask], n_keep


def build_model(ckpt, device):
    saved_args = ckpt["args"]
    model = GNNVAEv4(
        robot_dim=592,
        follower_embed_dim=8,
        bystander_embed_dim=8,
        hidden_dim=saved_args.get("hidden_dim", 128),
        latent_dim=saved_args.get("latent_dim", 64),
        num_mp_layers=saved_args.get("num_mp_layers", 2),
        edge_transformer_layers=saved_args.get("edge_transformer_layers", 2),
        edge_transformer_heads=saved_args.get("edge_transformer_heads", 2),
        dropout=saved_args.get("dropout", 0.1),
    ).to(device)
    skipped, missing, unexpected = load_compatible_state_dict(model, ckpt["model_state_dict"])
    if skipped:
        print(f"  skipped incompatible keys: {skipped}")
    if missing:
        print(f"  missing model keys left at init: {list(missing)}")
    if unexpected:
        print(f"  unexpected keys ignored: {list(unexpected)}")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def train_sean_heads(tr_z, tr_y, va_z, va_y, input_dim, lr, epochs, patience, batch_size, device):
    heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(3)]).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=1e-4)
    tz, ty = tr_z.to(device), tr_y.to(device).float()
    vz, vy = va_z.to(device), va_y.to(device).float()

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        heads.train()
        perm = torch.randperm(tz.shape[0], device=device)
        for start in range(0, tz.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = sum(
                F.binary_cross_entropy_with_logits(heads[d](tz[idx]).squeeze(-1), ty[idx, d])
                for d in range(3)
            ) / 3
            loss.backward()
            opt.step()

        heads.eval()
        with torch.no_grad():
            val_loss = sum(
                F.binary_cross_entropy_with_logits(heads[d](vz).squeeze(-1), vy[:, d])
                for d in range(3)
            ).item() / 3

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in heads.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    heads.load_state_dict(best_state)
    return heads


def sean_metrics(heads, z, y, device):
    heads.eval()
    with torch.no_grad():
        logits = torch.stack([
            heads[d](z.to(device)).squeeze(-1) for d in range(3)
        ], dim=1).cpu().numpy()
    preds = (logits > 0).astype(float)
    y_np = y.numpy()
    row, f1s, aucs, accs = {}, [], [], []
    for d, name in enumerate(LABEL_NAMES):
        acc = accuracy_score(y_np[:, d], preds[:, d])
        f1 = f1_score(y_np[:, d], preds[:, d], zero_division=0)
        try:
            auc = roc_auc_score(y_np[:, d], logits[:, d])
        except ValueError:
            auc = float("nan")
        row[f"acc_{name}"] = acc
        row[f"f1_{name}"] = f1
        row[f"auc_{name}"] = auc
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)
    row["acc_macro"] = float(np.mean(accs))
    row["f1_macro"] = float(np.mean(f1s))
    row["auc_macro"] = float(np.nanmean(aucs))
    return row


def resolve_split_path(vae_dir, ckpt):
    split_path = ckpt.get("split_path", os.path.join(vae_dir, "split_indices_v4.pt"))
    if not os.path.isabs(split_path) and not os.path.exists(split_path):
        candidate = os.path.join(vae_dir, os.path.basename(split_path))
        if os.path.exists(candidate):
            split_path = candidate
    return split_path


def process_vae_dir(vae_dir, args, device):
    ckpt_path = os.path.join(vae_dir, args.checkpoint_name)
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] No checkpoint: {ckpt_path}")
        return

    print(f"\n{'=' * 60}\nVAE: {vae_dir}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    split_path = resolve_split_path(vae_dir, ckpt)
    if not os.path.exists(split_path):
        print(f"[SKIP] No split file: {split_path}")
        return
    splits = torch.load(split_path, map_location="cpu", weights_only=False)
    print(f"Split: {split_path}")

    dataset = SEANGraphDatasetV4(
        args.h5_path,
        robot_mean=ckpt["robot_mean"],
        robot_std=ckpt["robot_std"],
    )
    scenes_all = load_scene_labels(args.h5_path, args.combined_h5)

    tr_idx = list(splits["train"])
    va_idx = list(splits["val"])
    te_idx = list(splits["test"])
    tr_scenes = [scenes_all[i] for i in tr_idx]

    train_loader = DataLoader(torch.utils.data.Subset(dataset, tr_idx),
                              batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, va_idx),
                            batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, te_idx),
                             batch_size=args.batch_size, shuffle=False)

    model = build_model(ckpt, device)
    raw_robot = args.ablation == "raw_robot"
    mode_name = "raw robot features" if raw_robot else "SEAN-only v4 embeddings"
    print(f"Extracting {mode_name}...")
    tr_z, tr_y = extract_embeddings(model, train_loader, device, raw_robot=raw_robot)
    va_z, va_y = extract_embeddings(model, val_loader, device, raw_robot=raw_robot)
    te_z, te_y = extract_embeddings(model, test_loader, device, raw_robot=raw_robot)
    tr_y = binarize_labels(tr_y, flip_surprise=args.flip_surprise)
    va_y = binarize_labels(va_y, flip_surprise=args.flip_surprise)
    te_y = binarize_labels(te_y, flip_surprise=args.flip_surprise)
    print(f"  train={tr_z.shape[0]} val={va_z.shape[0]} test={te_z.shape[0]} dim={tr_z.shape[1]}")

    split_seed = os.path.basename(vae_dir)
    # Unseeded per-run scene permutations: each run-unit draws a fresh random
    # shuffle (NO seed); the 6 fraction subsets nest within one run.
    unique_scenes = list(dict.fromkeys(tr_scenes))
    run_perms = [np.random.default_rng().permutation(unique_scenes).tolist()
                 for _ in range(args.n_subset_seeds)]
    total = len(args.fractions) * args.n_subset_seeds
    done = 0
    for fraction in args.fractions:
        for subset_seed in range(args.n_subset_seeds):
            done += 1
            print(f"  [{done}/{total}] frac={fraction:.0%} run={subset_seed}", end=" ", flush=True)
            sub_z, sub_y, n_scenes = subsample_embeddings_by_scene(
                tr_z, tr_y, tr_scenes, fraction, run_perms[subset_seed]
            )
            heads = train_sean_heads(
                sub_z, sub_y, va_z, va_y, tr_z.shape[1],
                args.lr, args.epochs, args.patience, args.batch_size, device
            )
            metrics = sean_metrics(heads, te_z, te_y, device)
            append_row(args.output, {
                "vae_dir": vae_dir,
                "split_seed": split_seed,
                "task": "sean",
                "subset_seed": subset_seed,
                "fraction": fraction,
                "n_train_scenes": n_scenes,
                "n_train_samples": len(sub_z),
                **metrics,
            })
            print(f"SEAN={metrics['f1_macro']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_dirs", nargs="+", required=True)
    parser.add_argument("--h5_path", default="dataloaders/sean_together_v2.h5")
    parser.add_argument("--combined_h5", default="dataloaders/combined_v1.h5",
                        help="Used only to map SEAN sample keys to scene IDs.")
    parser.add_argument("--checkpoint_name", default="best_gnn_vae_v4.pt")
    parser.add_argument("--output", default="sean_label_eff_v4.csv")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--n_subset_seeds", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--flip_surprise", action="store_true")
    parser.add_argument("--ablation", choices=["raw_robot"], default=None,
                        help="Probe-only ablation. raw_robot probes normalized robot input features directly.")
    args = parser.parse_args()

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Output: {args.output}")
    for vae_dir in args.vae_dirs:
        process_vae_dir(vae_dir, args, device)
    print(f"\nDone. {args.output}")


if __name__ == "__main__":
    main()
