"""
Supervised Social-STGCNN baseline for SEAN + SNSB tasks.

Trains a Social-STGCNN classifier end-to-end with ground-truth labels.
Uses the same data splits as the STARS model for fair comparison.

SEAN task: predict competence / surprise / intention (3 binary labels per scene).
SNSB task: predict action class per bystander node (5 classes).

Usage:
  python train_social_stgcnn_baseline.py --task sean \
      --sean_h5 dataloaders/combined_v1.h5 \
      --split_file checkpoints/split_indices_final.pt

  python train_social_stgcnn_baseline.py --task snsb \
      --snsb_h5 dataloaders/social_nav_v3.h5 \
      --split_file checkpoints/split_indices_final.pt

  python train_social_stgcnn_baseline.py --task both ...
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from dataloaders.unified_gnn_vae_datasets_v1 import (
    UnifiedSEANDataset, UnifiedSocialNavDataset, SNSB_ID,
)
from dataloaders.socialnav_hetero_graph_dataset_v1 import SocialNavGraphDatasetV1
from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset
from train_snsb_supervised_v1 import InterpSocialNavDataset
from models.social_stgcnn_classifier import SocialSTGCNNClassifier, build_stgcnn_input

BINARIZE_THRESH = 3
SEAN_LABEL_NAMES = ["competence", "surprise", "intention"]
SEAN_TIMESTEPS = 40
SNSB_TIMESTEPS = 20
SNSB_IGNORE = -1
SNSB_MACRO3_LABELS = [0, 1, 4]  # avoid, follow, not_consider


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binarize_sean(y_raw, flip_surprise=True):
    """y_raw: Tensor (3,) float → binary float Tensor (3,)."""
    labels = (y_raw.float() > BINARIZE_THRESH).float()
    if flip_surprise:
        labels[1] = 1.0 - labels[1]
    return labels


# ── SEAN ─────────────────────────────────────────────────────────────────────

def eval_sean(model, dataset, indices, device, flip_surprise):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for idx in indices:
            data = dataset[int(idx)]
            if not hasattr(data["robot"], "y"):
                continue
            y = binarize_sean(data["robot"].y.squeeze(), flip_surprise)
            try:
                v, A, T, N, _ = build_stgcnn_input(data, device)
            except ValueError:
                continue
            logits = model.forward_sean(v, A)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    if not all_logits:
        return {}

    logits = torch.stack(all_logits).numpy()
    labels = torch.stack(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    f1s, aucs = [], []
    metrics = {}
    for i, name in enumerate(SEAN_LABEL_NAMES):
        preds = (probs[:, i] > 0.5).astype(int)
        f1 = f1_score(labels[:, i], preds, zero_division=0)
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except Exception:
            auc = float("nan")
        metrics[f"f1_{name}"] = f1
        metrics[f"auc_{name}"] = auc
        f1s.append(f1)
        aucs.append(auc)
    metrics["f1_macro"] = float(np.mean(f1s))
    metrics["auc_macro"] = float(np.nanmean(aucs))
    return metrics


def train_sean(args, device):
    print("=== SEAN task ===")
    dataset = UnifiedSEANDataset(h5_path=args.sean_h5)
    split = torch.load(args.split_file, map_location="cpu", weights_only=False)
    train_idx = list(split["train"])
    val_idx   = list(split["val"])
    test_idx  = list(split["test"])
    dataset.set_robot_stats(train_idx)

    model = SocialSTGCNNClassifier(
        n_stgcnn=args.n_stgcnn, input_feat=8, hidden_feat=args.hidden_feat,
        seq_len=SEAN_TIMESTEPS, kernel_size=args.kernel_size, dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    no_improve = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, "best_sean.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for idx in np.random.permutation(train_idx):
            data = dataset[int(idx)]
            if not hasattr(data["robot"], "y"):
                continue
            y = binarize_sean(data["robot"].y.squeeze(), args.flip_surprise).to(device)
            try:
                v, A, T, N, _ = build_stgcnn_input(data, device)
            except ValueError:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.forward_sean(v, A), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss); n += 1

        val_m = eval_sean(model, dataset, val_idx, device, args.flip_surprise)
        val_f1 = val_m.get("f1_macro", 0.0)
        scheduler.step(val_f1)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={total_loss/max(n,1):.4f} | "
              f"val f1_macro={val_f1:.4f} auc_macro={val_m.get('auc_macro',0):.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_f1_macro": val_f1, "args": vars(args)}, best_path)
            print(f"  Saved best → {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}"); break

    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])
    test_m = eval_sean(model, dataset, test_idx, device, args.flip_surprise)
    print("\n=== SEAN TEST ===")
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")
    return test_m


# ── SNSB ─────────────────────────────────────────────────────────────────────

def eval_snsb(model, dataset, indices, device, use_all=False):
    """
    If use_all=True, iterate all samples in dataset (for oracle h5 which is all test).
    PA is computed when action_probs are available (oracle h5 only).
    """
    model.eval()
    all_preds, all_labels, all_pa_num, all_pa_den = [], [], 0.0, 0
    iterator = range(len(dataset)) if use_all else indices
    with torch.no_grad():
        for idx in iterator:
            data = dataset[int(idx)]
            n_b = int(data["bystander"].num_nodes)
            if n_b == 0:
                continue
            y = data["bystander"].y.view(-1).long()
            valid = y != SNSB_IGNORE
            if valid.sum() == 0:
                continue
            try:
                v, A, T, N, _ = build_stgcnn_input(data, device)
            except ValueError:
                continue
            logits = model.forward_snsb(v, A, n_b)
            preds = logits.argmax(dim=-1).cpu()
            probs = torch.softmax(logits, dim=-1).cpu()
            all_preds.append(preds[valid])
            all_labels.append(y[valid])
            # PA: human_action_probs[i, predicted_class]
            if hasattr(data["bystander"], "action_probs"):
                ap = data["bystander"].action_probs[valid]  # (n_valid, 5)
                p = preds[valid]
                all_pa_num += float(ap[torch.arange(len(p)), p].sum())
                all_pa_den += len(p)

    if not all_preds:
        return {}
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = {
        "f1_macro":  f1_score(labels, preds, average="macro", zero_division=0),
        "f1_macro3": f1_score(labels, preds, average="macro",
                              labels=SNSB_MACRO3_LABELS, zero_division=0),
        "acc": float((preds == labels).mean()),
    }
    if all_pa_den > 0:
        metrics["pa"] = all_pa_num / all_pa_den
    return metrics


def _get_snsb_split_indices(combined_h5, split_file, snsb_ds):
    """
    Replicate get_snsb_split_indices() from train_unified_gnn_vae_v3.py exactly.
    Derives SNSB train/val/test indices by matching scenes from combined_v1.h5
    + split_indices_final.pt to social_nav_v3.h5 (first N_REAL=3052 samples).
    Unmatched samples = oracle/test set.
    """
    N_REAL = 3052
    combined_ds = UnifiedGraphDataset(combined_h5, ssub=False)
    split = torch.load(split_file, map_location="cpu", weights_only=False)

    snsb_train_combined = [i for i in split["train"] if combined_ds._dataset_ids[i] == SNSB_ID]
    snsb_val_combined   = [i for i in split["val"]   if combined_ds._dataset_ids[i] == SNSB_ID]

    train_scenes = set(combined_ds._scenes[i] for i in snsb_train_combined)
    val_scenes   = set(combined_ds._scenes[i] for i in snsb_val_combined)
    assert not (train_scenes & val_scenes), "Scene overlap between train and val!"

    train_idx, val_idx, test_idx = [], [], []
    for i in range(N_REAL):
        sample = snsb_ds.base_dataset[i]
        scene = getattr(sample, "scene", None) or ""
        if scene in train_scenes:
            train_idx.append(i)
        elif scene in val_scenes:
            val_idx.append(i)
        else:
            test_idx.append(i)

    assert len(train_idx) == len(snsb_train_combined), \
        f"SNSB train mismatch: combined={len(snsb_train_combined)} snsb={len(train_idx)}"
    assert len(val_idx) == len(snsb_val_combined), \
        f"SNSB val mismatch: combined={len(snsb_val_combined)} snsb={len(val_idx)}"

    print(f"SNSB scene split verified: train={len(train_idx)} val={len(val_idx)} "
          f"test(unmatched)={len(test_idx)}")
    return train_idx, val_idx, test_idx


def train_snsb(args, device):
    print("=== SNSB task ===")
    dataset = UnifiedSocialNavDataset(h5_path=args.snsb_h5)
    train_idx, val_idx, test_idx = _get_snsb_split_indices(
        args.combined_h5, args.split_file, dataset)
    dataset.set_feature_stats(train_idx)

    model = SocialSTGCNNClassifier(
        n_stgcnn=args.n_stgcnn, input_feat=8, hidden_feat=args.hidden_feat,
        seq_len=SNSB_TIMESTEPS, kernel_size=args.kernel_size, dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=SNSB_IGNORE)

    best_val_f1 = -1.0
    no_improve = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, "best_snsb.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for idx in np.random.permutation(train_idx):
            data = dataset[int(idx)]
            n_b = int(data["bystander"].num_nodes)
            if n_b == 0:
                continue
            y = data["bystander"].y.view(-1).long().to(device)
            try:
                v, A, T, N, _ = build_stgcnn_input(data, device)
            except ValueError:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.forward_snsb(v, A, n_b), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss); n += 1

        val_m = eval_snsb(model, dataset, val_idx, device)
        val_f1 = val_m.get("f1_macro", 0.0)
        scheduler.step(val_f1)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={total_loss/max(n,1):.4f} | "
              f"val f1_macro={val_f1:.4f} f1_macro3={val_m.get('f1_macro3',0):.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_f1_macro": val_f1, "args": vars(args)}, best_path)
            print(f"  Saved best → {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}"); break

    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])

    # Test on oracle h5 (has action_probs for PA) — same 60 test episodes
    if args.oracle_h5 and os.path.isfile(args.oracle_h5):
        oracle_base = SocialNavGraphDatasetV1(args.oracle_h5)
        oracle_interp = InterpSocialNavDataset(oracle_base, target_t=SNSB_TIMESTEPS)
        oracle_ds = UnifiedSocialNavDataset(base_dataset=oracle_interp)
        oracle_ds.bystander_mean = dataset.bystander_mean
        oracle_ds.bystander_std  = dataset.bystander_std
        oracle_ds.robot_mean     = dataset.robot_mean
        oracle_ds.robot_std      = dataset.robot_std
        test_m = eval_snsb(model, oracle_ds, None, device, use_all=True)
        print("\n=== SNSB TEST (oracle h5, with PA) ===")
    else:
        test_m = eval_snsb(model, dataset, test_idx, device)
        print("\n=== SNSB TEST (social_nav_v3, no PA) ===")

    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")
    return test_m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Supervised Social-STGCNN baseline")
    parser.add_argument("--task", choices=["sean", "snsb", "both"], default="both")
    parser.add_argument("--sean_h5",      default="dataloaders/sean_together_v2.h5")
    parser.add_argument("--combined_h5",  default="dataloaders/combined_v1.h5",
                        help="combined_v1.h5 used only for SNSB scene split lookup")
    parser.add_argument("--snsb_h5",   default="dataloaders/social_nav_v3.h5")
    parser.add_argument("--oracle_h5", default="dataloaders/socialnav_human_oracle_v3.h5",
                        help="Oracle h5 for SNSB test eval with PA metric")
    parser.add_argument("--split_file",
                        default="checkpoints/split_indices_final.pt")
    parser.add_argument("--checkpoint_dir",
                        default="checkpoints/social_stgcnn_baseline")
    parser.add_argument("--n_stgcnn",   type=int,   default=2)
    parser.add_argument("--hidden_feat",type=int,   default=64)
    parser.add_argument("--kernel_size",type=int,   default=3)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--grad_clip",  type=float, default=5.0)
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--patience",   type=int,   default=30)
    parser.add_argument("--flip_surprise", action="store_true", default=True)
    parser.add_argument("--no_flip_surprise", dest="flip_surprise",
                        action="store_false")
    parser.add_argument("--gpu",  type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={args.seed}  task={args.task}")

    if args.task in ("sean", "both"):
        train_sean(args, device)
    if args.task in ("snsb", "both"):
        train_snsb(args, device)


if __name__ == "__main__":
    main()
