"""
Heterogeneous GNN-VAE pipeline — Optuna search for Social-STGCNN baseline.

Searches: lr, dropout, weight_decay, hidden_feat, n_stgcnn, kernel_size, patience
Fixed:    grad_clip=5.0, epochs=200, task=snsb (SNSB objective)

Objective: maximize best val_f1_macro (consistent with training script ES criterion).

Usage (parallel workers, one per GPU):
  CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4,5,6,7 N_TRIALS=10 \\
      bash hetero_scripts/run_optuna_stgcnn.sh
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset
from dataloaders.unified_gnn_vae_datasets_v1 import SNSB_ID, UnifiedSocialNavDataset
from models.social_stgcnn_classifier import SocialSTGCNNClassifier, build_stgcnn_input
from train_social_stgcnn_baseline import eval_snsb
from train_snsb_supervised_v1 import get_snsb_split_indices as _get_snsb_split_indices

# ── Fixed constants ───────────────────────────────────────────────────────────
SNSB_TIMESTEPS = 20
GRAD_CLIP      = 5.0
SNSB_IGNORE    = -1


# ── Storage helpers (identical to optuna_unified_v1.py) ──────────────────────

def resolve_storage(storage):
    if not storage.startswith("journal://"):
        return storage
    journal_path = storage[len("journal://"):]
    if not journal_path:
        raise ValueError("journal storage requires a path, e.g. journal://optuna_run.log")
    backend_cls = getattr(optuna.storages.journal, "JournalFileBackend", None)
    if backend_cls is not None:
        backend = backend_cls(journal_path)
    else:
        backend = optuna.storages.JournalFileStorage(journal_path)
    return optuna.storages.JournalStorage(backend)


# ── Trial ─────────────────────────────────────────────────────────────────────

def run_trial(trial, args, dataset, train_idx, val_idx, device):
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout     = trial.suggest_categorical("dropout", [0.2, 0.3, 0.4, 0.5])
    weight_decay= trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2])
    hidden_feat = trial.suggest_categorical("hidden_feat", [32, 64, 128])
    n_stgcnn    = trial.suggest_categorical("n_stgcnn", [1, 2, 3])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    patience    = trial.suggest_categorical("patience", [20, 50, 100])

    model = SocialSTGCNNClassifier(
        n_stgcnn=n_stgcnn, input_feat=8, hidden_feat=hidden_feat,
        seq_len=SNSB_TIMESTEPS, kernel_size=kernel_size, dropout=dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=SNSB_IGNORE)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

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
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss); n += 1

        # Val loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for idx in val_idx:
                data = dataset[int(idx)]
                n_b = int(data["bystander"].num_nodes)
                if n_b == 0:
                    continue
                y = data["bystander"].y.view(-1).long().to(device)
                try:
                    v, A, T, N, _ = build_stgcnn_input(data, device)
                except ValueError:
                    continue
                val_losses.append(criterion(model.forward_snsb(v, A, n_b), y).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        scheduler.step(-val_loss)  # scheduler mode=max, negate loss

        if epoch % 20 == 0:
            print(f"  [Trial {trial.number}] epoch {epoch:03d}  "
                  f"val_loss={val_loss:.4f}  no_improve={no_improve}/{patience}")

        trial.set_user_attr("epoch", epoch)
        trial.set_user_attr("val_loss", round(val_loss, 5))
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                trial.set_user_attr("stop_epoch", epoch)
                break

    trial.set_user_attr("best_val_loss", round(best_val_loss, 5))
    return best_val_loss


def objective(trial, args, dataset, train_idx, val_idx, device):
    try:
        return run_trial(trial, args, dataset, train_idx, val_idx, device)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  [Trial {trial.number}] FAILED: {type(e).__name__}: {e}")
        raise optuna.exceptions.TrialPruned()


# ── Report ────────────────────────────────────────────────────────────────────

def report(args):
    study = optuna.load_study(study_name=args.study_name,
                               storage=resolve_storage(args.storage))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial
    print(f"Completed: {len(completed)} / {len(study.trials)}")
    print(f"Best trial #{best.number}  val_loss={best.value:.5f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print("  User attrs:")
    for k, v in best.user_attrs.items():
        print(f"    {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        default="search", choices=["search", "report"])
    parser.add_argument("--study_name",  default="social_stgcnn_v1")
    parser.add_argument("--storage",     default="sqlite:///optuna_social_stgcnn_v1.db")
    parser.add_argument("--n_trials",    type=int, default=10)
    parser.add_argument("--epochs",      type=int, default=200)
    parser.add_argument("--snsb_h5",     default="/data/social_nav_v3.h5")
    parser.add_argument("--combined_h5", default="/data/combined_v1.h5")
    parser.add_argument("--split_file",  default="checkpoints/split_indices_final.pt")
    parser.add_argument("--gpu",         type=int, default=0)
    args = parser.parse_args()

    if args.mode == "report":
        report(args)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset + split once — shared across all trials
    combined_ds = UnifiedGraphDataset(args.combined_h5, ssub=False)
    split       = torch.load(args.split_file, map_location="cpu", weights_only=False)
    dataset     = UnifiedSocialNavDataset(h5_path=args.snsb_h5)
    train_idx, val_idx = _get_snsb_split_indices(combined_ds, split, dataset)
    dataset.set_feature_stats(train_idx)
    print(f"Train: {len(train_idx)} graphs  Val: {len(val_idx)} graphs")

    study = optuna.create_study(
        study_name     = args.study_name,
        storage        = resolve_storage(args.storage),
        direction      = "minimize",
        sampler        = TPESampler(seed=42),
        pruner         = MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: objective(trial, args, dataset, train_idx, val_idx, device),
        n_trials  = args.n_trials,
        callbacks = [lambda study, trial: print(
            f"Trial {trial.number} finished: val_f1_macro={trial.value:.5f}  params={trial.params}"
        )],
    )

    print("\n=== BEST TRIAL ===")
    best = study.best_trial
    print(f"  val_loss: {best.value:.5f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
