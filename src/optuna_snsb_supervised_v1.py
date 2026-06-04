"""
Heterogeneous GNN-VAE pipeline — Optuna search for supervised SNSB ablation.

Searches: lr, dropout, es_patience, adaptive_warmup_epochs, adaptive_max_alpha
Fixed:    hidden_dim=256, latent_dim=128, num_mp_layers=2, batch_size=32,
          adaptive_ramp_epochs=100, weight_decay=1e-4

Objective: minimize best val_loss (CE) over training run.

Usage (parallel workers, one per GPU):
  for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU python -u optuna_snsb_supervised_v1.py \\
        --study_name snsb_supervised_v1 \\
        --storage sqlite:///optuna_snsb_supervised_v1.db \\
        --n_trials 10 &
  done
  wait
"""

import argparse
import os
import sys

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from collections import Counter
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.unified_graph_dataset_v1 import DATASET_SSUB_REAL, UnifiedGraphDataset
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.unified_gnn_vae_v1 import SNSB_TIMESTEPS, UnifiedGNNEncoder
from train_snsb_supervised_v1 import (
    SupervisedSNSBGNN,
    build_stratified_weights,
    compute_per_sample_ce_losses,
    get_snsb_split_indices,
)

# ── Fixed constants ───────────────────────────────────────────────────────────
NUM_CLASSES          = 5
ACTION_CLASS_NAMES   = ["avoid", "follow", "overtake", "yield", "not_consider"]
SNSB_ID              = DATASET_SSUB_REAL
NUM_MP_LAYERS        = 2
BATCH_SIZE           = 32
WEIGHT_DECAY         = 1e-4
ADAPTIVE_RAMP_EPOCHS = 100   # fixed, same as VAE Optuna


# ── Optuna storage helpers (identical to optuna_unified_v1.py) ────────────────

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

def run_trial(trial, args, snsb_ds, train_idx, val_idx, ce_weights, device):
    # Search space — same ranges as optuna_unified_v1.py, plus es_patience
    lr                     = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout                = trial.suggest_categorical("dropout", [0.3, 0.4, 0.5, 0.6])
    weight_decay           = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2])
    es_patience            = trial.suggest_categorical("es_patience", [20, 50, 100])
    adaptive_warmup_epochs = trial.suggest_categorical("adaptive_warmup_epochs", [10, 20])
    adaptive_max_alpha     = trial.suggest_categorical("adaptive_max_alpha", [0.25, 0.5, 0.75])

    train_sub = torch.utils.data.Subset(snsb_ds, train_idx)
    val_sub   = torch.utils.data.Subset(snsb_ds, val_idx)

    stratified_weights = build_stratified_weights(train_sub)
    sampler = WeightedRandomSampler(stratified_weights, num_samples=len(train_sub), replacement=True)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, persistent_workers=True, sampler=sampler)
    val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, persistent_workers=True)

    model = SupervisedSNSBGNN(
        hidden_dim     = args.hidden_dim,
        latent_dim     = args.latent_dim,
        num_mp_layers  = NUM_MP_LAYERS,
        dropout        = dropout,
        snsb_timesteps = SNSB_TIMESTEPS,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val   = float("inf")
    best_state = None
    es_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Adaptive sampler update
        if epoch > adaptive_warmup_epochs:
            epochs_since = epoch - adaptive_warmup_epochs
            alpha = min(adaptive_max_alpha,
                        adaptive_max_alpha * epochs_since / max(ADAPTIVE_RAMP_EPOCHS, 1))
            loss_w   = compute_per_sample_ce_losses(model, train_sub, device,
                                                     BATCH_SIZE, 4)
            combined = (1.0 - alpha) * stratified_weights + alpha * loss_w
            sampler.weights = (combined / combined.sum()).double()

        # Train
        model.train()
        for batch in train_loader:
            batch  = batch.to(device)
            logits = model(batch)
            labels = batch["bystander"].y.long()
            loss   = F.cross_entropy(logits, labels, weight=ce_weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                labels = batch["bystander"].y.long()
                val_losses.append(F.cross_entropy(logits, labels, weight=ce_weights).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if epoch % 20 == 0:
            print(f"  [Trial {trial.number}] epoch {epoch:04d}  val_loss={val_loss:.4f}  "
                  f"es={es_counter}/{es_patience}")

        trial.set_user_attr("epoch", epoch)
        trial.set_user_attr("val_loss", round(val_loss, 5))
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= es_patience:
                trial.set_user_attr("stop_epoch", epoch)
                break

    trial.set_user_attr("best_val_loss", round(best_val, 5))
    return best_val


def objective(trial, args, snsb_ds, train_idx, val_idx, ce_weights, device):
    try:
        return run_trial(trial, args, snsb_ds, train_idx, val_idx, ce_weights, device)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  [Trial {trial.number}] FAILED: {type(e).__name__}: {e}")
        raise optuna.exceptions.TrialPruned()


# ── Report ────────────────────────────────────────────────────────────────────

def report(args):
    study = optuna.load_study(study_name=args.study_name,
                               storage=resolve_storage(args.storage))
    best  = study.best_trial
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
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
    parser.add_argument("--study_name",  default="snsb_supervised_v1")
    parser.add_argument("--storage",     default="sqlite:///optuna_snsb_supervised_v1.db")
    parser.add_argument("--n_trials",    type=int, default=50)
    parser.add_argument("--epochs",      type=int, default=1000)
    parser.add_argument("--hidden_dim",  type=int, default=256)
    parser.add_argument("--latent_dim",  type=int, default=128)
    parser.add_argument("--snsb_h5",     default="dataloaders/social_nav_v3.h5")
    parser.add_argument("--combined_h5", default="dataloaders/combined_v1.h5")
    parser.add_argument("--split_file",  default="checkpoints/split_indices_final.pt")
    parser.add_argument("--gpu",         type=int, default=0)
    args = parser.parse_args()

    if args.mode == "report":
        report(args)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset + fixed split once — shared across all trials
    combined_ds = UnifiedGraphDataset(args.combined_h5, ssub=False)
    split       = torch.load(args.split_file, map_location="cpu", weights_only=False)
    snsb_ds     = UnifiedSocialNavDataset(args.snsb_h5, zero_node_features=True)
    train_idx, val_idx = get_snsb_split_indices(combined_ds, split, snsb_ds)

    # CE class weights from train labels — fixed across all trials
    train_labels = torch.cat([snsb_ds[i]["bystander"].y for i in train_idx])
    counts     = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    ce_weights = (train_labels.shape[0] / (NUM_CLASSES * counts.clamp(min=1))).to(device)
    print(f"Train: {len(train_idx)} graphs  Val: {len(val_idx)} graphs")

    study = optuna.create_study(
        study_name  = args.study_name,
        storage     = resolve_storage(args.storage),
        direction   = "minimize",
        sampler     = TPESampler(seed=42),
        pruner      = MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: objective(trial, args, snsb_ds, train_idx, val_idx, ce_weights, device),
        n_trials  = args.n_trials,
        callbacks = [lambda study, trial: print(
            f"Trial {trial.number} finished: val_loss={trial.value:.5f}  params={trial.params}"
        )],
    )

    print("\n=== BEST TRIAL ===")
    best = study.best_trial
    print(f"  val_loss: {best.value:.5f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
