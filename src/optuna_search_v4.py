"""
Heterogeneous GNN-VAE pipeline — SEAN-only v4 Optuna search.

Optuna hyperparameter search for SEAN v4 GNN-VAE pipeline.

Each trial = Stage 1 VAE training with truncated epochs and early stopping.
Objective = best VAE validation loss.

Run 5 parallel workers, one per GPU:
    for i in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=$i python optuna_search_v4.py \\
            --study_name sean_v4 \\
            --storage sqlite:///optuna_sean_v4.db &
    done
    wait

After search, report best hyperparams and save JSON:
    python optuna_search_v4.py --mode report \\
        --study_name sean_v4 --storage sqlite:///optuna_sean_v4.db

Args:
    --mode          search (default) | report
    --study_name    Optuna study name (shared across workers)
    --storage       SQLite path, e.g. sqlite:///optuna_sean_v4.db
    --n_trials      Trials this worker will attempt (default 50)
    --vae_epochs    Max VAE epochs per trial (default 50)
    --probe_epochs  Max probe epochs per trial (default 1000; early stopping usually kicks in first)
    --h5_path       Path to HDF5 dataset
    --flip_surprise Flip surprise label polarity (default False)
    --out_json      Where to write best_hyperparams.json (default ./best_hyperparams_v4.json)
"""

import os
import sys
import json
import csv
import argparse
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloaders.sean_graph_dataset_v4 import SEANGraphDatasetV4, EDGE_TYPES as DATASET_EDGE_TYPES
from models.gnn_vae_v4 import GNNVAEv4, NODE_TYPES

# ------- constants copied from the v4 training scripts -------
EDGE_LOSS_WEIGHTS = {
    ('robot',     'to', 'follower'):  5.0,
    ('follower',  'to', 'robot'):     5.0,
    ('robot',     'to', 'bystander'): 2.0,
    ('bystander', 'to', 'robot'):     2.0,
    ('follower',  'to', 'bystander'): 1.0,
    ('bystander', 'to', 'follower'):  1.0,
    ('bystander', 'to', 'bystander'): 0.1,
}
LABEL_NAMES      = ['competence', 'surprise', 'intention']
LABEL_THRESHOLD  = 3
VAE_ES_PATIENCE   = 20  # early stopping patience for VAE val_total
PROBE_ES_PATIENCE = 20
VAE_BATCH_SIZE = 32
VAE_WEIGHT_DECAY = 1e-4
PROBE_BATCH_SIZE = 64
PROBE_WEIGHT_DECAY = 1e-4


# ===================================================================
# Shared helpers
# ===================================================================

def compute_vae_loss(robot_recon, edge_recons, data, mu_dict, logvar_dict, beta, device):
    recon_robot = F.mse_loss(robot_recon, data['robot'].x)

    edge_losses, edge_weights = [], []
    for et in DATASET_EDGE_TYPES:
        if et in edge_recons and hasattr(data[et], 'edge_attr'):
            edge_losses.append(F.mse_loss(edge_recons[et], data[et].edge_attr))
            edge_weights.append(EDGE_LOSS_WEIGHTS[et])
    if edge_losses:
        w           = torch.tensor(edge_weights, device=device)
        losses      = torch.stack(edge_losses)
        recon_edges = (w * losses).sum() / w.sum()
    else:
        recon_edges = torch.tensor(0.0, device=device)

    all_mu     = torch.cat([mu_dict[t] for t in NODE_TYPES if mu_dict[t].shape[0] > 0], dim=0)
    all_logvar = torch.cat([logvar_dict[t] for t in NODE_TYPES if logvar_dict[t].shape[0] > 0], dim=0)
    kl         = -0.5 * torch.mean(1 + all_logvar - all_mu.pow(2) - all_logvar.exp())

    total = recon_robot + recon_edges + beta * kl
    return total, recon_robot, recon_edges, kl


def get_z_concat(model, loader, device):
    """Extract [z_robot || z_follower || mean(z_bystanders)] for all samples."""
    all_z, all_y = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch   = batch.to(device)
            mu_dict, _ = model.encoder(batch)
            B       = int(batch['robot'].batch.max()) + 1
            z_robot    = mu_dict['robot']
            z_follower = mu_dict['follower']
            if mu_dict['bystander'].shape[0] > 0:
                z_b_mean = scatter(mu_dict['bystander'], batch['bystander'].batch,
                                   dim=0, reduce='mean', dim_size=B)
            else:
                z_b_mean = torch.zeros(B, model.latent_dim, device=device)
            z = torch.cat([z_robot, z_follower, z_b_mean], dim=-1)
            all_z.append(z.cpu())
            all_y.append(batch['robot'].y.cpu())
    return torch.cat(all_z), torch.cat(all_y)


def binarize_labels(y, flip_surprise=False):
    y_bin = (y >= LABEL_THRESHOLD).float()
    if flip_surprise:
        y_bin[:, 1] = 1.0 - y_bin[:, 1]
    return y_bin


def mean_val_auc(heads, val_z, val_y, device):
    """Compute mean AUC across 3 labels on pre-extracted val_z."""
    heads.eval()
    with torch.no_grad():
        logits = torch.stack([heads[d](val_z).squeeze(-1) for d in range(3)], dim=1).cpu().numpy()
    labels = val_y.cpu().numpy()
    aucs = []
    for d in range(3):
        try:
            aucs.append(roc_auc_score(labels[:, d], logits[:, d]))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs))


# ===================================================================
# Stage 1: VAE training
# ===================================================================

def train_vae(trial, args, dataset, splits, device):
    lr            = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_dim    = args.hidden_dim  # fixed per study
    latent_dim    = args.latent_dim  # fixed per study
    beta_max      = trial.suggest_categorical('beta_max', [0.1, 0.5, 0.9])
    warmup_epochs = trial.suggest_categorical('warmup_epochs', [10, 25, 50])
    num_mp_layers = 2    # fixed
    weight_decay  = VAE_WEIGHT_DECAY
    batch_size    = VAE_BATCH_SIZE
    dropout       = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2])

    train_set = torch.utils.data.Subset(dataset, splits['train'])
    val_set   = torch.utils.data.Subset(dataset, splits['val'])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = GNNVAEv4(
        robot_dim=592,
        follower_embed_dim=8,
        bystander_embed_dim=8,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_mp_layers=num_mp_layers,
        dropout=dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss   = float('inf')
    best_state      = None
    es_counter      = 0

    for epoch in range(1, args.vae_epochs + 1):
        beta = min(beta_max, beta_max * epoch / max(1, warmup_epochs))

        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            robot_recon, edge_recons, mu_dict, logvar_dict = model(batch)
            loss, *_ = compute_vae_loss(robot_recon, edge_recons, batch, mu_dict, logvar_dict, beta, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()

        model.eval()
        v_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                robot_recon, edge_recons, mu_dict, logvar_dict = model(batch)
                loss, r_rob, r_edge, kl = compute_vae_loss(
                    robot_recon, edge_recons, batch, mu_dict, logvar_dict, beta, device
                )
                v_total += (r_rob + r_edge + beta * kl).item()
        v_total /= len(val_loader)

        scheduler.step(v_total)

        trial.set_user_attr('vae_epoch', epoch)
        trial.set_user_attr('vae_val', round(v_total, 5))
        trial.report(v_total, epoch)
        if trial.should_prune():
            trial.set_user_attr('pruned_with_vae_val', round(best_val_loss, 5))
            raise optuna.exceptions.TrialPruned()

        if v_total < best_val_loss:
            best_val_loss = v_total
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_counter    = 0
        else:
            es_counter += 1
            if es_counter >= VAE_ES_PATIENCE:
                print(f"[Trial {trial.number}] VAE early stop at epoch {epoch} (patience={VAE_ES_PATIENCE})")
                trial.set_user_attr('vae_stop_epoch', epoch)
                break
    else:
        trial.set_user_attr('vae_stop_epoch', args.vae_epochs)

    model.load_state_dict(best_state)
    trial.set_user_attr('best_vae_val', round(best_val_loss, 5))
    return model, latent_dim, hidden_dim, num_mp_layers, best_val_loss


# ===================================================================
# Stage 2: Linear probe
# ===================================================================

def train_probe(trial, args, model, dataset, splits, latent_dim, device):
    lr_probe = trial.suggest_float('lr_probe', 1e-4, 1e-2, log=True)
    wd_probe = PROBE_WEIGHT_DECAY

    batch_size = PROBE_BATCH_SIZE

    train_set  = torch.utils.data.Subset(dataset, splits['train'])
    val_set    = torch.utils.data.Subset(dataset, splits['val'])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)

    train_z, train_y = get_z_concat(model, train_loader, device)
    val_z,   val_y   = get_z_concat(model, val_loader,   device)

    train_y = binarize_labels(train_y, flip_surprise=args.flip_surprise).to(device)
    val_y   = binarize_labels(val_y,   flip_surprise=args.flip_surprise).to(device)
    train_z = train_z.to(device)
    val_z   = val_z.to(device)

    input_dim = 3 * latent_dim
    heads     = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1) for _ in range(3)]).to(device)
    optimizer = AdamW(heads.parameters(), lr=lr_probe, weight_decay=wd_probe)

    best_auc     = -1.0
    best_state   = None
    es_counter   = 0

    for epoch in range(1, args.probe_epochs + 1):
        heads.train()
        perm = torch.randperm(train_z.shape[0], device=device)
        for i in range(0, train_z.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = sum(
                F.binary_cross_entropy_with_logits(heads[d](train_z[idx]).squeeze(-1), train_y[idx, d])
                for d in range(3)
            ) / 3
            loss.backward()
            optimizer.step()

        auc = mean_val_auc(heads, val_z, val_y, device)

        trial.report(auc, args.vae_epochs + epoch)
        if trial.should_prune():
            print(f"[Trial {trial.number}] Probe pruned at epoch {epoch}")
            trial.set_user_attr('probe_stop_epoch', epoch)
            raise optuna.exceptions.TrialPruned()

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= PROBE_ES_PATIENCE:
                print(f"[Trial {trial.number}] Probe early stop at epoch {epoch} (patience={PROBE_ES_PATIENCE})")
                trial.set_user_attr('probe_stop_epoch', epoch)
                break
    else:
        trial.set_user_attr('probe_stop_epoch', args.probe_epochs)

    return best_auc


# ===================================================================
# Optuna objective
# ===================================================================

def objective(trial, args, dataset, splits, device):
    try:
        print(f"\n[Trial {trial.number}] Starting VAE ...")
        _, _, _, _, vae_val = train_vae(trial, args, dataset, splits, device)
        print(f"[Trial {trial.number}] VAE done. best_vae_val={vae_val:.4f}")
        return vae_val
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        import traceback
        print(f"[Trial {trial.number}] FAILED: {e}")
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()


# ===================================================================
# Main
# ===================================================================

def make_splits(dataset):
    N          = len(dataset)
    test_size  = int(0.04 * N)
    val_size   = int(0.16 * N)
    train_size = N - val_size - test_size
    generator  = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    return {'train': train_set.indices, 'val': val_set.indices, 'test': test_set.indices}


def report_best(args):
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    best  = study.best_trial
    print(f"\nBest trial #{best.number}  |  VAE val loss = {best.value:.4f}")
    print("\nBest hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    with open(args.out_json, 'w') as f:
        json.dump({'best_trial': best.number, 'best_vae_val_loss': best.value, 'params': best.params}, f, indent=2)
    print(f"\nSaved -> {args.out_json}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nCompleted trials: {len(completed)} / {len(study.trials)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',         default='search', choices=['search', 'report'])
    parser.add_argument('--study_name',   default='sean_v4')
    parser.add_argument('--storage',      default='sqlite:///optuna_sean_v4.db')
    parser.add_argument('--n_trials',     type=int,   default=50)
    parser.add_argument('--vae_epochs',   type=int,   default=1000)
    parser.add_argument('--probe_epochs', type=int,   default=1000)
    parser.add_argument('--hidden_dim',   type=int,   default=256)
    parser.add_argument('--latent_dim',   type=int,   default=128)
    parser.add_argument('--h5_path',      default='dataloaders/sean_together_v2.h5')
    parser.add_argument('--flip_surprise', action='store_true')
    parser.add_argument('--out_json',     default='best_hyperparams_v4.json')
    args = parser.parse_args()

    if args.mode == 'report':
        report_best(args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')
    print(f"Worker starting | device={device} | CUDA_VISIBLE_DEVICES={gpu_id} | study={args.study_name}")

    print(f"Loading dataset from {args.h5_path} ...")
    dataset = SEANGraphDatasetV4(args.h5_path)
    splits  = make_splits(dataset)
    dataset.set_robot_stats(splits['train'])
    print(f"Dataset loaded. Train={len(splits['train'])}  Val={len(splits['val'])}  Test={len(splits['test'])}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        sampler=TPESampler(seed=None),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=10),
        load_if_exists=True,
    )

    def _objective(trial):
        return objective(trial, args, dataset, splits, device)

    gpu_id   = os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')
    csv_path = args.storage.replace('sqlite:///', '').replace('.db', f'_trials_gpu{gpu_id}.csv')

    # Fixed column order so all workers write consistent headers
    PARAM_COLS = [
        'vae_lr', 'vae_hidden_dim', 'vae_beta_max', 'vae_warmup_epochs',
        'vae_num_mp_layers', 'vae_dropout',
    ]
    PARAM_KEY_MAP = {
        'lr': 'vae_lr', 'hidden_dim': 'vae_hidden_dim',
        'beta_max': 'vae_beta_max', 'warmup_epochs': 'vae_warmup_epochs',
        'num_mp_layers': 'vae_num_mp_layers', 'dropout': 'vae_dropout',
    }
    FIELDNAMES = ['trial', 'vae_val_loss', 'best_vae_val_so_far', 'is_best'] + PARAM_COLS

    def fmt(v):
        return round(v, 6) if isinstance(v, float) else v

    def csv_callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        best    = study.best_trial
        is_best = trial.number == best.number
        params  = {PARAM_KEY_MAP.get(k, k): fmt(v) for k, v in trial.params.items()}
        row = {
            'trial':           trial.number,
            'vae_val_loss':    round(trial.value, 4),
            'best_vae_val_so_far': round(best.value, 4),
            'is_best':         '***BEST***' if is_best else '',
            **{col: params.get(col, '') for col in PARAM_COLS},
        }
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        flag  = '  *** NEW BEST ***' if is_best else ''
        parts = '  '.join(f'{PARAM_KEY_MAP.get(k, k)}={fmt(v)}' for k, v in trial.params.items())
        print(f"[Trial {trial.number:03d}] DB SAVED  vae_val_loss={trial.value:.4f}  best={best.value:.4f}{flag}")
        print(f"           {parts}")

    print(f"Logging trial results -> {csv_path}")
    study.optimize(_objective, n_trials=args.n_trials, show_progress_bar=False, callbacks=[csv_callback])

    # Append best-params summary block to CSV
    best = study.best_trial
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({col: '' for col in FIELDNAMES})  # blank separator row
        best_params = {PARAM_KEY_MAP.get(k, k): fmt(v) for k, v in best.params.items()}
        writer.writerow({
            'trial':           f'BEST=trial_{best.number}',
            'vae_val_loss':    round(best.value, 4),
            'best_vae_val_so_far': round(best.value, 4),
            'is_best':         '***BEST***',
            **{col: best_params.get(col, '') for col in PARAM_COLS},
        })

    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE  |  Best trial #{best.number}  VAE val loss={best.value:.4f}")
    print(f"Best hyperparams:")
    for k, v in best.params.items():
        print(f"  {PARAM_KEY_MAP.get(k, k)}: {v}")
    print(f"Results saved -> {csv_path}")
    print(f"Run --mode report to save best_hyperparams.json")


if __name__ == '__main__':
    main()
