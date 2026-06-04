"""
Heterogeneous GNN-VAE pipeline — label-efficiency sweep for supervised SNSB ablation.

Loads dataset ONCE per worker process, then loops over all assigned
(model_seed, label_fraction, subsample_seed) combinations — eliminates the
3052-graph reload overhead of launching 600 separate processes.

Each worker handles every N-th combination (round-robin by worker_id).
600 total runs = 10 seeds × 6 fractions × 10 subsample draws.

Usage (one worker per GPU):
  for i in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$((i+3)) python -u eval_snsb_supervised_scratch_label_eff.py \\
        --worker_id $i --n_workers 5 --gpu 0 &
  done
  wait
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.unified_gnn_vae_v1 import SNSB_TIMESTEPS
from train_snsb_supervised_v1 import (
    ACTION_CLASS_NAMES, NUM_CLASSES,
    SupervisedSNSBGNN,
    InterpSocialNavDataset,
    build_stratified_weights,
    compute_metrics,
    compute_per_sample_ce_losses,
    get_snsb_split_indices,
    make_loader,
    run_eval,
    run_oracle_forward,
)
from dataloaders.socialnav_hetero_graph_dataset_v1 import SocialNavGraphDatasetV1

FRACTIONS    = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
N_SEEDS      = 10
N_SUB_SEEDS  = 10

FIELDNAMES = [
    "model_seed", "label_fraction", "subsample_seed",
    "n_train_scenes", "n_train_samples", "stop_epoch",
    "oracle/acc", "oracle/f1_macro", "oracle/f1_macro3",
    "oracle/f1_avoid", "oracle/f1_follow", "oracle/f1_overtake",
    "oracle/f1_yield", "oracle/f1_not_consider",
    "oracle/f1_weighted", "oracle/pa",
]


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def subsample_train_by_scene(train_idx, snsb_ds, label_fraction, subsample_seed):
    if label_fraction >= 1.0:
        return train_idx, len(set(
            getattr(snsb_ds.base_dataset[i], "scene", "") for i in train_idx
        ))
    train_scenes  = [getattr(snsb_ds.base_dataset[i], "scene", "") for i in train_idx]
    unique_scenes = sorted(set(train_scenes))
    rng           = np.random.default_rng(subsample_seed)
    shuffled      = rng.permutation(unique_scenes).tolist()
    n_keep        = max(1, round(label_fraction * len(unique_scenes)))
    keep          = set(shuffled[:n_keep])
    sub_idx       = [i for i, sc in zip(train_idx, train_scenes) if sc in keep]
    return sub_idx, n_keep


def train_one_run(
    snsb_ds, train_idx, val_idx, oracle_loader,
    model_seed, label_fraction, subsample_seed,
    args, device,
):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    sub_idx, n_scenes = subsample_train_by_scene(
        train_idx, snsb_ds, label_fraction, subsample_seed)
    n_samples = len(sub_idx)

    train_sub = torch.utils.data.Subset(snsb_ds, sub_idx)
    val_sub   = torch.utils.data.Subset(snsb_ds, val_idx)

    # Adaptive stratified sampler
    stratified_weights = build_stratified_weights(train_sub)
    sampler = WeightedRandomSampler(
        stratified_weights, num_samples=len(train_sub), replacement=True)

    train_loader = make_loader(train_sub, args.batch_size, True,  args.num_workers, sampler=sampler)
    val_loader   = make_loader(val_sub,   args.batch_size, False, args.num_workers)

    # CE class weights
    train_labels = torch.cat([snsb_ds[i]["bystander"].y for i in sub_idx])
    counts     = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    ce_weights = (train_labels.shape[0] / (NUM_CLASSES * counts.clamp(min=1))).to(device)

    model = SupervisedSNSBGNN(
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        num_mp_layers=args.num_mp_layers, dropout=args.dropout,
        snsb_timesteps=SNSB_TIMESTEPS,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state    = None
    es_counter    = 0
    stop_epoch    = args.epochs

    for epoch in range(1, args.epochs + 1):
        if epoch > args.adaptive_warmup_epochs:
            epochs_since = epoch - args.adaptive_warmup_epochs
            alpha = min(args.adaptive_max_alpha,
                        args.adaptive_max_alpha * epochs_since / max(args.adaptive_ramp_epochs, 1))
            loss_w   = compute_per_sample_ce_losses(model, train_sub, device,
                                                     args.batch_size, args.num_workers)
            combined = (1.0 - alpha) * stratified_weights + alpha * loss_w
            sampler.weights = (combined / combined.sum()).double()

        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch  = batch.to(device)
            logits = model(batch)
            labels = batch["bystander"].y.long()
            loss   = F.cross_entropy(logits, labels, weight=ce_weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1

        _, _, val_loss = run_eval(model, val_loader, device, ce_weights=ce_weights)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_counter    = 0
        else:
            es_counter += 1
            if es_counter >= args.es_patience:
                stop_epoch = epoch
                break

    model.load_state_dict(best_state)
    oracle_logits, oracle_labels, oracle_probs = run_oracle_forward(model, oracle_loader, device)
    oracle_m = compute_metrics(oracle_logits, oracle_labels, "oracle")
    if oracle_probs is not None:
        preds = oracle_logits.argmax(axis=1)
        oracle_m["oracle/pa"] = float(oracle_probs[np.arange(len(preds)), preds].mean())

    print(f"  seed={model_seed} frac={label_fraction:.0%} sub={subsample_seed} "
          f"ep={stop_epoch} f1_macro3={oracle_m.get('oracle/f1_macro3', 0):.3f} "
          + (f"pa={oracle_m.get('oracle/pa', 0):.3f}" if 'oracle/pa' in oracle_m else ""))

    return {
        "model_seed": model_seed, "label_fraction": label_fraction,
        "subsample_seed": subsample_seed, "n_train_scenes": n_scenes,
        "n_train_samples": n_samples, "stop_epoch": stop_epoch,
        **{k: float(v) for k, v in oracle_m.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id",   type=int, required=True)
    parser.add_argument("--n_workers",   type=int, required=True)
    parser.add_argument("--snsb_h5",     default="dataloaders/social_nav_v3.h5")
    parser.add_argument("--combined_h5", default="dataloaders/combined_v1.h5")
    parser.add_argument("--split_file",  default="checkpoints/split_indices_final.pt")
    parser.add_argument("--oracle_h5",   default="dataloaders/socialnav_human_oracle_v3.h5")
    parser.add_argument("--results_csv", default="label_eff_supervised_scratch_snsb.csv")
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--epochs",      type=int,   default=1000)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=0.00011564)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--es_patience", type=int,   default=20)
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--latent_dim",  type=int,   default=128)
    parser.add_argument("--num_mp_layers",type=int,  default=2)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--adaptive_warmup_epochs", type=int,   default=20)
    parser.add_argument("--adaptive_max_alpha",     type=float, default=0.75)
    parser.add_argument("--adaptive_ramp_epochs",   type=int,   default=100)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[worker {args.worker_id}/{args.n_workers}] device={device}")

    # Load everything once
    print("Loading dataset...")
    combined_ds = UnifiedGraphDataset(args.combined_h5, ssub=False)
    split       = torch.load(args.split_file, map_location="cpu", weights_only=False)
    snsb_ds     = UnifiedSocialNavDataset(args.snsb_h5, zero_node_features=True)
    train_idx, val_idx = get_snsb_split_indices(combined_ds, split, snsb_ds)
    print(f"Split: train={len(train_idx)}  val={len(val_idx)}")

    print("Loading oracle...")
    oracle_base   = SocialNavGraphDatasetV1(args.oracle_h5)
    oracle_interp = InterpSocialNavDataset(oracle_base, target_t=SNSB_TIMESTEPS)
    oracle_ds     = UnifiedSocialNavDataset(base_dataset=oracle_interp, zero_node_features=True)
    oracle_loader = DataLoader(oracle_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build all combinations, take every n_workers-th one
    combos = [
        (seed, frac, sub_seed)
        for seed     in range(1, N_SEEDS + 1)
        for frac     in FRACTIONS
        for sub_seed in range(N_SUB_SEEDS)
    ]
    my_combos = [c for i, c in enumerate(combos) if i % args.n_workers == args.worker_id]
    print(f"Assigned {len(my_combos)}/{len(combos)} combinations")

    for i, (seed, frac, sub_seed) in enumerate(my_combos):
        print(f"\n[{i+1}/{len(my_combos)}] seed={seed} frac={frac:.0%} sub_seed={sub_seed}")
        row = train_one_run(
            snsb_ds, train_idx, val_idx, oracle_loader,
            seed, frac, sub_seed, args, device,
        )
        append_row(args.results_csv, row)

    print(f"\n[worker {args.worker_id}] done → {args.results_csv}")


if __name__ == "__main__":
    main()
