import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

try:
    import wandb
except ImportError:
    wandb = None

optuna.logging.set_verbosity(optuna.logging.WARNING)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from dataloaders.unified_graph_dataset_all import UnifiedGraphDataset
from models.hgnn_combined_v2 import HGNNCombinedVAE as Model
from train_hgnn_unified import (
    run_epoch,
    compute_stratified_weights,
    compute_per_sample_losses,
    seed_everything,
)

STUDY_META_KEYS = {
    "study_name", "storage", "gpus", "n_trials",
    "wandb_project", "experiments_root",
}


def resolve_storage(study_meta, experiments_root):
    journal_path = os.path.join(experiments_root, study_meta["storage"])
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path),
    )


def load_optuna_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    section = cfg["optuna_hgnn"]

    study_meta = {}
    fixed_params = {}
    tunable_params = {}

    for key, val in section.items():
        if key in STUDY_META_KEYS:
            study_meta[key] = val
        elif isinstance(val, dict) and "type" in val:
            tunable_params[key] = val
        else:
            fixed_params[key] = val

    return study_meta, fixed_params, tunable_params


def suggest_param(trial, name, spec):
    ptype = spec["type"]
    if ptype == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False),
        )
    elif ptype == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], log=spec.get("log", False),
        )
    elif ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unknown tunable type '{ptype}' for param '{name}'")


def build_args(trial, fixed_params, tunable_params, device_str, run_dir,
               study_meta):
    args = dict(fixed_params)
    for name, spec in tunable_params.items():
        args[name] = suggest_param(trial, name, spec)

    hd = args.get("hidden_dim")
    nh = args.get("num_heads")
    if hd is not None and nh is not None and hd % nh != 0:
        raise optuna.TrialPruned(
            f"hidden_dim={hd} not divisible by num_heads={nh}"
        )

    args["device"] = device_str
    args["run_dir"] = run_dir
    args["run_name"] = f"optuna_v2_{trial.number:04d}"
    args["wandb"] = True
    args["wandb_project"] = study_meta.get("wandb_project", "social_world_model")
    return SimpleNamespace(**args)


def objective(trial, fixed_params, tunable_params, study_meta, device_str,
              dataset, splits):
    experiments_root = study_meta["experiments_root"]
    trial_dir = os.path.join(experiments_root, f"optuna_v2_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    args = build_args(
        trial, fixed_params, tunable_params, device_str, trial_dir, study_meta,
    )

    try:
        device = torch.device(args.device)
        seed_everything(args.seed)

        if not args.ssub:
            dataset.set_robot_stats(splits["train"])

        train_set = torch.utils.data.Subset(dataset, splits["train"])
        val_set = torch.utils.data.Subset(dataset, splits["val"])

        sampler = None
        stratified_weights = None
        if args.use_weighted_sampling:
            stratified_weights = compute_stratified_weights(train_set)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=stratified_weights,
                num_samples=len(train_set),
                replacement=True,
            )

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size,
            shuffle=(sampler is None), sampler=sampler,
            num_workers=args.num_workers, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, persistent_workers=True,
        )

        model = Model(
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_mp_layers=args.num_mp_layers,
            edge_transformer_layers=args.edge_transformer_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)

        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )

        wb_run = None
        if wandb is not None:
            try:
                wb_run = wandb.init(
                    project=args.wandb_project,
                    name=args.run_name,
                    group=study_meta.get("study_name", "optuna"),
                    config=vars(args),
                    dir=trial_dir,
                    reinit=True,
                )
            except Exception:
                wb_run = None

        with open(os.path.join(trial_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        node_loss_weight = args.node_loss_weight
        beta_max = args.beta_max
        beta_warmup = args.beta_warmup_epochs

        for epoch in range(1, args.epochs + 1):
            beta = min(beta_max, beta_max * epoch / max(1, beta_warmup))

            if (args.use_weighted_sampling and args.adaptive_reweight
                    and sampler is not None
                    and stratified_weights is not None
                    and epoch > args.adaptive_warmup_epochs):
                loss_weights = compute_per_sample_losses(
                    model, train_set, device, args.batch_size, args.ssub,
                )
                loss_weights = loss_weights / loss_weights.sum()
                epochs_since = epoch - args.adaptive_warmup_epochs
                ramp_epochs = args.adaptive_ramp_epochs
                alpha = min(
                    args.adaptive_max_alpha,
                    args.adaptive_max_alpha * epochs_since / max(ramp_epochs, 1),
                )
                combined = ((1.0 - alpha) * stratified_weights
                            + alpha * loss_weights)
                combined = combined / combined.sum()
                sampler.weights = combined.double()

            train_loss, train_edge, train_node, train_kl = run_epoch(
                model, train_loader, device, args.ssub, node_loss_weight, beta,
                optimizer=optimizer, grad_clip=args.grad_clip,
            )
            with torch.no_grad():
                val_loss, val_edge, val_node, val_kl = run_epoch(
                    model, val_loader, device, args.ssub, node_loss_weight, beta,
                )

            if not math.isfinite(val_loss):
                print(f"  [Trial {trial.number}] NaN/Inf at epoch {epoch}")
                if wb_run:
                    wandb.finish(quiet=True)
                raise optuna.TrialPruned("val_loss is NaN/Inf")

            if wb_run:
                try:
                    wandb.log({
                        "epoch": epoch, "beta": beta,
                        "train/loss": train_loss,
                        "train/loss_edge": train_edge,
                        "train/loss_node": train_node,
                        "train/kl": train_kl,
                        "val/loss": val_loss,
                        "val/loss_edge": val_edge,
                        "val/loss_node": val_node,
                        "val/kl": val_kl,
                    }, step=epoch)
                except Exception:
                    pass

            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"  [Trial {trial.number}] Pruned at epoch {epoch}")
                if wb_run:
                    wandb.finish(quiet=True)
                raise optuna.TrialPruned()

            if val_loss < best_val_loss - args.early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "trial_number": trial.number,
                    "args": vars(args),
                }
                if not args.ssub:
                    save_dict["robot_mean"] = dataset.robot_mean
                    save_dict["robot_std"] = dataset.robot_std
                torch.save(save_dict, os.path.join(trial_dir, "best_model.pt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stopping_patience:
                    print(
                        f"  [Trial {trial.number}] Early stop at epoch {epoch} "
                        f"(patience={args.early_stopping_patience})"
                    )
                    break

        # ── Objective = VAE reconstruction val loss (no linear probe) ─────────
        best_ckpt_path = os.path.join(trial_dir, "best_model.pt")
        if not os.path.exists(best_ckpt_path):
            if wb_run:
                wandb.finish(quiet=True)
            raise optuna.TrialPruned("No best model saved (training may have diverged)")

        trial.set_user_attr("hgnn_stop_epoch", epoch)
        trial.set_user_attr("hgnn_best_val_loss", best_val_loss)

        if wb_run:
            try:
                wandb.log({"best_val_loss": best_val_loss})
            except Exception:
                pass
            wandb.finish(quiet=True)

        print(
            f"  [Trial {trial.number}] HGNN done. "
            f"recon_val_loss={best_val_loss:.6f} stop_epoch={epoch}"
        )
        return best_val_loss

    except optuna.TrialPruned:
        raise
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if wandb is not None and wandb.run is not None:
            wandb.finish(quiet=True)
        print(f"  [Trial {trial.number}] CUDA OOM — skipping")
        raise optuna.TrialPruned("CUDA OOM")
    except Exception as e:
        traceback.print_exc()
        if wandb is not None and wandb.run is not None:
            wandb.finish(quiet=True)
        raise optuna.TrialPruned(f"Trial failed: {e}")


def make_csv_callback(csv_path, tunable_names):
    fieldnames = (
        ["trial", "val_loss", "best_val_loss_so_far", "is_best"]
        + sorted(tunable_names)
    )

    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        best = study.best_trial
        is_best = trial.number == best.number
        row = {
            "trial": trial.number,
            "val_loss": round(trial.value, 6),
            "best_val_loss_so_far": round(best.value, 6),
            "is_best": "***BEST***" if is_best else "",
        }
        for name in sorted(tunable_names):
            v = trial.params.get(name, "")
            row[name] = round(v, 6) if isinstance(v, float) else v

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        flag = "  *** NEW BEST ***" if is_best else ""
        print(
            f"Trial {trial.number:04d}  val_loss={trial.value:.6f}  "
            f"best={best.value:.6f}{flag}"
        )

    return callback


def worker_fn(gpu_id, config_path, n_per_worker):
    study_meta, fixed_params, tunable_params = load_optuna_config(config_path)

    h5_path = fixed_params["h5_path"]
    if not os.path.isabs(h5_path):
        h5_path = os.path.join(SRC_DIR, h5_path)
    ssub = fixed_params.get("ssub", False)

    dataset = UnifiedGraphDataset(h5_path, ssub=ssub)
    test_ratio = 1.0 - fixed_params["train_ratio"] - fixed_params["val_ratio"]
    train_idx, val_idx, test_idx = dataset.scene_split(
        val_ratio=fixed_params["val_ratio"],
        test_ratio=test_ratio,
        seed=fixed_params.get("seed", 42),
    )
    if not ssub:
        dataset.set_robot_stats(train_idx)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    print(
        f"[GPU {gpu_id}] Dataset loaded. "
        f"Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}"
    )

    experiments_root = study_meta["experiments_root"]
    if not os.path.isabs(experiments_root):
        experiments_root = os.path.join(SRC_DIR, experiments_root)
    study_meta["experiments_root"] = experiments_root
    os.makedirs(experiments_root, exist_ok=True)

    storage = resolve_storage(study_meta, experiments_root)

    study = optuna.create_study(
        study_name=study_meta["study_name"],
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=None),
        pruner=MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10,
        ),
        load_if_exists=True,
    )

    device_str = f"cuda:{gpu_id}"

    def _objective(trial):
        return objective(
            trial, fixed_params, tunable_params, study_meta,
            device_str, dataset, splits,
        )

    csv_path = os.path.join(experiments_root, f"trials_gpu{gpu_id}.csv")
    callback = make_csv_callback(csv_path, set(tunable_params.keys()))

    print(f"[GPU {gpu_id}] Starting {n_per_worker} trials")
    study.optimize(_objective, n_trials=n_per_worker, callbacks=[callback])
    print(f"[GPU {gpu_id}] Finished")


def report_best(study_meta, experiments_root):
    storage = resolve_storage(study_meta, experiments_root)
    study = optuna.load_study(
        study_name=study_meta["study_name"],
        storage=storage,
    )
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    print(
        f"\nCompleted: {len(completed)} | Pruned: {len(pruned)} "
        f"| Total: {len(study.trials)}"
    )

    if not completed:
        print("No completed trials yet.")
        return

    best = study.best_trial
    print(f"\nBest trial #{best.number} | val_loss = {best.value:.6f}")
    print("\nBest hyperparameters:")
    for k, v in sorted(best.params.items()):
        print(f"  {k}: {v}")

    experiments_root = study_meta.get("experiments_root", ".")
    if not os.path.isabs(experiments_root):
        experiments_root = os.path.join(SRC_DIR, experiments_root)
    out_path = os.path.join(experiments_root, "best_hyperparams.json")
    result = {
        "best_trial": best.number,
        "best_val_loss": best.value,
        "params": best.params,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for HGNN combined AE",
    )
    parser.add_argument("--config", default="params/config.yaml")
    parser.add_argument("--gpus", default=None,
                        help="Override GPU list (comma-separated, e.g. '0,2,4')")
    parser.add_argument("--n_trials", type=int, default=None,
                        help="Override total trial count")
    parser.add_argument("--mode", default="search",
                        choices=["search", "report"])
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(SRC_DIR, config_path)

    study_meta, fixed_params, tunable_params = load_optuna_config(config_path)

    if args.gpus is not None:
        study_meta["gpus"] = [int(g) for g in args.gpus.split(",")]
    if args.n_trials is not None:
        study_meta["n_trials"] = args.n_trials

    if args.mode == "report":
        report_best(study_meta, experiments_root)
        return

    gpu_ids = study_meta["gpus"]
    n_total = study_meta["n_trials"]
    n_gpus = len(gpu_ids)

    experiments_root = study_meta.get("experiments_root", "experiments/optuna")
    if not os.path.isabs(experiments_root):
        experiments_root = os.path.join(SRC_DIR, experiments_root)
    study_meta["experiments_root"] = experiments_root
    os.makedirs(experiments_root, exist_ok=True)

    storage = resolve_storage(study_meta, experiments_root)

    optuna.create_study(
        study_name=study_meta["study_name"],
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=None),
        pruner=MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10,
        ),
        load_if_exists=True,
    )

    journal_path = os.path.join(experiments_root, study_meta["storage"])
    print(f"Study: {study_meta['study_name']}")
    print(f"Storage: {journal_path}")
    print(f"GPUs: {gpu_ids}  |  Total trials: {n_total}")
    print(f"Tunable params: {sorted(tunable_params.keys())}")
    print()

    mp.set_start_method("spawn", force=True)
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        n_per_worker = n_total // n_gpus
        if i < n_total % n_gpus:
            n_per_worker += 1

        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, config_path, n_per_worker),
        )
        p.start()
        processes.append((gpu_id, p))

    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"[GPU {gpu_id}] Worker exited with code {p.exitcode}")

    print("\n" + "=" * 60)
    print("ALL WORKERS FINISHED")
    print("=" * 60)
    report_best(study_meta, experiments_root)


if __name__ == "__main__":
    main()
