"""
Unified Linear Probe Training Pipeline.

Evaluates a single frozen HGNNCombinedVAE encoder across three downstream tasks:
1. SEAN-T (emotion classification)
2. SNS (action classification)

Each probe is trained num_probe_runs times with different random seeds (probe
weight initialization and training shuffles). Results are reported as mean ± std.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset
from probes.common import (
    create_run_dir,
    init_wandb,
    load_frozen_encoder,
    resolve_device,
    seed_everything,
)
from probes.sean_probe import train_sean_probe
from probes.sns_probe import train_sns_probe


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "linear_probe_all_params" not in cfg:
        raise ValueError("Missing 'linear_probe_all_params' in config.")

    args_dict = cfg["linear_probe_all_params"]
    args = SimpleNamespace(**args_dict)

    args.checkpoint = os.path.abspath(args.checkpoint)
    args.h5_path = os.path.abspath(args.h5_path)
    args.split_indices_path = os.path.abspath(args.split_indices_path)
    args.experiments_root = os.path.abspath(args.experiments_root)

    return args


def _is_numeric(v):
    if not isinstance(v, (int, float)):
        return False
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return False
    return True


def _leaf_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
        "values": [float(x) for x in arr.tolist()],
    }


def aggregate_metric_dicts(metric_dicts):
    """Recursively aggregate identically-structured metric dicts → mean/std at leaves."""
    metric_dicts = [d for d in metric_dicts if d is not None]
    if not metric_dicts:
        return {}

    keys = set()
    for d in metric_dicts:
        keys.update(d.keys())

    out = {}
    for k in sorted(keys):
        vals = [d[k] for d in metric_dicts if k in d and d[k] is not None]
        if not vals:
            continue
        if all(_is_numeric(v) for v in vals):
            out[k] = _leaf_stats([float(v) for v in vals])
        elif all(isinstance(v, dict) for v in vals):
            out[k] = aggregate_metric_dicts(vals)
        else:
            # e.g. mixed types — keep numeric-only subset if possible
            numeric = [float(v) for v in vals if _is_numeric(v)]
            if numeric:
                out[k] = _leaf_stats(numeric)
    return out


def fmt_mean_std(stats):
    return f"{stats['mean']:.4f} ± {stats['std']:.4f}"


def get_leaf(stats_dict, *keys, default=None):
    d = stats_dict
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    if isinstance(d, dict) and "mean" in d:
        return d
    return default


def run_probe_repeats(train_fn, num_runs, base_seed):
    """Run train_fn num_runs times with seeds base_seed, base_seed+1, ..."""
    per_run = []
    for i in range(num_runs):
        run_seed = int(base_seed) + i
        seed_everything(run_seed)
        print(f"\n{'=' * 60}")
        print(f"Probe run {i + 1}/{num_runs}  (seed={run_seed})")
        print("=" * 60)
        per_run.append(train_fn())
    return per_run, aggregate_metric_dicts(per_run)


def save_results(run_dir, per_run_results, aggregated, task_name):
    path = os.path.join(run_dir, f"{task_name}_probe_results.json")
    payload = {"per_run": per_run_results, "aggregate": aggregated}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved {task_name} results → {path}")


def print_sean_summary(agg):
    print("SEAN-T (binary emotion classification - [Competence, Surprise, Intention]):")
    for split in ("val", "test"):
        block = agg.get(split)
        if not block:
            continue
        loss = get_leaf(block, "loss")
        if loss:
            print(f"  {split.capitalize()} Loss: {fmt_mean_std(loss)}")
        f1_keys = ["f1_competence", "f1_surprise", "f1_intention"]
        f1_stats = [get_leaf(block, k) for k in f1_keys]
        if all(s for s in f1_stats):
            f1_str = ", ".join(fmt_mean_std(s) for s in f1_stats)
            print(f"  {split.capitalize()} F1:  [{f1_str}]")
        auc_keys = ["auc_competence", "auc_surprise", "auc_intention"]
        auc_stats = [get_leaf(block, k) for k in auc_keys]
        if all(s for s in auc_stats):
            auc_str = ", ".join(fmt_mean_std(s) for s in auc_stats)
            print(f"  {split.capitalize()} AUC: [{auc_str}]")


def print_sns_summary(agg):
    print("SNS (SSUB human action classification):")
    for split in ("val", "test"):
        block = agg.get(split)
        if not block:
            if split == "test":
                print("  Test Split:          skipped (no valid SSUB labels)")
            continue
        loss = get_leaf(block, "loss")
        if loss:
            print(f"  {split.capitalize()} Loss:            {fmt_mean_std(loss)}")
        wf1 = get_leaf(block, "weighted_f1")
        if wf1:
            print(f"  {split.capitalize()} Weighted F1:     {fmt_mean_std(wf1)}")
        wauc = get_leaf(block, "weighted_auc")
        if wauc:
            print(f"  {split.capitalize()} Weighted AUC:    {fmt_mean_std(wauc)}")



def main():
    parser = argparse.ArgumentParser(description="Unified Linear Probe Training")
    parser.add_argument("--config", default="params/config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (overrides config)")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Number of probe runs per task (different seeds); default from config or 10",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Path to an existing run directory to resume from (skips already completed stages/runs)",
    )
    parser.add_argument(
        "--use-first-mp-layer",
        action="store_true",
        help="Use representations from after the first message passing layer instead of the compressed latent"
    )
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Unfreeze encoder parameters to allow end-to-end weight updates during downstream training"
    )
    cmd_args = parser.parse_args()

    args = load_config(cmd_args.config)
    if cmd_args.device:
        args.device = cmd_args.device
    if cmd_args.use_first_mp_layer:
        args.use_first_mp_layer = True
    if not hasattr(args, "use_first_mp_layer"):
        args.use_first_mp_layer = False
    if cmd_args.unfreeze_encoder:
        args.unfreeze_encoder = True
    if not hasattr(args, "unfreeze_encoder"):
        args.unfreeze_encoder = False

    num_runs = (
        cmd_args.num_runs
        if cmd_args.num_runs is not None
        else int(getattr(args, "num_probe_runs", 10))
    )
    base_seed = int(args.seed)

    seed_everything(base_seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Probe runs per task: {num_runs}  (seeds {base_seed}..{base_seed + num_runs - 1})")

    if cmd_args.resume_dir:
        run_dir = os.path.abspath(cmd_args.resume_dir)
        print(f"Resuming with existing run directory: {run_dir}")
    else:
        run_name = getattr(args, "run_name", "all_probes")
        if getattr(args, "use_first_mp_layer", False):
            run_name += "_first_mp_layer"
        if getattr(args, "unfreeze_encoder", False):
            run_name += "_unfrozen_encoder"
        run_dir = create_run_dir(args.experiments_root, run_name)
        print(f"Created new run directory: {run_dir}")

    wb_run = init_wandb(
        project=getattr(args, "wandb_project", "social_world_model"),
        name=getattr(args, "run_name", os.path.basename(run_dir)),
        config={**vars(args), "num_probe_runs": num_runs},
        run_dir=run_dir,
        enabled=bool(getattr(args, "wandb", False)) and num_runs == 1,
    )

    print("\nLoading frozen encoder...")
    encoder, robot_mean, robot_std, _ = load_frozen_encoder(
        checkpoint_path=args.checkpoint,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_mp_layers=args.num_mp_layers,
        edge_transformer_layers=args.edge_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=device,
    )

    if getattr(args, "use_first_mp_layer", False):
        print("\n>>> CONFIGURING ENCODER TO USE FIRST MESSAGE PASSING LAYER REPRESENTATION <<<")
        encoder.use_first_mp_layer = True
        encoder.encoder.use_first_mp_layer = True

    if getattr(args, "unfreeze_encoder", False):
        print("\n>>> CONFIGURING ENCODER PARAMETERS FOR FINE-TUNING (UNFROZEN) <<<")
        for p in encoder.parameters():
            p.requires_grad = True
        encoder.train()

    dataset = None
    splits = None
    if getattr(args, "run_sean", True) or getattr(args, "run_sns", True):
        # Optimize loading of massive unified dataset: only load if SEAN or SNS actually needs training.
        sean_results_path = os.path.join(run_dir, "sean_probe_results.json")
        sns_results_path = os.path.join(run_dir, "sns_probe_results.json")
        need_sean = getattr(args, "run_sean", True) and not os.path.exists(sean_results_path)
        need_sns = getattr(args, "run_sns", True) and not os.path.exists(sns_results_path)

        if need_sean or need_sns:
            print("\nLoading unified dataset...")
            dataset = UnifiedGraphDataset(
                h5_path=args.h5_path,
                ssub=False,
                robot_mean=robot_mean,
                robot_std=robot_std,
            )
            print(f"Loaded {len(dataset)} unified graphs.")

            splits = torch.load(args.split_indices_path, map_location="cpu", weights_only=False)
            print(
                f"Loaded splits: train={len(splits['train'])} "
                f"val={len(splits['val'])} test={len(splits['test'])}"
            )
            dataset.set_robot_stats(splits["train"])
        else:
            print("\nSEAN and SNS results already exist on disk. Skipping loading of unified dataset.")

    aggregated_results = {}

    if getattr(args, "run_sean", True):
        sean_results_path = os.path.join(run_dir, "sean_probe_results.json")
        if os.path.exists(sean_results_path):
            print(f"\nFound existing SEAN results at {sean_results_path}. Loading and skipping SEAN training.")
            with open(sean_results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                sean_runs = payload["per_run"]
                sean_agg = payload["aggregate"]
        else:
            sean_runs, sean_agg = run_probe_repeats(
                lambda: train_sean_probe(
                    encoder=encoder,
                    dataset=dataset,
                    splits=splits,
                    args=args,
                    device=device,
                    run_dir=run_dir,
                    wb_run=None,
                ),
                num_runs,
                base_seed,
            )
            save_results(run_dir, sean_runs, sean_agg, "sean")
        aggregated_results["sean"] = sean_agg

    if getattr(args, "run_sns", True):
        sns_results_path = os.path.join(run_dir, "sns_probe_results.json")
        if os.path.exists(sns_results_path):
            print(f"\nFound existing SNS results at {sns_results_path}. Loading and skipping SNS training.")
            with open(sns_results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                sns_runs = payload["per_run"]
                sns_agg = payload["aggregate"]
        else:
            sns_runs, sns_agg = run_probe_repeats(
                lambda: train_sns_probe(
                    encoder=encoder,
                    dataset=dataset,
                    splits=splits,
                    args=args,
                    device=device,
                    run_dir=run_dir,
                    wb_run=None,
                ),
                num_runs,
                base_seed,
            )
            save_results(run_dir, sns_runs, sns_agg, "sns")
        aggregated_results["sns"] = sns_agg

    summary_path = os.path.join(run_dir, "all_probes_aggregate.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2)

    if wb_run:
        wb_run.finish()

    print("\n" + "=" * 60)
    print("ALL PROBES COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nSUMMARY OF RESULTS (mean ± std over {num_runs} runs):")
    print("-" * 60)

    if "sean" in aggregated_results and aggregated_results["sean"]:
        print_sean_summary(aggregated_results["sean"])
        print("-" * 60)

    if "sns" in aggregated_results and aggregated_results["sns"]:
        print_sns_summary(aggregated_results["sns"])
        print("-" * 60)


    print(f"Results saved to: {run_dir}")
    print(f"Aggregate summary: {summary_path}\n")


if __name__ == "__main__":
    main()
