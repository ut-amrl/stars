"""
Stage 2: Frozen linear probe for SWM pedestrian forecasting.

Protocol:
- Load pretrained HGNNCombinedAE checkpoint.
- Freeze the full pretrained model (encoder + decoder).
- Precompute [z_human || z_robot] features and regression targets once per split.
- Train only a linear regression head on those frozen features.

Target:
- Pedestrian position at target_offset_frame (default: frame 105).

Metrics:
- FDE: final displacement error at target frame.
- ADE: average displacement over clip_end+1 .. target_offset.

ADE is computed by linearly interpolating from last observed point to the
predicted endpoint. A linear extrapolation baseline is reported as reference.

Usage:
  python train_swm_forecasting_linear_probe_v1.py
    python train_swm_forecasting_linear_probe_v1.py --h5_path dataloaders/swm_data_dense_bag_200s/swm_segments_forecasting.h5
  python train_swm_forecasting_linear_probe_v1.py --split_mode temporal_gap --temporal_gap_seconds 10.0
  python train_swm_forecasting_linear_probe_v1.py --epochs 120 --batch_size 8
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

try:
    import wandb
except ImportError:
    wandb = None

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from dataloaders.swm_segment_forecasting_dataset import SWMSegmentForecastingDataset
from models.hgnn_combined_v1 import HGNNCombinedAE


def create_run_dir(experiments_root: str, run_name: str) -> str:
    os.makedirs(experiments_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(experiments_root, f"{timestamp}_{run_name}" if run_name else timestamp)
    run_dir = base
    suffix = 1
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = f"{base}_{suffix}"
    os.makedirs(run_dir)
    return run_dir


def resolve_device(device_cfg: str) -> torch.device:
    cfg = str(device_cfg).strip().lower()
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but no CUDA device is available; falling back to CPU.")
        return torch.device("cpu")
    if cfg.startswith("cuda:") and not torch.cuda.is_available():
        print(f"Requested {cfg} but no CUDA device is available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(cfg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_source(args: argparse.Namespace) -> Path:
    if args.h5_path:
        h5_path = Path(args.h5_path).expanduser().resolve()
        if h5_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"--h5_path must point to an .h5/.hdf5 file: {h5_path}")
        if not h5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")
        return h5_path

    segments_root = Path(args.segments_root).expanduser().resolve()
    auto_h5 = segments_root.parent / "swm_segments_forecasting.h5"
    if auto_h5.is_file():
        print(f"Auto-detected HDF5 dataset -> {auto_h5}")
        return auto_h5
    return segments_root


def split_indices_random(
    n_total: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if n_total <= 2:
        raise ValueError("Need at least 3 samples for train/val/test split.")
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.arange(n_total)).tolist()

    n_train = int(round(train_ratio * n_total))
    n_val = int(round(val_ratio * n_total))
    n_test = n_total - n_train - n_val

    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = max(1, n_test)

    while n_train + n_val + n_test > n_total:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def split_indices_episode(
    dataset: SWMSegmentForecastingDataset,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict[str, object]]:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    episode_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(dataset.records):
        episode_key = str(record.get("episode_key", "default_episode"))
        if episode_key not in episode_to_indices:
            episode_to_indices[episode_key] = []
        episode_to_indices[episode_key].append(idx)

    episode_keys = sorted(episode_to_indices.keys())
    n_episodes = len(episode_keys)
    if n_episodes <= 2:
        raise ValueError(
            "Episode split requires at least 3 distinct episodes. "
            "Use --split_mode random or --split_mode temporal_gap for single-episode data."
        )

    rng = np.random.default_rng(seed)
    episode_perm = [episode_keys[i] for i in rng.permutation(np.arange(n_episodes)).tolist()]

    n_train = int(round(train_ratio * n_episodes))
    n_val = int(round(val_ratio * n_episodes))
    n_test = n_episodes - n_train - n_val

    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = max(1, n_test)

    while n_train + n_val + n_test > n_episodes:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    train_eps = episode_perm[:n_train]
    val_eps = episode_perm[n_train : n_train + n_val]
    test_eps = episode_perm[n_train + n_val : n_train + n_val + n_test]

    train_idx = sorted(idx for episode in train_eps for idx in episode_to_indices[episode])
    val_idx = sorted(idx for episode in val_eps for idx in episode_to_indices[episode])
    test_idx = sorted(idx for episode in test_eps for idx in episode_to_indices[episode])

    split_meta: dict[str, object] = {
        "n_episodes": int(n_episodes),
        "train_episode_count": int(len(train_eps)),
        "val_episode_count": int(len(val_eps)),
        "test_episode_count": int(len(test_eps)),
        "train_episode_keys": train_eps,
        "val_episode_keys": val_eps,
        "test_episode_keys": test_eps,
    }
    return train_idx, val_idx, test_idx, split_meta


def split_indices_temporal_gap(
    dataset: SWMSegmentForecastingDataset,
    *,
    train_ratio: float,
    val_ratio: float,
    gap_frames: int,
) -> tuple[list[int], list[int], list[int], dict[str, int | bool | float]]:
    episode_keys = {str(record.get("episode_key", "default_episode")) for record in dataset.records}
    if len(episode_keys) > 1:
        raise ValueError(
            "temporal_gap split is only supported for a single episode timeline. "
            "Use --split_mode episode for multi-episode data."
        )

    if gap_frames < 0:
        raise ValueError("gap_frames must be >= 0")

    n_total = len(dataset)
    if n_total <= 2:
        raise ValueError("Need at least 3 samples for train/val/test split.")
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    ordered = sorted(
        [(int(record["segment_index"]), idx) for idx, record in enumerate(dataset.records)],
        key=lambda item: item[0],
    )
    timeline = [item[0] for item in ordered]
    sorted_idx = [item[1] for item in ordered]

    desired_eval = int(round((1.0 - train_ratio) * n_total))
    desired_eval = min(max(desired_eval, 2), n_total - 1)
    target_train = min(max(n_total - desired_eval, 1), n_total - 2)

    def _find_split(min_eval_count: int) -> tuple[int, int] | None:
        for n_train in range(target_train, 0, -1):
            train_end_t = timeline[n_train - 1]
            eval_start_t = int(train_end_t + gap_frames + 1)
            eval_start_pos = bisect.bisect_left(timeline, eval_start_t)
            n_eval = n_total - eval_start_pos
            if n_eval >= min_eval_count:
                return n_train, eval_start_pos
        return None

    split = _find_split(desired_eval)
    used_fallback = False
    if split is None:
        split = _find_split(2)
        used_fallback = True
    if split is None:
        raise ValueError(
            "Unable to create temporal split with the requested gap. "
            "Try reducing --temporal_gap_seconds or changing train/val ratios."
        )

    n_train, eval_start_pos = split
    train_idx = sorted_idx[:n_train]
    eval_idx = sorted_idx[eval_start_pos:]
    dropped_gap_count = eval_start_pos - n_train

    eval_total_ratio = val_ratio + test_ratio
    val_frac = val_ratio / eval_total_ratio
    n_val = int(round(val_frac * len(eval_idx)))
    n_val = min(max(n_val, 1), len(eval_idx) - 1)
    val_idx = eval_idx[:n_val]
    test_idx = eval_idx[n_val:]

    split_meta: dict[str, int | bool | float] = {
        "gap_frames": int(gap_frames),
        "dropped_gap_count": int(dropped_gap_count),
        "desired_eval_count": int(desired_eval),
        "actual_eval_count": int(len(eval_idx)),
        "used_min_eval_fallback": bool(used_fallback),
        "timeline_min": int(timeline[0]),
        "timeline_max": int(timeline[-1]),
        "train_end_timeline": int(timeline[n_train - 1]),
        "eval_start_timeline": int(timeline[eval_start_pos]) if eval_start_pos < len(timeline) else -1,
    }
    return train_idx, val_idx, test_idx, split_meta


def load_frozen_hgnn(args: argparse.Namespace, device: torch.device) -> tuple[HGNNCombinedAE, dict, int]:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})

    model = HGNNCombinedAE(
        hidden_dim=int(saved_args.get("hidden_dim", args.hidden_dim)),
        latent_dim=int(saved_args.get("latent_dim", args.latent_dim)),
        num_mp_layers=int(saved_args.get("num_mp_layers", args.num_mp_layers)),
        edge_transformer_layers=int(saved_args.get("edge_transformer_layers", args.edge_transformer_layers)),
        num_heads=int(saved_args.get("num_heads", args.num_heads)),
        dropout=float(saved_args.get("dropout", args.dropout)),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"Missing checkpoint keys: {list(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {list(unexpected)}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    latent_dim = int(saved_args.get("latent_dim", args.latent_dim))
    return model, ckpt, latent_dim


def extract_probe_tensors(
    model: HGNNCombinedAE,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x_list: list[torch.Tensor] = []
    target_xy_list: list[torch.Tensor] = []
    baseline_xy_list: list[torch.Tensor] = []
    obs_last_xy_list: list[torch.Tensor] = []
    future_xy_list: list[torch.Tensor] = []
    future_valid_list: list[torch.Tensor] = []

    model.eval()
    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            z_nodes = model.encode_nodes(batch)

        node_batch = getattr(batch, "batch", torch.zeros(z_nodes.shape[0], dtype=torch.long, device=device))
        n_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1

        node_type = batch.node_type.long()
        robot_mask = node_type == 0
        human_mask = (node_type == 1) & batch.target_valid.bool()

        if int(human_mask.sum().item()) == 0:
            continue

        robot_z_by_graph = scatter(
            z_nodes[robot_mask],
            node_batch[robot_mask],
            dim=0,
            reduce="mean",
            dim_size=n_graphs,
        )

        human_z = z_nodes[human_mask]
        human_graph = node_batch[human_mask]
        features = torch.cat([human_z, robot_z_by_graph[human_graph]], dim=-1)

        x_list.append(features.cpu())
        target_xy_list.append(batch.target_xy[human_mask].cpu())
        baseline_xy_list.append(batch.baseline_xy[human_mask].cpu())
        obs_last_xy_list.append(batch.obs_xy[human_mask, -1, :].cpu())
        future_xy_list.append(batch.future_xy[human_mask].cpu())
        future_valid_list.append(batch.future_valid[human_mask].float().cpu())

    if len(x_list) == 0:
        raise RuntimeError("No supervised human nodes were found while extracting probe tensors.")

    return {
        "x": torch.cat(x_list, dim=0),
        "target_xy": torch.cat(target_xy_list, dim=0),
        "baseline_xy": torch.cat(baseline_xy_list, dim=0),
        "obs_last_xy": torch.cat(obs_last_xy_list, dim=0),
        "future_xy": torch.cat(future_xy_list, dim=0),
        "future_valid": torch.cat(future_valid_list, dim=0),
    }


def apply_feature_normalization(
    train_tensors: dict[str, torch.Tensor],
    val_tensors: dict[str, torch.Tensor],
    test_tensors: dict[str, torch.Tensor],
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_mean = train_tensors["x"].mean(dim=0)
    feature_std = train_tensors["x"].std(dim=0, unbiased=False).clamp(min=eps)

    for split in (train_tensors, val_tensors, test_tensors):
        split["x"] = (split["x"] - feature_mean) / feature_std

    return feature_mean.cpu(), feature_std.cpu()


def _compute_displacement_metrics(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    baseline_xy: torch.Tensor,
    obs_last_xy: torch.Tensor,
    future_xy: torch.Tensor,
    future_valid: torch.Tensor,
    horizon_alphas: torch.Tensor,
) -> dict[str, torch.Tensor]:
    fde_pred = torch.linalg.norm(pred_xy - target_xy, dim=1)
    fde_base = torch.linalg.norm(baseline_xy - target_xy, dim=1)

    pred_path = obs_last_xy.unsqueeze(1) + horizon_alphas * (pred_xy.unsqueeze(1) - obs_last_xy.unsqueeze(1))
    base_path = obs_last_xy.unsqueeze(1) + horizon_alphas * (baseline_xy.unsqueeze(1) - obs_last_xy.unsqueeze(1))

    err_pred = torch.linalg.norm(pred_path - future_xy, dim=-1)
    err_base = torch.linalg.norm(base_path - future_xy, dim=-1)

    mask = future_valid
    ade_pred = (err_pred * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    ade_base = (err_base * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return {
        "fde_pred": fde_pred,
        "fde_base": fde_base,
        "ade_pred": ade_pred,
        "ade_base": ade_base,
    }


def run_probe_epoch(
    head: nn.Module,
    split_tensors: dict[str, torch.Tensor],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    horizon_alphas: torch.Tensor,
) -> dict[str, float]:
    training = optimizer is not None
    head.train(training)

    x_all = split_tensors["x"]
    target_all = split_tensors["target_xy"]
    baseline_all = split_tensors["baseline_xy"]
    obs_last_all = split_tensors["obs_last_xy"]
    future_all = split_tensors["future_xy"]
    future_valid_all = split_tensors["future_valid"]

    n_total = int(x_all.shape[0])
    if n_total == 0:
        raise RuntimeError("No probe examples available for this split.")

    order = torch.randperm(n_total) if training else torch.arange(n_total)

    sums = {
        "loss": 0.0,
        "fde": 0.0,
        "ade": 0.0,
        "baseline_fde": 0.0,
        "baseline_ade": 0.0,
        "count": 0,
    }

    for start in range(0, n_total, batch_size):
        idx = order[start : start + batch_size]

        x_b = x_all[idx].to(device)
        target_b = target_all[idx].to(device)
        baseline_b = baseline_all[idx].to(device)
        obs_last_b = obs_last_all[idx].to(device)
        future_b = future_all[idx].to(device)
        future_valid_b = future_valid_all[idx].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred_b = head(x_b)
            loss = F.mse_loss(pred_b, target_b)
            if training:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            metrics = _compute_displacement_metrics(
                pred_xy=pred_b.detach(),
                target_xy=target_b,
                baseline_xy=baseline_b,
                obs_last_xy=obs_last_b,
                future_xy=future_b,
                future_valid=future_valid_b,
                horizon_alphas=horizon_alphas,
            )

        n_batch = int(pred_b.shape[0])
        sums["loss"] += float(loss.detach().cpu()) * n_batch
        sums["fde"] += float(metrics["fde_pred"].mean().detach().cpu()) * n_batch
        sums["ade"] += float(metrics["ade_pred"].mean().detach().cpu()) * n_batch
        sums["baseline_fde"] += float(metrics["fde_base"].mean().detach().cpu()) * n_batch
        sums["baseline_ade"] += float(metrics["ade_base"].mean().detach().cpu()) * n_batch
        sums["count"] += n_batch

    count = float(sums["count"])
    return {
        "loss": sums["loss"] / count,
        "fde": sums["fde"] / count,
        "ade": sums["ade"] / count,
        "baseline_fde": sums["baseline_fde"] / count,
        "baseline_ade": sums["baseline_ade"] / count,
        "count": sums["count"],
    }


def save_history_csv(history_rows: list[dict[str, float]], out_path: str) -> None:
    if len(history_rows) == 0:
        return
    fieldnames = list(history_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def _extract_scalar(value) -> int:
    if torch.is_tensor(value):
        return int(value.view(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _forward_human_predictions_for_visualization(
    model: HGNNCombinedAE,
    head: nn.Module,
    batch,
    *,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    with torch.no_grad():
        z_nodes = model.encode_nodes(batch)

        node_batch = getattr(batch, "batch", torch.zeros(z_nodes.shape[0], dtype=torch.long, device=device))
        n_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 1

        node_type = batch.node_type.long()
        robot_mask = node_type == 0
        human_mask = (node_type == 1) & batch.target_valid.bool()

        if int(human_mask.sum().item()) == 0:
            return None

        robot_z_by_graph = scatter(
            z_nodes[robot_mask],
            node_batch[robot_mask],
            dim=0,
            reduce="mean",
            dim_size=n_graphs,
        )

        human_z = z_nodes[human_mask]
        human_graph = node_batch[human_mask]
        features = torch.cat([human_z, robot_z_by_graph[human_graph]], dim=-1)

        mean = feature_mean.to(device)
        std = feature_std.to(device)
        features = (features - mean) / std

        pred_xy = head(features)

    return {
        "pred_xy": pred_xy,
        "human_node_idx": torch.nonzero(human_mask, as_tuple=False).view(-1),
    }


def visualize_predictions(
    model: HGNNCombinedAE,
    head: nn.Module,
    subset,
    *,
    device: torch.device,
    out_dir: str,
    n_samples: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> int:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    head.eval()

    plotted = 0
    for batch in loader:
        if plotted >= n_samples:
            break

        batch = batch.to(device)
        outputs = _forward_human_predictions_for_visualization(
            model,
            head,
            batch,
            device=device,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        if outputs is None:
            continue

        human_node_idx = outputs["human_node_idx"].detach().cpu().numpy()
        if human_node_idx.size == 0:
            continue

        pred_xy = outputs["pred_xy"].detach().cpu().numpy()
        target_xy = batch.target_xy.detach().cpu().numpy()
        baseline_xy = batch.baseline_xy.detach().cpu().numpy()
        obs_xy = batch.obs_xy.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.get_cmap("tab20")

        for local_i, node_idx in enumerate(human_node_idx.tolist()):
            color = cmap(local_i % 20)
            traj = obs_xy[node_idx]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.6, alpha=0.85)

            gt = target_xy[node_idx]
            pr = pred_xy[local_i]
            bl = baseline_xy[node_idx]

            ax.scatter(gt[0], gt[1], marker="*", s=130, color=color, edgecolors="black")
            ax.scatter(pr[0], pr[1], marker="X", s=65, color=color)
            ax.scatter(bl[0], bl[1], marker="o", s=40, facecolors="none", edgecolors=color)

        ax.scatter([0.0], [0.0], marker="x", s=90, c="black")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", linewidth=0.7)

        segment_index = _extract_scalar(batch.segment_index)
        sample_idx = _extract_scalar(getattr(batch, "sample_idx", segment_index))
        frame_idx = _extract_scalar(batch.frame_idx)
        ax.set_title(f"Segment {segment_index} | frame_idx {frame_idx}")

        from matplotlib.lines import Line2D

        legend_items = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=12, label="GT future (star)"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="gray", markersize=9, label="Predicted (*)"),
            Line2D([0], [0], marker="o", color="gray", markerfacecolor="none", markersize=7, label="Linear baseline"),
            Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=9, label="Robot"),
        ]
        ax.legend(handles=legend_items, loc="best")

        out_path = os.path.join(
            out_dir,
            f"sample_{sample_idx:06d}_segment_{segment_index:04d}_frame_{frame_idx}.png",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        plotted += 1

    return plotted


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    dataset_source = resolve_dataset_source(args)
    print(f"Dataset source: {dataset_source}")

    run_dir = create_run_dir(args.experiments_root, args.run_name)
    print(f"Run directory: {run_dir}")

    if args.wandb and wandb is None:
        print("wandb requested but not installed; continuing without wandb.")
    use_wandb = bool(args.wandb and wandb is not None)
    if use_wandb:
        wandb.init(project=args.wandb_project, name=os.path.basename(run_dir), config=vars(args), dir=run_dir)

    dataset = SWMSegmentForecastingDataset(
        segments_root=dataset_source,
        use_smoothed_episode=not args.use_raw_episode,
        clip_end_frame=args.clip_end_frame,
        target_offset_frame=args.target_offset_frame,
        history_steps=args.history_steps,
        model_timesteps=args.model_timesteps,
        min_history_points=args.min_history_points,
        require_future_labels=args.require_future_labels,
    )

    if not args.require_future_labels:
        print(
            "Dataset mode: future labels optional. "
            "Linear probe uses only nodes where target_valid=True."
        )

    resolved_split_mode = str(args.split_mode)
    if resolved_split_mode == "auto":
        episode_count = len(getattr(dataset, "episode_keys", []))
        if episode_count >= 3:
            resolved_split_mode = "episode"
        else:
            resolved_split_mode = "temporal_gap"
        print(
            f"Auto split mode selected '{resolved_split_mode}' "
            f"(detected episodes={episode_count})."
        )

    split_meta: dict[str, object] = {
        "mode": resolved_split_mode,
        "requested_mode": str(args.split_mode),
    }
    if resolved_split_mode == "random":
        train_idx, val_idx, test_idx = split_indices_random(len(dataset), args.train_ratio, args.val_ratio, args.seed)
    elif resolved_split_mode == "temporal_gap":
        gap_frames = int(round(float(args.temporal_gap_seconds) * float(args.source_fps)))
        train_idx, val_idx, test_idx, temporal_meta = split_indices_temporal_gap(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            gap_frames=gap_frames,
        )
        split_meta.update(temporal_meta)
        split_meta["gap_seconds"] = float(args.temporal_gap_seconds)
        split_meta["source_fps"] = float(args.source_fps)

        def _timeline_range(indices: list[int]) -> tuple[int, int]:
            vals = sorted(int(dataset.records[i]["segment_index"]) for i in indices)
            return vals[0], vals[-1]

        tr_lo, tr_hi = _timeline_range(train_idx)
        va_lo, va_hi = _timeline_range(val_idx)
        te_lo, te_hi = _timeline_range(test_idx)
        print(
            f"Temporal split: gap={args.temporal_gap_seconds:.2f}s "
            f"({gap_frames} frames @ {args.source_fps:.2f} fps), "
            f"unused_gap_graphs={split_meta.get('dropped_gap_count', 0)}"
        )
        print(
            f"Timeline ranges (segment_index) | "
            f"train=[{tr_lo}, {tr_hi}] val=[{va_lo}, {va_hi}] test=[{te_lo}, {te_hi}]"
        )
        if bool(split_meta.get("used_min_eval_fallback", False)):
            print(
                "Warning: requested train/val proportions cannot be fully preserved "
                "with this temporal gap; using maximum feasible train size."
            )
    else:
        train_idx, val_idx, test_idx, episode_meta = split_indices_episode(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        split_meta.update(episode_meta)
        print(
            "Episode split: "
            f"episodes(train/val/test)={split_meta['train_episode_count']}/{split_meta['val_episode_count']}/{split_meta['test_episode_count']}"
        )

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    print(
        f"Split sizes | train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"(graphs)"
    )

    split_path = os.path.join(run_dir, "split_indices_swm_forecasting_linear_probe_v1.pt")
    torch.save(
        {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
            "split_mode": resolved_split_mode,
            "requested_split_mode": str(args.split_mode),
            "dataset_source": str(dataset_source),
            "split_meta": split_meta,
        },
        split_path,
    )
    print(f"Split indices saved -> {split_path}")

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=persistent,
    )

    model, ckpt, latent_dim = load_frozen_hgnn(args, device)
    print("Encoder mode: frozen linear probe (full pretrained model is frozen)")

    print("Precomputing frozen probe tensors...")
    train_tensors = extract_probe_tensors(model, train_loader, device=device)
    val_tensors = extract_probe_tensors(model, val_loader, device=device)
    test_tensors = extract_probe_tensors(model, test_loader, device=device)
    print(
        "Probe tensor sizes | "
        f"train={train_tensors['x'].shape[0]} val={val_tensors['x'].shape[0]} test={test_tensors['x'].shape[0]}"
    )

    if args.normalize_features:
        feature_mean, feature_std = apply_feature_normalization(train_tensors, val_tensors, test_tensors)
        print("Feature normalization: enabled (train-set mean/std)")
    else:
        feature_mean = torch.zeros((2 * latent_dim,), dtype=torch.float32)
        feature_std = torch.ones((2 * latent_dim,), dtype=torch.float32)
        print("Feature normalization: disabled")

    input_dim = int(train_tensors["x"].shape[1])
    head = nn.Linear(input_dim, 2).to(device)

    optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)

    horizon = args.target_offset_frame - args.clip_end_frame
    horizon_alphas = torch.linspace(
        1.0 / float(horizon),
        1.0,
        steps=horizon,
        device=device,
        dtype=torch.float32,
    ).view(1, horizon, 1)

    best_val_fde = float("inf")
    best_epoch = 0
    no_improve = 0

    best_path = os.path.join(run_dir, "best_swm_forecasting_linear_probe_v1.pt")
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_stats = run_probe_epoch(
            head,
            train_tensors,
            device=device,
            optimizer=optimizer,
            batch_size=args.batch_size,
            horizon_alphas=horizon_alphas,
        )
        val_stats = run_probe_epoch(
            head,
            val_tensors,
            device=device,
            optimizer=None,
            batch_size=args.batch_size,
            horizon_alphas=horizon_alphas,
        )

        scheduler.step(val_stats["fde"])

        lr_value = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_stats['loss']:.4f} ade={train_stats['ade']:.4f} fde={train_stats['fde']:.4f} | "
            f"val loss={val_stats['loss']:.4f} ade={val_stats['ade']:.4f} fde={val_stats['fde']:.4f} | "
            f"val baseline ade={val_stats['baseline_ade']:.4f} fde={val_stats['baseline_fde']:.4f} | "
            f"lr={lr_value:.2e}"
        )

        row = {
            "epoch": epoch,
            "lr": lr_value,
            "train_loss": train_stats["loss"],
            "train_ade": train_stats["ade"],
            "train_fde": train_stats["fde"],
            "val_loss": val_stats["loss"],
            "val_ade": val_stats["ade"],
            "val_fde": val_stats["fde"],
            "val_baseline_ade": val_stats["baseline_ade"],
            "val_baseline_fde": val_stats["baseline_fde"],
        }
        history_rows.append(row)

        if use_wandb:
            wandb.log(row, step=epoch)

        if val_stats["fde"] < best_val_fde - args.early_stopping_min_delta:
            best_val_fde = val_stats["fde"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "linear_head_state_dict": head.state_dict(),
                    "best_val_fde": best_val_fde,
                    "args": vars(args),
                    "source_checkpoint": str(args.checkpoint),
                    "split_indices_path": split_path,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "input_dim": input_dim,
                    "probe_type": "frozen_linear_regression",
                },
                best_path,
            )
            print(f"  Saved best checkpoint -> {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val FDE improvement for {args.early_stopping_patience} epochs)."
                )
                break

    history_csv_path = os.path.join(run_dir, "training_history_swm_forecasting_linear_probe_v1.csv")
    save_history_csv(history_rows, history_csv_path)
    print(f"Training history saved -> {history_csv_path}")

    print("\n=== TEST EVALUATION (best checkpoint) ===")
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    head.load_state_dict(best_ckpt["linear_head_state_dict"], strict=True)
    feature_mean = best_ckpt.get("feature_mean", feature_mean).float().cpu()
    feature_std = best_ckpt.get("feature_std", feature_std).float().cpu()

    test_stats = run_probe_epoch(
        head,
        test_tensors,
        device=device,
        optimizer=None,
        batch_size=args.batch_size,
        horizon_alphas=horizon_alphas,
    )

    print(
        f"Best epoch: {best_epoch} | "
        f"Test ADE={test_stats['ade']:.4f} FDE={test_stats['fde']:.4f} | "
        f"Baseline ADE={test_stats['baseline_ade']:.4f} FDE={test_stats['baseline_fde']:.4f}"
    )

    summary_path = os.path.join(run_dir, "test_metrics_swm_forecasting_linear_probe_v1.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "best_epoch",
                "test_loss",
                "test_ade",
                "test_fde",
                "baseline_ade",
                "baseline_fde",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "best_epoch": best_epoch,
                "test_loss": test_stats["loss"],
                "test_ade": test_stats["ade"],
                "test_fde": test_stats["fde"],
                "baseline_ade": test_stats["baseline_ade"],
                "baseline_fde": test_stats["baseline_fde"],
            }
        )
    print(f"Test metrics saved -> {summary_path}")

    vis_dir = os.path.join(run_dir, "visualizations")
    n_plotted = visualize_predictions(
        model,
        head,
        test_set,
        device=device,
        out_dir=vis_dir,
        n_samples=args.plot_samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    print(f"Saved {n_plotted} visualization samples -> {vis_dir}")

    if use_wandb:
        wandb.log(
            {
                "test/loss": test_stats["loss"],
                "test/ade": test_stats["ade"],
                "test/fde": test_stats["fde"],
                "test/baseline_ade": test_stats["baseline_ade"],
                "test/baseline_fde": test_stats["baseline_fde"],
            }
        )
        wandb.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen linear probe for SWM pedestrian forecasting")

    parser.add_argument(
        "--h5_path",
        default=None,
        help=(
            "Optional SWM forecasting HDF5 dataset path. If omitted, the trainer uses "
            "--segments_root (and auto-detects swm_segments_forecasting.h5 beside it when present)."
        ),
    )
    parser.add_argument(
        "--segments_root",
        default="dataloaders/swm_data_dense_bag_200s/segments",
        help="Fallback JSON root containing segment_*/segment_*.json files (supports nested episode folders)",
    )
    parser.add_argument(
        "--checkpoint",
        default="src/experiments/hgnn_combined_v2/20260514_170835_hgnn_combined_v2/best_model.pt",
        help="Pretrained HGNN checkpoint path",
    )
    parser.add_argument(
        "--experiments_root",
        default="src/experiments/swm_forecasting_linear_probe_v1",
        help="Directory where run artifacts are saved",
    )
    parser.add_argument("--run_name", default="swm_forecasting_linear_probe_v1")

    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lr_patience", type=int, default=8)
    parser.add_argument("--early_stopping_patience", type=int, default=14)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--split_mode",
        choices=("auto", "episode", "temporal_gap", "random"),
        default="auto",
        help="Data split strategy. auto picks episode split when multiple episodes exist, otherwise temporal_gap.",
    )
    parser.add_argument(
        "--temporal_gap_seconds",
        type=float,
        default=10.0,
        help="Unused dead-zone duration after train split boundary (seconds) when split_mode=temporal_gap on single-episode data.",
    )
    parser.add_argument(
        "--source_fps",
        type=float,
        default=30.0,
        help="FPS used to convert temporal_gap_seconds into frame units.",
    )

    parser.add_argument("--clip_end_frame", type=int, default=75)
    parser.add_argument("--target_offset_frame", type=int, default=105)
    parser.add_argument("--history_steps", type=int, default=21)
    parser.add_argument("--model_timesteps", type=int, default=40)
    parser.add_argument("--min_history_points", type=int, default=2)
    parser.add_argument("--use_raw_episode", action="store_true", help="Use raw episodes instead of interpolated+kalman episodes")
    parser.add_argument(
        "--require_future_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require full future supervision when building dataset samples. "
            "Use --no-require_future_labels to keep tracks without future labels "
            "(they are ignored by linear-probe supervision via target_valid)."
        ),
    )

    parser.add_argument(
        "--normalize_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize frozen probe features using train-set mean/std before head training.",
    )
    parser.add_argument("--plot_samples", type=int, default=6)

    # Fallback architecture args if checkpoint args are missing.
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_mp_layers", type=int, default=3)
    parser.add_argument("--edge_transformer_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="social_world_model")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
