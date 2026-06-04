"""
Shared utilities for linear probe training.
"""

import csv
import os
import random
from datetime import datetime

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SRC_DIR)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg):
    s = str(device_cfg).lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def create_run_dir(experiments_root, run_name):
    os.makedirs(experiments_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = str(run_name).strip()
    run_dir = os.path.join(experiments_root, f"{timestamp}_{name}" if name else timestamp)
    suffix = 1
    base = run_dir
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = f"{base}_{suffix}"
    os.makedirs(run_dir)
    return run_dir


def load_frozen_encoder(checkpoint_path, hidden_dim, latent_dim, num_mp_layers,
                        edge_transformer_layers, num_heads, dropout, device):
    """Load HGNNCombinedVAE from checkpoint with all params frozen."""
    from models.hgnn_combined_v2 import HGNNCombinedVAE

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    robot_mean = ckpt.get("robot_mean", None)
    robot_std = ckpt.get("robot_std", None)

    model = HGNNCombinedVAE(
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        num_mp_layers=int(num_mp_layers),
        edge_transformer_layers=int(edge_transformer_layers),
        num_heads=int(num_heads),
        dropout=float(dropout),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"Missing checkpoint keys: {list(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {list(unexpected)}")

    # Validate against saved args
    saved_args = ckpt.get("args", {})
    for key in ["hidden_dim", "latent_dim", "num_mp_layers", "edge_transformer_layers", "num_heads"]:
        if isinstance(saved_args, dict) and key in saved_args:
            expected = int(saved_args[key])
            got = int(locals()[key])
            if expected != got:
                print(f"WARNING: config {key}={got} differs from checkpoint {key}={expected}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    print(f"Loaded frozen HGNNCombinedVAE from {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', 'N/A')}")
    return model, robot_mean, robot_std, ckpt


def flatten_wandb_metrics(metrics, prefix):
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def save_metrics_csv(metrics_dict, path):
    """Save {split: {metric: value}} to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        for split, m in metrics_dict.items():
            for k, v in m.items():
                writer.writerow({"split": split, "metric": k, "value": v})


def init_wandb(project, name, config, run_dir, enabled=True):
    """Initialize wandb if available and enabled."""
    if not enabled:
        print("wandb logging disabled.")
        return None
    if wandb is None:
        print("wandb not installed — continuing without W&B.")
        return None
    return wandb.init(
        project=project,
        name=name,
        config=config,
        dir=run_dir,
    )
