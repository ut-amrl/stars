import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from dataloaders.unified_graph_dataset_v1 import UnifiedGraphDataset
from models.hgnn_combined_v2 import HGNNCombinedVAE


ACTION_CLASSES = ["avoid", "follow", "overtake", "yield", "not_consider"]
NUM_CLASSES    = len(ACTION_CLASSES)
IGNORE_LABEL   = -100


def resolve_path(path, base_dir=SRC_DIR):
    path = str(path)
    if os.path.isabs(path):
        return path
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(os.path.join(base_dir, path))


def resolve_device(device_cfg):
    device_cfg = str(device_cfg).lower()
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_config(config_path):
    config_path = resolve_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    section = "eval_ssub"
    if section not in cfg:
        raise ValueError(f"Config missing section '{section}'")
    cfg  = cfg[section]
    args = SimpleNamespace(**cfg)
    args.ae_checkpoint    = resolve_path(args.ae_checkpoint)
    args.probe_checkpoint = resolve_path(args.probe_checkpoint)
    args.eval_h5_path     = resolve_path(args.eval_h5_path)
    return args


def load_encoder(args, device):
    ckpt       = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    robot_mean = ckpt.get("robot_mean", None)
    robot_std  = ckpt.get("robot_std",  None)

    model = HGNNCombinedVAE(
        hidden_dim              = int(args.hidden_dim),
        latent_dim              = int(args.latent_dim),
        num_mp_layers           = int(args.num_mp_layers),
        edge_transformer_layers = int(args.edge_transformer_layers),
        num_heads               = int(args.num_heads),
        dropout                 = float(args.dropout),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {list(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {list(unexpected)}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, robot_mean, robot_std


def load_probe(args, device):
    ckpt  = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
    probe = nn.Linear(int(args.latent_dim), NUM_CLASSES).to(device)
    probe.load_state_dict(ckpt["probe_state_dict"])
    for p in probe.parameters():
        p.requires_grad = False
    probe.eval()
    return probe


def extract_human_embeddings(encoder, loader, device):
    all_z, all_y, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z    = encoder.encode_nodes(batch)
            keep = batch.action_label != IGNORE_LABEL
            if keep.any():
                all_z.append(z[keep].cpu())
                all_y.append(batch.action_label[keep].cpu())
                all_probs.append(batch.action_probs[keep].cpu())
    if not all_z:
        return torch.zeros(0, 0), torch.zeros(0, dtype=torch.long), torch.zeros(0, NUM_CLASSES)
    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0).long(), torch.cat(all_probs, dim=0)


def compute_metrics(probe, z, y, gt_probs, device, batch_size):
    logits_list = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            z_b = z[start:start + batch_size].to(device)
            logits_list.append(probe(z_b).cpu())

    logits  = torch.cat(logits_list, dim=0)
    probs   = F.softmax(logits, dim=-1).numpy()
    y_true  = y.numpy()
    y_pred  = probs.argmax(axis=1)
    support = np.bincount(y_true, minlength=NUM_CLASSES)

    per_class_f1 = f1_score(
        y_true, y_pred,
        labels=np.arange(NUM_CLASSES),
        average=None,
        zero_division=0,
    )

    gt_probs_np = gt_probs.numpy()                          # [M, 5]
    pa = float(gt_probs_np[np.arange(len(y_pred)), y_pred].mean())

    return {
        "per_class_f1": {ACTION_CLASSES[i]: float(per_class_f1[i]) for i in range(NUM_CLASSES)},
        "weighted_f1":  float(f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES), average="weighted", zero_division=0)),
        "pa":           pa,
        "support":      {ACTION_CLASSES[i]: int(support[i]) for i in range(NUM_CLASSES)},
        "total":        int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params/config.yaml")
    cli_args = parser.parse_args()

    args   = load_config(cli_args.config)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    encoder, robot_mean, robot_std = load_encoder(args, device)
    print("Encoder loaded.")

    probe = load_probe(args, device)
    print("Probe loaded.\n")

    dataset = UnifiedGraphDataset(
        args.eval_h5_path,
        ssub=True,
        robot_mean=robot_mean,
        robot_std=robot_std,
    )
    print(f"Eval dataset: {len(dataset)} episodes from {args.eval_h5_path}")

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        persistent_workers=(int(args.num_workers) > 0),
    )

    z, y, gt_probs = extract_human_embeddings(encoder, loader, device)
    if z.shape[0] == 0:
        print("No human nodes found in dataset.")
        return

    print(f"Human nodes: {z.shape[0]}  latent dim: {z.shape[1]}")
    results = compute_metrics(probe, z, y, gt_probs, device, int(args.batch_size))

    col_w = 14
    header = f"{'class':<{col_w}}{'F1':>{col_w}}{'support':>{col_w}}"
    print(f"\n{header}")
    print("-" * len(header))
    for c in ACTION_CLASSES:
        print(f"{c:<{col_w}}{results['per_class_f1'][c]:>{col_w}.4f}{results['support'][c]:>{col_w}}")

    print("-" * len(header))
    print(f"{'weighted F1':<{col_w}}{results['weighted_f1']:>{col_w}.4f}")
    print(f"{'PA':<{col_w}}{results['pa']:>{col_w}.4f}  (N={results['total']})")


if __name__ == "__main__":
    main()
