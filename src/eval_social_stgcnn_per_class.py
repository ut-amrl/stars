#!/usr/bin/env python
"""
Eval saved Social-STGCNN SNSB checkpoints on the human oracle set.

Prints per-class F1, macro3, macro, accuracy, and PA for each run plus aggregate
mean/std. The training script only prints aggregate SNSB metrics, so this script
recomputes predictions from best_snsb.pt checkpoints.
"""

import argparse
import os
import statistics as st

import numpy as np
import torch
from sklearn.metrics import f1_score

from dataloaders.socialnav_hetero_graph_dataset_v1 import SocialNavGraphDatasetV1
from dataloaders.unified_gnn_vae_datasets_v1 import UnifiedSocialNavDataset
from models.social_stgcnn_classifier import SocialSTGCNNClassifier, build_stgcnn_input
from train_snsb_supervised_v1 import InterpSocialNavDataset
from train_social_stgcnn_baseline import SNSB_TIMESTEPS, _get_snsb_split_indices


CLASSES = ["avoid", "follow", "overtake", "yield", "not_consider"]
MACRO3_LABELS = [0, 1, 4]  # avoid, follow, not_consider
IGNORE_INDEX = -1


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def build_model(checkpoint, device):
    args = checkpoint.get("args", {})
    model = SocialSTGCNNClassifier(
        n_stgcnn=int(args.get("n_stgcnn", 2)),
        input_feat=8,
        hidden_feat=int(args.get("hidden_feat", 128)),
        seq_len=SNSB_TIMESTEPS,
        kernel_size=int(args.get("kernel_size", 3)),
        dropout=float(args.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_oracle_dataset(args):
    snsb_h5 = args.get("snsb_h5", "/data/social_nav_v3.h5")
    combined_h5 = args.get("combined_h5", "/data/combined_v1.h5")
    split_file = args.get("split_file", "checkpoints/split_indices_final.pt")
    oracle_h5 = args.get("oracle_h5", "/data/socialnav_human_oracle_v3.h5")

    train_ds = UnifiedSocialNavDataset(h5_path=snsb_h5)
    train_idx, _, _ = _get_snsb_split_indices(combined_h5, split_file, train_ds)
    train_ds.set_feature_stats(train_idx)

    oracle_base = SocialNavGraphDatasetV1(oracle_h5)
    oracle_interp = InterpSocialNavDataset(oracle_base, target_t=SNSB_TIMESTEPS)
    oracle_ds = UnifiedSocialNavDataset(base_dataset=oracle_interp)

    oracle_ds.bystander_mean = train_ds.bystander_mean
    oracle_ds.bystander_std = train_ds.bystander_std
    oracle_ds.robot_mean = train_ds.robot_mean
    oracle_ds.robot_std = train_ds.robot_std
    return oracle_ds


def eval_one(model, dataset, device):
    preds, labels = [], []
    pa_num, pa_den = 0.0, 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            n_bystanders = int(data["bystander"].num_nodes)
            if n_bystanders == 0:
                continue

            y = data["bystander"].y.view(-1).long()
            valid = y != IGNORE_INDEX
            if valid.sum() == 0:
                continue

            try:
                v, adj, _, _, _ = build_stgcnn_input(data, device)
            except ValueError:
                continue

            logits = model.forward_snsb(v, adj, n_bystanders)
            pred = logits.argmax(dim=-1).cpu()

            preds.extend(pred[valid].tolist())
            labels.extend(y[valid].tolist())

            if hasattr(data["bystander"], "action_probs"):
                action_probs = data["bystander"].action_probs[valid]
                valid_pred = pred[valid]
                pa_num += float(action_probs[torch.arange(len(valid_pred)), valid_pred].sum())
                pa_den += len(valid_pred)

    if not preds:
        return {}

    preds = np.asarray(preds)
    labels = np.asarray(labels)
    per_class = f1_score(
        labels,
        preds,
        labels=list(range(len(CLASSES))),
        average=None,
        zero_division=0,
    )

    metrics = {CLASSES[i]: float(per_class[i]) for i in range(len(CLASSES))}
    metrics["macro3"] = float(
        f1_score(labels, preds, labels=MACRO3_LABELS, average="macro", zero_division=0)
    )
    metrics["macro"] = float(
        f1_score(
            labels,
            preds,
            labels=list(range(len(CLASSES))),
            average="macro",
            zero_division=0,
        )
    )
    metrics["acc"] = float((preds == labels).mean())
    if pa_den:
        metrics["pa"] = pa_num / pa_den
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Social-STGCNN SNSB checkpoints with per-class F1."
    )
    parser.add_argument(
        "--root",
        default="experiments/social_stgcnn_v1_optuned",
        help="Directory containing run*/best_snsb.pt checkpoints.",
    )
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    first_ckpt_path = os.path.join(args.root, "run1", "best_snsb.pt")
    first_ckpt = load_checkpoint(first_ckpt_path, device)
    oracle_ds = build_oracle_dataset(first_ckpt.get("args", {}))

    rows = []
    for run in range(1, args.n_runs + 1):
        ckpt_path = os.path.join(args.root, f"run{run}", "best_snsb.pt")
        if not os.path.isfile(ckpt_path):
            print(f"run {run}: missing {ckpt_path}")
            continue

        checkpoint = load_checkpoint(ckpt_path, device)
        model = build_model(checkpoint, device)
        metrics = eval_one(model, oracle_ds, device)
        if not metrics:
            print(f"run {run}: no valid predictions")
            continue

        rows.append(metrics)
        pretty = {k: round(v, 4) for k, v in metrics.items()}
        print(f"run {run}: {pretty}")

    if not rows:
        raise SystemExit("No completed runs found.")

    print("\nAGG")
    for key in CLASSES + ["macro3", "macro", "acc", "pa"]:
        vals = [row[key] for row in rows if key in row]
        if vals:
            print(f"{key}: {st.mean(vals):.4f} +- {st.pstdev(vals):.4f}")


if __name__ == "__main__":
    main()
