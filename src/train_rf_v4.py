"""
RF baseline on the exact v4 split.

Uses the same sample ordering as sean_to_hdf5_v2.py / SEANGraphDatasetV4, then
loads split_indices_v4.pt so train/val/test match the current v4 GNN pipeline.

Features: 3512D Nav-Only vector:
  spatial.follower         [40, 4]     -> 160
  spatial.nearby           [40, 16, 4] -> 2560
  spatial.robot            [40, 2]     -> 80
  spatial.goal             [40, 2]     -> 80
  map_resnet18 (mean)      [512]       -> 512
  viz.gaze_direction_local [40, 3]     -> 120

Task:
  3 independent binary RF classifiers for competence / surprise / intention.
  By default labels are binarized as >=3 -> 1.
  With --flip_surprise, surprise is flipped so <3 -> 1.

Outputs:
  - per-run metrics CSV
  - final mean/std summary CSV
  - mean ROC plot with std shading over 5 runs
  - NPZ with test labels and per-run RF scores

Usage:
  cd src/
  python train_rf_v4.py
  python train_rf_v4.py --flip_surprise
  python train_rf_v4.py --split_path checkpoints/split_indices_v4.pt
"""

import os
import csv
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


LABEL_NAMES = ["competence", "surprise", "intention"]
LABEL_THRESHOLD = 3
N_RUNS = 5
N_CLASSES = 5


def extract_nav_features(pt_path):
    """Extract the 3512D Nav-Only feature vector from one .pt file."""
    x = torch.load(str(pt_path), weights_only=False)

    follower = np.array(x["spatial"]["follower"]).astype(np.float32)          # [40, 4]
    nearby   = np.array(x["spatial"]["nearby"]).astype(np.float32)            # [40, 16, 4]
    robot    = np.array(x["spatial"]["robot"]).astype(np.float32)             # [40, 2]
    goal     = np.array(x["spatial"]["goal"]).astype(np.float32)              # [40, 2]
    map_r18  = np.array(x["physical"]["map_resnet18"]).astype(np.float32)     # [40, 512]
    gaze     = np.array(x["viz"]["gaze_direction_local"]).astype(np.float32)  # [40, 3]

    return np.concatenate([
        follower.flatten(),
        nearby.flatten(),
        robot.flatten(),
        goal.flatten(),
        map_r18.mean(axis=0),
        gaze.flatten(),
    ])


def load_dataset_in_hdf5_order(sean_root):
    """
    Iterate .pt files in the same order as sean_to_hdf5_v2.py.
    Returns:
      feats  : [N, 3434]
      labels : [N, 3] raw Likert
    """
    sean_root = Path(sean_root)
    folders = sorted([p for p in sean_root.iterdir() if p.is_dir()])

    all_feats = []
    all_labels = []
    for folder in folders:
        pt_files = sorted(
            folder.glob("temporal_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        for pt_file in pt_files:
            x = torch.load(str(pt_file), weights_only=False)
            all_feats.append(extract_nav_features(pt_file))
            all_labels.append(np.array(x["true_label"], dtype=np.float32))

    feats = np.stack(all_feats).astype(np.float32)
    labels = np.stack(all_labels).astype(np.float32)
    return feats, labels


def binarize_labels(y, flip_surprise=False):
    """Convert raw Likert labels to binary targets, optionally flipping surprise only."""
    y_bin = (y >= LABEL_THRESHOLD).astype(np.float32)
    if flip_surprise:
        y_bin[:, 1] = 1.0 - y_bin[:, 1]
    return y_bin


def get_mode_tag(args):
    tag = "rf_v4"
    if args.flip_surprise:
        tag += "_flip_surprise"
    return tag


def train_one_run(X_train, y_train, X_test, y_test, random_state):
    """
    Train 3 independent binary RF classifiers and return:
      test_metrics : dict
      scores_test  : [N_test, 3] positive-class scores
      models       : list[RandomForestClassifier]
    """
    test_metrics = {}
    scores_test = np.zeros((X_test.shape[0], len(LABEL_NAMES)), dtype=np.float32)
    models = []

    for dim, name in enumerate(LABEL_NAMES):
        rf = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
        rf.fit(X_train, y_train[:, dim].astype(np.int64))
        models.append(rf)

        preds = rf.predict(X_test).astype(np.float32)
        if hasattr(rf, "predict_proba"):
            class_scores = rf.predict_proba(X_test)
            if class_scores.shape[1] == 2:
                probs = class_scores[:, 1]
            else:
                # Degenerate case: only one class seen in training
                cls = int(rf.classes_[0])
                probs = np.full(X_test.shape[0], float(cls), dtype=np.float32)
        else:
            probs = preds.copy()

        scores_test[:, dim] = probs

        acc = accuracy_score(y_test[:, dim], preds)
        f1 = f1_score(y_test[:, dim], preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test[:, dim], probs)
        except ValueError:
            auc = float("nan")

        test_metrics[f"test/acc_{name}"] = float(acc)
        test_metrics[f"test/f1_{name}"] = float(f1)
        test_metrics[f"test/auc_{name}"] = float(auc)

    test_metrics["test/f1_macro"] = float(np.mean([test_metrics[f"test/f1_{n}"] for n in LABEL_NAMES]))
    test_metrics["test/auc_macro"] = float(np.nanmean([test_metrics[f"test/auc_{n}"] for n in LABEL_NAMES]))
    return test_metrics, scores_test, models


def save_mean_roc_artifacts(gt_test, all_scores, out_dir, mode_tag):
    """
    Save mean ROC with std shading over runs.
    all_scores: [N_RUNS, N_test, 3]
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"roc_test_outputs_{mode_tag}.npz")
    np.savez_compressed(npz_path, gt_test=gt_test, scores_test_runs=all_scores, label_names=np.array(LABEL_NAMES))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    mean_fpr = np.linspace(0.0, 1.0, 200)

    for d, name in enumerate(LABEL_NAMES):
        tprs = []
        aucs = []
        for run_scores in all_scores:
            scores = run_scores[:, d]
            fpr, tpr, _ = roc_curve(gt_test[:, d], scores)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            try:
                aucs.append(roc_auc_score(gt_test[:, d], scores))
            except ValueError:
                aucs.append(np.nan)

        tprs = np.stack(tprs, axis=0)
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_tpr[-1] = 1.0

        ax = axes[d]
        ax.plot(mean_fpr, mean_tpr, linewidth=2, label=f"AUC={float(np.nanmean(aucs)):.3f}")
        ax.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0.0, 1.0),
            np.clip(mean_tpr + std_tpr, 0.0, 1.0),
            alpha=0.2,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="random")
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right")

    fig.suptitle(f"Test ROC Curves - {mode_tag}")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"roc_test_{mode_tag}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ROC figure -> {png_path}")
    print(f"Saved ROC arrays  -> {npz_path}")


def save_metrics_csvs(all_test_metrics, out_dir, mode_tag):
    per_run_path = os.path.join(out_dir, f"rf_results_{mode_tag}.csv")
    summary_path = os.path.join(out_dir, f"rf_summary_{mode_tag}.csv")

    fieldnames = ["run_idx"] + list(all_test_metrics[0].keys())
    with open(per_run_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_idx, metrics in enumerate(all_test_metrics, start=1):
            row = {"run_idx": run_idx}
            row.update(metrics)
            writer.writerow(row)

    summary_row = {}
    for key in all_test_metrics[0].keys():
        vals = [m[key] for m in all_test_metrics]
        summary_row[f"{key}_mean"] = float(np.nanmean(vals))
        summary_row[f"{key}_std"] = float(np.nanstd(vals))

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved per-run metrics -> {per_run_path}")
    print(f"Saved summary metrics -> {summary_path}")


def train(args):
    mode_tag = get_mode_tag(args)
    out_dir = os.path.join(args.checkpoint_dir, mode_tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset from {args.sean_root} ...")
    feats, labels_raw = load_dataset_in_hdf5_order(args.sean_root)
    labels_bin = binarize_labels(labels_raw, flip_surprise=args.flip_surprise)
    print(f"Loaded {len(feats)} samples.")

    print(f"Loading split from {args.split_path} ...")
    splits = torch.load(args.split_path, weights_only=False)
    train_idx = np.array(splits["train"], dtype=np.int64)
    val_idx = np.array(splits["val"], dtype=np.int64)
    test_idx = np.array(splits["test"], dtype=np.int64)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print("RF uses the exact v4 split; val is preserved but not used for fitting.")

    X_train = feats[train_idx]
    y_train = labels_bin[train_idx]
    X_test = feats[test_idx]
    y_test = labels_bin[test_idx]

    all_test_metrics = []
    all_scores = []

    for run_idx in range(1, N_RUNS + 1):
        seed = args.seed + run_idx - 1
        print(f"\n=== RF RUN {run_idx}/{N_RUNS} | seed={seed} ===")
        test_metrics, scores_test, models = train_one_run(
            X_train, y_train, X_test, y_test, random_state=seed
        )

        for name in LABEL_NAMES:
            print(
                f"  {name}: "
                f"acc={test_metrics[f'test/acc_{name}']:.3f} "
                f"f1={test_metrics[f'test/f1_{name}']:.3f} "
                f"auc={test_metrics[f'test/auc_{name}']:.3f}"
            )
        print(
            f"  macro: "
            f"f1={test_metrics['test/f1_macro']:.3f} "
            f"auc={test_metrics['test/auc_macro']:.3f}"
        )

        all_test_metrics.append(test_metrics)
        all_scores.append(scores_test)

        model_path = os.path.join(out_dir, f"rf_models_run{run_idx}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(models, f)

    all_scores = np.stack(all_scores, axis=0)  # [N_RUNS, N_test, 3]

    print(f"\n=== FINAL SUMMARY ({mode_tag}) OVER {N_RUNS} RUNS ===")
    for name in LABEL_NAMES:
        acc_vals = [m[f"test/acc_{name}"] for m in all_test_metrics]
        f1_vals = [m[f"test/f1_{name}"] for m in all_test_metrics]
        auc_vals = [m[f"test/auc_{name}"] for m in all_test_metrics]
        print(
            f"  {name}: "
            f"acc={float(np.mean(acc_vals)):.3f} +- {float(np.std(acc_vals)):.3f} "
            f"f1={float(np.mean(f1_vals)):.3f} +- {float(np.std(f1_vals)):.3f} "
            f"auc={float(np.nanmean(auc_vals)):.3f} +- {float(np.nanstd(auc_vals)):.3f}"
        )
    macro_f1_vals = [m["test/f1_macro"] for m in all_test_metrics]
    macro_auc_vals = [m["test/auc_macro"] for m in all_test_metrics]
    print(
        f"  macro: "
        f"f1={float(np.mean(macro_f1_vals)):.3f} +- {float(np.std(macro_f1_vals)):.3f} "
        f"auc={float(np.nanmean(macro_auc_vals)):.3f} +- {float(np.nanstd(macro_auc_vals)):.3f}"
    )

    save_metrics_csvs(all_test_metrics, out_dir, mode_tag)
    save_mean_roc_artifacts(y_test, all_scores, out_dir, mode_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sean_root", default="dataloaders/SEAN_TOGETHER_Dataset/SEAN_TOGETHER")
    parser.add_argument("--split_path", default="checkpoints/split_indices_v4.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--flip_surprise", action="store_true",
                        help="Flip surprise so that <3 is the positive class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; runs use seed, seed+1, ..., seed+4")
    args = parser.parse_args()
    train(args)
