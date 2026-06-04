"""
Train the same RF configuration 10 times (different random seeds) on the fixed
80/16/4 split and report mean ± std of binary F1 on the test set.

Usage:
  cd src/
  python eval_rf_stability.py --h5_path dataloaders/sean_together_v2.h5
  python eval_rf_stability.py --h5_path dataloaders/sean_together_v2.h5 \
      --n_estimators 85 --max_depth 5 --min_samples_split 17 \
      --min_samples_leaf 7 --max_features 0.3
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Reuse feature extraction and data loading from train_rf_split
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_rf_split import (
    load_dataset_features,
    normalize_features,
    LABEL_NAMES,
    LABEL_THRESHOLD,
)

N_RUNS = 10


def binarize(y_raw):
    y_bin = (y_raw >= LABEL_THRESHOLD).astype(int)
    y_bin[:, 1] = (y_raw[:, 1] < LABEL_THRESHOLD).astype(int)  # surprise inverted
    return y_bin


def run(args):
    print(f"Loading dataset from {args.h5_path}...")
    X_all, y_raw_all, _ = load_dataset_features(args.h5_path)
    N = len(X_all)

    # Same 80/16/4 split as train_rf_split.py
    test_size  = int(0.04 * N)
    val_size   = int(0.16 * N)
    train_size = N - val_size - test_size
    generator  = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(N, generator=generator).numpy()
    train_idx = all_indices[:train_size]
    test_idx  = all_indices[train_size + val_size:]
    print(f"Train: {len(train_idx)}  |  Test: {len(test_idx)}")

    X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
    y_train_raw, y_test_raw = y_raw_all[train_idx], y_raw_all[test_idx]

    X_train, _, X_test, _, _ = normalize_features(X_train_raw, X_test_raw, X_test_raw)

    y_train_bin = binarize(y_train_raw)
    y_test_bin  = binarize(y_test_raw)

    rf_params = {
        "n_estimators":      args.n_estimators,
        "max_depth":         args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf":  args.min_samples_leaf,
        "max_features":      args.max_features,
        "class_weight":      args.class_weight if args.class_weight != "None" else None,
        "n_jobs":            -1,
    }
    print(f"\nRF params: {rf_params}")
    print(f"Running {N_RUNS} seeds...\n")

    # f1s[run, label]
    f1s = np.zeros((N_RUNS, 3))

    header = f"{'Seed':>4} | {'comp_f1':>7} {'surp_f1':>7} {'int_f1':>7} {'mean_f1':>8}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for seed in range(N_RUNS):
        for d in range(3):
            clf = RandomForestClassifier(**rf_params, random_state=seed)
            clf.fit(X_train, y_train_bin[:, d])
            preds = clf.predict(X_test)
            f1s[seed, d] = f1_score(y_test_bin[:, d], preds, zero_division=0)

        mean_f1 = f1s[seed].mean()
        print(
            f"{seed:>4} | "
            f"{f1s[seed, 0]:>7.3f} {f1s[seed, 1]:>7.3f} {f1s[seed, 2]:>7.3f} {mean_f1:>8.3f}"
        )

    print(sep)
    means = f1s.mean(axis=0)
    stds  = f1s.std(axis=0)
    print(
        f"{'mean':>4} | "
        f"{means[0]:>7.3f} {means[1]:>7.3f} {means[2]:>7.3f} {means.mean():>8.3f}"
    )
    print(
        f"{'std':>4} | "
        f"{stds[0]:>7.3f} {stds[1]:>7.3f} {stds[2]:>7.3f} {stds.mean():>8.3f}"
    )
    print(sep)

    print("\n=== FINAL RESULTS ===")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {means[i]:.3f} ± {stds[i]:.3f}")

    # ── Plot ────────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: bar chart with error bars (mean ± std per label)
    x = np.arange(3)
    bars = axes[0].bar(x, means, yerr=stds, capsize=6, color=["#4C72B0", "#DD8452", "#55A868"],
                       width=0.5, error_kw={"linewidth": 2})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LABEL_NAMES)
    axes[0].set_ylabel("Binary F1")
    axes[0].set_title(f"Mean ± Std F1 over {N_RUNS} seeds (test set)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02,
                     f"{mean:.3f}±{std:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: scatter of all 10 individual runs per label
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for d, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        jitter = (np.random.default_rng(0).random(N_RUNS) - 0.5) * 0.15
        axes[1].scatter(np.full(N_RUNS, d) + jitter, f1s[:, d],
                        color=color, alpha=0.7, s=50, label=name)
        axes[1].hlines(means[d], d - 0.25, d + 0.25, color=color, linewidth=2.5)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(LABEL_NAMES)
    axes[1].set_ylabel("Binary F1")
    axes[1].set_title(f"Individual run F1 scores ({N_RUNS} seeds)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "rf_stability.pdf")
    fig.savefig(plot_path, format="pdf")
    png_path = os.path.join(args.output_dir, "rf_stability.png")
    fig.savefig(png_path, format="png", dpi=150)
    plt.close(fig)
    print(f"\nSaved plot -> {plot_path}")
    print(f"Saved plot -> {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path",           default="dataloaders/sean_together_v2.h5")
    parser.add_argument("--output_dir",         default="results/rf_stability")
    parser.add_argument("--n_estimators",        type=int,   default=85)
    parser.add_argument("--max_depth",           type=int,   default=5)
    parser.add_argument("--min_samples_split",   type=int,   default=17)
    parser.add_argument("--min_samples_leaf",    type=int,   default=7)
    parser.add_argument("--max_features",        default="0.3",
                        help="float (e.g. 0.3), 'sqrt', or 'log2'")
    parser.add_argument("--class_weight",        default="None",
                        help="'balanced' or 'None'")
    args = parser.parse_args()

    # Convert max_features to float if numeric
    try:
        args.max_features = float(args.max_features)
    except ValueError:
        pass  # keep as string ("sqrt" or "log2")

    run(args)
