"""
RF split baseline — same 80/16/4 train/val/test split as GNN-VAE v5.

Feature vector per sample (fixed-size, for RF compatibility):
  - robot node features:                   592D  (normalized by train-split mean/std)
  - mean-pooled edge attrs per edge type:  7 × 320D = 2240D  (zeros if type absent)
  Total:                                   2832D per sample

Uses Optuna to tune RF hyperparameters; each trial is logged as its own W&B run
(grouped so all trials appear together in the W&B UI). After optimization, evaluates
the best hyperparameters on the held-out test set and saves results in the same
format as train_linear_probe_v5.py (CSVs, ROC/PR curves, summary PDF).

Label binarization: y >= 3 for competence and intention; surprise is inverted
(y < 3 = positive), matching train_rf_baseline.py.

Config section: rf_split (in params/config.yaml)

Usage:
  cd src/
  python train_rf_split.py
  python train_rf_split.py --config params/config.yaml
"""

import os
import sys
import csv
import pickle
import argparse
import shutil
import shlex
import subprocess
from datetime import datetime
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import optuna
import yaml

try:
    import wandb
except ImportError:
    wandb = None

# ── Constants ──────────────────────────────────────────────────────────────────

LABEL_NAMES     = ["competence", "surprise", "intention"]
LABEL_THRESHOLD = 3          # y >= 3 → positive class (matches linear_probe_v5)

# The seven directed edge types; order matters for the feature vector layout
# Node-index semantics (positional, not stored in H5): 0=robot, 1=follower, 2+=bystander
EDGE_TYPE_NAMES = [
    "robot_to_follower",
    "follower_to_robot",
    "robot_to_bystander",
    "bystander_to_robot",
    "follower_to_bystander",
    "bystander_to_follower",
    "bystander_to_bystander",
]

ROBOT_FEAT_DIM  = 592
EDGE_FEAT_FLAT  = 40 * 8    # 40 timesteps × 8 features = 320 per edge type
N_EDGE_TYPES    = len(EDGE_TYPE_NAMES)
TOTAL_FEAT_DIM  = ROBOT_FEAT_DIM + N_EDGE_TYPES * EDGE_FEAT_FLAT  # 2832

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── Utility: reproducible run directory ────────────────────────────────────────

def create_run_dir(experiments_root):
    os.makedirs(experiments_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(experiments_root, timestamp)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(experiments_root, f"{timestamp}_{suffix}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def get_git_commit_hash(project_root):
    env_hash = os.environ.get("GIT_COMMIT_SHA", "").strip()
    if env_hash:
        if len(env_hash) == 40 and all(c in "0123456789abcdef" for c in env_hash.lower()):
            return env_hash
        raise RuntimeError("GIT_COMMIT_SHA is set but is not a valid 40-char hex SHA1.")
    try:
        commit_hash = subprocess.check_output(
            ["git", "-C", project_root, "rev-parse", "--verify", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as e:
        raise RuntimeError("Failed to resolve git commit hash.") from e
    if len(commit_hash) != 40 or any(c not in "0123456789abcdef" for c in commit_hash.lower()):
        raise RuntimeError(f"Resolved git commit hash is invalid: {commit_hash!r}")
    return commit_hash


def get_launch_command():
    return f"{sys.executable} {shlex.join(sys.argv)}"


# ── Config loading ──────────────────────────────────────────────────────────────

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a top-level mapping: {config_path}")
    if "rf_split" not in cfg:
        raise ValueError(
            f"Config is missing required top-level section 'rf_split': {config_path}\n"
            "Add the section shown in the rf_split config template."
        )
    cfg = cfg["rf_split"]
    if not isinstance(cfg, dict):
        raise ValueError(f"Config section 'rf_split' must be a mapping: {config_path}")

    required_keys = [
        "h5_path",
        "experiments_root",
        "wandb",
        "wandb_project",
        "n_trials",
        "n_jobs_rf",
        "save_plots",
        "plot_formats",
        "trials_csv_filename",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(
            f"Config section 'rf_split' is missing keys: {', '.join(missing)}"
        )
    return SimpleNamespace(**cfg)


def normalize_plot_formats(plot_formats):
    if isinstance(plot_formats, str):
        tokens = [p.strip().lower() for p in plot_formats.replace(",", " ").split() if p.strip()]
    elif isinstance(plot_formats, (list, tuple)):
        tokens = [str(p).strip().lower() for p in plot_formats if str(p).strip()]
    else:
        tokens = ["pdf"]
    return list(dict.fromkeys(tokens)) or ["pdf"]


# ── Feature extraction ──────────────────────────────────────────────────────────

def _classify_node(node_idx):
    """Return 'robot', 'follower', or 'bystander'."""
    if node_idx == 0:
        return "robot"
    if node_idx == 1:
        return "follower"
    return "bystander"


def _edge_type_index(src, dst):
    """Map (src_type, dst_type) → index into EDGE_TYPE_NAMES, or -1 if unknown."""
    src_t = _classify_node(src)
    dst_t = _classify_node(dst)
    key = f"{src_t}_to_{dst_t}"
    try:
        return EDGE_TYPE_NAMES.index(key)
    except ValueError:
        return -1


def extract_sample_features(grp):
    """
    Build the 2832D flat feature vector for one HDF5 sample group.

    Returns:
      feat  np.ndarray [TOTAL_FEAT_DIM]  (un-normalized robot features)
      label np.ndarray [3]               (raw integer Likert scores)
    """
    node_features  = grp["node_features"][:]   # [18, 592]
    valid_mask     = grp["valid_mask"][:]       # [18, 40]
    edge_index_all = grp["edge_index"][:]       # [2, E_max]
    edge_attr_all  = grp["edge_attr"][:]        # [E_max, 40, 8]
    label          = grp["label"][:]            # [3]

    # Robot features (un-normalized here; normalized later from train split stats)
    robot_feat = node_features[0].astype(np.float32)   # [592]

    # Determine which nodes are present across at least one timestep
    node_presence = (valid_mask.sum(axis=1) > 0)       # [18] bool

    # Filter edges: both endpoints must be present
    src = edge_index_all[0]
    dst = edge_index_all[1]
    keep = node_presence[src] & node_presence[dst]
    src = src[keep]
    dst = dst[keep]
    edge_attr = edge_attr_all[keep].astype(np.float32)  # [E_valid, 40, 8]

    # Accumulate edge attrs per type, then mean-pool
    type_attrs = [[] for _ in range(N_EDGE_TYPES)]
    for e in range(len(src)):
        ti = _edge_type_index(src[e], dst[e])
        if ti >= 0:
            type_attrs[ti].append(edge_attr[e])   # [40, 8]

    edge_parts = []
    for ti in range(N_EDGE_TYPES):
        if type_attrs[ti]:
            pooled = np.stack(type_attrs[ti]).mean(axis=0)   # [40, 8]
        else:
            pooled = np.zeros((40, 8), dtype=np.float32)
        edge_parts.append(pooled.flatten())   # [320]

    feat = np.concatenate([robot_feat] + edge_parts)   # [2832]
    return feat, label.astype(np.int64)


def load_dataset_features(h5_path):
    """
    Load all samples from the graph-format HDF5 in stable key order.

    Returns:
      X      np.ndarray [N, 2832]  (un-normalized)
      y_raw  np.ndarray [N, 3]     (raw Likert integer scores)
      keys   list[str]             (sample keys, same order as X/y)
    """
    import re

    def _key_sort(k):
        m = re.search(r"(\d+)$", k)
        return (int(m.group(1)) if m else 0, k)

    with h5py.File(h5_path, "r") as hf:
        required = {"node_features", "valid_mask", "edge_index", "edge_attr", "label"}
        keys = sorted(
            [k for k in hf.keys() if isinstance(hf[k], h5py.Group) and required.issubset(hf[k].keys())],
            key=_key_sort,
        )
        if not keys:
            raise ValueError(f"No valid sample groups found in {h5_path}")

        X_list, y_list = [], []
        for k in keys:
            feat, lbl = extract_sample_features(hf[k])
            X_list.append(feat)
            y_list.append(lbl)

    X = np.stack(X_list)        # [N, 2832]
    y = np.stack(y_list)        # [N, 3]
    return X, y, keys


def normalize_features(X_train, X_val, X_test):
    """
    Z-score normalize using train-split statistics only.
    Avoids division by zero with a small epsilon.
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    X_train_n = (X_train - mean) / (std + 1e-8)
    X_val_n   = (X_val   - mean) / (std + 1e-8)
    X_test_n  = (X_test  - mean) / (std + 1e-8)
    return X_train_n, X_val_n, X_test_n, mean, std


# ── Metrics (mirrors train_linear_probe_v5.py) ─────────────────────────────────

def compute_metrics(scores_np, gt_np):
    """
    scores_np: [N, 3]  real-valued decision scores (RF predict_proba positive class)
    gt_np:     [N, 3]  binary ground truth (0/1)
    Returns dict of per-label {acc, f1, auc}.
    """
    metrics = {}
    preds = (scores_np >= 0.5).astype(float)
    for d, name in enumerate(LABEL_NAMES):
        acc = accuracy_score(gt_np[:, d], preds[:, d])
        f1  = f1_score(gt_np[:, d], preds[:, d], zero_division=0)
        try:
            auc = roc_auc_score(gt_np[:, d], scores_np[:, d])
        except ValueError:
            auc = float("nan")
        metrics[name] = {"acc": acc, "f1": f1, "auc": auc}
    return metrics


def rf_predict_scores(models, X):
    """
    Returns positive-class probability scores: [N, 3].
    Uses the probability of class 1 from each RF.
    """
    scores = []
    for clf in models:
        proba = clf.predict_proba(X)          # [N, n_classes]
        classes = list(clf.classes_)
        if 1 in classes:
            pos_idx = classes.index(1)
            scores.append(proba[:, pos_idx])
        else:
            scores.append(np.zeros(len(X), dtype=np.float32))
    return np.stack(scores, axis=1)           # [N, 3]


# ── CSV / plot helpers ──────────────────────────────────────────────────────────

def save_trials_csv(trial_rows, csv_path):
    if not trial_rows:
        return
    fieldnames = list(trial_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)


def save_test_metrics_csv(test_metrics, csv_path):
    rows = []
    for name in LABEL_NAMES:
        m = test_metrics[name]
        rows.append({"label": name, "accuracy": m["acc"], "f1": m["f1"], "auc": m["auc"]})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "accuracy", "f1", "auc"])
        writer.writeheader()
        writer.writerows(rows)


def compute_test_curve_data(scores_np, gt_np):
    curve_data = {}
    for d, name in enumerate(LABEL_NAMES):
        y_true  = gt_np[:, d]
        y_score = scores_np[:, d]
        label_data = {"roc": None, "pr": None}
        if np.unique(y_true).size >= 2:
            fpr, tpr, roc_thr = roc_curve(y_true, y_score)
            prec, rec, pr_thr = precision_recall_curve(y_true, y_score)
            label_data["roc"] = {
                "fpr": fpr, "tpr": tpr, "thresholds": roc_thr,
                "auc": roc_auc_score(y_true, y_score),
            }
            label_data["pr"] = {
                "precision": prec, "recall": rec, "thresholds": pr_thr,
                "ap": average_precision_score(y_true, y_score),
            }
        curve_data[name] = label_data
    return curve_data


def save_curve_csvs(curve_data, run_dir):
    roc_path = os.path.join(run_dir, "test_roc_curve_data.csv")
    with open(roc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "fpr", "tpr", "threshold", "auc"])
        writer.writeheader()
        for name in LABEL_NAMES:
            d = curve_data[name]["roc"]
            if d is None:
                continue
            for fpr, tpr, thr in zip(d["fpr"], d["tpr"], d["thresholds"]):
                writer.writerow({"label": name, "fpr": fpr, "tpr": tpr, "threshold": thr, "auc": d["auc"]})
    print(f"Saved ROC curve data -> {roc_path}")

    pr_path = os.path.join(run_dir, "test_pr_curve_data.csv")
    with open(pr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "recall", "precision", "threshold", "ap"])
        writer.writeheader()
        for name in LABEL_NAMES:
            d = curve_data[name]["pr"]
            if d is None:
                continue
            for i, (rec, prec) in enumerate(zip(d["recall"], d["precision"])):
                thr = d["thresholds"][i] if i < len(d["thresholds"]) else ""
                writer.writerow({"label": name, "recall": rec, "precision": prec, "threshold": thr, "ap": d["ap"]})
    print(f"Saved PR curve data -> {pr_path}")


def save_plots(trial_rows, curve_data, run_dir, plot_formats):
    # ── Optuna trial history ────────────────────────────────────────────────────
    if trial_rows:
        trial_nums = [r["trial"] for r in trial_rows]
        val_f1s    = [r["val_mean_f1"] for r in trial_rows]
        best_so_far = [max(val_f1s[:i+1]) for i in range(len(val_f1s))]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(trial_nums, val_f1s, s=20, alpha=0.6, label="trial val mean-F1")
        ax.plot(trial_nums, best_so_far, color="tab:orange", linewidth=2, label="best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Val mean-F1 (3 labels)")
        ax.set_title("Optuna Optimization History")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for fmt in plot_formats:
            p = os.path.join(run_dir, f"optuna_history.{fmt}")
            fig.savefig(p, format=fmt)
            print(f"Saved Optuna history -> {p}")
        plt.close(fig)

    # ── Test ROC curves ─────────────────────────────────────────────────────────
    roc_fig, roc_ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for name in LABEL_NAMES:
        d = curve_data[name]["roc"]
        if d is None:
            continue
        plotted = True
        roc_ax.plot(d["fpr"], d["tpr"], label=f"{name} (AUC={d['auc']:.3f})")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    roc_ax.set_title("Test ROC Curves")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.grid(True, alpha=0.3)
    if plotted:
        roc_ax.legend()
    for fmt in plot_formats:
        p = os.path.join(run_dir, f"test_roc_curves.{fmt}")
        roc_fig.savefig(p, format=fmt)
        print(f"Saved ROC curves -> {p}")
    plt.close(roc_fig)

    # ── Test PR curves ──────────────────────────────────────────────────────────
    pr_fig, pr_ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for name in LABEL_NAMES:
        d = curve_data[name]["pr"]
        if d is None:
            continue
        plotted = True
        pr_ax.plot(d["recall"], d["precision"], label=f"{name} (AP={d['ap']:.3f})")
    pr_ax.set_title("Test Precision-Recall Curves")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.grid(True, alpha=0.3)
    if plotted:
        pr_ax.legend()
    for fmt in plot_formats:
        p = os.path.join(run_dir, f"test_pr_curves.{fmt}")
        pr_fig.savefig(p, format=fmt)
        print(f"Saved PR curves -> {p}")
    plt.close(pr_fig)

    # ── Summary PDF (6-panel, mirrors linear_probe_summary.pdf) ────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    ax = axes.flatten()

    # Panel 0: Optuna history
    if trial_rows:
        ax[0].scatter(trial_nums, val_f1s, s=15, alpha=0.5, label="trial")
        ax[0].plot(trial_nums, best_so_far, color="tab:orange", linewidth=2, label="best")
        ax[0].set_title("Optuna History (val mean-F1)")
        ax[0].set_xlabel("Trial")
        ax[0].set_ylabel("Mean F1")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

    # Panel 1: Per-label val F1 across trials
    if trial_rows:
        for name in LABEL_NAMES:
            ax[1].scatter(trial_nums, [r[f"val_f1_{name}"] for r in trial_rows], s=12, alpha=0.4, label=name)
        ax[1].set_title("Val F1 per Label (all trials)")
        ax[1].set_xlabel("Trial")
        ax[1].set_ylabel("F1")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

    # Panel 2: Per-label val AUC across trials
    if trial_rows:
        for name in LABEL_NAMES:
            ax[2].scatter(trial_nums, [r[f"val_auc_{name}"] for r in trial_rows], s=12, alpha=0.4, label=name)
        ax[2].set_title("Val AUC per Label (all trials)")
        ax[2].set_xlabel("Trial")
        ax[2].set_ylabel("AUC")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend()

    # Panel 3: blank (placeholder for future use)
    ax[3].axis("off")

    # Panel 4: Test ROC curves
    plotted_roc = False
    for name in LABEL_NAMES:
        d = curve_data[name]["roc"]
        if d is None:
            continue
        plotted_roc = True
        ax[4].plot(d["fpr"], d["tpr"], label=f"{name} (AUC={d['auc']:.3f})")
    ax[4].plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax[4].set_title("Test ROC Curves")
    ax[4].set_xlabel("FPR")
    ax[4].set_ylabel("TPR")
    ax[4].grid(True, alpha=0.3)
    if plotted_roc:
        ax[4].legend()

    # Panel 5: Test PR curves
    plotted_pr = False
    for name in LABEL_NAMES:
        d = curve_data[name]["pr"]
        if d is None:
            continue
        plotted_pr = True
        ax[5].plot(d["recall"], d["precision"], label=f"{name} (AP={d['ap']:.3f})")
    ax[5].set_title("Test Precision-Recall Curves")
    ax[5].set_xlabel("Recall")
    ax[5].set_ylabel("Precision")
    ax[5].grid(True, alpha=0.3)
    if plotted_pr:
        ax[5].legend()

    fig.tight_layout()
    summary_path = os.path.join(run_dir, "rf_summary.pdf")
    fig.savefig(summary_path, format="pdf")
    print(f"Saved RF summary -> {summary_path}")
    plt.close(fig)


# ── Optuna objective ────────────────────────────────────────────────────────────

def make_objective(X_train, y_train_bin, X_val, y_val_bin, args, trial_rows, wb_run):
    """
    Returns an Optuna objective function.
    wb_run is the single active wandb Run (or None), shared across all trials.
    Each trial is logged as one step so W&B shows charts over trial number.
    """

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "max_depth":         trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
            "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),
        }

        models = []
        train_scores_list = []
        val_scores_list   = []
        for d in range(3):
            clf = RandomForestClassifier(
                **params,
                n_jobs=args.n_jobs_rf,
                random_state=trial.number,
            )
            clf.fit(X_train, y_train_bin[:, d])
            models.append(clf)

            classes = list(clf.classes_)
            pos_col = classes.index(1) if 1 in classes else None

            for scores_list, X in [(train_scores_list, X_train), (val_scores_list, X_val)]:
                proba = clf.predict_proba(X)
                if pos_col is not None:
                    scores_list.append(proba[:, pos_col])
                else:
                    scores_list.append(np.zeros(len(X)))

        train_scores  = np.stack(train_scores_list, axis=1)
        val_scores    = np.stack(val_scores_list,   axis=1)
        train_metrics = compute_metrics(train_scores, y_train_bin)
        val_metrics   = compute_metrics(val_scores,   y_val_bin)

        val_f1s = [val_metrics[n]["f1"] for n in LABEL_NAMES]
        mean_f1 = float(np.mean(val_f1s))

        row = {
            "trial":         trial.number,
            "train_mean_f1": float(np.mean([train_metrics[n]["f1"] for n in LABEL_NAMES])),
            "val_mean_f1":   mean_f1,
        }
        for name in LABEL_NAMES:
            for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
                m = metrics[name]
                row[f"{split}_f1_{name}"]  = m["f1"]
                row[f"{split}_auc_{name}"] = m["auc"]
                row[f"{split}_acc_{name}"] = m["acc"]
        row.update(params)
        trial_rows.append(row)

        print(
            f"  Trial {trial.number:03d} | "
            f"comp_f1={val_f1s[0]:.3f} surp_f1={val_f1s[1]:.3f} int_f1={val_f1s[2]:.3f} | "
            f"mean_f1={mean_f1:.3f} | "
            f"n_est={params['n_estimators']} depth={params['max_depth']} "
            f"feat={params['max_features']} cw={params['class_weight']}"
        )

        if wb_run is not None:
            log = {
                "trial":         trial.number,
                "val/mean_f1":   mean_f1,
                "train/mean_f1": float(np.mean([train_metrics[n]["f1"] for n in LABEL_NAMES])),
            }
            for name in LABEL_NAMES:
                for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
                    m = metrics[name]
                    log[f"{split}/{name}_f1"]  = m["f1"]
                    log[f"{split}/{name}_auc"] = m["auc"]
                    log[f"{split}/{name}_acc"] = m["acc"]
            # Log hyperparams as metrics too so they appear in W&B charts
            log["params/n_estimators"]      = params["n_estimators"]
            log["params/min_samples_split"] = params["min_samples_split"]
            log["params/min_samples_leaf"]  = params["min_samples_leaf"]
            wb_run.log(log, step=trial.number)

        return mean_f1

    return objective


# ── Main training function ──────────────────────────────────────────────────────

def train(args, console_stream=None):
    use_wandb = bool(getattr(args, "wandb", False))
    if use_wandb and wandb is None:
        print("wandb requested but package not installed; continuing without W&B logging.")
        use_wandb = False
    elif not use_wandb:
        print("W&B logging disabled via config.")

    run_name = f"rf_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Loading dataset from {args.h5_path}...")
    X_all, y_raw_all, sample_keys = load_dataset_features(args.h5_path)
    N = len(sample_keys)
    print(f"Loaded {N} samples  |  feature dim={X_all.shape[1]}")

    # Replicate the exact same 80/16/4 split used in train_gnn_vae_v5.py:
    #   test_size  = 4% of N, val_size = 16% of N, train_size = remainder
    #   seed = 42 via torch.Generator (both scripts sort H5 keys numerically)
    test_size  = int(0.04 * N)
    val_size   = int(0.16 * N)
    train_size = N - val_size - test_size
    generator  = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(N, generator=generator).numpy()
    train_idx = all_indices[:train_size]
    val_idx   = all_indices[train_size:train_size + val_size]
    test_idx  = all_indices[train_size + val_size:]
    print(f"Train: {len(train_idx)}  |  Val: {len(val_idx)}  |  Test: {len(test_idx)}")

    X_train_raw = X_all[train_idx]
    X_val_raw   = X_all[val_idx]
    X_test_raw  = X_all[test_idx]

    y_train_raw = y_raw_all[train_idx]
    y_val_raw   = y_raw_all[val_idx]
    y_test_raw  = y_raw_all[test_idx]

    # Normalize features using train-split statistics only
    X_train, X_val, X_test, feat_mean, feat_std = normalize_features(
        X_train_raw, X_val_raw, X_test_raw
    )
    print(f"Feature normalization: mean in [{feat_mean.min():.3f}, {feat_mean.max():.3f}]  "
          f"std in [{feat_std.min():.5f}, {feat_std.max():.3f}]")

    # Binarize labels: >= 3 → 1 for competence and intention.
    # Surprise (index 1) is inverted: < 3 → 1 (matches train_rf_baseline.py).
    def binarize(y_raw):
        y_bin = (y_raw >= LABEL_THRESHOLD).astype(int)
        y_bin[:, 1] = (y_raw[:, 1] < LABEL_THRESHOLD).astype(int)
        return y_bin

    y_train_bin = binarize(y_train_raw)
    y_val_bin   = binarize(y_val_raw)
    y_test_bin  = binarize(y_test_raw)

    for split_name, yb in [("train", y_train_bin), ("val", y_val_bin), ("test", y_test_bin)]:
        pos_rates = yb.mean(axis=0)
        print(f"  {split_name} positive rates: " +
              "  ".join(f"{n}={r:.2f}" for n, r in zip(LABEL_NAMES, pos_rates)))

    # ── Optuna optimization ─────────────────────────────────────────────────────
    print(f"\nRunning Optuna with {args.n_trials} trials (maximize val mean-F1)...")
    trial_rows = []
    wb_run = None
    if use_wandb and wandb is not None:
        wb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={"n_trials": args.n_trials, "n_jobs_rf": args.n_jobs_rf},
        )

    objective = make_objective(
        X_train, y_train_bin, X_val, y_val_bin, args, trial_rows, wb_run
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    best_trial  = study.best_trial
    best_params = best_trial.params
    print(f"\nBest trial: {best_trial.number}  val mean-F1={best_trial.value:.4f}")
    print(f"Best params: {best_params}")

    # ── Retrain best model on train split ──────────────────────────────────────
    print("\nRetraining best model on train split for test evaluation...")
    best_models = []
    for d in range(3):
        clf = RandomForestClassifier(
            **best_params,
            n_jobs=args.n_jobs_rf,
            random_state=42,
        )
        clf.fit(X_train, y_train_bin[:, d])
        best_models.append(clf)

    val_scores  = rf_predict_scores(best_models, X_val)
    test_scores = rf_predict_scores(best_models, X_test)

    val_metrics  = compute_metrics(val_scores,  y_val_bin)
    test_metrics = compute_metrics(test_scores, y_test_bin)

    print("\n=== VALIDATION SET (best model) ===")
    for name in LABEL_NAMES:
        m = val_metrics[name]
        print(f"  {name}: acc={m['acc']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}")

    print("\n=== TEST SET (best model) ===")
    for name in LABEL_NAMES:
        m = test_metrics[name]
        print(f"  {name}: acc={m['acc']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}")

    # ── Log final test metrics + best params to the same W&B run ──────────────
    if wb_run is not None:
        summary = {"best_trial": best_trial.number}
        for name in LABEL_NAMES:
            for split, metrics in [("val", val_metrics), ("test", test_metrics)]:
                m = metrics[name]
                summary[f"{split}/best_{name}_acc"] = m["acc"]
                summary[f"{split}/best_{name}_f1"]  = m["f1"]
                summary[f"{split}/best_{name}_auc"] = m["auc"]
        summary["val/best_mean_f1"]  = float(np.mean([val_metrics[n]["f1"]  for n in LABEL_NAMES]))
        summary["test/best_mean_f1"] = float(np.mean([test_metrics[n]["f1"] for n in LABEL_NAMES]))
        for k, v in {**summary, **best_params}.items():
            wb_run.summary[k] = v
        wb_run.finish()

    test_curve_data = compute_test_curve_data(test_scores, y_test_bin)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    # Best RF models
    models_path = os.path.join(args.run_dir, "best_rf_models.pkl")
    with open(models_path, "wb") as f:
        pickle.dump(best_models, f)
    print(f"\nSaved best RF models -> {models_path}")

    # Best params
    params_path = os.path.join(args.run_dir, "best_rf_params.yaml")
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"best_trial": best_trial.number, "val_mean_f1": best_trial.value, **best_params},
            f, sort_keys=False,
        )
    print(f"Saved best params -> {params_path}")

    # Normalization stats
    norm_path = os.path.join(args.run_dir, "feature_norm_stats.npz")
    np.savez(norm_path, mean=feat_mean, std=feat_std)
    print(f"Saved normalization stats -> {norm_path}")

    # Trial CSV
    trials_csv = os.path.join(args.run_dir, args.trials_csv_filename)
    save_trials_csv(trial_rows, trials_csv)
    print(f"Saved trial results -> {trials_csv}")

    # Test metrics CSV
    test_csv = os.path.join(args.run_dir, "test_metrics.csv")
    save_test_metrics_csv(test_metrics, test_csv)
    print(f"Saved test metrics -> {test_csv}")

    # ROC / PR CSVs
    save_curve_csvs(test_curve_data, args.run_dir)

    # Plots
    if args.save_plots:
        plot_formats = normalize_plot_formats(args.plot_formats)
        save_plots(trial_rows, test_curve_data, args.run_dir, plot_formats)

    print(f"\nDone. Run directory: {args.run_dir}")
    print(f"Best val mean-F1: {best_trial.value:.4f}")
    print(f"Test mean-F1:     {np.mean([test_metrics[n]['f1'] for n in LABEL_NAMES]):.4f}")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "--cfg", "--c",
        dest="config",
        default="params/config.yaml",
        help="Path to YAML config file (must contain an 'rf_split' section).",
    )
    cli_args = parser.parse_args()
    args = load_config(cli_args.config)

    args.config_path      = os.path.abspath(cli_args.config)
    args.experiments_root = os.path.abspath(args.experiments_root)
    args.run_dir          = create_run_dir(args.experiments_root)

    # Snapshot config and metadata into run dir
    config_snapshot = os.path.join(args.run_dir, "config.yaml")
    shutil.copy2(args.config_path, config_snapshot)

    resolved_config_path = os.path.join(args.run_dir, "config_resolved.yaml")
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    git_hash_path = os.path.join(args.run_dir, "git_commit.txt")
    with open(git_hash_path, "w", encoding="utf-8") as f:
        f.write(get_git_commit_hash(PROJECT_ROOT) + "\n")

    command_path = os.path.join(args.run_dir, "run_command.txt")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write(get_launch_command() + "\n")

    log_path        = os.path.join(args.run_dir, "train.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "a", encoding="utf-8") as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        try:
            print(f"Run directory: {args.run_dir}")
            print(f"Config snapshot: {config_snapshot}")
            train(args, console_stream=original_stdout)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr