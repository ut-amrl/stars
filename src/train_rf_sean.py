import os
import sys
import csv
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

FEATURES_SPATIAL  = ["goal", "robot", "follower", "nearby"]
FEATURES_PHYSICAL = ["map_resnet18"]
LABEL_NAMES     = ["competence", "surprise", "intention"]
LABEL_THRESHOLD = 3


def _to_np(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    return np.asarray(v)

def extract_features_from_pt(pt_path):
    x = torch.load(str(pt_path), weights_only=False)

    parts = []
    for key in FEATURES_SPATIAL:
        parts.append(_to_np(x["spatial"][key]).astype(np.float32).flatten())
    for key in FEATURES_PHYSICAL:
        parts.append(_to_np(x["physical"][key]).astype(np.float32).flatten())

    feat = np.concatenate(parts)
    label = _to_np(x["y"]).astype(np.float32)
    return feat, label


def _load_scenes(scene_dirs):
    X_list, y_list, scene_list = [], [], []
    for scene_dir in scene_dirs:
        pt_files = sorted(
            scene_dir.glob("temporal_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for pt_file in pt_files:
            feat, label = extract_features_from_pt(pt_file)
            X_list.append(feat)
            y_list.append(label)
            scene_list.append(scene_dir.name)

    if not X_list:
        return np.empty((0, 0)), np.empty((0, 3)), []
    return np.stack(X_list), np.stack(y_list), scene_list


def load_split(split_dir):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    scenes = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    return _load_scenes(scenes)


def load_train_with_pct(train_dir, train_pct, seed=42):
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    all_scenes = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    n_total = len(all_scenes)
    import math
    n_use = max(1, math.ceil(n_total * train_pct))

    if train_pct >= 1.0:
        selected = all_scenes
    else:
        rng = random.Random(seed)
        selected = sorted(rng.sample(all_scenes, n_use), key=lambda p: p.name)

    print(f"  Training scenes: {n_use}/{n_total} ({train_pct*100:.1f}%)")
    return _load_scenes(selected), n_use, n_total


def binarize(y_raw):
    y_bin = (y_raw >= LABEL_THRESHOLD).astype(int)
    y_bin[:, 1] = (y_raw[:, 1] < LABEL_THRESHOLD).astype(int)
    return y_bin



def run(args):
    train_sean = os.path.join(args.data_root, "train", "sean")
    val_sean   = os.path.join(args.data_root, "val", "sean")
    test_sean  = os.path.join(args.data_root, "test", "sean")

    print(f"Data root: {args.data_root}")
    print(f"Train %:   {args.train_pct*100:.1f}%")

    (X_train, y_train_raw, train_scenes), n_scenes_used, n_scenes_total = \
        load_train_with_pct(train_sean, args.train_pct, seed=args.seed)
    X_val, y_val_raw, val_scenes = load_split(val_sean)
    X_test, y_test_raw, test_scenes = load_split(test_sean)

    print(f"  Train samples: {X_train.shape[0]}  (feat dim={X_train.shape[1]})")
    print(f"  Val samples:   {X_val.shape[0]}")
    print(f"  Test samples:  {X_test.shape[0]}")

    y_train = binarize(y_train_raw)
    y_val   = binarize(y_val_raw)
    y_test  = binarize(y_test_raw)

    for split_name, yb in [("train", y_train), ("val", y_val), ("test", y_test)]:
        pos = yb.mean(axis=0)
        print(f"  {split_name} pos rates: " +
              "  ".join(f"{n}={r:.2f}" for n, r in zip(LABEL_NAMES, pos)))

    # ── Train & evaluate ──────────────────────────────────────────────────────
    results = {}
    for d, name in enumerate(LABEL_NAMES):
        clf = RandomForestClassifier(
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        clf.fit(X_train, y_train[:, d])

        val_pred  = clf.predict(X_val)
        test_pred = clf.predict(X_test)

        val_f1  = f1_score(y_val[:, d],  val_pred,  zero_division=0)
        test_f1 = f1_score(y_test[:, d], test_pred, zero_division=0)

        results[name] = {"val_f1": val_f1, "test_f1": test_f1}
        print(f"  {name:12s}  val_f1={val_f1:.3f}  test_f1={test_f1:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    val_mean  = np.mean([results[n]["val_f1"]  for n in LABEL_NAMES])
    test_mean = np.mean([results[n]["test_f1"] for n in LABEL_NAMES])
    print(f"\n  Mean F1    val={val_mean:.3f}  test={test_mean:.3f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pct_tag = f"pct{int(args.train_pct*100)}"
    run_dir = os.path.join(args.experiments_root, f"{ts}_{pct_tag}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "val_f1", "test_f1"])
        writer.writeheader()
        for name in LABEL_NAMES:
            writer.writerow({
                "label":   name,
                "val_f1":  results[name]["val_f1"],
                "test_f1": results[name]["test_f1"],
            })
        writer.writerow({
            "label":   "mean",
            "val_f1":  val_mean,
            "test_f1": test_mean,
        })
    print(f"Saved metrics -> {csv_path}")

    config_path = os.path.join(run_dir, "run_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"data_root: {args.data_root}\n")
        f.write(f"train_pct: {args.train_pct}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"n_jobs: {args.n_jobs}\n")
        f.write(f"train_scenes_used: {n_scenes_used}/{n_scenes_total}\n")
        f.write(f"train_samples: {X_train.shape[0]}\n")
        f.write(f"val_samples: {X_val.shape[0]}\n")
        f.write(f"test_samples: {X_test.shape[0]}\n")
        f.write(f"feature_dim: {X_train.shape[1]}\n")
        f.write(f"features_spatial: {FEATURES_SPATIAL}\n")
        f.write(f"features_physical: {FEATURES_PHYSICAL}\n")
        f.write(f"command: {sys.executable} {' '.join(sys.argv)}\n")
    print(f"Saved config  -> {config_path}")

    print(f"\nDone. Results in {run_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF baseline on pre-split SEAN dataset")
    parser.add_argument("--data_root", type=str,
                        default="dataloaders/dataset_split",
                        help="Path to dataset_split/ directory")
    parser.add_argument("--train_pct", type=float, default=0.01,
                        help="Fraction of training scenes to use (0.01=1%%, 0.05=5%%, 1.0=all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for scene selection and RF")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Number of parallel jobs for RF (-1=all cores)")
    parser.add_argument("--experiments_root", type=str,
                        default="experiments/rf_sean",
                        help="Root directory for saving experiment results")
    args = parser.parse_args()

    run(args)
