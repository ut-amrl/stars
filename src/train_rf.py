"""
Train Random Forest classifiers on Nav-Only features to predict
follower's perceived robot performance (competence, surprise, intention).

Nav features (3,434D flattened):
  spatial.follower         [40, 4]     → 160   follower pose in robot frame
  spatial.nearby           [40, 16, 4] → 2560  bystander poses in robot frame
  spatial.robot            [40, 2]     → 80    robot orientation (cos, sin)
  spatial.goal             [40, 2]     → 80    goal position all timesteps in robot frame
  map_resnet18 (mean)      [512]       → 512   mean occupancy embedding
  viz.gaze_direction_local [40, 3]     → 120   follower gaze direction
  Total: 3,434

3 RF classifiers (competence, surprise, intention), Likert 1-5 (5-class).
RF params: 100 trees, min_samples_leaf=2 (per paper).

Output (saved to checkpoints/):
  rf_models.pkl        3 trained sklearn RF models
  rf_output.npy     [N, 15] predict_proba outputs for ALL samples in HDF5 order
                       → written into follower node features (15D) in the GNN

Usage:
  cd src/
  python train_rf.py
  python train_rf.py --sean_root dataloaders/SEAN_TOGETHER_Dataset/SEAN_TOGETHER
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

LABEL_NAMES = ["competence", "surprise", "intention"]
N_CLASSES   = 5   # Likert 1-5


def extract_nav_features(pt_path):
    """Extract 3,434D Nav-Only feature vector from a single .pt file."""
    x = torch.load(str(pt_path), weights_only=False)

    follower = np.array(x["spatial"]["follower"]).astype(np.float32)          # [40, 4]
    nearby   = np.array(x["spatial"]["nearby"]).astype(np.float32)            # [40, 16, 4]
    robot    = np.array(x["spatial"]["robot"]).astype(np.float32)             # [40, 2]
    goal     = np.array(x["spatial"]["goal"]).astype(np.float32)              # [40, 2]
    map_r18  = np.array(x["physical"]["map_resnet18"]).astype(np.float32)     # [40, 512]
    gaze     = np.array(x["viz"]["gaze_direction_local"]).astype(np.float32)  # [40, 3]

    feat = np.concatenate([
        follower.flatten(),      # 160
        nearby.flatten(),        # 2560
        robot.flatten(),         # 80
        goal.flatten(),          # 80  (all 40 timesteps × 2)
        map_r18.mean(axis=0),    # 512
        gaze.flatten(),          # 120
    ])                           # 3434 total
    return feat


def load_dataset(sean_root):
    """
    Iterate .pt files in the same order as sean_to_hdf5_v2.py.
    Returns:
      all_feats  : [N_all, 3434]  features for every .pt file
      all_labels : [N_labeled, 3] labels (only samples with have_label=True)
      label_mask : [N_all]        bool, True where sample has a label
      pt_paths   : [N_all]        Path objects (for debugging)
    """
    sean_root = Path(sean_root)
    folders   = sorted([p for p in sean_root.iterdir() if p.is_dir()])

    all_feats  = []
    all_labels = []
    label_mask = []
    pt_paths   = []

    for folder in folders:
        pt_files = sorted(
            folder.glob("temporal_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        for pt_file in pt_files:
            try:
                x    = torch.load(str(pt_file), weights_only=False)
                feat = extract_nav_features(pt_file)
                all_feats.append(feat)
                pt_paths.append(pt_file)

                # have_label is always False in this dataset — use true_label directly
                all_labels.append(np.array(x["true_label"], dtype=np.int64))
                label_mask.append(True)
            except Exception as e:
                print(f"  SKIP {pt_file.name}: {e}")

    return (
        np.stack(all_feats),           # [N_all, 3434]
        np.stack(all_labels),          # [N_labeled, 3]
        np.array(label_mask, dtype=bool),
        pt_paths,
    )


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading dataset from {args.sean_root} ...")
    all_feats, labeled_labels, label_mask, pt_paths = load_dataset(args.sean_root)

    N_all     = all_feats.shape[0]
    N_labeled = label_mask.sum()
    print(f"Total samples: {N_all}  |  Labeled: {N_labeled}")

    # ---- Split labeled samples into train/val/test ----
    labeled_feats = all_feats[label_mask]   # [N_labeled, 3434]
    rng           = np.random.default_rng(42)
    idx           = rng.permutation(N_labeled)

    n_test  = int(0.1 * N_labeled)
    n_val   = int(0.1 * N_labeled)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    X_train, y_train = labeled_feats[train_idx], labeled_labels[train_idx]
    X_val,   y_val   = labeled_feats[val_idx],   labeled_labels[val_idx]
    X_test,  y_test  = labeled_feats[test_idx],  labeled_labels[test_idx]
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    # ---- Train one RF per label dimension ----
    rf_models = []
    for dim, name in enumerate(LABEL_NAMES):
        print(f"\nTraining RF for {name} ...")
        rf = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train[:, dim])
        rf_models.append(rf)

        for split_name, X, y in [("val", X_val, y_val[:, dim]), ("test", X_test, y_test[:, dim])]:
            preds = rf.predict(X)
            acc   = accuracy_score(y, preds)
            f1    = f1_score(y, preds, average="macro", zero_division=0)
            print(f"  {split_name}: acc={acc:.3f}  macro-f1={f1:.3f}")

    # ---- Save models ----
    models_path = os.path.join(args.checkpoint_dir, "rf_models.pkl")
    with open(models_path, "wb") as f:
        pickle.dump(rf_models, f)
    print(f"\nSaved RF models → {models_path}")

    # ---- Generate probabilities for ALL samples (in HDF5 order) ----
    # Shape: [N_all, 15] = concat of 3 × [N_all, 5] predict_proba outputs
    output_parts = []
    for rf in rf_models:
        class_scores = rf.predict_proba(all_feats)   # [N_all, n_classes_seen]
        # Ensure full 5-class output even if some classes missing in training data
        class_output = np.zeros((N_all, N_CLASSES), dtype=np.float32)
        for col_idx, cls in enumerate(rf.classes_):
            class_output[:, int(cls) - 1] = class_scores[:, col_idx]   # cls is 1-indexed
        output_parts.append(class_output)

    rf_output = np.concatenate(output_parts, axis=1)   # [N_all, 15]
    output_path   = os.path.join(args.checkpoint_dir, "rf_output.npy")
    np.save(output_path, rf_output)
    print(f"Saved RF probabilities → {output_path}  shape: {rf_output.shape}")
    print("These are the follower node features (15D) for the GNN.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sean_root",      default="dataloaders/SEAN_TOGETHER_Dataset/SEAN_TOGETHER")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()
    train(args)
