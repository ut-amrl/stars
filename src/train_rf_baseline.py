"""
RF LOOCV baseline — mirrors random_forest_loocv.py from navigation_foundation.

For each of 60 participants as subject_under_test:
  train = all samples from the other 59 participants concatenated
  test  = all samples from that participant

Per fold per label: binary F1 (gt>2, pt>2) + per-sample gt and pt.
Final summary:      mean ± SD of binary F1 across 60 folds per label.
"""

import os
import datetime
import numpy as np
import h5py
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

H5_PATH = "dataloaders/sean_together_v3.h5"
FEATURES = ['eye', 'follower', 'gaze', 'goal', 'lip', 'map_resnet18', 'nearby']
LABELS      = [0, 1, 2]
LABEL_NAMES = ['competence', 'surprise', 'intention']
PARALLELISM = 5


def load_participant_data(hf, participant_id, features):
    """Load all samples for one participant. Returns X [N, D], y [N, 3]."""
    X_list, y_list = [], []
    for sample_key in hf[participant_id].keys():
        grp = hf[participant_id][sample_key]
        x = np.concatenate([grp[f"feat_{f}"][:] for f in features])
        y = grp["label"][:]
        X_list.append(x)
        y_list.append(y)
    return np.stack(X_list), np.stack(y_list)


def load_train_test(hf, subject_under_test, features):
    """Mirror of load_train_test() in random_forest_loocv.py."""
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    for participant_id in hf.keys():
        X, y = load_participant_data(hf, participant_id, features)
        if participant_id == subject_under_test:
            X_test.append(X)
            y_test.append(y)
        else:
            X_train.append(X)
            y_train.append(y)

    if not X_train:
        raise RuntimeError(f"no train data for subject_under_test={subject_under_test}")
    if not X_test:
        raise RuntimeError(f"no test data for subject_under_test={subject_under_test}")

    return (
        np.concatenate(X_train),  # [N_train, D]
        np.concatenate(y_train),  # [N_train, 3]
        np.concatenate(X_test),   # [N_test, D]
        np.concatenate(y_test),   # [N_test, 3]
    )


def run_one_fold(subject_under_test, output_dir):
    """Train and evaluate one LOOCV fold. Mirrors by_featureset() in random_forest_loocv.py."""
    with h5py.File(H5_PATH, "r") as hf:
        X_train, y_train, X_test, y_test = load_train_test(hf, subject_under_test, FEATURES)

    print(f"Held out {subject_under_test}: train={X_train.shape[0]}  test={X_test.shape[0]}")
    clf = RandomForestClassifier(class_weight='balanced')

    fold_results = {}
    for lbl in LABELS:
        out_path = os.path.join(output_dir, f"held-{subject_under_test}_cls-{lbl}.npy")
        if os.path.exists(out_path):
            fold_results[lbl] = np.load(out_path, allow_pickle=True).item()
            continue

        clf.fit(X_train, y_train[:, lbl])
        pt = clf.predict(X_test)
        gt = y_test[:, lbl]

        # surprise is inverted: low score (<3) is the positive class
        if lbl == 1:
            bin_f1 = f1_score(gt < 3, pt < 3, zero_division=0)
        else:
            bin_f1 = f1_score(gt > 2, pt > 2, zero_division=0)

        results = {
            'subject_under_test': subject_under_test,
            'label_name':         LABEL_NAMES[lbl],
            'gt':                 gt,
            'pt':                 pt,
            'bin_f1':             bin_f1,
        }
        np.save(out_path, results)
        print(f"  {LABEL_NAMES[lbl]}: bin_f1={results['bin_f1']:.3f}  n_test={gt.shape[0]}")
        fold_results[lbl] = results

    return fold_results


def print_summary(all_fold_results, participants):
    print(f"\n=== LOOCV Summary ({len(participants)} folds × {len(LABELS)} labels = {len(participants)*len(LABELS)} total) ===")
    for lbl in LABELS:
        bin_f1s = [all_fold_results[p][lbl]['bin_f1'] for p in participants]
        print(f"  {LABEL_NAMES[lbl]}: binary F1 = {np.mean(bin_f1s):.3f} ± {np.std(bin_f1s):.3f}  (n={len(bin_f1s)} folds)")


if __name__ == "__main__":
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/rf-loocv-{ts}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")

    with h5py.File(H5_PATH, "r") as hf:
        participants = list(hf.keys())
    print(f"Loaded {len(participants)} participants from {H5_PATH}")

    fold_results_list = Parallel(n_jobs=PARALLELISM)(
        delayed(run_one_fold)(p, output_dir) for p in participants
    )

    all_fold_results = {p: r for p, r in zip(participants, fold_results_list)}
    print_summary(all_fold_results, participants)
