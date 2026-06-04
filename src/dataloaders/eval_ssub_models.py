"""
Evaluate LLM model predictions on the SSUB test set.

Metrics per model (gemini, gpt4o, o4-mini):
  - F1 per action class (binary one-vs-rest, using majority ground truth label)
  - Macro-averaged F1 across all classes
  - PA (Prediction Accuracy from the paper):
      PA = (1/NQ) * sum_q [ (1/NH) * sum_i I[Aq == Aq_h,i] ]
    which simplifies to: mean over all (episode, person) pairs of
      ground_truth_action_probabilities[person][predicted_label]

Only (episode, person) pairs where ground truth exists are included.
Extra model-only person IDs (hallucinated) are skipped.
"""

import json
import os
from collections import defaultdict

from sklearn.metrics import f1_score

SSUB_TEST_DIR = os.path.join(os.path.dirname(__file__), "ssub_test")
MODEL_FOLDERS = ["gemini", "gpt4o", "o4-mini"]
ACTION_CLASSES = ["avoid", "follow", "overtake", "yield", "not_consider"]


def collect_samples(model):
    """Return lists of (true_label, pred_label, gt_probs, pred_probs) per person."""
    model_dir = os.path.join(SSUB_TEST_DIR, model)
    true_labels, pred_labels, pa_scores = [], [], []

    for scene in sorted(os.listdir(model_dir)):
        scene_dir = os.path.join(model_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        for ep_file in sorted(os.listdir(scene_dir)):
            if not ep_file.endswith(".json"):
                continue
            with open(os.path.join(scene_dir, ep_file)) as f:
                data = json.load(f)

            gt_probs = data.get("ground_truth_action_probabilities", {})
            gt_labels = data.get("ground_truth_action_labels", {})
            pred_label_map = data.get("people_action_labels", {})

            for pid in gt_labels:
                if pid not in pred_label_map:
                    continue
                true_labels.append(gt_labels[pid])
                pred = pred_label_map[pid]
                pred_labels.append(pred)
                # PA contribution: fraction of annotators who agree with prediction
                pa_scores.append(gt_probs[pid].get(pred, 0.0))

    return true_labels, pred_labels, pa_scores


def evaluate(model):
    true_labels, pred_labels, pa_scores = collect_samples(model)
    n = len(true_labels)

    # F1 per class (binary one-vs-rest)
    f1_per_class = {}
    for cls in ACTION_CLASSES:
        y_true = [1 if t == cls else 0 for t in true_labels]
        y_pred = [1 if p == cls else 0 for p in pred_labels]
        f1_per_class[cls] = f1_score(y_true, y_pred, zero_division=0)

    pa = sum(pa_scores) / n if n > 0 else 0.0

    return {
        "n_samples": n,
        "f1_per_class": f1_per_class,
        "pa": pa,
    }


def print_gt_distribution():
    # Read ground truth directly from human_oracle — independent of any model
    oracle_dir = os.path.join(SSUB_TEST_DIR, "human_oracle")
    true_labels = []
    for scene in sorted(os.listdir(oracle_dir)):
        scene_dir = os.path.join(oracle_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        for ep_file in sorted(os.listdir(scene_dir)):
            if not ep_file.endswith(".json"):
                continue
            with open(os.path.join(oracle_dir, scene, ep_file)) as f:
                data = json.load(f)
            for label in data["people_action_labels"].values():
                true_labels.append(label)
    n = len(true_labels)
    counts = {cls: true_labels.count(cls) for cls in ACTION_CLASSES}

    col_w = 16
    print(f"\nGround truth label distribution (N={n}):")
    print(f"  {'Class':<{col_w}}", end="")
    for cls in ACTION_CLASSES:
        print(f"{cls:>{col_w}}", end="")
    print()
    print(f"  {'Count':<{col_w}}", end="")
    for cls in ACTION_CLASSES:
        print(f"{counts[cls]:>{col_w}}", end="")
    print()
    print(f"  {'%':<{col_w}}", end="")
    for cls in ACTION_CLASSES:
        print(f"{counts[cls]/n*100:>{col_w}.1f}", end="")
    print()


def main():
    print_gt_distribution()

    col_w = 16
    print(f"\n{'Model':<{col_w}}", end="")
    for cls in ACTION_CLASSES:
        print(f"{cls+' F1':>{col_w}}", end="")
    print(f"{'PA':>{col_w}}{'N':>{col_w}}")
    print("-" * (col_w * (len(ACTION_CLASSES) + 3)))

    for model in MODEL_FOLDERS:
        res = evaluate(model)
        print(f"{model:<{col_w}}", end="")
        for cls in ACTION_CLASSES:
            print(f"{res['f1_per_class'][cls]:>{col_w}.4f}", end="")
        print(f"{res['pa']:>{col_w}.4f}{res['n_samples']:>{col_w}}")

    print()


if __name__ == "__main__":
    main()
