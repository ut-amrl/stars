"""
Add ground truth labels from human_oracle to model prediction files.

For each model folder (gemini, gpt4o, o4-mini), this script:
1. Loads each episode JSON and the corresponding human_oracle JSON
2. Verifies robot_trajectory and people_trajectories match exactly
3. Adds ground truth fields from human_oracle:
   - ground_truth_action_probabilities
   - ground_truth_action_labels
4. Overwrites the model JSON in-place with the combined data
"""

import json
import os
import sys

SSUB_TEST_DIR = os.path.join(os.path.dirname(__file__), "ssub_test")
MODEL_FOLDERS = ["gemini", "gpt4o", "o4-mini"]
ORACLE_FOLDER = "human_oracle"


def trajectories_match(model_data, oracle_data):
    if model_data["robot_trajectory"] != oracle_data["robot_trajectory"]:
        return False
    # Model may have extra person IDs (hallucinated); only verify oracle IDs match
    for pid, traj in oracle_data["people_trajectories"].items():
        if model_data["people_trajectories"].get(pid) != traj:
            return False
    return True


def process_file(model_path, oracle_path):
    with open(model_path) as f:
        model_data = json.load(f)
    with open(oracle_path) as f:
        oracle_data = json.load(f)

    if not trajectories_match(model_data, oracle_data):
        return False, "trajectory mismatch"

    model_data["ground_truth_action_probabilities"] = oracle_data["people_action_probabilities"]
    model_data["ground_truth_action_labels"] = oracle_data["people_action_labels"]

    with open(model_path, "w") as f:
        json.dump(model_data, f)

    return True, None


def main():
    total = 0
    updated = 0
    mismatches = []

    for model in MODEL_FOLDERS:
        model_dir = os.path.join(SSUB_TEST_DIR, model)
        for scene in sorted(os.listdir(model_dir)):
            scene_dir = os.path.join(model_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            for ep_file in sorted(os.listdir(scene_dir)):
                if not ep_file.endswith(".json"):
                    continue
                total += 1
                model_path = os.path.join(scene_dir, ep_file)
                oracle_path = os.path.join(SSUB_TEST_DIR, ORACLE_FOLDER, scene, ep_file)

                if not os.path.exists(oracle_path):
                    mismatches.append(f"MISSING oracle: {model}/{scene}/{ep_file}")
                    continue

                ok, err = process_file(model_path, oracle_path)
                if ok:
                    updated += 1
                else:
                    mismatches.append(f"{err}: {model}/{scene}/{ep_file}")

    print(f"Processed {total} files, updated {updated}")
    if mismatches:
        print(f"\n{len(mismatches)} issues:")
        for m in mismatches:
            print(f"  - {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
