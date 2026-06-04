#!/usr/bin/env python3
"""
Sweep runner: train_rf_sean.py across 10 splits × 6 train percentages × 10 seeds.

Saves all results to /data/experiments/rf_sean_sweep/ with a structured layout:
  rf_sean_sweep/
    s{1..10}/
      pct{1,5,10,25,50,100}/
        run{0..9}/
          metrics.csv
          run_config.txt
    aggregated_results.csv   (one row per split×pct×seed)
    summary_table.csv        (mean±std F1 per pct, averaged over splits and seeds)

Usage (inside container):
  python3 src/run_rf_sean_sweep.py
"""

import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

# Ensure src/ is importable
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from train_rf_sean import run as run_rf_sean

SPLITS_ROOT     = "/data/dataset_splits"
EXPERIMENTS_OUT = "/data/experiments/rf_sean_sweep"
N_SPLITS        = 10          # s1..s10
N_REPETITIONS   = 10          # 10 random seeds per (split, pct)
TRAIN_PCTS      = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
N_JOBS          = 4
LABEL_NAMES     = ["competence", "surprise", "intention"]


def main():
    os.makedirs(EXPERIMENTS_OUT, exist_ok=True)

    all_rows = []           # one dict per run for the aggregated CSV
    total_runs = N_SPLITS * len(TRAIN_PCTS) * N_REPETITIONS
    run_count = 0
    t0 = time.time()

    for split_num in range(1, N_SPLITS + 1):
        data_root = os.path.join(SPLITS_ROOT, f"s{split_num}")
        if not os.path.isdir(data_root):
            print(f"WARNING: {data_root} not found, skipping")
            continue

        for pct in TRAIN_PCTS:
            pct_tag = f"pct{int(pct * 100)}"

            for rep in range(N_REPETITIONS):
                run_count += 1
                # Use a different seed for each repetition.
                # Encoding: split_num * 1000 + rep  keeps seeds unique & reproducible.
                seed = split_num * 1000 + rep

                run_dir = os.path.join(
                    EXPERIMENTS_OUT, f"s{split_num}", pct_tag, f"run{rep}"
                )
                os.makedirs(run_dir, exist_ok=True)

                elapsed = time.time() - t0
                eta = (elapsed / run_count) * (total_runs - run_count) if run_count else 0
                print(
                    f"\n{'='*70}\n"
                    f"[{run_count}/{total_runs}] "
                    f"split=s{split_num}  pct={pct*100:.0f}%  rep={rep}  seed={seed}  "
                    f"ETA={eta/60:.1f}min\n"
                    f"{'='*70}"
                )

                args = SimpleNamespace(
                    data_root=data_root,
                    train_pct=pct,
                    seed=seed,
                    n_jobs=N_JOBS,
                    experiments_root=run_dir,   # run() creates a timestamped subdir
                )

                try:
                    results = run_rf_sean(args)
                except Exception as e:
                    print(f"ERROR: {e}")
                    results = {n: {"val_f1": float("nan"), "test_f1": float("nan")}
                               for n in LABEL_NAMES}

                row = {
                    "split": f"s{split_num}",
                    "train_pct": pct,
                    "rep": rep,
                    "seed": seed,
                }
                for name in LABEL_NAMES:
                    row[f"val_f1_{name}"]  = results[name]["val_f1"]
                    row[f"test_f1_{name}"] = results[name]["test_f1"]
                all_rows.append(row)

    # ── Save aggregated CSV ────────────────────────────────────────────────────
    agg_path = os.path.join(EXPERIMENTS_OUT, "aggregated_results.csv")
    fieldnames = (
        ["split", "train_pct", "rep", "seed"]
        + [f"val_f1_{n}"  for n in LABEL_NAMES]
        + [f"test_f1_{n}" for n in LABEL_NAMES]
    )
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nAggregated results -> {agg_path}")

    # ── Build summary table ────────────────────────────────────────────────────
    import numpy as np

    summary_rows = []
    for pct in TRAIN_PCTS:
        matching = [r for r in all_rows if r["train_pct"] == pct]
        n_scenes = max(1, math.ceil(48 * pct))
        row = {"train_pct": f"{int(pct*100)}% ({n_scenes} Scenes)"}
        for name in LABEL_NAMES:
            vals = [r[f"test_f1_{name}"] for r in matching
                    if not (isinstance(r[f"test_f1_{name}"], float)
                            and math.isnan(r[f"test_f1_{name}"]))]
            if vals:
                mean = np.mean(vals)
                std  = np.std(vals)
                row[f"{name}_mean"] = round(mean, 3)
                row[f"{name}_std"]  = round(std, 3)
                row[name] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[f"{name}_mean"] = float("nan")
                row[f"{name}_std"]  = float("nan")
                row[name] = "N/A"
        summary_rows.append(row)

    summary_path = os.path.join(EXPERIMENTS_OUT, "summary_table.csv")
    summary_fields = (
        ["train_pct"]
        + LABEL_NAMES
        + [f"{n}_mean" for n in LABEL_NAMES]
        + [f"{n}_std" for n in LABEL_NAMES]
    )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary table   -> {summary_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("F1 SCORES ON TEST (mean ± std over 10 splits × 10 seeds = 100 runs per pct)")
    print("=" * 80)
    header = f"{'train_pct':>22s}"
    for name in LABEL_NAMES:
        header += f"  {name:>20s}"
    print(header)
    for row in summary_rows:
        line = f"{row['train_pct']:>22s}"
        for name in LABEL_NAMES:
            line += f"  {row[name]:>20s}"
        print(line)

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_runs} runs)")


if __name__ == "__main__":
    main()
