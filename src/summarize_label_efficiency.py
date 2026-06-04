"""
Heterogeneous GNN-VAE pipeline — label-efficiency results summarizer.

summarize_label_efficiency.py

Reads label_eff_results.csv (raw 1200 rows) and produces:
  - Printed table (mean±std per metric × fraction)
  - summary_sean.csv and summary_snsb.csv (4×6 and 6×6 respectively)

Usage:
    python summarize_label_efficiency.py --csv label_eff_results.csv
"""

import argparse
import csv
import os
import numpy as np
from collections import defaultdict

SEAN_METRICS = [
    "f1_competence", "auc_competence",
    "f1_surprise",   "auc_surprise",
    "f1_intention",  "auc_intention",
    "f1_macro",      "auc_macro",
]
SNSB_METRICS = [
    "f1_avoid",        "prec_avoid",        "acc_avoid",
    "f1_follow",       "prec_follow",       "acc_follow",
    "f1_overtake",     "prec_overtake",     "acc_overtake",
    "f1_yield",        "prec_yield",        "acc_yield",
    "f1_not_consider", "prec_not_consider", "acc_not_consider",
    "f1_macro3", "f1_macro", "acc", "pa",
]


def load_csv(path):
    rows = []
    skip = {"vae_dir", "split_seed", "task"}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                parsed[k] = float(v) if k not in skip and v not in ("", None) else v
            rows.append(parsed)
    return rows


def summarize(rows, task, metrics, out_csv):
    task_rows = [r for r in rows if r["task"] == task]
    fractions  = sorted(set(r["fraction"] for r in task_rows))

    by_frac = defaultdict(list)
    for r in task_rows:
        by_frac[r["fraction"]].append(r)

    n_runs = len(by_frac[fractions[0]]) if fractions else 0
    print(f"\n{'='*70}")
    print(f"TASK: {task.upper()}   ({n_runs} runs/cell)")
    print(f"{'='*70}")

    frac_labels = []
    for frac in fractions:
        n_sc = int(np.mean([r["n_train_scenes"] for r in by_frac[frac]]))
        frac_labels.append(f"{frac:.0%}({n_sc}sc)")

    col_w = 15
    print(f"{'metric':<24}" + "".join(f"{l:>{col_w}}" for l in frac_labels))
    print("-" * (24 + col_w * len(fractions)))

    summary_rows = []
    for metric in metrics:
        row_str = f"{metric:<24}"
        summary_row = {"metric": metric}
        for frac in fractions:
            vals = [
                r[metric] for r in by_frac[frac]
                if r.get(metric, "") not in ("", None)
            ]
            mean, std = (
                (float(np.mean(vals)), float(np.std(vals)))
                if vals else (float("nan"), float("nan"))
            )
            row_str += f"{mean:.3f}±{std:.3f}".rjust(col_w)
            summary_row[f"{frac:.2f}_mean"] = round(mean, 4)
            summary_row[f"{frac:.2f}_std"]  = round(std,  4)
        print(row_str)
        summary_rows.append(summary_row)

    # write summary CSV
    fieldnames = ["metric"] + [f"{frac:.2f}_mean" for frac in fractions] + \
                              [f"{frac:.2f}_std"  for frac in fractions]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved → {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True)
    parser.add_argument("--task",    choices=["sean", "snsb", "both"], default="both")
    parser.add_argument("--out_dir", default=None, help="Dir for summary CSVs (default: same as --csv)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    if args.task in ("sean", "both"):
        summarize(rows, "sean", SEAN_METRICS,
                  os.path.join(out_dir, "summary_sean.csv"))
    if args.task in ("snsb", "both"):
        summarize(rows, "snsb", SNSB_METRICS,
                  os.path.join(out_dir, "summary_snsb.csv"))


if __name__ == "__main__":
    main()
