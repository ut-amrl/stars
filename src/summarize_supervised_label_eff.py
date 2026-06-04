"""
Summarize label_eff_supervised_snsb.csv → mean +- std per fraction.

Usage:
  python summarize_supervised_label_eff.py --csv label_eff_supervised_snsb.csv
"""

import argparse
import csv
import numpy as np
from collections import defaultdict

METRICS = [
    # scratch label-eff CSV uses oracle/ prefix
    "oracle/f1_macro", "oracle/f1_macro3", "oracle/acc", "oracle/pa",
    "oracle/f1_avoid", "oracle/f1_follow", "oracle/f1_overtake",
    "oracle/f1_yield", "oracle/f1_not_consider",
    # frozen-encoder LP CSV uses no prefix
    "f1_macro", "f1_macro3", "acc", "pa",
    "f1_avoid", "f1_follow", "f1_overtake", "f1_yield", "f1_not_consider",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    args = parser.parse_args()

    # fraction -> metric -> [values]
    data = defaultdict(lambda: defaultdict(list))
    n_rows = 0
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            frac = float(row.get("fraction") or row.get("label_fraction"))
            for m in args.metrics:
                if m in row and row[m] != "":
                    data[frac][m].append(float(row[m]))
            n_rows += 1

    # strip metrics prefix if present (oracle/f1_macro -> f1_macro)
    fractions = sorted(data.keys())
    n_ckpts   = len(set(open(args.csv).readlines()))  # rough
    print(f"Loaded {n_rows} rows  {len(fractions)} fractions\n")

    for metric in args.metrics:
        vals_exist = any(data[f][metric] for f in fractions)
        if not vals_exist:
            continue
        print(f"── {metric} ──")
        parts = []
        for frac in fractions:
            vals = data[frac][metric]
            parts.append(f"{frac:.0%}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")
        print("  " + "   ".join(parts))
        print()

if __name__ == "__main__":
    main()
