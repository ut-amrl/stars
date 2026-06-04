#!/usr/bin/env bash
# run_summarize.sh — Heterogeneous GNN-VAE pipeline.
# Merge the per-VAE label-efficiency CSVs into one raw CSV
# and produce the mean+-std summary tables.
#
#   merged raw CSV -> summarize_label_efficiency.py -> summary_sean.csv / summary_snsb.csv
#
# Expected merged size:  combined 1201 lines (1 + 10*120)   sean 601 (1 + 10*60).
set -euo pipefail
cd "$(dirname "$0")/.."

PIPELINE=""
IN_PREFIX="label_eff_results"
VAE_PREFIX="run"
N_VAES=10
MERGED=""
TASK=""

usage() {
  cat <<'EOF'
Usage: ./run_summarize.sh --pipeline combined|sean [options]

  --pipeline     combined | sean                         [required]
  --in-prefix S  per-VAE CSV prefix (<prefix>_<vae>.csv)  (default: label_eff_results)
  --vae-prefix S per-VAE token in the CSV name            (default: run)
  --n-vaes N     number of per-VAE CSVs                   (default: 10)
  --merged FILE  merged raw CSV path             (default: <in-prefix>.csv)
  --task T       sean | snsb | both             (default: combined->both, sean->sean)
  -h | --help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --pipeline)   PIPELINE="$2"; shift 2 ;;
    --in-prefix)  IN_PREFIX="$2"; shift 2 ;;
    --vae-prefix) VAE_PREFIX="$2"; shift 2 ;;
    --n-vaes)     N_VAES="$2"; shift 2 ;;
    --merged)     MERGED="$2"; shift 2 ;;
    --task)       TASK="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ "$PIPELINE" = combined ] || [ "$PIPELINE" = sean ] || {
  echo "ERROR: --pipeline must be 'combined' or 'sean'" >&2; usage; exit 1; }

: "${MERGED:=${IN_PREFIX}.csv}"
if [ "$PIPELINE" = combined ]; then
  : "${TASK:=both}"; EXPECT_LINES=$(( 1 + N_VAES * 120 ))
else
  : "${TASK:=sean}"; EXPECT_LINES=$(( 1 + N_VAES * 60 ))
fi

# collect the per-VAE CSVs in index order
CSVS=()
for (( i=0; i<N_VAES; i++ )); do
  f="${IN_PREFIX}_${VAE_PREFIX}$i.csv"
  [ -s "$f" ] || { echo "ERROR: missing/empty $f — run run_label_eff.sh first" >&2; exit 1; }
  CSVS+=("$f")
done

# merge: header from the first file, data rows from all
head -1 "${CSVS[0]}" > "$MERGED"
for f in "${CSVS[@]}"; do
  tail -n +2 "$f" >> "$MERGED"
done

lines=$(wc -l < "$MERGED")
echo "merged ${#CSVS[@]} CSVs -> $MERGED  ($lines lines, expect $EXPECT_LINES)"
[ "$lines" -eq "$EXPECT_LINES" ] || echo "WARNING: line count != expected — check for short per-VAE CSVs"

python -u summarize_label_efficiency.py --csv "$MERGED" --task "$TASK"
echo "Done: wrote summary_sean.csv / summary_snsb.csv (task=$TASK)"
