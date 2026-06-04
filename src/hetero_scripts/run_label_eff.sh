#!/usr/bin/env bash
# run_label_eff.sh — Heterogeneous GNN-VAE pipeline.
# Label-efficiency LP sweep over N frozen VAEs.
#
#   combined -> eval_label_efficiency.py        (SEAN + SNSB downstream)
#   sean     -> eval_sean_label_efficiency_v4.py (SEAN downstream only)
#
# Each VAE -> 6 fractions x 10 unseeded run-units. For combined that is
# 2 tasks => 120 rows/VAE; for sean 60 rows/VAE. Over 10 VAEs:
#   combined: 600 SEAN LPs + 600 SNSB LPs   sean: 600 SEAN LPs.
#
# Writes one CSV per VAE, verifies line counts, reruns stragglers.
# At most one eval process per GPU (wait -n pool). Requires bash >= 4.3.
set -euo pipefail
cd "$(dirname "$0")/.."

PIPELINE=""
N_VAES=10
VAE_ROOT=""
VAE_PREFIX="run"
OUT_PREFIX="label_eff_results"
FLIP_SURPRISE=1        # sean only
EXTRA=""

usage() {
  cat <<'EOF'
Usage: ./run_label_eff.sh --pipeline combined|sean [options]

  --pipeline       combined | sean                       [required]
  --n-vaes N       number of VAEs                  (default: 10)
  --vae-root DIR   VAE checkpoint root; per-VAE dir = DIR/<prefix><i>
                   default: combined -> checkpoints/v3_hetero_optuned128
                            sean     -> checkpoints/v4_sean_optuned128
  --vae-prefix S   per-VAE subdir prefix                 (default: run)
  --out-prefix S   per-VAE CSV prefix -> <prefix>_<vae>.csv (default: label_eff_results)
  --no-flip-surprise   sean only: disable --flip_surprise
  --extra "..."    extra args passed verbatim to the eval
  -h | --help

GPU pool comes from CUDA_VISIBLE_DEVICES, e.g.
  CUDA_VISIBLE_DEVICES=5,6,7 ./run_label_eff.sh --pipeline combined
If CUDA_VISIBLE_DEVICES is unset, defaults to 0,1,2,3,4,5,6,7.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --pipeline)         PIPELINE="$2"; shift 2 ;;
    --n-vaes)           N_VAES="$2"; shift 2 ;;
    --vae-root)         VAE_ROOT="$2"; shift 2 ;;
    --vae-prefix)       VAE_PREFIX="$2"; shift 2 ;;
    --out-prefix)       OUT_PREFIX="$2"; shift 2 ;;
    --no-flip-surprise) FLIP_SURPRISE=0; shift ;;
    --extra)            EXTRA="$2"; shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ "$PIPELINE" = combined ] || [ "$PIPELINE" = sean ] || {
  echo "ERROR: --pipeline must be 'combined' or 'sean'" >&2; usage; exit 1; }

if [ "$PIPELINE" = combined ]; then
  : "${VAE_ROOT:=checkpoints/v3_hetero_optuned128}"
  EXPECT_LINES=121          # 1 header + 120 rows
else
  : "${VAE_ROOT:=checkpoints/v4_sean_optuned128}"
  EXPECT_LINES=61           # 1 header + 60 rows
fi
read -ra EXTRA_ARR <<< "$EXTRA"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if command -v nvidia-smi &> /dev/null; then
    num_gpus=$(nvidia-smi -L | wc -l)
  else
    num_gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
  fi
  [ "$num_gpus" -ge 1 ] || num_gpus=1
  GPU_POOL=$(seq -s, 0 $((num_gpus - 1)))
else
  GPU_POOL="$CUDA_VISIBLE_DEVICES"
fi
GPU_POOL="${GPU_POOL//,/ }"
read -ra GPU_ARR <<< "$GPU_POOL"
[ ${#GPU_ARR[@]} -gt 0 ] || { echo "ERROR: no GPUs given" >&2; exit 1; }
mkdir -p logs

build_cmd() {   # $1 = vae index -> sets global CMD array
  local i="$1" dir="$VAE_ROOT/${VAE_PREFIX}$1" out="${OUT_PREFIX}_${VAE_PREFIX}$1.csv"
  if [ "$PIPELINE" = combined ]; then
    CMD=( eval_label_efficiency.py --vae_dirs "$dir" --output "$out" --gpu 0 )
  else
    CMD=( eval_sean_label_efficiency_v4.py --vae_dirs "$dir" --output "$out" --gpu 0 )
    [ "$FLIP_SURPRISE" = 1 ] && CMD+=( --flip_surprise )
  fi
  CMD+=( "${EXTRA_ARR[@]}" )
}

pool_run() {    # args: list of vae indices — runs them, <=1 process per GPU
  declare -A PID2GPU
  local FREE=("${GPU_ARR[@]}") i gpu log pid
  for i in "$@"; do
    while [ ${#FREE[@]} -eq 0 ]; do
      wait -n || true
      for pid in "${!PID2GPU[@]}"; do
        kill -0 "$pid" 2>/dev/null || { FREE+=("${PID2GPU[$pid]}"); unset 'PID2GPU[$pid]'; }
      done
    done
    gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
    build_cmd "$i"
    log="logs/eval_${PIPELINE}_${VAE_PREFIX}${i}.log"
    echo "[eval $i] GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u "${CMD[@]}" > "$log" 2>&1 &
    PID2GPU[$!]="$gpu"
  done
  wait || true
}

echo "pipeline=$PIPELINE  n_vaes=$N_VAES  vae_root=$VAE_ROOT  CUDA_VISIBLE_DEVICES=(${GPU_ARR[*]})"

# eval scripts APPEND to their CSV — clear stale files first
rm -f "${OUT_PREFIX}"_*.csv

ALL=(); for (( i=0; i<N_VAES; i++ )); do ALL+=("$i"); done
pool_run "${ALL[@]}"

# verify per-VAE CSVs and rerun any short/missing ones
BAD=()
for (( i=0; i<N_VAES; i++ )); do
  csv="${OUT_PREFIX}_${VAE_PREFIX}$i.csv"
  lines=$(wc -l < "$csv" 2>/dev/null || echo 0)
  if [ "$lines" -ne "$EXPECT_LINES" ]; then
    echo "  $csv: $lines lines (expect $EXPECT_LINES) — rerunning"
    rm -f "$csv"; BAD+=("$i")
  fi
done
if [ ${#BAD[@]} -gt 0 ]; then
  echo "Rerunning stragglers: ${BAD[*]}"
  pool_run "${BAD[@]}"
fi

echo "--- per-VAE CSV line counts ---"
wc -l "${OUT_PREFIX}"_*.csv
echo "Done. Next: ./run_summarize.sh --pipeline $PIPELINE --in-prefix $OUT_PREFIX"
