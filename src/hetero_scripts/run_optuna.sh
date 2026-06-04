#!/usr/bin/env bash
# run_optuna.sh — Heterogeneous GNN-VAE pipeline.
# Parallel Optuna hyperparameter search for the hetero pipeline.
#
#   combined -> optuna_unified_v1.py   (unified SEAN+SNSB VAE)
#   sean     -> optuna_search_v4.py    (SEAN-only v4 VAE)
#
# Launches ONE worker per GPU, all sharing the same sqlite --storage + study
# name; Optuna's RDB storage coordinates trials across the workers. Exactly one
# process per GPU.  Requires bash >= 4.3.
set -euo pipefail
cd "$(dirname "$0")/.."

PIPELINE=""
N_TRIALS=10            # trials PER worker (total = N_TRIALS * #gpus)
HIDDEN_DIM=256
LATENT_DIM=128
STUDY=""               # empty -> python default
STORAGE=""             # empty -> python default
FLIP_SURPRISE=1        # sean only
EXTRA=""

usage() {
  cat <<'EOF'
Usage: ./run_optuna.sh --pipeline combined|sean [options]

  --pipeline       combined | sean                       [required]
  --n-trials N     trials PER worker; total = N * #gpus  (default: 10)
  --hidden-dim N    fixed VAE hidden dimension      (default: 256)
  --latent-dim N    fixed per-node latent dimension (default: 128)
  --study NAME     Optuna study name               (default: script's own)
  --storage URL    Optuna storage url              (default: script's own)
  --no-flip-surprise   sean only: disable --flip_surprise
  --extra "..."    extra args passed verbatim
  -h | --help

GPU pool comes from CUDA_VISIBLE_DEVICES, e.g.
  CUDA_VISIBLE_DEVICES=5,6,7 ./run_optuna.sh --pipeline combined
If CUDA_VISIBLE_DEVICES is unset, defaults to 0,1,2,3,4,5,6,7.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --pipeline)          PIPELINE="$2"; shift 2 ;;
    --n-trials)          N_TRIALS="$2"; shift 2 ;;
    --hidden-dim)        HIDDEN_DIM="$2"; shift 2 ;;
    --latent-dim)        LATENT_DIM="$2"; shift 2 ;;
    --study)             STUDY="$2"; shift 2 ;;
    --storage)           STORAGE="$2"; shift 2 ;;
    --no-flip-surprise)  FLIP_SURPRISE=0; shift ;;
    --extra)             EXTRA="$2"; shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ "$PIPELINE" = combined ] || [ "$PIPELINE" = sean ] || {
  echo "ERROR: --pipeline must be 'combined' or 'sean'" >&2; usage; exit 1; }

if [ "$PIPELINE" = combined ]; then
  BASE=( optuna_unified_v1.py --mode search --n_trials "$N_TRIALS"
         --hidden_dim "$HIDDEN_DIM" --latent_dim "$LATENT_DIM" )
else
  BASE=( optuna_search_v4.py  --mode search --n_trials "$N_TRIALS"
         --hidden_dim "$HIDDEN_DIM" --latent_dim "$LATENT_DIM" )
  [ "$FLIP_SURPRISE" = 1 ] && BASE+=( --flip_surprise )
fi
[ -n "$STUDY" ]   && BASE+=( --study_name "$STUDY" )
[ -n "$STORAGE" ] && BASE+=( --storage "$STORAGE" )
read -ra EXTRA_ARR <<< "$EXTRA"

mkdir -p logs
GPU_POOL="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_POOL="${GPU_POOL//,/ }"
read -ra GPU_ARR <<< "$GPU_POOL"
[ ${#GPU_ARR[@]} -gt 0 ] || { echo "ERROR: no GPUs given" >&2; exit 1; }

echo "pipeline=$PIPELINE  workers=${#GPU_ARR[@]}  trials/worker=$N_TRIALS" \
     " hidden_dim=$HIDDEN_DIM latent_dim=$LATENT_DIM" \
     " total=$(( N_TRIALS * ${#GPU_ARR[@]} ))  CUDA_VISIBLE_DEVICES=(${GPU_ARR[*]})"

# one worker per GPU — exactly one process per GPU
for gpu in "${GPU_ARR[@]}"; do
  log="logs/optuna_${PIPELINE}_gpu${gpu}.log"
  echo "[worker] GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "${BASE[@]}" "${EXTRA_ARR[@]}" > "$log" 2>&1 &
done
wait
echo "Done: Optuna search complete ($PIPELINE). Run with --mode report for the best trial."
