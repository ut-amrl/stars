#!/usr/bin/env bash
# run_optuna_supervised.sh — Optuna search for supervised SNSB ablation.
#
# Runs one worker per GPU in parallel, sharing a single SQLite study.
# Each worker contributes --n_trials trials.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 ./hetero_scripts/run_optuna_supervised.sh
#   GPUS=4,5,6,7 N_TRIALS=10 STUDY=my_study ./hetero_scripts/run_optuna_supervised.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-0}"
N_TRIALS="${N_TRIALS:-10}"
STUDY="${STUDY:-snsb_supervised_v1}"
STORAGE="${STORAGE:-sqlite:///optuna_snsb_supervised_v1.db}"
EPOCHS="${EPOCHS:-1000}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LATENT_DIM="${LATENT_DIM:-128}"
EXTRA="${EXTRA:-}"

mkdir -p logs

IFS=',' read -r -a GPU_ARR <<< "$GPUS"

echo "[run_optuna_supervised] study=${STUDY}  storage=${STORAGE}"
echo "[run_optuna_supervised] gpus=${GPU_ARR[*]}  n_trials_per_worker=${N_TRIALS}"

for gpu in "${GPU_ARR[@]}"; do
  log="logs/optuna_supervised_gpu${gpu}.log"
  echo "[run_optuna_supervised] launching worker on gpu=${gpu}  log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" python -u optuna_snsb_supervised_v1.py \
    --mode       search \
    --study_name "$STUDY" \
    --storage    "$STORAGE" \
    --n_trials   "$N_TRIALS" \
    --epochs     "$EPOCHS" \
    --hidden_dim "$HIDDEN_DIM" \
    --latent_dim "$LATENT_DIM" \
    --gpu        0 \
    $EXTRA > "$log" 2>&1 &
done

wait
echo "[run_optuna_supervised] all workers done"
echo ""
echo "Report:"
python -u optuna_snsb_supervised_v1.py \
  --mode report \
  --study_name "$STUDY" \
  --storage    "$STORAGE"
