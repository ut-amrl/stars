#!/usr/bin/env bash
# run_optuna_stgcnn.sh — Optuna search for Social-STGCNN baseline.
#
# One worker per GPU, sharing a single SQLite study.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4,5,6,7 N_TRIALS=10 bash hetero_scripts/run_optuna_stgcnn.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-0}"
N_TRIALS="${N_TRIALS:-10}"
STUDY="${STUDY:-social_stgcnn_v1}"
STORAGE="${STORAGE:-sqlite:///optuna_social_stgcnn_v1.db}"
EPOCHS="${EPOCHS:-200}"
EXTRA="${EXTRA:-}"

mkdir -p logs

IFS=',' read -r -a GPU_ARR <<< "$GPUS"

echo "[run_optuna_stgcnn] study=${STUDY}  storage=${STORAGE}"
echo "[run_optuna_stgcnn] gpus=${GPU_ARR[*]}  n_trials_per_worker=${N_TRIALS}"

for gpu in "${GPU_ARR[@]}"; do
  log="logs/optuna_stgcnn_gpu${gpu}.log"
  echo "[run_optuna_stgcnn] launching worker on gpu=${gpu}  log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" python -u optuna_social_stgcnn_v1.py \
    --mode       search \
    --study_name "$STUDY" \
    --storage    "$STORAGE" \
    --n_trials   "$N_TRIALS" \
    --epochs     "$EPOCHS" \
    --gpu        0 \
    $EXTRA > "$log" 2>&1 &
done

wait
echo "[run_optuna_stgcnn] all workers done"
echo ""
echo "Report:"
python -u optuna_social_stgcnn_v1.py \
  --mode report \
  --study_name "$STUDY" \
  --storage    "$STORAGE"
