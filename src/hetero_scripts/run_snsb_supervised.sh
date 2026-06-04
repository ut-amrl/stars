#!/usr/bin/env bash
# run_snsb_supervised.sh — Supervised HeteroGNN on SNSB (ablation).
#
# Trains N independent runs of train_snsb_supervised_v1.py.
# Each run gets its own scene-based 85/15 split (seeded by run index).
# Oracle eval (60-episode human oracle) is done at the end of each run.
#
# Usage:
#   ./hetero_scripts/run_snsb_supervised.sh
#   N_RUNS=5 GPUS=0,1 ./hetero_scripts/run_snsb_supervised.sh
#
# Key env overrides (all optional):
#   N_RUNS        number of independent runs        (default: 10)
#   GPUS          comma-separated GPU ids           (default: 0)
#   H5_PATH       SNSB real train h5                (default: dataloaders/socialnav_real_train_v3.h5)
#   ORACLE_H5     human oracle h5                   (default: dataloaders/socialnav_human_oracle_v3.h5)
#   OUT_ROOT      experiment root dir               (default: experiments/snsb_supervised_v1)
#   EPOCHS        max epochs per run                (default: 500)
#   BATCH_SIZE                                      (default: 32)
#   HIDDEN_DIM                                      (default: 256)
#   LATENT_DIM                                      (default: 128)
#   EXTRA         extra args passed to trainer      (default: "")

set -euo pipefail
cd "$(dirname "$0")/.."

N_RUNS="${N_RUNS:-10}"
if [ -z "${GPUS:-}" ]; then
  if command -v nvidia-smi &> /dev/null; then
    num_gpus=$(nvidia-smi -L | wc -l)
  else
    num_gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
  fi
  [ "$num_gpus" -ge 1 ] || num_gpus=1
  GPUS=$(seq -s, 0 $((num_gpus - 1)))
fi
SNSB_H5="${SNSB_H5:-dataloaders/social_nav_v3.h5}"
COMBINED_H5="${COMBINED_H5:-dataloaders/combined_v1.h5}"
SPLIT_FILE="${SPLIT_FILE:-checkpoints/split_indices_final.pt}"
ORACLE_H5="${ORACLE_H5:-dataloaders/socialnav_human_oracle_v3.h5}"
OUT_ROOT="${OUT_ROOT:-experiments/snsb_supervised_v1}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LATENT_DIM="${LATENT_DIM:-128}"
LR="${LR:-0.00011564}"
DROPOUT="${DROPOUT:-0.2}"
EXTRA="${EXTRA:-}"

mkdir -p logs "$OUT_ROOT"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"

for ((i = 0; i < N_RUNS; i++)); do
  gpu="${GPU_ARR[$((i % N_GPUS))]}"
  run=$((i + 1))
  out_dir="${OUT_ROOT}/run${run}"
  log="logs/snsb_supervised_run${run}.log"

  mkdir -p "$out_dir"
  echo "[run_snsb_supervised] run=${run}/${N_RUNS}  gpu=${gpu}  out=${out_dir}  log=${log}"

  CUDA_VISIBLE_DEVICES="$gpu" python -u train_snsb_supervised_v1.py \
    --snsb_h5     "$SNSB_H5" \
    --combined_h5 "$COMBINED_H5" \
    --split_file  "$SPLIT_FILE" \
    --oracle_h5   "$ORACLE_H5" \
    --out_dir     "$out_dir" \
    --seed        "$run" \
    --epochs      "$EPOCHS" \
    --batch_size  "$BATCH_SIZE" \
    --hidden_dim  "$HIDDEN_DIM" \
    --latent_dim  "$LATENT_DIM" \
    --lr          "$LR" \
    --dropout     "$DROPOUT" \
    --weighted_sampling \
    --adaptive_warmup_epochs 20 \
    --adaptive_max_alpha     0.75 \
    --adaptive_ramp_epochs   100 \
    --device "cuda:0" \
    $EXTRA > "$log" 2>&1 &

  if (( (i + 1) % N_GPUS == 0 )); then
    wait
  fi
done

wait
echo "[run_snsb_supervised] all ${N_RUNS} runs done"
echo "Val results:    ${OUT_ROOT}/val_results.csv"
echo "Oracle results: ${OUT_ROOT}/oracle_results.csv"
