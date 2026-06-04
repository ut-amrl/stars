#!/usr/bin/env bash
# run_snsb_supervised_label_eff.sh — Label-efficiency sweep for supervised SNSB ablation.
#
# Mirrors run_label_eff.sh (SNSB path). One process per GPU, each handling a
# subset of the 10 checkpoint dirs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4,5,6,7 bash hetero_scripts/run_snsb_supervised_label_eff.sh
#
# Env overrides:
#   CKPT_ROOT   root of supervised runs      (default: experiments/snsb_supervised_v1_optuned)
#   N_RUNS      number of ckpt dirs          (default: 10)
#   GPUS        comma-separated GPU ids      (default: 0)
#   OUTPUT      output CSV path              (default: label_eff_supervised_snsb.csv)
#   N_SUBSET_SEEDS                           (default: 10)

set -euo pipefail
cd "$(dirname "$0")/.."

CKPT_ROOT="${CKPT_ROOT:-experiments/snsb_supervised_v1_optuned}"
N_RUNS="${N_RUNS:-10}"
GPUS="${GPUS:-0}"
OUTPUT="${OUTPUT:-label_eff_supervised_snsb.csv}"
N_SUBSET_SEEDS="${N_SUBSET_SEEDS:-10}"

mkdir -p logs

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"

# Build list of ckpt dirs
CKPT_DIRS=()
for ((i = 1; i <= N_RUNS; i++)); do
  CKPT_DIRS+=("${CKPT_ROOT}/run${i}")
done

# Distribute ckpt dirs across GPUs — one process per GPU, each gets N_RUNS/N_GPUS dirs
for ((g = 0; g < N_GPUS; g++)); do
  gpu="${GPU_ARR[$g]}"
  log="logs/label_eff_supervised_gpu${gpu}.log"

  # Collect every Nth ckpt for this GPU
  GPU_CKPTS=()
  for ((i = g; i < N_RUNS; i += N_GPUS)); do
    GPU_CKPTS+=("${CKPT_DIRS[$i]}")
  done

  echo "[run_snsb_supervised_label_eff] gpu=${gpu}  ckpts=${GPU_CKPTS[*]}  log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" python -u eval_snsb_supervised_label_eff.py \
    --ckpt_dirs     "${GPU_CKPTS[@]}" \
    --output        "$OUTPUT" \
    --n_subset_seeds "$N_SUBSET_SEEDS" \
    --gpu           0 > "$log" 2>&1 &
done

wait
echo "[run_snsb_supervised_label_eff] done → ${OUTPUT}"
