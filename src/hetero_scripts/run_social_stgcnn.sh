#!/usr/bin/env bash
# run_social_stgcnn.sh — Train N independent Social-STGCNN runs (post-Optuna).
set -euo pipefail
cd "$(dirname "$0")/.."

N_RUNS="${N_RUNS:-10}"
GPUS="${GPUS:-0}"
TASK="${TASK:-snsb}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_FEAT="${HIDDEN_FEAT:-64}"
N_STGCNN="${N_STGCNN:-2}"
KERNEL_SIZE="${KERNEL_SIZE:-3}"
PATIENCE="${PATIENCE:-30}"
EPOCHS="${EPOCHS:-200}"
OUT_ROOT="${OUT_ROOT:-experiments/social_stgcnn_v1}"
EXTRA="${EXTRA:-}"

mkdir -p logs "$OUT_ROOT"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS="${#GPU_ARR[@]}"

for ((i = 0; i < N_RUNS; i++)); do
  gpu="${GPU_ARR[$((i % N_GPUS))]}"
  run=$((i + 1))
  ckpt_dir="${OUT_ROOT}/run${run}"
  log="logs/social_stgcnn_run${run}.log"

  mkdir -p "$ckpt_dir"
  echo "[run_social_stgcnn] run=${run}/${N_RUNS}  gpu=${gpu}  ckpt=${ckpt_dir}  log=${log}"

  CUDA_VISIBLE_DEVICES="$gpu" python -u train_social_stgcnn_baseline.py \
    --task           "$TASK" \
    --checkpoint_dir "$ckpt_dir" \
    --seed           "$run" \
    --lr             "$LR" \
    --dropout        "$DROPOUT" \
    --weight_decay   "$WEIGHT_DECAY" \
    --hidden_feat    "$HIDDEN_FEAT" \
    --n_stgcnn       "$N_STGCNN" \
    --kernel_size    "$KERNEL_SIZE" \
    --patience       "$PATIENCE" \
    --epochs         "$EPOCHS" \
    --gpu            0 \
    $EXTRA > "$log" 2>&1 &

  if (( (i + 1) % N_GPUS == 0 )); then
    wait
  fi
done

wait
echo "[run_social_stgcnn] all ${N_RUNS} runs done"
