#!/usr/bin/env bash
# run_snsb_supervised_scratch_label_eff.sh
#
# Label-efficiency sweep for supervised SNSB ablation — full retrain from scratch.
# Loads dataset ONCE per GPU worker, then loops over assigned combinations.
# 10 seeds × 6 fractions × 10 draws = 600 runs total.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3,4,5,6,7 GPUS=3,4,5,6,7 \
#       bash hetero_scripts/run_snsb_supervised_scratch_label_eff.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${GPUS:-}" ]; then
  if command -v nvidia-smi &> /dev/null; then
    num_gpus=$(nvidia-smi -L | wc -l)
  else
    num_gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
  fi
  [ "$num_gpus" -ge 1 ] || num_gpus=1
  GPUS=$(seq -s, 0 $((num_gpus - 1)))
fi
RESULTS_CSV="${RESULTS_CSV:-label_eff_supervised_scratch_snsb.csv}"
EXTRA="${EXTRA:-}"

mkdir -p logs

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_WORKERS="${#GPU_ARR[@]}"

echo "[run_snsb_supervised_scratch_label_eff] ${N_WORKERS} workers  csv=${RESULTS_CSV}"

for ((worker_id = 0; worker_id < N_WORKERS; worker_id++)); do
  gpu="${GPU_ARR[$worker_id]}"
  log="logs/scratch_label_eff_worker${worker_id}.log"
  echo "  worker=${worker_id}  gpu=${gpu}  log=${log}"

  CUDA_VISIBLE_DEVICES="$gpu" python -u eval_snsb_supervised_scratch_label_eff.py \
    --worker_id  "$worker_id" \
    --n_workers  "$N_WORKERS" \
    --results_csv "$RESULTS_CSV" \
    --gpu        0 \
    $EXTRA > "$log" 2>&1 &
done

wait
echo "[run_snsb_supervised_scratch_label_eff] all workers done → ${RESULTS_CSV}"
echo ""
echo "Summarize:"
echo "  python summarize_supervised_label_eff.py --csv ${RESULTS_CSV}"
