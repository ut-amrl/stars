#!/usr/bin/env bash
# Stage-A launcher for the transfer ablation (STARS-He).
#
# Runs, in parallel across two disjoint GPU pools:
#   SEAN-only pretrains  (--snsb_loss_weight 0)  -> $SEAN_ROOT/run{0..N-1}
#   SNSB-only pretrains  (--sean_loss_weight 0)  -> $SNSB_ROOT/run{0..N-1}
#
# Each pool is a job queue: at most ONE training process per GPU at a time
# (wait -n), so N VAEs on a 3-GPU pool auto-runs as ceil(N/3) waves.
# All hyperparameters match the STARS-He 256x128 config (run_train_vae.sh).
#
# Usage (from src/, venv active, inside tmux):
#   ./hetero_scripts/run_stageA_transfer.sh                 # 10 each, GPUs 2 3 4 / 5 6 7
#   ./hetero_scripts/run_stageA_transfer.sh --n-vaes 9
#   ./hetero_scripts/run_stageA_transfer.sh --sean-gpus "2 3 4" --snsb-gpus "5 6 7"
set -uo pipefail

N_VAES=10
SEAN_GPUS="2 3 4"
SNSB_GPUS="5 6 7"
SEAN_ROOT="checkpoints/xfer_seanonly"
SNSB_ROOT="checkpoints/xfer_snsbonly"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-vaes)    N_VAES="$2";    shift 2 ;;
    --sean-gpus) SEAN_GPUS="$2"; shift 2 ;;
    --snsb-gpus) SNSB_GPUS="$2"; shift 2 ;;
    --sean-root) SEAN_ROOT="$2"; shift 2 ;;
    --snsb-root) SNSB_ROOT="$2"; shift 2 ;;
    -h|--help)
      echo "flags: --n-vaes N  --sean-gpus \"a b c\"  --snsb-gpus \"a b c\"  --sean-root DIR  --snsb-root DIR"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p logs "$SEAN_ROOT" "$SNSB_ROOT"

COMMON=( --seed none --gpu 0
         --hidden_dim 256 --latent_dim 128 --num_mp_layers 2
         --lr 0.00011564 --beta_max 0.9 --warmup_epochs 25 --dropout 0.2 --batch_size 32
         --weighted_sampling --adaptive_warmup_epochs 20 --adaptive_max_alpha 0.75 --adaptive_ramp_epochs 100 )

run_pool() {   # $1=tag  $2=out_root  $3=loss_flag  $4=loss_val  $5...=gpus
  local tag="$1" out_root="$2" loss_flag="$3" loss_val="$4"; shift 4
  local gpus=("$@")
  declare -A PID2GPU
  local FREE=("${gpus[@]}") i gpu log pid
  echo "[$tag] launching $N_VAES VAEs on GPUs: ${gpus[*]}  -> $out_root"
  for (( i=0; i<N_VAES; i++ )); do
    # block until a GPU in this pool frees (1 process per GPU)
    while [ ${#FREE[@]} -eq 0 ]; do
      wait -n || true
      for pid in "${!PID2GPU[@]}"; do
        kill -0 "$pid" 2>/dev/null || { FREE+=("${PID2GPU[$pid]}"); unset 'PID2GPU[$pid]'; }
      done
    done
    gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
    log="logs/stageA_${tag}_run${i}.log"
    echo "[$tag run $i] GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u train_unified_gnn_vae_v3.py \
      "${COMMON[@]}" "$loss_flag" "$loss_val" \
      --checkpoint_dir "$out_root/run$i" > "$log" 2>&1 &
    PID2GPU[$!]="$gpu"
  done
  wait
  echo "[$tag] DONE: $N_VAES VAEs -> $out_root/run{0..$((N_VAES-1))}"
}

# Both directions run concurrently on their own GPU pools.
run_pool sean "$SEAN_ROOT" --snsb_loss_weight 0 $SEAN_GPUS &
SEAN_PID=$!
run_pool snsb "$SNSB_ROOT" --sean_loss_weight 0 $SNSB_GPUS &
SNSB_PID=$!

wait "$SEAN_PID" "$SNSB_PID"
echo "ALL STAGE-A DONE  ($N_VAES SEAN-only + $N_VAES SNSB-only)"
