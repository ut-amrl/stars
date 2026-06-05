#!/usr/bin/env bash
# run_train_vae.sh — Heterogeneous GNN-VAE pipeline.
# Train N independent VAEs on one fixed data split with random weight
# initializations, no seeds, for the hetero GNN-VAE pipeline.
#
#   combined -> train_unified_gnn_vae_v3.py  (SEAN+SNSB unified hetero VAE)
#   sean     -> train_gnn_vae_v4.py          (SEAN-only hetero VAE)
#
# No seeds are passed anywhere: combined uses `--seed none`, train_gnn_vae_v4
# has no seed flag — every VAE gets a fresh random init. The only fixed thing
# is the base datasplit.
#
# GPU safety: at most ONE training process per GPU at any time (wait -n pool).
# Requires bash >= 4.3.
set -euo pipefail
cd "$(dirname "$0")/.."

PIPELINE=""
N_VAES=10
OUT_ROOT=""
EPOCHS=1000
HIDDEN_DIM=256
LATENT_DIM=128
LR=0.00011564
BETA_MAX=0.9
WARMUP_EPOCHS=25
DROPOUT=0.2
BATCH_SIZE=32
EXTRA=""

usage() {
  cat <<'EOF'
Usage: ./run_train_vae.sh --pipeline combined|sean|homo [options]

  --pipeline      combined | sean | homo                   [required]
  --n-vaes  N     number of Protocol-B VAEs          (default: 10)
  --out-root DIR  checkpoint root (per-VAE: DIR/run{i})
                  default: combined -> checkpoints/v3_hetero_optuned128
                           sean     -> checkpoints/v4_sean_optuned128
  --epochs --hidden-dim --latent-dim --lr --beta-max --warmup-epochs
  --dropout --batch-size                             (default: optuned 256x128)
  --extra "..."   extra args passed verbatim to the trainer
  -h | --help

Both pipelines default to the combined 256x128 Optuna config.
GPU pool comes from CUDA_VISIBLE_DEVICES, e.g.
  CUDA_VISIBLE_DEVICES=5,6,7 ./run_train_vae.sh --pipeline combined
If CUDA_VISIBLE_DEVICES is unset, defaults to 0,1,2,3,4,5,6,7.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --pipeline)      PIPELINE="$2"; shift 2 ;;
    --n-vaes)        N_VAES="$2"; shift 2 ;;
    --out-root)      OUT_ROOT="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --hidden-dim)    HIDDEN_DIM="$2"; shift 2 ;;
    --latent-dim)    LATENT_DIM="$2"; shift 2 ;;
    --lr)            LR="$2"; shift 2 ;;
    --beta-max)      BETA_MAX="$2"; shift 2 ;;
    --warmup-epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
    --dropout)       DROPOUT="$2"; shift 2 ;;
    --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
    --extra)         EXTRA="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ "$PIPELINE" = combined ] || [ "$PIPELINE" = sean ] || [ "$PIPELINE" = homo ] || {
  echo "ERROR: --pipeline must be 'combined', 'sean', or 'homo'" >&2; usage; exit 1; }

COMMON=( --epochs "$EPOCHS" --hidden_dim "$HIDDEN_DIM" --latent_dim "$LATENT_DIM"
         --lr "$LR" --beta_max "$BETA_MAX" --warmup_epochs "$WARMUP_EPOCHS"
         --dropout "$DROPOUT" --batch_size "$BATCH_SIZE" --num_mp_layers 2 )
read -ra EXTRA_ARR <<< "$EXTRA"

if [ "$PIPELINE" = combined ]; then
  : "${OUT_ROOT:=checkpoints/v3_hetero_optuned128}"
  BASE=( train_unified_gnn_vae_v3.py
         --h5_path      /data/combined_v1.h5
         --snsb_h5_path /data/social_nav_v3.h5
         --split_file   ../data/split_indices_final.pt
         --seed none
         --weighted_sampling
         --adaptive_warmup_epochs 20 --adaptive_max_alpha 0.75 --adaptive_ramp_epochs 100
         "${COMMON[@]}" )
elif [ "$PIPELINE" = homo ]; then
  : "${OUT_ROOT:=checkpoints/hgnn_combined_v8}"
  BASE=( train_hgnn_unified_v2.py
         --h5_path      /data/combined_v1.h5
         --split_file   ../data/split_indices_final.pt
         --edge_transformer_layers 3 --num_heads 4
         --seed none
         --weighted_sampling
         --adaptive_warmup_epochs 20 --adaptive_max_alpha 0.75 --adaptive_ramp_epochs 100
         "${COMMON[@]}" )
else
  : "${OUT_ROOT:=checkpoints/v4_sean_optuned128}"
  BASE=( train_gnn_vae_v4.py
         --h5_path    /data/sean_together_v2.h5
         --split_file checkpoints/split_indices_final_sean_only.pt
         --edge_transformer_layers 3 --edge_transformer_heads 4
         "${COMMON[@]}" )
fi

mkdir -p logs "$OUT_ROOT"
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

echo "pipeline=$PIPELINE  n_vaes=$N_VAES  CUDA_VISIBLE_DEVICES=(${GPU_ARR[*]})  out_root=$OUT_ROOT"

declare -A PID2GPU
FREE=("${GPU_ARR[@]}")

for (( i=0; i<N_VAES; i++ )); do
  # block until a GPU frees (guarantees 1 process per GPU)
  while [ ${#FREE[@]} -eq 0 ]; do
    wait -n || true
    for pid in "${!PID2GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { FREE+=("${PID2GPU[$pid]}"); unset 'PID2GPU[$pid]'; }
    done
  done
  gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
  log="logs/train_${PIPELINE}_run${i}.log"
  echo "[run $i] GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "${BASE[@]}" \
    --checkpoint_dir "$OUT_ROOT/run$i" "${EXTRA_ARR[@]}" > "$log" 2>&1 &
  PID2GPU[$!]="$gpu"
done
wait
echo "Done: $N_VAES $PIPELINE VAEs trained -> $OUT_ROOT/run{0..$((N_VAES-1))}"
