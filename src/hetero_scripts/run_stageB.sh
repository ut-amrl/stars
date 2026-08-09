#!/usr/bin/env bash
# Stage-B launcher for the transfer ablation (STARS-He).
#
# For each Stage-A run dir ($VAE_ROOT/run{i}), trains ONLY the held-out dataset's
# front-end adapter into the frozen trunk (train_stage_b_adapter_align.py),
# writing an aligned checkpoint + copied split into $OUT_ROOT/run{i}. Fanned
# across a GPU pool, one process per GPU (wait -n queue).
#
# The $OUT_ROOT/run{i} dirs are drop-in inputs for run_label_eff.sh.
#
# Usage (from src/, venv active, inside tmux — run AFTER Stage A finishes):
#   # Dir 1: SEAN-only pretrain -> align SNSB adapter
#   ./hetero_scripts/run_stageB.sh --align snsb \
#       --vae-root checkpoints/xfer_seanonly \
#       --out-root checkpoints/xfer_seanonly_alignSNSB
#
#   # Dir 2: SNSB-only pretrain -> align SEAN adapter
#   ./hetero_scripts/run_stageB.sh --align sean \
#       --vae-root checkpoints/xfer_snsbonly \
#       --out-root checkpoints/xfer_snsbonly_alignSEAN
set -uo pipefail

ALIGN=""
VAE_ROOT=""
OUT_ROOT=""
N_VAES=10
GPUS="0 1 2 3 4 5 6 7"
CKPT_NAME="best_unified_gnn_vae_v3_seednone.pt"
EXTRA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --align)     ALIGN="$2";     shift 2 ;;
    --vae-root)  VAE_ROOT="$2";  shift 2 ;;
    --out-root)  OUT_ROOT="$2";  shift 2 ;;
    --n-vaes)    N_VAES="$2";     shift 2 ;;
    --gpus)      GPUS="$2";      shift 2 ;;
    --ckpt-name) CKPT_NAME="$2"; shift 2 ;;
    --extra)     EXTRA="$2";     shift 2 ;;   # verbatim to train_stage_b_adapter_align.py
    -h|--help)
      echo "flags: --align {sean|snsb}  --vae-root DIR  --out-root DIR  [--n-vaes N]  [--gpus \"a b c\"]  [--extra \"...\"]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ "$ALIGN" == "sean" || "$ALIGN" == "snsb" ]] || { echo "ERROR: --align must be 'sean' or 'snsb'" >&2; exit 1; }
[[ -n "$VAE_ROOT" ]] || { echo "ERROR: --vae-root required" >&2; exit 1; }
[[ -n "$OUT_ROOT" ]] || { echo "ERROR: --out-root required" >&2; exit 1; }

mkdir -p logs "$OUT_ROOT"
read -ra GPU_ARR <<< "$GPUS"
read -ra EXTRA_ARR <<< "$EXTRA"

declare -A PID2GPU
FREE=("${GPU_ARR[@]}")
launched=0

echo "Stage-B align=$ALIGN  vae_root=$VAE_ROOT  out_root=$OUT_ROOT  n_vaes=$N_VAES  gpus=(${GPU_ARR[*]})"

for (( i=0; i<N_VAES; i++ )); do
  ckpt="$VAE_ROOT/run$i/$CKPT_NAME"
  if [[ ! -s "$ckpt" ]]; then
    echo "[align $i] SKIP: missing Stage-A checkpoint $ckpt" >&2
    continue
  fi
  # block until a GPU frees (1 process per GPU)
  while [ ${#FREE[@]} -eq 0 ]; do
    wait -n || true
    for pid in "${!PID2GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || { FREE+=("${PID2GPU[$pid]}"); unset 'PID2GPU[$pid]'; }
    done
  done
  gpu="${FREE[0]}"; FREE=("${FREE[@]:1}")
  log="logs/stageB_${ALIGN}_run${i}.log"
  echo "[align $i] GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train_stage_b_adapter_align.py \
    --align "$ALIGN" --gpu 0 \
    --stage_a_ckpt "$ckpt" \
    --out_dir "$OUT_ROOT/run$i" \
    "${EXTRA_ARR[@]}" > "$log" 2>&1 &
  PID2GPU[$!]="$gpu"
  launched=$((launched+1))
done
wait
echo "Stage-B DONE: $launched aligned checkpoints -> $OUT_ROOT/run{0..$((N_VAES-1))}"
echo "Next: ./hetero_scripts/run_label_eff.sh --pipeline combined --vae-root $OUT_ROOT --out-prefix <prefix>"
