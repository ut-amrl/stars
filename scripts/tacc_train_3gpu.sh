#!/bin/bash
#SBATCH -J hgnn_vae_3gpu
#SBATCH -o logs/hgnn_vae_3gpu_%j.out
#SBATCH -e logs/hgnn_vae_3gpu_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 48:00:00
#SBATCH -A IRI25030

module load tacc-apptainer

SRC_DIR=/scratch/10995/rishabmah/social_world_model/src
mkdir -p $SRC_DIR/../logs

for gpu_idx in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$gpu_idx apptainer exec --nv \
    --env WANDB_API_KEY=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 \
    --bind $SCRATCH:$SCRATCH,$WORK:$WORK \
    $WORK/swm.sif \
    python $SRC_DIR/train_hgnn_unified.py \
      --config $SRC_DIR/params/config.yaml \
    > $SRC_DIR/../logs/gpu${gpu_idx}_${SLURM_JOBID}.log 2>&1 &
  sleep 30
done
wait
