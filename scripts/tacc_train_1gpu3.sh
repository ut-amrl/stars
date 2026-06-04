#!/bin/bash
#SBATCH -J hgnn_vae_3runs
#SBATCH -o logs/hgnn_vae_3runs_%j.out
#SBATCH -e logs/hgnn_vae_3runs_%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A IRI25030

module load tacc-apptainer

SRC_DIR=/scratch/10995/rishabmah/social_world_model/src
mkdir -p $SRC_DIR/../logs

for run in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 apptainer exec --nv \
    --env WANDB_API_KEY=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 \
    --bind $SCRATCH:$SCRATCH,$WORK:$WORK \
    $WORK/swm.sif \
    python $SRC_DIR/train_hgnn_unified.py \
      --config $SRC_DIR/params/config2.yaml \
    > $SRC_DIR/../logs/run${run}_${SLURM_JOBID}.log 2>&1
done
