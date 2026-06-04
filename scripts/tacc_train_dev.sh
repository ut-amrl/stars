#!/bin/bash
#SBATCH -J hgnn_dev_test
#SBATCH -o logs/hgnn_dev_test_%j.out
#SBATCH -e logs/hgnn_dev_test_%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 02:00:00
#SBATCH -A IRI25030

export WANDB_API_KEY=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
source $WORK/.venvs/swm/bin/activate

SRC_DIR=/scratch/10995/rishabmah/social_world_model/src
cd $SRC_DIR
mkdir -p ../logs

for gpu_idx in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$gpu_idx python train_hgnn_unified.py \
    --config params/config.yaml \
    > ../logs/gpu${gpu_idx}_${SLURM_JOBID}.log 2>&1 &
  sleep 2
done
wait
