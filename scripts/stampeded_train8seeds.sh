#!/bin/bash
#SBATCH -J hgnn_8seeds
#SBATCH -o logs/hgnn_8seeds_%j.out
#SBATCH -e logs/hgnn_8seeds_%j.err
#SBATCH -p amd-rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A IRI26004

module load tacc-apptainer

SRC_DIR=/scratch/10995/rishabmah/social_world_model/src
LOG_DIR=$SRC_DIR/../logs
mkdir -p $LOG_DIR

for i in $(seq 0 7); do
    apptainer exec --nv \
      --env WANDB_API_KEY=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 \
      --bind $SCRATCH:$SCRATCH,$WORK:$WORK \
      $WORK/swm.sif \
      python $SRC_DIR/train_hgnn_unified.py \
        --config $SRC_DIR/params/config.yaml \
        --device cuda:$i \
        --run_name swm_1pct_run_$i \
      > $LOG_DIR/run_${i}_${SLURM_JOBID}.log 2>&1 &
done

wait
echo "All 8 training runs finished."
