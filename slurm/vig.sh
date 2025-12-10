#!/bin/bash
#SBATCH --job-name=vigbasic
#SBATCH --output=slurm/log/train/vigbasic_%j.out
#SBATCH --error=slurm/log/train/vigbasic_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/vision
#SBATCH --container-mounts=/stratch:/stratch


set -euo pipefail

# Always start in the directory you ran `sbatch` from (the repo root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure the log directory exists (under slurm/)
mkdir -p slurm/log/train

# resolve the same filename sbatch will use by expanding %j → $SLURM_JOB_ID
export SLURM_STDOUT_PATH="$SLURM_SUBMIT_DIR/slurm/log/train/vigbasic_${SLURM_JOB_ID}.out"
export SLURM_STDERR_PATH="$SLURM_SUBMIT_DIR/slurm/log/train/vigbasic_${SLURM_JOB_ID}.err"

>&2 echo "SLURM_STDOUT_PATH=$SLURM_STDOUT_PATH"
>&2 echo "SLURM_STDERR_PATH=$SLURM_STDERR_PATH"

python -m pip install --upgrade --force-reinstall --no-cache-dir \
    numpy==1.25.1 scikit-learn==1.3.2 scipy

python -m pip install --no-cache-dir timm

# Base settings
EPOCHS=300
BS=512
NW=8

python train_vig.py --epochs $EPOCHS --train-batch-size $BS --test-batch-size $BS --num-workers $NW \
  --lr 1e-5 --tag "lr_1e-5" --tag "bs_$BS"