#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=slurm/log/sweep/vitcls_%j.out
#SBATCH --error=slurm/log/sweep/vitcls_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/vision
#SBATCH --container-mounts=/stratch:/stratch


set -euo pipefail

# Always start in the directory you ran `sbatch` from (the repo root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure the log directory exists (under slurm/)
mkdir -p slurm/log/sweep

# resolve the same filename sbatch will use by expanding %j → $SLURM_JOB_ID
export SLURM_STDOUT_PATH="$SLURM_SUBMIT_DIR/slurm/log/sweep/vitcls_${SLURM_JOB_ID}.out"
export SLURM_STDERR_PATH="$SLURM_SUBMIT_DIR/slurm/log/sweep/vitcls_${SLURM_JOB_ID}.err"

>&2 echo "SLURM_STDOUT_PATH=$SLURM_STDOUT_PATH"
>&2 echo "SLURM_STDERR_PATH=$SLURM_STDERR_PATH"

python -m pip install --upgrade --force-reinstall --no-cache-dir \
    numpy==1.25.1 scikit-learn==1.3.2 scipy

# Base settings
EPOCHS=5
BS=128
NW=8

python train_neptune.py --epochs $EPOCHS --train-batch-size $BS --test-batch-size $BS --num-workers $NW \
  --lr 1e-5 --tag "lr_1e-5" --tag "bs_$BS"

python train_neptune.py --epochs $EPOCHS --train-batch-size $BS --test-batch-size $BS --num-workers $NW \
  --lr 3e-5 --tag "lr_3e-5" --tag "bs_$BS"

python train_neptune.py --epochs $EPOCHS --train-batch-size $BS --test-batch-size $BS --num-workers $NW \
  --lr 1e-4 --tag "lr_1e-4" --tag "bs_$BS"

python train_neptune.py --epochs $EPOCHS --train-batch-size $BS --test-batch-size $BS --num-workers $NW \
  --lr 3e-4 --tag "lr_3e-4" --tag "bs_$BS"

python train_neptune.py --epochs $EPOCHS --train-batch-size $BS --test-batch-size $BS --num-workers $NW \
  --lr 1e-3 --tag "lr_1e-3" --tag "bs_$BS"

