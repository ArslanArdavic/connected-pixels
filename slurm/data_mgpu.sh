#!/bin/bash
#SBATCH --job-name=data_mgpu
#SBATCH --output=slurm/log/data/mpgu_%j.out
#SBATCH --error=slurm/log/data/mpgu_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/vision
#SBATCH --container-mounts=/stratch:/stratch

set -euo pipefail

# Always start in the directory you ran `sbatch` from (the repo root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure the log directory exists (under slurm/)
mkdir -p slurm/log/data

# resolve the same filename sbatch will use by expanding %j → $SLURM_JOB_ID
export SLURM_STDOUT_PATH="$SLURM_SUBMIT_DIR/slurm/log/data/mgpu_${SLURM_JOB_ID}.out"
export SLURM_STDERR_PATH="$SLURM_SUBMIT_DIR/slurm/log/data/mgpu_${SLURM_JOB_ID}.err"

>&2 echo "SLURM_STDOUT_PATH=$SLURM_STDOUT_PATH"
>&2 echo "SLURM_STDERR_PATH=$SLURM_STDERR_PATH"

#python -m pip install --upgrade --force-reinstall --no-cache-dir \
#    numpy==1.25.1 scikit-learn==1.3.2

python -m pip install --upgrade --force-reinstall --no-cache-dir \
    scipy

# Run the distributed data loader smoke test (one process per GPU)
NUM_GPUS="${SLURM_GPUS:-4}"

echo "Launching imagenet_parallel.py with ${NUM_GPUS} GPUs"

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    imagenet_parallel.py
