#!/bin/bash
#SBATCH --job-name=vit_imgnet
#SBATCH --output=slurm/log/train/multigpu_%j.out
#SBATCH --error=slurm/log/train/multigpu_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/vision
#SBATCH --container-mounts=/stratch:/stratch

echo "=== GPU info at job start ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv


set -euo pipefail

# Always start in the directory you ran `sbatch` from (the repo root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure the log directory exists (under slurm/)
mkdir -p slurm/log/train

# resolve the same filename sbatch will use by expanding %j → $SLURM_JOB_ID
export SLURM_STDOUT_PATH="$SLURM_SUBMIT_DIR/slurm/log/train/multigpu_${SLURM_JOB_ID}.out"
export SLURM_STDERR_PATH="$SLURM_SUBMIT_DIR/slurm/log/train/multigpu_${SLURM_JOB_ID}.err"

>&2 echo "SLURM_STDOUT_PATH=$SLURM_STDOUT_PATH"
>&2 echo "SLURM_STDERR_PATH=$SLURM_STDERR_PATH"

python -m pip install --upgrade --force-reinstall --no-cache-dir \
    numpy==1.25.1 scikit-learn==1.3.2 scipy

#python -u train_direct.py --tag no_weight_decay --tag no_lr_warmup --tag no_lr_decay --tag no_dropout 
python -u train_direct.py --w-decay 0.3 --lr-decay --lr-warm --dropout 0.1 --g-clip --tag official_paper_settings --tag 300_epochs