#!/bin/bash
#SBATCH --job-name=coco
#SBATCH --output=slurm/log/data/coco_%j.out
#SBATCH --error=slurm/log/data/coco_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --gpus=1
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
export SLURM_STDOUT_PATH="$SLURM_SUBMIT_DIR/slurm/log/data/coco_${SLURM_JOB_ID}.out"
export SLURM_STDERR_PATH="$SLURM_SUBMIT_DIR/slurm/log/data/coco_${SLURM_JOB_ID}.err"

>&2 echo "SLURM_STDOUT_PATH=$SLURM_STDOUT_PATH"
>&2 echo "SLURM_STDERR_PATH=$SLURM_STDERR_PATH"

# ---- COCO 2017 download + extract into <submit_dir>/data ----

DATA_DIR="$SLURM_SUBMIT_DIR/data"
COCO_DIR="$DATA_DIR/coco2017"

mkdir -p "$COCO_DIR"
>&2 echo "COCO_DIR=$COCO_DIR"

# quick disk check on the filesystem backing <submit_dir>
df -h "$SLURM_SUBMIT_DIR" >&2 || true

cd "$COCO_DIR"

download() {
  local url="$1"
  local out="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    # resume if possible
    curl -L --fail --continue-at - -o "$out" "$url"
  else
    # python fallback (no resume)
    python - <<'PY'
import sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, out)
PY
    "$url" "$out"
  fi
}

# COCO official URLs
TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


#download "$TRAIN_URL" "train2017.zip"
#download "$VAL_URL"   "val2017.zip"
#download "$ANN_URL"   "annotations_trainval2017.zip"


extract_zip() {
  local zip_path="$1"
  local dest_dir="$2"

  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$zip_path" -d "$dest_dir"
  else
    python - <<'PY' "$zip_path" "$dest_dir"
import sys, os, zipfile
zip_path, dest_dir = sys.argv[1], sys.argv[2]
os.makedirs(dest_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as z:
    # Extract streaming; no big RAM usage
    z.extractall(dest_dir)
print(f"Extracted {zip_path} -> {dest_dir}")
PY
  fi
}

# extract (skip if already extracted)
if [[ ! -d train2017 ]]; then extract_zip train2017.zip .; else >&2 echo "train2017/ exists, skipping"; fi
if [[ ! -d val2017   ]]; then extract_zip val2017.zip   .; else >&2 echo "val2017/ exists, skipping";   fi
if [[ ! -d annotations ]]; then extract_zip annotations_trainval2017.zip .; else >&2 echo "annotations/ exists, skipping"; fi

# optional cleanup to save space (comment out if you want to keep zips)
rm -f train2017.zip val2017.zip annotations_trainval2017.zip

# show what we ended up with
>&2 echo "COCO install complete."
ls -al "$COCO_DIR" >&2
du -sh train2017 val2017 annotations 2>/dev/null || true
