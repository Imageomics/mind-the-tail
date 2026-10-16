#!/usr/bin/env bash
#SBATCH --job-name=train_cc12m
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.log
# Build the nested top-k% WebDataset subsets (once) and train CLIP ViT-B/16 on one.
# Usage: DATA_ROOT=... OUTPUT_ROOT=... RATIO=10 sbatch slurm/cc12m/train.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${DATA_ROOT:?}" "${OUTPUT_ROOT:?}"
RATIO=${RATIO:-10}
SUBSETS="$OUTPUT_ROOT/cc12m_subsets"
if [[ ! -d "$SUBSETS" ]]; then
  python clip/build_subset.py --ranking "$OUTPUT_ROOT/selection/cc12m/ranked_examples.parquet" \
    --source-dir "$DATA_ROOT/cc12m-wds" --output "$SUBSETS" --ratios 10 20 30 50 70
fi
SUBSET_DIR="$SUBSETS/top_${RATIO}pct" RATIO="$RATIO" OUTPUT_ROOT="$OUTPUT_ROOT" NPROC=4 bash clip/train_clip.sh
