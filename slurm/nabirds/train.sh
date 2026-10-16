#!/usr/bin/env bash
#SBATCH --job-name=train_nabirds
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
# Usage: DATA_ROOT=... OUTPUT_ROOT=... RATIO=0.1 sbatch --array=0-2 slurm/nabirds/train.sh
# One array task per seed (SEED defaults to the array index); results are averaged by eval.sh.
DATASET=nabirds
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${DATA_ROOT:?}" "${OUTPUT_ROOT:?}"
export DATA_ROOT OUTPUT_ROOT
RATIO=${RATIO:-0.1}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
PCT=$(python -c "print(round(${RATIO} * 100))")
OUT="$OUTPUT_ROOT/downstream/${DATASET}_${PCT}pct/seed${SEED}"
# Recipe: configs/downstream/nabirds.yaml (ranking from configs/selection/nabirds.yaml).
python -m downstream.train --config configs/downstream/nabirds.yaml --set ratio="$RATIO" seed="$SEED" output_dir="$OUT"
