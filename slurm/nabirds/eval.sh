#!/usr/bin/env bash
#SBATCH --job-name=eval_nabirds
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# Usage: DATA_ROOT=... OUTPUT_ROOT=... RATIO=0.1 sbatch slurm/nabirds/eval.sh
# Evaluates every seed in SEEDS (default "0 1 2") and writes the mean/std to <out>/summary.json.
DATASET=nabirds
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${DATA_ROOT:?}" "${OUTPUT_ROOT:?}"
export DATA_ROOT OUTPUT_ROOT
RATIO=${RATIO:-0.1}
SEEDS=${SEEDS:-0 1 2}
PCT=$(python -c "print(round(${RATIO} * 100))")
OUT="$OUTPUT_ROOT/downstream/${DATASET}_${PCT}pct"
SUMMARIES=()
for SEED in $SEEDS; do
  python -m downstream.evaluate --config configs/downstream/eval_nabirds.yaml \
    --set checkpoint="$OUT/seed${SEED}/checkpoints/best.pt" output_dir="$OUT/seed${SEED}/eval"
  SUMMARIES+=("$OUT/seed${SEED}/eval/summary.json")
done
python -m downstream.aggregate --runs "${SUMMARIES[@]}" --output "$OUT/summary.json"
