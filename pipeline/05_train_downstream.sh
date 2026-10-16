#!/usr/bin/env bash
# Stage 5: train on the top-RATIO subset (default 0.1) and evaluate.
#   Classification: one ResNet-50 run per seed in SEEDS (default "0 1 2"), each evaluated
#   on the test split; the mean/std over seeds is written to $OUT/summary.json.
#   CC12M: the nested top-k% subsets are built once, then CLIP is trained on top-RATIO.
set -euo pipefail
cd "$(dirname "$0")/.."
RATIO=${RATIO:-0.1}
PCT=$(python -c "print(round(${RATIO} * 100))")
if [[ "${DATASET:?}" == cc12m ]]; then
  SUBSETS="$OUTPUT_ROOT/cc12m_subsets"
  # build_subset.py writes every ratio in one pass and refuses an existing output dir.
  if [[ ! -d "$SUBSETS" ]]; then
    python clip/build_subset.py --ranking "$OUTPUT_ROOT/selection/cc12m/ranked_examples.parquet" \
      --source-dir "$DATA_ROOT/cc12m-wds" --output "$SUBSETS" --ratios 10 20 30 50 70
  fi
  SUBSET_DIR="$SUBSETS/top_${PCT}pct"
  if [[ ! -d "$SUBSET_DIR" ]]; then
    echo "$SUBSET_DIR not found; $SUBSETS was built with other ratios." >&2
    exit 1
  fi
  SUBSET_DIR="$SUBSET_DIR" RATIO=$PCT bash clip/train_clip.sh
  exit
fi
SEEDS=${SEEDS:-0 1 2}
OUT="$OUTPUT_ROOT/downstream/${DATASET}_${PCT}pct"
SUMMARIES=()
for SEED in $SEEDS; do
  RUN="$OUT/seed${SEED}"
  python -m downstream.train --config "configs/downstream/${DATASET}.yaml" \
    --set ratio="$RATIO" seed="$SEED" output_dir="$RUN"
  # Test-split accuracy (and worst-group accuracy for Waterbirds / Spawrious).
  python -m downstream.evaluate --config "configs/downstream/eval_${DATASET}.yaml" \
    --set checkpoint="$RUN/checkpoints/best.pt" output_dir="$RUN/eval"
  SUMMARIES+=("$RUN/eval/summary.json")
done
python -m downstream.aggregate --runs "${SUMMARIES[@]}" --output "$OUT/summary.json"
