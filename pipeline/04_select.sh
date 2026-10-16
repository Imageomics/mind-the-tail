#!/usr/bin/env bash
# Stage 4: rank the pool with the concept-coverage objective.
#   labelled pools: within-class ranking (configs/selection/<dataset>.yaml)
#   cc12m:          one-shot global ranking with the large-scale path
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${DATASET:?}" == cc12m ]]; then
  python -m selection.large_scale \
    --activations "$OUTPUT_ROOT"/image_activations/cc12m* \
    --output "$OUTPUT_ROOT/selection/cc12m" --beta 0.05 --k 3 --max-fraction 1.0
else
  python -m selection.select --config "configs/selection/${DATASET}.yaml"
fi
