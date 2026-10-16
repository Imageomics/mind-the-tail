#!/usr/bin/env bash
# Stage 3: image-level SAE activations (max over patches) of a candidate pool.
#   DATASET in: nabirds stanford_cars waterbirds spawrious imagenet1k cc12m
# Equivalent two-step route for pools whose token shards exist:
#   python -m sae.infer tokens --config configs/sae/infer_tokens.yaml
#   python -m sae.infer pool   --config configs/sae/pool_images.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sae.infer images --config "configs/sae/infer_images_${DATASET:?}.yaml"
