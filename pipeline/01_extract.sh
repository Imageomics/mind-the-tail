#!/usr/bin/env bash
# Stage 1: DINOv3-B layer-11 patch tokens of ImageNet-1k (SAE train/val data),
# then a global shuffle of the training examples across shards.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sae.extract --config configs/sae/extract_imagenet1k_train.yaml
python -m sae.extract --config configs/sae/extract_imagenet1k_val.yaml
python -m sae.remap --config configs/sae/remap_imagenet1k_train.yaml
