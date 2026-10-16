#!/usr/bin/env bash
# Stage 2: train the TopK-32 SAE (100M tokens) and evaluate it on ImageNet-1k val tokens.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sae.train --config configs/sae/train_topk32.yaml
python -m sae.evaluate --config configs/sae/evaluate.yaml
