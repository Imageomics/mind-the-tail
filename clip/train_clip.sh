#!/usr/bin/env bash
# CLIP ViT-B/16 pre-training on a CC12M subset with the open_clip package.
#
#   SUBSET_DIR=${DATA_ROOT}/cc12m_subsets/ours/top_10pct RATIO=10 bash clip/train_clip.sh
#
# Every run trains for 15 "epochs" of 10,977,280 samples (the size of full CC12M),
# so all subset ratios get the same compute. Global batch: 4 GPUs x 4096 = 16,384.
set -euo pipefail

: "${SUBSET_DIR:?set SUBSET_DIR to .../top_<ratio>pct}"
: "${RATIO:?set RATIO to 10, 20, 30, 50 or 70}"
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
NPROC=${NPROC:-4}
EPOCHS=${EPOCHS:-15}
RUN_NAME=${RUN_NAME:-cc12m-ours-top${RATIO}pct-vitb16}

n_shards=$(find "$SUBSET_DIR/shards" -name '*.tar' | wc -l)
last=$(printf '%08d' $((n_shards - 1)))

export TOKENIZERS_PARALLELISM=false
exec torchrun --standalone --nproc_per_node="$NPROC" -m open_clip_train.main \
  --train-data "$SUBSET_DIR/shards/{00000000..$last}.tar" \
  --train-num-samples 10977280 \
  --dataset-type webdataset \
  --dataset-resampled \
  --model ViT-B-16 \
  --batch-size 4096 \
  --epochs "$EPOCHS" \
  --workers 5 \
  --opt adamw \
  --lr 1e-3 \
  --beta1 0.9 \
  --beta2 0.98 \
  --eps 1e-8 \
  --wd 0.5 \
  --lr-scheduler cosine \
  --warmup 400 \
  --precision amp \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  --ddp-static-graph \
  --seed 0 \
  --torchcompile \
  --save-frequency 1 \
  --save-most-recent \
  --logs "$OUTPUT_ROOT/clip" \
  --name "$RUN_NAME" \
  --log-every-n-steps 20
