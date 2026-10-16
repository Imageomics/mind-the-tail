#!/usr/bin/env bash
#SBATCH --job-name=eval_cc12m
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log
# ImageNet-1k zero-shot and linear probe, Flickr30k retrieval of the final checkpoint.
# Usage: DATA_ROOT=... OUTPUT_ROOT=... RATIO=10 sbatch slurm/cc12m/eval.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}/clip"
: "${DATA_ROOT:?}" "${OUTPUT_ROOT:?}"
RATIO=${RATIO:-10}
RUN="$OUTPUT_ROOT/clip/cc12m-ours-top${RATIO}pct-vitb16"
CKPT="$RUN/checkpoints/epoch_15.pt"
python eval_zeroshot.py --checkpoint "$CKPT" --val-dir "$DATA_ROOT/imagenet/val" --output "$RUN/zeroshot_imagenet1k.json"
python eval_linear_probe.py --checkpoint "$CKPT" --imagenet "$DATA_ROOT/imagenet" \
  --cache-dir "$OUTPUT_ROOT/linear_probe_cache" --output "$RUN/linear_probe_imagenet1k.json"
python eval_flickr30k.py --checkpoint "$CKPT" --root "$DATA_ROOT/flickr30k" --output "$RUN/flickr30k_retrieval.json"
