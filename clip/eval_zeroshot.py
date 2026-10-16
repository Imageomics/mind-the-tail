#!/usr/bin/env python3
"""ImageNet-1k zero-shot top-1/top-5 with the 80 OpenAI prompt templates.

    python clip/eval_zeroshot.py --checkpoint RUN/checkpoints/epoch_5.pt --val-dir ${DATA_ROOT}/imagenet/val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from checkpoint import load_model


@torch.inference_mode()
def classifier(model, tokenizer, device: str) -> torch.Tensor:
    weights = []
    for name in IMAGENET_CLASSNAMES:
        text = tokenizer([template(name) for template in OPENAI_IMAGENET_TEMPLATES]).to(device)
        weights.append(F.normalize(F.normalize(model.encode_text(text), dim=-1).mean(0), dim=0))
    return torch.stack(weights, dim=1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True, help="ImageFolder with WordNet-id class dirs")
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("zeroshot_imagenet1k.json"))
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args.model, args.checkpoint, args.device)
    weights = classifier(model, tokenizer, args.device)
    loader = DataLoader(ImageFolder(args.val_dir, preprocess), batch_size=args.batch_size, num_workers=args.workers)
    top1 = top5 = total = 0
    for images, targets in loader:
        with torch.autocast("cuda", enabled=args.device.startswith("cuda")):
            logits = F.normalize(model.encode_image(images.to(args.device)), dim=-1) @ weights
        hits = logits.topk(5, dim=1).indices.cpu() == targets[:, None]
        top1 += hits[:, 0].sum().item()
        top5 += hits.any(1).sum().item()
        total += len(targets)
    result = {"checkpoint": str(args.checkpoint), "acc1": top1 / total, "acc5": top5 / total, "n": total}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
