#!/usr/bin/env python3
"""ImageNet-1k linear probe on frozen, L2-normalized CLIP image embeddings.

One eval-preprocessed embedding per image is cached (float16); a zero-initialized
linear layer is trained with AdamW (lr 1e-2, no weight decay, batch 32768, 100
epochs, 1 warmup epoch + cosine) and evaluated once on the validation split.

    python clip/eval_linear_probe.py --checkpoint RUN/checkpoints/epoch_5.pt --imagenet ${DATA_ROOT}/imagenet
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from checkpoint import load_model


@torch.inference_mode()
def embed(model, preprocess, folder: Path, cache: Path, args) -> tuple[np.ndarray, np.ndarray]:
    if cache.with_suffix(".labels.npy").exists():
        return np.load(cache.with_suffix(".features.npy")), np.load(cache.with_suffix(".labels.npy"))
    loader = DataLoader(ImageFolder(folder, preprocess), batch_size=args.batch_size, num_workers=args.workers)
    features, labels = [], []
    for images, targets in loader:
        with torch.autocast("cuda", enabled=args.device.startswith("cuda")):
            out = model.encode_image(images.to(args.device))
        features.append(F.normalize(out.float(), dim=-1).cpu().half().numpy())
        labels.append(targets.numpy())
    features, labels = np.concatenate(features), np.concatenate(labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache.with_suffix(".features.npy"), features)
    np.save(cache.with_suffix(".labels.npy"), labels)
    return features, labels


def fit_probe(train_x, train_y, args) -> torch.nn.Linear:
    x = torch.tensor(train_x, dtype=torch.float32, device=args.device)
    y = torch.tensor(train_y, dtype=torch.long, device=args.device)
    probe = torch.nn.Linear(x.shape[1], int(y.max()) + 1, device=args.device)
    torch.nn.init.zeros_(probe.weight)
    torch.nn.init.zeros_(probe.bias)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(y) / args.probe_batch_size)
    total, warmup = steps_per_epoch * args.epochs, steps_per_epoch

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min((step - warmup) / max(1, total - warmup), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    for _ in range(args.epochs):
        for batch in torch.randperm(len(y), device=args.device).split(args.probe_batch_size):
            loss = F.cross_entropy(probe(x[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imagenet", type=Path, required=True, help="directory with train/ and val/ ImageFolders")
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/linear_probe_cache"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--probe-batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("linear_probe_imagenet1k.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model, preprocess, _ = load_model(args.model, args.checkpoint, args.device)
    # Grandparent, parent and stem keep caches apart for both RUN/checkpoints/epoch_N.pt
    # and the released <dataset>/<ratio>pct/clip.pt layout.
    ckpt = args.checkpoint
    prefix = args.cache_dir / f"{ckpt.parent.parent.name}_{ckpt.parent.name}_{ckpt.stem}"
    train_x, train_y = embed(model, preprocess, args.imagenet / "train", prefix.with_name(prefix.name + "_train"), args)
    val_x, val_y = embed(model, preprocess, args.imagenet / "val", prefix.with_name(prefix.name + "_val"), args)
    del model
    probe = fit_probe(train_x, train_y, args)
    with torch.inference_mode():
        logits = probe(torch.tensor(val_x, dtype=torch.float32, device=args.device)).cpu()
    hits = logits.topk(5, dim=1).indices == torch.tensor(val_y)[:, None]
    result = {"checkpoint": str(args.checkpoint), "acc1": hits[:, 0].float().mean().item(),
              "acc5": hits.any(1).float().mean().item()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
