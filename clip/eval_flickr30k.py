#!/usr/bin/env python3
"""Zero-shot Flickr30k retrieval (Karpathy test split, 1000 images x 5 captions).

Expects ``<root>/flickr_annotations_30k.csv`` (columns ``raw`` = JSON list of five
captions, ``filename``, ``split``) and ``<root>/images``. Reports Recall@1/5/10 for
image->text and text->image.

    python clip/eval_flickr30k.py --checkpoint RUN/checkpoints/epoch_5.pt --root ${DATA_ROOT}/flickr30k
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from checkpoint import load_model


def load_split(root: Path, split: str) -> tuple[list[str], list[str]]:
    with open(root / "flickr_annotations_30k.csv", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    captions = [caption for row in rows for caption in json.loads(row["raw"])]
    assert all(len(json.loads(row["raw"])) == 5 for row in rows)
    return [row["filename"] for row in rows], captions


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("flickr30k_retrieval.json"))
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args.model, args.checkpoint, args.device)
    files, captions = load_split(args.root, args.split)
    amp = torch.autocast("cuda", enabled=args.device.startswith("cuda"))
    image_feats, text_feats = [], []
    for start in range(0, len(files), args.batch_size):
        images = torch.stack([preprocess(Image.open(args.root / "images" / f).convert("RGB"))
                              for f in files[start : start + args.batch_size]])
        with amp:
            image_feats.append(F.normalize(model.encode_image(images.to(args.device)).float(), dim=-1))
    for start in range(0, len(captions), args.batch_size):
        with amp:
            tokens = tokenizer(captions[start : start + args.batch_size]).to(args.device)
            text_feats.append(F.normalize(model.encode_text(tokens).float(), dim=-1))
    scores = torch.cat(image_feats) @ torch.cat(text_feats).T  # [images, captions]
    owner = torch.arange(len(files), device=scores.device).repeat_interleave(5)

    result = {"checkpoint": str(args.checkpoint)}
    i2t = owner[scores.topk(10, dim=1).indices]
    t2i = scores.T.topk(10, dim=1).indices
    for k in (1, 5, 10):
        result[f"image_to_text_R@{k}"] = (i2t[:, :k] == torch.arange(len(files), device=scores.device)[:, None]).any(1).float().mean().item()
        result[f"text_to_image_R@{k}"] = (t2i[:, :k] == owner[:, None]).any(1).float().mean().item()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
