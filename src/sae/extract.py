"""Extract frozen DINOv3 layer-11 patch tokens into token shards.

    python -m sae.extract --config configs/sae/extract_imagenet1k_train.yaml
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

from sae.backbone import DEFAULT_BACKBONE, Dinov3PatchTokens, Dinov3Transform
from sae.config import cli_config, dump_json
from sae.images import ManifestImages
from sae.manifests import build_manifest
from sae.shards import Metadata, open_shard, write_shard_sizes

logger = logging.getLogger("sae.extract")


@dataclasses.dataclass(frozen=True)
class ExtractConfig:
    output_dir: pathlib.Path
    dataset: str
    root: pathlib.Path
    split: str | None = None
    backbone: str = DEFAULT_BACKBONE
    layer: int = 11
    n_patches: int = 196
    d_model: int = 768
    max_tokens_per_shard: int = 2_400_000
    batch_size: int = 128
    num_workers: int = 8
    device: str = "cuda"


def run(cfg: ExtractConfig) -> pathlib.Path:
    shards_dir = cfg.output_dir / "shards"
    manifest_path = cfg.output_dir / "sample_manifest.parquet"
    if manifest_path.exists():
        manifest = pl.read_parquet(manifest_path)
    else:
        manifest = build_manifest(cfg.dataset, cfg.root, cfg.split)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_parquet(manifest_path)

    md = Metadata(
        ckpt=cfg.backbone,
        layers=(cfg.layer,),
        content_tokens_per_example=cfg.n_patches,
        d_model=cfg.d_model,
        n_examples=manifest.height,
        max_tokens_per_shard=cfg.max_tokens_per_shard,
    )
    md.dump(shards_dir)
    dump_json(cfg.output_dir / "run_config.json", cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    backbone = Dinov3PatchTokens(cfg.backbone, cfg.layer).to(cfg.device)
    dataset = ManifestImages(manifest, Dinov3Transform(cfg.backbone))
    state_path = shards_dir / "extraction_state.json"
    done = set(json.loads(state_path.read_text())) if state_path.exists() else set()

    sizes = []
    for shard_idx in range(md.n_shards):
        start = shard_idx * md.examples_per_shard
        end = min(start + md.examples_per_shard, md.n_examples)
        name = f"acts{shard_idx:06}.bin"
        sizes.append((name, end - start))
        if name in done:
            continue
        logger.info("Writing %s with examples [%d, %d).", name, start, end)
        shard = open_shard(shards_dir, md, name, mode="w+")
        loader = DataLoader(
            Subset(dataset, range(start, end)),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        for pixels, _, example_idx in loader:
            tokens = backbone(pixels.to(cfg.device, non_blocking=True)).cpu().numpy()
            shard[example_idx.numpy() - start, 0] = tokens
        shard.flush()
        del shard
        done.add(name)
        state_path.write_text(json.dumps(sorted(done)))

    write_shard_sizes(shards_dir, sizes)
    logger.info("Token shards written to %s", shards_dir)
    return shards_dir


if __name__ == "__main__":
    run(cli_config(ExtractConfig, __doc__))
