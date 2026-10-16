"""SAE inference and image-level max pooling.

Three commands, all producing sparse ``.npz`` shards plus a parquet shard manifest:

    python -m sae.infer tokens --config ...  # token shards -> sparse token latents
    python -m sae.infer pool   --config ...  # token latents -> image vectors (max over patches)
    python -m sae.infer images --config ...  # dataset -> DINOv3 -> SAE -> image vectors
                                             # in one pass (no token files; used for CC12M)

An *image-activation artifact* (the input to selection) is a directory with
``image_shards/*.npz`` (CSR, one row per image), ``image_shard_manifest.parquet``
and ``sample_manifest.parquet``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import shutil
import sys

import numpy as np
import polars as pl
import scipy.sparse
import torch
from torch.utils.data import DataLoader, Subset

from sae.backbone import DEFAULT_BACKBONE, Dinov3PatchTokens, Dinov3Transform
from sae.config import dump_json, load_config
from sae.images import ManifestImages, collate_valid
from sae.manifests import build_manifest
from sae.model import load
from sae.shards import Metadata, patch_tokens, shard_sizes

logger = logging.getLogger("sae.infer")


@dataclasses.dataclass(frozen=True)
class TokenInferConfig:
    shards_dir: pathlib.Path
    sample_manifest: pathlib.Path
    checkpoint: pathlib.Path
    output_dir: pathlib.Path
    layer: int = 11
    batch_size: int = 8192
    device: str = "cuda"


@dataclasses.dataclass(frozen=True)
class PoolConfig:
    token_dir: pathlib.Path
    output_dir: pathlib.Path


@dataclasses.dataclass(frozen=True)
class ImageInferConfig:
    dataset: str
    root: pathlib.Path
    checkpoint: pathlib.Path
    output_dir: pathlib.Path
    split: str | None = None
    backbone: str = DEFAULT_BACKBONE
    layer: int = 11
    images_per_shard: int = 10_000
    batch_size: int = 32
    num_workers: int = 8
    skip_invalid_images: bool = False
    device: str = "cuda"


def to_csr(dense: torch.Tensor) -> scipy.sparse.csr_matrix:
    """Sparsify on the device; only the nonzeros are copied to the host."""
    rows, cols = dense.nonzero(as_tuple=True)
    values = dense[rows, cols]
    return scipy.sparse.csr_matrix(
        (values.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())), shape=tuple(dense.shape), dtype=np.float32
    )


def max_pool(latents: torch.Tensor) -> scipy.sparse.csr_matrix:
    """``[images, patches, d_sae]`` -> CSR ``[images, d_sae]`` of positive patch maxima."""
    return to_csr(latents.max(dim=1).values.clamp_min(0))


def _encode_images(sae, patches: torch.Tensor) -> torch.Tensor:
    n_images, n_patches, d_model = patches.shape
    return sae.encode(patches.reshape(-1, d_model)).reshape(n_images, n_patches, -1)


def _write_manifest(rows: list[dict], path: pathlib.Path) -> None:
    pl.DataFrame(rows).sort("shard_id").write_parquet(path)


@torch.inference_mode()
def run_tokens(cfg: TokenInferConfig) -> pathlib.Path:
    torch.backends.cuda.matmul.allow_tf32 = False
    md = Metadata.load(cfg.shards_dir)
    sae = load(cfg.checkpoint, cfg.device)
    (cfg.output_dir / "token_shards").mkdir(parents=True, exist_ok=True)
    rows = []
    for shard_id, (name, n_examples) in enumerate(shard_sizes(cfg.shards_dir)):
        tokens = patch_tokens(cfg.shards_dir, md, name, n_examples, cfg.layer).reshape(-1, md.d_model)
        parts = []
        for start in range(0, len(tokens), cfg.batch_size):
            x = torch.from_numpy(np.array(tokens[start : start + cfg.batch_size])).to(cfg.device)
            parts.append(to_csr(sae.encode(x)))
        path = pathlib.Path("token_shards") / f"{pathlib.Path(name).stem}.npz"
        scipy.sparse.save_npz(cfg.output_dir / path, scipy.sparse.vstack(parts, format="csr"))
        rows.append({"shard_id": shard_id, "token_shard_path": str(path), "n_examples": n_examples,
                     "rows_per_example": md.content_tokens_per_example})
        logger.info("Wrote %s", path)
    _write_manifest(rows, cfg.output_dir / "token_shard_manifest.parquet")
    shutil.copy2(cfg.sample_manifest, cfg.output_dir / "sample_manifest.parquet")
    dump_json(cfg.output_dir / "run_config.json", cfg)
    return cfg.output_dir


def run_pool(cfg: PoolConfig) -> pathlib.Path:
    (cfg.output_dir / "image_shards").mkdir(parents=True, exist_ok=True)
    rows = []
    for row in pl.read_parquet(cfg.token_dir / "token_shard_manifest.parquet").sort("shard_id").iter_rows(named=True):
        coo = scipy.sparse.load_npz(cfg.token_dir / row["token_shard_path"]).tocoo()
        n_images = row["n_examples"]
        image = coo.row // row["rows_per_example"]
        # Max over patches per (image, latent); only positive activations are stored.
        keys = image.astype(np.int64) * coo.shape[1] + coo.col
        order = np.lexsort((-coo.data, keys))
        first = np.unique(keys[order], return_index=True)[1]
        keep = order[first][coo.data[order[first]] > 0]
        pooled = scipy.sparse.csr_matrix(
            (coo.data[keep], (image[keep], coo.col[keep])), shape=(n_images, coo.shape[1]), dtype=np.float32
        )
        path = pathlib.Path("image_shards") / pathlib.Path(row["token_shard_path"]).name
        scipy.sparse.save_npz(cfg.output_dir / path, pooled)
        rows.append({"shard_id": row["shard_id"], "image_shard_path": str(path), "n_images": n_images,
                     "image_nnz": pooled.nnz})
    _write_manifest(rows, cfg.output_dir / "image_shard_manifest.parquet")
    shutil.copy2(cfg.token_dir / "sample_manifest.parquet", cfg.output_dir / "sample_manifest.parquet")
    dump_json(cfg.output_dir / "run_config.json", cfg)
    return cfg.output_dir


@torch.inference_mode()
def run_images(cfg: ImageInferConfig) -> pathlib.Path:
    torch.backends.cuda.matmul.allow_tf32 = True
    out = cfg.output_dir
    (out / "image_shards").mkdir(parents=True, exist_ok=True)
    source_path = out / "source_manifest.parquet"
    if not source_path.exists():
        build_manifest(cfg.dataset, cfg.root, cfg.split).write_parquet(source_path)
    source = pl.read_parquet(source_path)

    backbone = Dinov3PatchTokens(cfg.backbone, cfg.layer).to(cfg.device)
    sae = load(cfg.checkpoint, cfg.device)
    dataset = ManifestImages(source, Dinov3Transform(cfg.backbone), skip_invalid=cfg.skip_invalid_images)
    state_path = out / "inference_state.json"
    done = {row["shard_id"]: row for row in json.loads(state_path.read_text())} if state_path.exists() else {}

    for shard_id in range(-(-source.height // cfg.images_per_shard)):
        if shard_id in done:
            continue
        start = shard_id * cfg.images_per_shard
        stop = min(start + cfg.images_per_shard, source.height)
        loader = DataLoader(Subset(dataset, range(start, stop)), batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_valid)
        parts, valid = [], []
        for batch in loader:
            if batch is None:
                continue
            pixels, _, example_idx = batch
            parts.append(max_pool(_encode_images(sae, backbone(pixels.to(cfg.device)))))
            valid.append(example_idx.numpy())
        path = pathlib.Path("image_shards") / f"image_acts{shard_id:06}.npz"
        matrix = scipy.sparse.vstack(parts, format="csr")
        scipy.sparse.save_npz(out / path, matrix)
        np.save(out / "image_shards" / f"valid_idx{shard_id:06}.npy", np.concatenate(valid))
        done[shard_id] = {"shard_id": shard_id, "image_shard_path": str(path), "n_images": matrix.shape[0],
                          "image_nnz": matrix.nnz, "n_invalid": (stop - start) - matrix.shape[0]}
        state_path.write_text(json.dumps(list(done.values())))
        logger.info("Wrote %s (%d images)", path, matrix.shape[0])

    _write_manifest(list(done.values()), out / "image_shard_manifest.parquet")
    valid = np.concatenate([np.load(out / "image_shards" / f"valid_idx{i:06}.npy") for i in sorted(done)])
    manifest = source[valid.tolist()].rename({"example_idx": "source_example_idx"}).with_row_index("example_idx")
    manifest.with_columns(pl.col("example_idx").cast(pl.Int64)).write_parquet(out / "sample_manifest.parquet")
    dump_json(out / "run_config.json", cfg)
    return out


COMMANDS = {"tokens": (TokenInferConfig, run_tokens), "pool": (PoolConfig, run_pool), "images": (ImageInferConfig, run_images)}


def main(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s")
    schema, fn = COMMANDS[args.command]
    fn(load_config(args.config, schema, args.set))


if __name__ == "__main__":
    main(sys.argv[1:])
