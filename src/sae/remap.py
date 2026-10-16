"""Rewrite token shards in a globally shuffled example order.

Extraction walks datasets class by class, so each shard would otherwise hold only a
few classes. Shuffling examples across shards lets the streaming SAE loader see a
class-mixed token stream.

    python -m sae.remap --config configs/sae/remap_imagenet1k_train.yaml
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import numpy as np
import polars as pl

from sae.config import cli_config, dump_json
from sae.shards import Metadata, open_shard, shard_sizes, write_shard_sizes

logger = logging.getLogger("sae.remap")


@dataclasses.dataclass(frozen=True)
class RemapConfig:
    input_dir: pathlib.Path
    output_dir: pathlib.Path
    seed: int = 42
    n_examples: int | None = None


def _source_locations(sizes: list[tuple[str, int]], examples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.cumsum([0] + [n for _, n in sizes[:-1]])
    shard = np.searchsorted(starts, examples, side="right") - 1
    return shard, examples - starts[shard]


def run(cfg: RemapConfig) -> pathlib.Path:
    src_dir = cfg.input_dir / "shards"
    md = Metadata.load(src_dir)
    sizes = shard_sizes(src_dir)
    order = np.random.default_rng(cfg.seed).permutation(md.n_examples)[: cfg.n_examples]

    out_dir = cfg.output_dir / "shards"
    out_md = dataclasses.replace(md, n_examples=len(order))
    out_md.dump(out_dir)
    manifest = pl.read_parquet(cfg.input_dir / "sample_manifest.parquet").sort("example_idx")
    remapped = manifest[order.tolist()].rename({"example_idx": "source_example_idx"}).with_row_index("example_idx")
    remapped.with_columns(pl.col("example_idx").cast(pl.Int64)).write_parquet(cfg.output_dir / "sample_manifest.parquet")
    np.save(out_dir / "remap_order.npy", order)
    dump_json(cfg.output_dir / "run_config.json", cfg)

    out_sizes = []
    for shard_idx in range(out_md.n_shards):
        start = shard_idx * out_md.examples_per_shard
        chunk = order[start : start + out_md.examples_per_shard]
        name = f"acts{shard_idx:06}.bin"
        logger.info("Writing %s (%d examples).", name, len(chunk))
        dst = open_shard(out_dir, out_md, name, mode="w+")
        src_shard, src_row = _source_locations(sizes, chunk)
        for s in np.unique(src_shard):
            mask = src_shard == s
            src = open_shard(src_dir, md, sizes[s][0])
            # Sorted reads keep the memmap access pattern sequential.
            dst_pos = np.flatnonzero(mask)[np.argsort(src_row[mask])]
            dst[dst_pos] = src[np.sort(src_row[mask])]
            del src
        dst.flush()
        del dst
        out_sizes.append((name, len(chunk)))
    write_shard_sizes(out_dir, out_sizes)
    return out_dir


if __name__ == "__main__":
    run(cli_config(RemapConfig, __doc__))
