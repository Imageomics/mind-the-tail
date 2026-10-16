"""Rank a pool of images with the neighborhood-weighted concept-coverage objective.

``mode: within_class`` builds the kNN graph and runs greedy separately inside each
class (labels available; subsets are then class-balanced). ``mode: global`` ranks
the whole pool at once without labels.

    python -m selection.select --config configs/selection/nabirds.yaml
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import typing as tp

import numpy as np
import polars as pl
import scipy.sparse

from sae.config import cli_config, dump_json
from selection.data import drop_dead_latents, load_image_activations
from selection.objective import HnswConfig, exact_knn, hnsw_knn, neighborhood_vectors, rank

logger = logging.getLogger("selection.select")


@dataclasses.dataclass(frozen=True)
class SelectConfig:
    image_activations: pathlib.Path
    output_dir: pathlib.Path
    mode: tp.Literal["within_class", "global"] = "within_class"
    beta: float = 0.05
    k: int = 3
    neighbors: tp.Literal["exact", "hnsw"] = "exact"
    hnsw: HnswConfig = HnswConfig()


def rank_pool(vectors: scipy.sparse.csr_matrix, cfg: SelectConfig) -> tuple[np.ndarray, np.ndarray]:
    if vectors.shape[0] == 1:
        return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.float32)
    knn = exact_knn if cfg.neighbors == "exact" else lambda v, k: hnsw_knn(v, k, cfg.hnsw)
    idx, sim = knn(vectors, cfg.k)
    return rank(neighborhood_vectors(vectors, idx, sim, cfg.beta))


def run(cfg: SelectConfig) -> pathlib.Path:
    vectors, manifest = load_image_activations(cfg.image_activations)
    vectors, active = drop_dead_latents(vectors)
    logger.info("Pool: %d images, %d active latents.", *vectors.shape)
    base = manifest.select("example_idx", "image_path", "class_idx", "class_name")

    if cfg.mode == "global":
        order, gain = rank_pool(vectors, cfg)
        ranked = base[order.tolist()].with_columns(
            rank_global=pl.int_range(len(order), dtype=pl.Int64), selection_gain=pl.Series(gain)
        )
    else:
        parts = []
        for class_idx, rows in manifest.group_by("class_idx", maintain_order=True).agg(pl.col("example_idx")).sort("class_idx").iter_rows():
            rows = np.asarray(rows)
            order, gain = rank_pool(vectors[rows], cfg)
            parts.append(
                base[rows[order].tolist()].with_columns(
                    rank_within_class=pl.int_range(len(rows), dtype=pl.Int64),
                    class_candidate_count=pl.lit(len(rows), dtype=pl.Int64),
                    selection_gain=pl.Series(gain),
                )
            )
            logger.info("class %d: ranked %d images", class_idx, len(rows))
        ranked = pl.concat(parts)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ranked.write_parquet(cfg.output_dir / "ranked_examples.parquet")
    np.save(cfg.output_dir / "active_latent_indices.npy", active)
    dump_json(cfg.output_dir / "run_config.json", cfg)
    return cfg.output_dir


if __name__ == "__main__":
    run(cli_config(SelectConfig, __doc__))
