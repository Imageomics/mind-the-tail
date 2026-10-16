"""Turn a ranking (``ranked_examples.parquet``) into a training subset."""

from __future__ import annotations

import os

import polars as pl


def class_balanced(ranked: pl.DataFrame, ratio: float) -> pl.DataFrame:
    """Top ``max(1, floor(ratio * n_c))`` examples of every class ``c``."""
    return (
        ranked.sort("class_idx", "rank_within_class")
        .with_columns(
            n_keep=(pl.col("class_candidate_count") * ratio).floor().cast(pl.Int64).clip(lower_bound=1)
        )
        .filter(pl.col("rank_within_class") < pl.col("n_keep"))
        .drop("n_keep")
    )


def global_top(ranked: pl.DataFrame, ratio: float) -> pl.DataFrame:
    """Top ``floor(ratio * n)`` examples of a global ranking."""
    n_keep = max(1, int(ranked.height * ratio))
    return ranked.sort("rank_global").head(n_keep)


def select_subset(ranked: pl.DataFrame, ratio: float) -> pl.DataFrame:
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    subset = class_balanced(ranked, ratio) if "rank_within_class" in ranked.columns else global_top(ranked, ratio)
    # Released rankings store paths as ``${DATA_ROOT}/...`` so they work with any dataset location.
    return subset.with_columns(pl.col("image_path").map_elements(os.path.expandvars, return_dtype=pl.Utf8))
