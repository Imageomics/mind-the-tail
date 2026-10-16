"""Load image-level SAE activations produced by ``sae.infer``."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import scipy.sparse


def load_image_activations(artifact_dir: pathlib.Path) -> tuple[scipy.sparse.csr_matrix, pl.DataFrame]:
    """Return the stacked CSR ``[n_images, d_sae]`` and its row-aligned sample manifest."""
    shards = pl.read_parquet(artifact_dir / "image_shard_manifest.parquet").sort("shard_id")
    matrix = scipy.sparse.vstack(
        [scipy.sparse.load_npz(artifact_dir / path) for path in shards["image_shard_path"]], format="csr"
    ).astype(np.float32)
    manifest = pl.read_parquet(artifact_dir / "sample_manifest.parquet").sort("example_idx")
    if manifest.height != matrix.shape[0] or not np.array_equal(manifest["example_idx"], np.arange(manifest.height)):
        raise ValueError(f"{artifact_dir}: sample manifest is not row-aligned with the activations.")
    return matrix, manifest


def drop_dead_latents(matrix: scipy.sparse.csr_matrix) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Remove latents that never fire in this pool; returns the kept column ids."""
    active = np.flatnonzero(matrix.getnnz(axis=0))
    return matrix[:, active].tocsr(), active
