"""Streaming, shuffled patch-token batches from token shards.

Shards are visited in random order and read sequentially in chunks by a background
thread; tokens enter a shuffle buffer from which batches are drawn uniformly.
"""

from __future__ import annotations

import pathlib
import queue
import threading
from collections.abc import Iterator

import numpy as np
import torch

from sae.shards import Metadata, patch_tokens, shard_sizes


class TokenLoader:
    def __init__(
        self,
        shards_dir: pathlib.Path,
        *,
        layer: int,
        batch_size: int,
        n_tokens: int | None = None,
        buffer_batches: int = 64,
        chunk_examples: int = 1024,
        seed: int = 42,
    ):
        self.shards_dir = shards_dir
        self.md = Metadata.load(shards_dir)
        self.sizes = shard_sizes(shards_dir)
        self.layer = layer
        self.batch_size = batch_size
        total = self.md.n_examples * self.md.content_tokens_per_example
        self.n_tokens = total if n_tokens is None else min(n_tokens, total)
        self.buffer_tokens = buffer_batches * batch_size
        self.chunk_examples = chunk_examples
        self.seed = seed

    def __len__(self) -> int:
        return -(-self.n_tokens // self.batch_size)

    def _chunks(self, out: queue.Queue, stop: threading.Event) -> None:
        rng = np.random.default_rng(self.seed)
        for shard in rng.permutation(len(self.sizes)):
            name, n_examples = self.sizes[shard]
            tokens = patch_tokens(self.shards_dir, self.md, name, n_examples, self.layer)
            for start in range(0, n_examples, self.chunk_examples):
                if stop.is_set():
                    return
                chunk = np.ascontiguousarray(tokens[start : start + self.chunk_examples])
                out.put(chunk.reshape(-1, self.md.d_model))
        out.put(None)

    def __iter__(self) -> Iterator[torch.Tensor]:
        rng = np.random.default_rng(self.seed + 1)
        chunks: queue.Queue = queue.Queue(maxsize=8)
        stop = threading.Event()
        threading.Thread(target=self._chunks, args=(chunks, stop), daemon=True).start()
        pool = np.empty((0, self.md.d_model), dtype=np.float32)
        emitted, exhausted = 0, False
        try:
            while emitted < self.n_tokens:
                while not exhausted and len(pool) < self.buffer_tokens:
                    chunk = chunks.get()
                    exhausted = chunk is None
                    if not exhausted:
                        pool = np.concatenate([pool, chunk])
                if len(pool) == 0:
                    return
                # Emit half of the shuffled pool, keep the rest mixed with new chunks.
                pool = pool[rng.permutation(len(pool))]
                n_out = len(pool) if exhausted else max(self.batch_size, len(pool) // 2)
                n_out -= n_out % self.batch_size if not exhausted else 0
                out, pool = pool[:n_out], pool[n_out:]
                for start in range(0, len(out), self.batch_size):
                    batch = out[start : start + self.batch_size][: self.n_tokens - emitted]
                    emitted += len(batch)
                    yield torch.from_numpy(batch)
                    if emitted >= self.n_tokens:
                        return
        finally:
            stop.set()
            while not chunks.empty():
                chunks.get_nowait()
