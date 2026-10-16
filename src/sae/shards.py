"""On-disk format of backbone token activations.

A shards directory holds ``metadata.json``, ``shards.json`` and float32 memmaps
``acts000000.bin, ...`` of shape ``[examples_per_shard, n_layers, tokens, d_model]``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class Metadata:
    ckpt: str
    layers: tuple[int, ...]
    content_tokens_per_example: int
    d_model: int
    n_examples: int
    max_tokens_per_shard: int
    cls_token: bool = False

    @classmethod
    def load(cls, shards_dir: pathlib.Path) -> "Metadata":
        raw = json.loads((shards_dir / "metadata.json").read_text(encoding="utf-8"))
        names = {field.name for field in dataclasses.fields(cls)}
        raw["layers"] = tuple(raw["layers"])
        return cls(**{key: value for key, value in raw.items() if key in names})

    def dump(self, shards_dir: pathlib.Path) -> None:
        shards_dir.mkdir(parents=True, exist_ok=True)
        payload = {**dataclasses.asdict(self), "layers": list(self.layers), "dtype": "float32"}
        (shards_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @property
    def tokens_per_example(self) -> int:
        return self.content_tokens_per_example + int(self.cls_token)

    @property
    def examples_per_shard(self) -> int:
        return self.max_tokens_per_shard // (self.tokens_per_example * len(self.layers))

    @property
    def shard_shape(self) -> tuple[int, int, int, int]:
        return (self.examples_per_shard, len(self.layers), self.tokens_per_example, self.d_model)

    @property
    def n_shards(self) -> int:
        return -(-self.n_examples // self.examples_per_shard)


def shard_sizes(shards_dir: pathlib.Path) -> list[tuple[str, int]]:
    entries = json.loads((shards_dir / "shards.json").read_text(encoding="utf-8"))
    return [(entry["name"], int(entry["n_examples"])) for entry in entries]


def write_shard_sizes(shards_dir: pathlib.Path, sizes: list[tuple[str, int]]) -> None:
    payload = [{"name": name, "n_examples": n} for name, n in sizes]
    (shards_dir / "shards.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def open_shard(shards_dir: pathlib.Path, md: Metadata, name: str, mode: str = "r") -> np.memmap:
    return np.memmap(shards_dir / name, mode=mode, dtype=np.float32, shape=md.shard_shape)


def patch_tokens(shards_dir: pathlib.Path, md: Metadata, name: str, n_examples: int, layer: int) -> np.ndarray:
    """Lazy ``[n_examples, n_patches, d_model]`` view of one shard's patch tokens."""
    array = open_shard(shards_dir, md, name)
    start = int(md.cls_token)
    return array[:n_examples, md.layers.index(layer), start : start + md.content_tokens_per_example]
