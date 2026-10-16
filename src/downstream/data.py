"""Datasets and transforms for downstream ResNet training/evaluation."""

from __future__ import annotations

import dataclasses
import pathlib

import polars as pl
from torchvision import transforms

from sae.manifests import build_manifest, class_names

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """``name``/``root`` as in :func:`sae.manifests.build_manifest`; ``split`` is the eval split."""

    name: str
    root: pathlib.Path
    split: str = "test"


def eval_manifest(cfg: DatasetConfig) -> tuple[pl.DataFrame, list[str]]:
    manifest = build_manifest(cfg.name, cfg.root, cfg.split)
    return manifest, class_names(manifest)


def check_class_order(subset: pl.DataFrame, classes: list[str]) -> None:
    """Selection-time labels must index the same classes as the evaluation split."""
    pairs = subset.select("class_idx", "class_name").unique().iter_rows()
    wrong = [(idx, name) for idx, name in pairs if idx >= len(classes) or classes[idx] != name]
    if wrong:
        raise ValueError(f"Class order differs between selection and evaluation data, e.g. {wrong[:5]}")
