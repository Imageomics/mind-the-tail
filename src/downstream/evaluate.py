"""Top-1 accuracy of a trained checkpoint; per-group and worst-group accuracy when the
dataset defines groups (Waterbirds, Spawrious).

    python -m downstream.evaluate --config configs/downstream/eval_waterbirds.yaml
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import polars as pl
import torch
from torch.utils.data import DataLoader

from downstream.data import EVAL_TRANSFORM, DatasetConfig, eval_manifest
from downstream.train import build_model
from sae.config import cli_config, dump_json
from sae.images import ManifestImages

logger = logging.getLogger("downstream.evaluate")


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    checkpoint: pathlib.Path
    dataset: DatasetConfig
    output_dir: pathlib.Path
    arch: str = "resnet50"
    batch_size: int = 256
    num_workers: int = 8
    device: str = "cuda"


@torch.inference_mode()
def run(cfg: EvalConfig) -> dict[str, object]:
    manifest, classes = eval_manifest(cfg.dataset)
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("classes", classes) != classes:
        raise ValueError("Checkpoint class order differs from the evaluation dataset.")
    model = build_model(cfg.arch, len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device).eval()

    loader = DataLoader(ManifestImages(manifest, EVAL_TRANSFORM), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    predictions = []
    for images, _, example_idx in loader:
        predictions.append(pl.DataFrame({
            "example_idx": example_idx.numpy(),
            "pred": model(images.to(cfg.device)).argmax(1).cpu().numpy(),
        }))
    results = manifest.join(pl.concat(predictions), on="example_idx").with_columns(
        correct=(pl.col("pred") == pl.col("class_idx"))
    )
    summary: dict[str, object] = {"checkpoint": cfg.checkpoint, "top1_acc": results["correct"].mean()}
    if "group_idx" in results.columns:
        groups = results.group_by("group_idx").agg(pl.col("correct").mean().alias("acc"), pl.len().alias("n"))
        worst = groups.sort("acc").row(0, named=True)
        summary |= {"worst_group_acc": worst["acc"], "worst_group_idx": worst["group_idx"],
                    "group_acc": groups.sort("group_idx").to_dicts()}
    logger.info("%s", {k: v for k, v in summary.items() if k != "group_acc"})
    dump_json(cfg.output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run(cli_config(EvalConfig, __doc__))
