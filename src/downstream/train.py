"""Train a ResNet on a selected subset and evaluate top-1 accuracy.

The subset is the top ``ratio`` of a selection ranking (class-balanced for
within-class rankings). The model is evaluated every ``eval_every_steps`` (rounded
to whole epochs) and at the end; the best checkpoint is kept as ``best.pt``.

    python -m downstream.train --config configs/downstream/nabirds.yaml
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import random
import typing as tp

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from downstream.data import EVAL_TRANSFORM, TRAIN_TRANSFORM, DatasetConfig, check_class_order, eval_manifest
from sae.config import cli_config, dump_json
from sae.images import ManifestImages
from selection.subsets import select_subset

logger = logging.getLogger("downstream.train")


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    selection_dir: pathlib.Path
    ratio: float
    dataset: DatasetConfig
    output_dir: pathlib.Path
    arch: tp.Literal["resnet50"] = "resnet50"
    pretrained: bool = True
    max_steps: int = 18_700
    batch_size: int = 128
    eval_batch_size: int = 256
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    scheduler: tp.Literal["none", "cosine"] = "none"
    eval_every_steps: int = 800
    log_every_steps: int = 1000
    num_workers: int = 16
    seed: int = 0
    device: str = "cuda"


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    builder = getattr(models, arch)
    if not pretrained:
        return builder(weights=None, num_classes=num_classes)
    model = builder(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for images, labels, _ in loader:
        logits = model(images.to(device, non_blocking=True))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += len(labels)
    model.train()
    return correct / total


def run(cfg: TrainConfig) -> dict[str, float]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(cfg.output_dir / "training.log"))

    subset = select_subset(pl.read_parquet(cfg.selection_dir / "ranked_examples.parquet"), cfg.ratio)
    subset.write_parquet(cfg.output_dir / "selected_examples.parquet")
    val_manifest, classes = eval_manifest(cfg.dataset)
    check_class_order(subset, classes)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(ManifestImages(subset, TRAIN_TRANSFORM), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(ManifestImages(val_manifest, EVAL_TRANSFORM), batch_size=cfg.eval_batch_size,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(cfg.arch, len(classes), cfg.pretrained).to(cfg.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=1e-4)
                 if cfg.scheduler == "cosine" else None)
    criterion = nn.CrossEntropyLoss()
    epochs_per_eval = max(cfg.eval_every_steps // len(train_loader), 1)
    logger.info("%d training images, %d classes, %d steps/epoch", subset.height, len(classes), len(train_loader))

    def save(name: str, step: int, acc: float) -> None:
        torch.save({"model_state_dict": model.state_dict(), "step": step, "val_top1_acc": acc,
                    "classes": classes, "arch": cfg.arch}, cfg.output_dir / "checkpoints" / name)

    (cfg.output_dir / "checkpoints").mkdir(exist_ok=True)
    step, epoch, best = 0, 0, -1.0
    model.train()
    while step < cfg.max_steps:
        for images, labels, _ in train_loader:
            loss = criterion(model(images.to(cfg.device, non_blocking=True)), labels.to(cfg.device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            if step % cfg.log_every_steps == 0:
                logger.info("step %d loss %.4f lr %.3g", step, loss.item(), optimizer.param_groups[0]["lr"])
            if step >= cfg.max_steps:
                break
        last_epoch = step >= cfg.max_steps
        if epoch % epochs_per_eval == 0 or last_epoch:
            acc = evaluate(model, val_loader, cfg.device)
            logger.info("epoch %d step %d val top-1 %.4f", epoch, step, acc)
            if acc >= best:
                best = acc
                save("best.pt", step, acc)
        epoch += 1

    save("last.pt", step, acc)
    metrics = {"final_val_top1_acc": acc, "best_val_top1_acc": best, "n_train": subset.height}
    dump_json(cfg.output_dir / "metrics.json", metrics)
    dump_json(cfg.output_dir / "run_config.json", cfg)
    return metrics


if __name__ == "__main__":
    run(cli_config(TrainConfig, __doc__))
