"""Evaluate an SAE checkpoint.

* Reconstruction metrics on held-out token shards: MSE, normalized MSE (vs. the
  mean predictor), cosine similarity, L0, dead / dense latents.
* Optional fidelity: splice SAE reconstructions of the layer-11 patch tokens back
  into DINOv3, finish the forward pass, and compare a linear probe on the
  mean-pooled final patch features (KL and top-1 accuracy drop). The probe is a
  directory with ``config.json`` (``in_features``, ``out_features``) and
  ``model.safetensors``.

    python -m sae.evaluate --config configs/sae/evaluate.yaml
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sae.backbone import DEFAULT_BACKBONE, Dinov3PatchTokens, Dinov3Transform
from sae.config import cli_config, dump_json
from sae.images import ManifestImages
from sae.manifests import build_manifest
from sae.model import TopKSAE, load
from sae.shards import Metadata, patch_tokens, shard_sizes

logger = logging.getLogger("sae.evaluate")


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    checkpoint: pathlib.Path
    val_shards: pathlib.Path
    output_dir: pathlib.Path
    layer: int = 11
    batch_size: int = 8192
    dead_fraction: float = 1e-9
    dense_fraction: float = 1e-2
    # Fidelity (skipped when probe is None).
    probe: pathlib.Path | None = None
    dataset: str = "imagefolder"
    root: pathlib.Path | None = None
    split: str | None = None
    backbone: str = DEFAULT_BACKBONE
    image_batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"


@torch.inference_mode()
def reconstruction_metrics(sae: TopKSAE, cfg: EvalConfig) -> dict[str, float]:
    md = Metadata.load(cfg.val_shards)
    n = 0
    sum_x = torch.zeros(md.d_model, dtype=torch.float64)
    sum_sq = sse = cos = l0 = 0.0
    fired = torch.zeros(sae.cfg.d_sae, dtype=torch.long)
    for name, n_examples in shard_sizes(cfg.val_shards):
        tokens = patch_tokens(cfg.val_shards, md, name, n_examples, cfg.layer).reshape(-1, md.d_model)
        for start in range(0, len(tokens), cfg.batch_size):
            x = torch.from_numpy(np.array(tokens[start : start + cfg.batch_size])).to(cfg.device)
            f, x_hat = sae(x)
            n += len(x)
            sum_x += x.sum(0, dtype=torch.float64).cpu()
            sum_sq += x.double().pow(2).sum().item()
            sse += (x_hat - x).double().pow(2).sum().item()
            cos += F.cosine_similarity(x, x_hat, dim=1).double().sum().item()
            active = f > 0
            l0 += active.sum().item()
            fired += active.sum(0).cpu()
    mean_sse = sum_sq - sum_x.pow(2).sum().item() / n
    freq = fired.double() / n
    return {
        "n_tokens": n,
        "mse": sse / (n * md.d_model),
        "normalized_mse": sse / mean_sse,
        "cosine": cos / n,
        "l0": l0 / n,
        "n_dead": int((freq < cfg.dead_fraction).sum()),
        "n_dense": int((freq > cfg.dense_fraction).sum()),
    }


def load_probe(path: pathlib.Path, device: str) -> torch.nn.Linear:
    from safetensors.torch import load_file

    spec = json.loads((path / "config.json").read_text())
    probe = torch.nn.Linear(spec["in_features"], spec["out_features"])
    probe.load_state_dict(load_file(str(path / "model.safetensors")))
    return probe.to(device).eval()


@torch.inference_mode()
def fidelity_metrics(sae: TopKSAE, cfg: EvalConfig) -> dict[str, float]:
    assert cfg.probe is not None and cfg.root is not None
    vit = Dinov3PatchTokens(cfg.backbone, cfg.layer).model.to(cfg.device)
    probe = load_probe(cfg.probe, cfg.device)
    start = 1 + int(getattr(vit.config, "num_register_tokens", 0))
    manifest = build_manifest(cfg.dataset, cfg.root, cfg.split)
    loader = DataLoader(
        ManifestImages(manifest, Dinov3Transform(cfg.backbone)),
        batch_size=cfg.image_batch_size,
        num_workers=cfg.num_workers,
    )
    n = correct_orig = correct_recon = 0
    kl_sum = 0.0
    for pixels, labels, _ in loader:
        pixels, labels = pixels.to(cfg.device), labels.to(cfg.device)
        rope = vit.rope_embeddings(pixels)
        hidden = vit.embeddings(pixels)
        for block in vit.layer[: cfg.layer + 1]:
            hidden = block(hidden, position_embeddings=rope)
        recon = hidden.clone()
        patches = hidden[:, start:]
        recon[:, start:] = sae(patches.reshape(-1, patches.shape[-1]))[1].reshape(patches.shape)
        logits = []
        for states in (hidden, recon):
            for block in vit.layer[cfg.layer + 1 :]:
                states = block(states, position_embeddings=rope)
            logits.append(probe(vit.norm(states)[:, start:].mean(dim=1)))
        orig, rec = logits
        kl_sum += F.kl_div(rec.log_softmax(-1), orig.softmax(-1), reduction="sum").item()
        correct_orig += (orig.argmax(-1) == labels).sum().item()
        correct_recon += (rec.argmax(-1) == labels).sum().item()
        n += len(labels)
    return {
        "n_images": n,
        "mean_kl": kl_sum / n,
        "top1_acc_orig": correct_orig / n,
        "top1_acc_recon": correct_recon / n,
        "top1_acc_drop": (correct_orig - correct_recon) / n,
    }


def run(cfg: EvalConfig) -> dict[str, float]:
    sae = load(cfg.checkpoint, cfg.device)
    summary = reconstruction_metrics(sae, cfg)
    if cfg.probe is not None:
        summary |= fidelity_metrics(sae, cfg)
    logger.info("%s", summary)
    dump_json(cfg.output_dir / "summary.json", {"checkpoint": cfg.checkpoint, **summary})
    pl.DataFrame([summary]).write_parquet(cfg.output_dir / "summary.parquet")
    return summary


if __name__ == "__main__":
    run(cli_config(EvalConfig, __doc__))
