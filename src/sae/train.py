"""Train a TopK SAE on patch-token shards with a Matryoshka + AuxK objective.

    python -m sae.train --config configs/sae/train_topk32.yaml
"""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib

import torch

from sae.config import cli_config, dump_json
from sae.loader import TokenLoader
from sae.model import SAEConfig, TopKSAE, dump
from sae.shards import Metadata

logger = logging.getLogger("sae.train")


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    train_shards: pathlib.Path
    output_dir: pathlib.Path
    layer: int = 11
    d_sae: int = 16_384
    top_k: int = 32
    aux_k: int = 512
    aux_alpha: float = 1 / 32
    n_prefixes: int = 10
    dead_threshold_tokens: int = 10_000_000
    reinit_blend: float = 0.8
    n_train: int = 100_000_000
    batch_size: int = 16_384
    lr: float = 1e-3
    n_lr_warmup: int = 500
    grad_clip: float = 1.0
    log_every: int = 25
    seed: int = 42
    device: str = "cuda"


def sample_prefixes(d_sae: int, n_prefixes: int) -> torch.Tensor:
    """Sorted Matryoshka prefix lengths drawn from a Pareto(0.5) law, ending at ``d_sae``."""
    if n_prefixes <= 1:
        return torch.tensor([d_sae])
    lengths = torch.arange(1, d_sae)
    cdf = 1 - (1 / lengths.float()) ** 0.5
    pdf = torch.cat([cdf[:1], cdf[1:] - cdf[:-1]])
    picks = lengths[torch.multinomial(pdf / pdf.sum(), n_prefixes - 1, replacement=False)]
    return torch.sort(torch.cat([picks, torch.tensor([d_sae])])).values


def aux_k_loss(sae: TopKSAE, x: torch.Tensor, x_hat: torch.Tensor, h: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
    """Reconstruct the residual from the top ``aux_k`` dead latents (revives dead features)."""
    k = min(sae.cfg.aux_k, int(dead.sum()))
    if k == 0:
        return x.new_zeros(())
    _, idx = h.masked_fill(~dead, float("-inf")).topk(k, dim=-1)
    aux = torch.zeros_like(h).scatter(-1, idx, h.gather(-1, idx))
    residual = (x - x_hat).detach()
    return sae.cfg.aux_alpha * (sae.decode(aux)[:, -1] - residual).pow(2).mean()


@torch.no_grad()
def datapoint_init(sae: TopKSAE, loader: TokenLoader, blend: float) -> None:
    """Initialise encoder rows as a blend of centred data points and Kaiming noise."""
    n = min(max(sae.cfg.d_sae, 65_536), loader.n_tokens)
    acts, seen = [], 0
    for batch in loader:
        acts.append(batch)
        seen += len(batch)
        if seen >= n:
            break
    acts = torch.cat(acts)[torch.randperm(seen)[:n]]
    centred = acts[: sae.cfg.d_sae] - acts.mean(dim=0, keepdim=True)
    noise = torch.nn.init.kaiming_uniform_(torch.empty_like(centred))
    rows = blend * centred + (1 - blend) * noise
    sae.W_dec.data.copy_(rows[torch.randperm(sae.cfg.d_sae)].to(sae.W_dec.device))
    sae.normalize_decoder()
    sae.W_enc.data.copy_(sae.W_dec.data.T)


def lr_at(step: int, cfg: TrainConfig, total: int) -> float:
    if step < cfg.n_lr_warmup:
        return cfg.lr * step / cfg.n_lr_warmup
    progress = (step - cfg.n_lr_warmup) / max(1, total - cfg.n_lr_warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def run(cfg: TrainConfig) -> pathlib.Path:
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    md = Metadata.load(cfg.train_shards)
    sae = TopKSAE(SAEConfig(md.d_model, cfg.d_sae, cfg.top_k, cfg.aux_k, cfg.aux_alpha))
    loader = TokenLoader(cfg.train_shards, layer=cfg.layer, batch_size=cfg.batch_size, n_tokens=cfg.n_train, seed=cfg.seed)
    if cfg.reinit_blend > 0:
        datapoint_init(sae, loader, cfg.reinit_blend)
    sae.to(cfg.device).train()
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.0, fused=cfg.device == "cuda")
    tokens_since_active = torch.zeros(cfg.d_sae, dtype=torch.long, device=cfg.device)
    total_steps = len(loader)

    for step, x in enumerate(loader):
        x = x.to(cfg.device, non_blocking=True)
        sae.normalize_decoder()
        h = sae.pre_activation(x)
        values, idx = h.topk(cfg.top_k, dim=-1, sorted=False)
        f = torch.zeros_like(h).scatter(-1, idx, values)
        with torch.no_grad():
            tokens_since_active += len(x)
            tokens_since_active[(f != 0).any(dim=0)] = 0
            dead = tokens_since_active >= cfg.dead_threshold_tokens

        x_hats = sae.decode(f, sample_prefixes(cfg.d_sae, cfg.n_prefixes))
        mse = (x_hats - x[:, None, :]).pow(2).mean()
        aux = aux_k_loss(sae, x, x_hats[:, -1], h, dead)
        (mse + aux).backward()
        sae.remove_parallel_grads()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step + 1, cfg, total_steps)

        if step % cfg.log_every == 0:
            logger.info("step %d/%d mse=%.4f aux=%.4f dead=%d", step, total_steps, mse.item(), aux.item(), int(dead.sum()))

    ckpt = cfg.output_dir / "sae.pt"
    dump(ckpt, sae.eval().cpu())
    dump_json(cfg.output_dir / "config.json", cfg)
    logger.info("Saved %s", ckpt)
    return ckpt


if __name__ == "__main__":
    run(cli_config(TrainConfig, __doc__))
