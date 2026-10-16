"""TopK sparse autoencoder with Matryoshka (nested-prefix) decoding.

Checkpoints are a one-line JSON header followed by a ``torch.save``'d state dict
(``W_enc, b_enc, W_dec, b_dec``).
"""

from __future__ import annotations

import dataclasses
import io
import json
import pathlib

import torch


@dataclasses.dataclass(frozen=True)
class SAEConfig:
    d_model: int = 768
    d_sae: int = 16_384
    top_k: int = 32
    aux_k: int = 512
    aux_alpha: float = 1 / 32


class TopKSAE(torch.nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        w_dec = torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_sae, cfg.d_model))
        self.W_dec = torch.nn.Parameter(w_dec)
        self.b_dec = torch.nn.Parameter(torch.zeros(cfg.d_model))
        self.normalize_decoder()
        self.W_enc = torch.nn.Parameter(self.W_dec.data.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(cfg.d_sae))

    def pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_activation(x)
        values, idx = h.topk(self.cfg.top_k, dim=-1, sorted=False)
        return torch.zeros_like(h).scatter(-1, idx, values)

    def decode(self, f: torch.Tensor, prefixes: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstructions using latents ``[0, p)`` for each prefix length ``p``.

        Returns ``[batch, n_prefixes, d_model]``; the last prefix is always ``d_sae``.
        """
        if prefixes is None:
            return (f @ self.W_dec + self.b_dec)[:, None, :]
        bounds = [0, *prefixes.tolist()]
        blocks = [f[:, a:b] @ self.W_dec[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        blocks[0] = blocks[0] + self.b_dec
        return torch.cumsum(torch.stack(blocks, dim=1), dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return f, self.decode(f)[:, -1]

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grads(self) -> None:
        """Drop the gradient component along each unit-norm decoder row."""
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
        self.W_dec.grad -= parallel * self.W_dec.data


def dump(path: pathlib.Path, sae: TopKSAE) -> None:
    cfg = sae.cfg
    header = {
        "schema": 5,
        "cfg": {
            "d_model": cfg.d_model,
            "d_sae": cfg.d_sae,
            "activation": {
                "cls": "TopK",
                "params": {
                    "top_k": cfg.top_k,
                    "aux": {"cls": "AuxK", "params": {"k_aux": cfg.aux_k, "alpha": cfg.aux_alpha}},
                },
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fd:
        fd.write((json.dumps(header) + "\n").encode())
        torch.save(sae.state_dict(), fd)


def load(path: pathlib.Path, device: str = "cpu") -> TopKSAE:
    with open(path, "rb") as fd:
        header = json.loads(fd.readline())
        state = torch.load(io.BytesIO(fd.read()), map_location=device, weights_only=True)
    params = header["cfg"]["activation"]["params"]
    aux = params.get("aux", {}).get("params", {})
    cfg = SAEConfig(
        d_model=header["cfg"]["d_model"],
        d_sae=header["cfg"]["d_sae"],
        top_k=params["top_k"],
        aux_k=aux.get("k_aux", 512),
        aux_alpha=aux.get("alpha", 1 / 32),
    )
    sae = TopKSAE(cfg)
    sae.load_state_dict(state)
    return sae.to(device).eval()
