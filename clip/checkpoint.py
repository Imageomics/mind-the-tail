"""Load open_clip training checkpoints into a fresh model."""

from __future__ import annotations

from pathlib import Path

import open_clip
import torch


def load_model(arch: str, checkpoint: Path, device: str):
    """Return ``(model, eval_preprocess, tokenizer)`` with the checkpoint weights."""
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=None, device=device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    # DDP and torch.compile wrap parameter names.
    state = {key.removeprefix("module.").removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.eval(), preprocess, open_clip.get_tokenizer(arch)
