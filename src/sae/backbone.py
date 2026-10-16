"""Frozen DINOv3 (Hugging Face) backbone that returns one layer's patch tokens."""

from __future__ import annotations

import torch
from PIL import Image

DEFAULT_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class Dinov3PatchTokens(torch.nn.Module):
    """``forward(pixels) -> [batch, n_patches, d_model]`` residual stream after ``layer``.

    The CLS and register tokens are dropped; only the spatial patch tokens are kept.
    """

    def __init__(self, ref: str = DEFAULT_BACKBONE, layer: int = 11):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(ref).eval().requires_grad_(False)
        self.layer = layer
        self.patch_start = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        self._captured: torch.Tensor | None = None
        self.model.layer[layer].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output) -> None:
        self._captured = output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        self._captured = None
        self.model(pixel_values=pixels)
        assert self._captured is not None, "layer hook did not fire"
        return self._captured[:, self.patch_start :, :].float()


class Dinov3Transform:
    """The backbone's own Hugging Face image preprocessing (224x224 -> 196 patches)."""

    def __init__(self, ref: str = DEFAULT_BACKBONE):
        from transformers import AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(ref)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]
