#!/usr/bin/env python

"""Image encoder module (DINOv2-only).

This module intentionally contains *only* the image backbone encoder.
Pooling / projection utilities live in `policies.modules.fusion`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from typing import Any, Literal

try:
    from transformers import AutoModel
except ImportError as e:
    raise ImportError(
        "`policies.modules.vision.DinoImageEncoder` requires `transformers` (DINOv2 backbone). "
        "Install with `uv add transformers` or `pip install transformers`."
    ) from e

_ImageEncoderOutput = Literal["patch", "cls", "both"]

class DinoImageEncoder(nn.Module):
    """
    DINOv2 image encoder for RGB observations.

    The encoder consumes RGB images and returns a per-image feature vector that can be
    concatenated with other conditioning signals (e.g., robot state).
    """

    def __init__(
        self,
        model_size: str = "small",
        input_shape: tuple[int, int, int] = (3, 256, 256),
        crop_shape: tuple[int, int] | None = (224, 224),
        crop_is_random: bool = True,
        freeze_backbone: bool = True,
        default_output: _ImageEncoderOutput = "patch",
        *,
        auto_resize_to_224: bool = False,
        imagenet_normalize: bool = True,
    ):
        super().__init__()
        self.model_size = model_size
        self.input_shape = input_shape
        self.crop_shape = crop_shape
        self.default_output: _ImageEncoderOutput = default_output
        self.auto_resize_to_224 = bool(auto_resize_to_224)
        self.imagenet_normalize = bool(imagenet_normalize)

        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # DINOv2 backbones expect ImageNet-normalized RGB in [0,1].
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", mean, persistent=False)
        self.register_buffer("rgb_std", std, persistent=False)

        self._setup_backbone(
            model_size,
            freeze_backbone,
        )

        # DINOv2 patch grid size for 224x224 inputs (ViT-S/B/L/G use 14x14 or 16x16 depending).
        # In this implementation we reshape tokens to a 16x16 map, matching prior behavior.
        self.feature_map_hw = (16, 16)

    def io_shapes(self, *, output: _ImageEncoderOutput | None = None) -> dict[str, Any]:
        """
        Report expected input/output shapes for this encoder.

        Args:
            output (Literal["patch","cls","both"]): Which representation `forward()` will return.

        Returns:
            dict: A small summary containing `input` and `output` shape strings.
                - input["x"]: "[B, C, H, W]" (with notes about crop + required final size)
                - output: depends on `output`:
                    - "patch": "[B, L, D]"
                    - "cls":   "[B, D]"
                    - "both":  {"cls": "[B, D]", "patch": "[B, L, D]"}
        """
        if output is None:
            output = self.default_output

        C = self.input_shape[0]
        Hf, Wf = self.feature_map_hw
        D = int(self.model_dim)
        L = int(Hf * Wf)
        info: dict[str, Any] = {
            "input": {
                "x": f"[B, {C}, H, W] (after optional crop must be [B, {C}, 224, 224])",
            },
        }
        if output == "cls":
            info["output"] = {"cls": f"[B, {D}]"}
        elif output == "patch":
            info["output"] = {"patch": f"[B, {L}, {D}]"}
        elif output == "both":
            info["output"] = {"cls": f"[B, {D}]", "patch": f"[B, {L}, {D}]"}
        else:
            raise ValueError(f"Unknown output={output!r}. Choose from 'patch', 'cls', 'both'.")
        return info

    def _setup_backbone(
        self,
        model_size: str,
        freeze_backbone: bool,
    ):
        model_name_map = {
            "small": "facebook/dinov2-small",
            "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large",
            "giant": "facebook/dinov2-giant",
        }
        model_name = model_name_map.get(model_size, "facebook/dinov2-small")
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        dim_map = {"small": 384, "base": 768, "large": 1024, "giant": 1536}
        self.model_dim = dim_map.get(model_size, 384)

    def forward(
        self,
        x: Tensor,
        *,
        output: _ImageEncoderOutput | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """
        Image encoder forward.

        Args:
            x (Tensor): Shape [B, C, H, W]. Input images (typically C=3).
            output (Literal["patch","cls","both"]): Select which representation to return.
                - "patch": patch tokens [B, L, D]
                - "cls": CLS token embedding [B, D]
                - "both": dict with keys {"cls","patch"}

        Returns:
            Tensor | dict[str, Tensor]:
                - If output="patch": Tensor of shape [B, L, D]
                - If output="cls":   Tensor of shape [B, D]
                - If output="both":  {"cls": [B, D], "patch": [B, L, D]}

        Details:
            1. Optional crop/resize depending on backbone requirements.
            2. DINOv2 produces token features which are reshaped into a spatial feature map.
        """
        if output is None:
            output = self.default_output

        # Accept uint8 images from datasets and normalize for DINOv2.
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div_(255.0)
        elif not torch.is_floating_point(x):
            x = x.to(dtype=torch.float32)

        # Optionally resize to 224x224 before cropping/backbone.
        if self.auto_resize_to_224 and (x.shape[2] != 224 or x.shape[3] != 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        if x.shape[2] != 224 or x.shape[3] != 224:
            raise ValueError(
                f"DINOv2 DinoImageEncoder expects input images of shape [B, C, 224, 224], "
                f"but got [B, C, {x.shape[2]}, {x.shape[3]}]."
            )

        if self.imagenet_normalize:
            # (B,3,224,224) normalized.
            x = (x - self.rgb_mean) / self.rgb_std

        outputs = self.backbone(x)
        hidden = outputs.last_hidden_state  # [B, 1 + L, D]
        cls = hidden[:, 0, :]
        patch_tokens = hidden[:, 1:, :]

        if output == "cls":
            return cls

        Hf, Wf = self.feature_map_hw
        if patch_tokens.shape[1] != Hf * Wf:
            raise ValueError(
                f"Unexpected DINOv2 token count L={patch_tokens.shape[1]} for feature_map_hw={self.feature_map_hw} "
                f"(expected {Hf*Wf})."
            )

        if output == "patch":
            return patch_tokens
        if output == "both":
            return {"cls": cls, "patch": patch_tokens}
        raise ValueError(f"Unknown output={output!r}. Choose from 'patch', 'cls', 'both'.")

