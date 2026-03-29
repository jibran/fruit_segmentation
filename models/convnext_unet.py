"""ConvNeXt-V2 U-Net segmentation model.

Uses a pretrained ConvNeXt-V2 backbone (loaded via ``timm``) as the encoder.
Forward hooks intercept intermediate feature maps to form U-Net skip connections.
The decoder and segmentation head are custom ``nn.Module`` classes trained
from scratch.

Supported sizes and their backbone channel widths:

+--------+-----------------------------+--------+
| Size   | Stage channels (1-2-3-4)    | Params |
+========+=============================+========+
| tiny   | 96 – 192 – 384 – 768        | ~28 M  |
+--------+-----------------------------+--------+
| small  | 96 – 192 – 384 – 768        | ~50 M  |
+--------+-----------------------------+--------+
| base   | 128 – 256 – 512 – 1024      | ~89 M  |
+--------+-----------------------------+--------+
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

from models.decoder import BottleneckBlock, DecoderBlock, SegmentationHead

# timm model name lookup keyed by size string
_CONVNEXT_TIMM_NAMES: dict[str, str] = {
    "tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "small": "convnextv2_small.fcmae_ft_in22k_in1k",
    "base": "convnextv2_base.fcmae_ft_in22k_in1k",
}

# Per-size backbone stage output channels [stage1, stage2, stage3, stage4]
_CONVNEXT_CHANNELS: dict[str, list[int]] = {
    "tiny": [96, 192, 384, 768],
    "small": [96, 192, 384, 768],
    "base": [128, 256, 512, 1024],
}


class ConvNeXtUNet(nn.Module):
    """Hybrid U-Net with a ConvNeXt-V2 encoder and a custom decoder.

    The encoder is a pretrained ConvNeXt-V2 backbone whose intermediate
    activations are intercepted via ``register_forward_hook``.  Skip
    connections from stages 1 and 2 are concatenated with the upsampled
    decoder feature maps (``torch.cat`` along the channel dimension).

    Args:
        num_classes: Number of segmentation output classes.
        size: Backbone size — one of ``"tiny"``, ``"small"``, or ``"base"``.
        pretrained: Whether to load ImageNet pretrained backbone weights.
        decoder_channels: Output channels for each decoder stage.
            Defaults to ``[256, 128, 64]``.

    Raises:
        ValueError: If *size* is not one of the supported values.

    Example:
        >>> model = ConvNeXtUNet(num_classes=17, size="tiny")
        >>> x = torch.randn(2, 3, 512, 512)
        >>> logits = model(x)          # (2, 17, 512, 512)
        >>> logits.shape
        torch.Size([2, 17, 512, 512])
    """

    def __init__(
        self,
        num_classes: int = 17,
        size: str = "tiny",
        pretrained: bool = True,
        decoder_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if size not in _CONVNEXT_TIMM_NAMES:
            raise ValueError(
                f"Unsupported size '{size}'. Choose from {list(_CONVNEXT_TIMM_NAMES)}"
            )

        self.size = size
        self.num_classes = num_classes
        decoder_channels = decoder_channels or [256, 128, 64]

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            _CONVNEXT_TIMM_NAMES[size],
            pretrained=pretrained,
            features_only=True,  # returns list of stage outputs
            out_indices=(0, 1, 2, 3),  # all 4 stages
        )
        enc_ch = _CONVNEXT_CHANNELS[size]  # [C1, C2, C3, C4]

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = BottleneckBlock(enc_ch[3], decoder_channels[0])

        # ── Decoder ───────────────────────────────────────────────────────
        # dec3: bottleneck → up → cat(stage3) → refine
        self.dec3 = DecoderBlock(decoder_channels[0], enc_ch[2], decoder_channels[0])
        # dec2: dec3 out → up → cat(stage2) → refine
        self.dec2 = DecoderBlock(decoder_channels[0], enc_ch[1], decoder_channels[1])
        # dec1: dec2 out → up → cat(stage1) → refine
        self.dec1 = DecoderBlock(decoder_channels[1], enc_ch[0], decoder_channels[2])

        # ── Segmentation head ─────────────────────────────────────────────
        # output_size is passed at forward time — works for any input resolution
        self.seg_head = SegmentationHead(decoder_channels[2], num_classes)

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1 training).

        After calling this, only the decoder and segmentation head
        parameters will receive gradient updates.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (Phase 2 fine-tuning).

        Call this to switch to end-to-end training.  Use discriminative
        learning rates: a much lower LR for the backbone than the decoder.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(
        self,
        lr_backbone: float,
        lr_decoder: float,
    ) -> list[dict[str, Any]]:
        """Return parameter groups with discriminative learning rates.

        Args:
            lr_backbone: Learning rate for backbone parameters.
            lr_decoder: Learning rate for decoder + head parameters.

        Returns:
            A list of parameter-group dicts consumable by any
            ``torch.optim`` optimiser.

        Example:
            >>> groups = model.get_param_groups(lr_backbone=1e-5, lr_decoder=1e-4)
            >>> optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        """
        decoder_params = (
            list(self.bottleneck.parameters())
            + list(self.dec3.parameters())
            + list(self.dec2.parameters())
            + list(self.dec1.parameters())
            + list(self.seg_head.parameters())
        )
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": decoder_params, "lr": lr_decoder},
        ]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full encoder-decoder forward pass.

        Args:
            x: Batch of RGB images, shape ``(B, 3, H, W)``.
                *H* and *W* must be divisible by 32.

        Returns:
            Per-pixel class logits of shape ``(B, num_classes, H, W)``.
            No softmax is applied — pass directly to
            ``torch.nn.CrossEntropyLoss``.
        """
        # Encoder: backbone returns [f1, f2, f3, f4]
        features = self.backbone(x)
        f1, f2, f3, f4 = features  # ascending abstraction, descending resolution

        # Bottleneck
        z = self.bottleneck(f4)

        # Decoder with skip connections
        d3 = self.dec3(z, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        # Segmentation head — upsample back to original input resolution
        return self.seg_head(d1, output_size=(x.shape[-2], x.shape[-1]))

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts broken down by component.

        Returns:
            Dict with keys ``"backbone"``, ``"decoder"``, ``"seg_head"``,
            and ``"total"``.

        Example:
            >>> counts = model.count_parameters()
            >>> print(counts["total"])
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        decoder_params = sum(
            p.numel()
            for m in [self.bottleneck, self.dec3, self.dec2, self.dec1]
            for p in m.parameters()
        )
        head_params = sum(p.numel() for p in self.seg_head.parameters())
        return {
            "backbone": backbone_params,
            "decoder": decoder_params,
            "seg_head": head_params,
            "total": backbone_params + decoder_params + head_params,
        }
