"""Swin Transformer V2 U-Net segmentation model.

Uses a pretrained Swin-V2 backbone (loaded via ``timm``) as the encoder.
Because Swin outputs non-spatial feature tokens, each stage output is
reshaped from ``(B, H*W, C)`` to ``(B, C, H, W)`` before being passed
to the decoder.  Skip connections from stages 1 and 2 are concatenated
with upsampled decoder feature maps via ``torch.cat``.

Supported sizes and their backbone channel widths:

+--------+-----------------------------+--------+
| Size   | Stage channels (1-2-3-4)    | Params |
+========+=============================+========+
| tiny   | 96 – 192 – 384 – 768        | ~28 M  |
+--------+-----------------------------+--------+
| small  | 96 – 192 – 384 – 768        | ~50 M  |
+--------+-----------------------------+--------+
| base   | 128 – 256 – 512 – 1024      | ~88 M  |
+--------+-----------------------------+--------+
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

from models.decoder import BottleneckBlock, DecoderBlock, SegmentationHead

_SWIN_TIMM_NAMES: dict[str, str] = {
    "tiny": "swinv2_tiny_window16_256.ms_in1k",
    "small": "swinv2_small_window16_256.ms_in1k",
    "base": "swinv2_base_window16_256.ms_in1k",
}

# Fallback names used when the primary variant is not available in the
# installed timm version (older timm ships window8 only)
_SWIN_TIMM_FALLBACKS: dict[str, str] = {
    "tiny": "swinv2_tiny_window8_256.ms_in1k",
    "small": "swinv2_small_window8_256.ms_in1k",
    "base": "swinv2_base_window8_256.ms_in1k",
}

_SWIN_CHANNELS: dict[str, list[int]] = {
    "tiny": [96, 192, 384, 768],
    "small": [96, 192, 384, 768],
    "base": [128, 256, 512, 1024],
}


def _token_to_spatial(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Reshape Swin token sequence back to a spatial feature map.

    Swin stages output tensors of shape ``(B, H*W, C)``.  This helper
    transposes and reshapes them to ``(B, C, H, W)`` so they are
    compatible with the convolutional decoder.

    Args:
        x: Token tensor of shape ``(B, N, C)`` where ``N = H * W``.
        h: Spatial height of the feature map.
        w: Spatial width of the feature map.

    Returns:
        Feature map tensor of shape ``(B, C, H, W)``.
    """
    B, N, C = x.shape
    assert N == h * w, f"Token count {N} does not match {h}×{w}={h*w}"
    return x.permute(0, 2, 1).reshape(B, C, h, w)


class SwinUNet(nn.Module):
    """Hybrid U-Net with a Swin Transformer V2 encoder and a custom decoder.

    The encoder is a pretrained Swin-V2 backbone whose stage outputs are
    intercepted via ``register_forward_hook``.  Swin stages emit token
    sequences; these are reshaped to spatial maps before entering the
    decoder.  Skip connections from stages 1 and 2 are concatenated with
    upsampled decoder outputs.

    Args:
        num_classes: Number of segmentation output classes.
        size: Backbone size — one of ``"tiny"``, ``"small"``, or ``"base"``.
        pretrained: Whether to load pretrained backbone weights.
        input_size: Expected spatial input size (height == width).
            Must match the window size the backbone was pretrained with.
            Defaults to 512.
        decoder_channels: Output channels for each decoder stage.
            Defaults to ``[256, 128, 64]``.

    Raises:
        ValueError: If *size* is not one of the supported values.

    Example:
        >>> model = SwinUNet(num_classes=17, size="tiny")
        >>> x = torch.randn(2, 3, 512, 512)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 17, 512, 512])
    """

    def __init__(
        self,
        num_classes: int = 17,
        size: str = "tiny",
        pretrained: bool = True,
        input_size: int = 512,
        decoder_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if size not in _SWIN_TIMM_NAMES:
            raise ValueError(
                f"Unsupported size '{size}'. Choose from {list(_SWIN_TIMM_NAMES)}"
            )

        self.size = size
        self.num_classes = num_classes
        self.input_size = input_size
        decoder_channels = decoder_channels or [256, 128, 64]

        # ── Backbone ──────────────────────────────────────────────────────
        # Try the preferred window16 variant first; fall back to window8 for
        # older timm installations.  Always pass img_size so timm interpolates
        # the relative position bias tables to the requested resolution.
        primary_name = _SWIN_TIMM_NAMES[size]
        fallback_name = _SWIN_TIMM_FALLBACKS[size]
        backbone_kwargs = dict(
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=input_size,
        )
        try:
            self.backbone = timm.create_model(primary_name, **backbone_kwargs)
        except (RuntimeError, ValueError):
            self.backbone = timm.create_model(fallback_name, **backbone_kwargs)
        enc_ch = _SWIN_CHANNELS[size]

        # Compute spatial sizes at each stage: Swin patch size=4, then ×2 merge
        # Stage 1: H/4, Stage 2: H/8, Stage 3: H/16, Stage 4: H/32
        self._spatial_sizes = [
            input_size // (4 * (2**i)) for i in range(4)
        ]  # [H/4, H/8, H/16, H/32]

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = BottleneckBlock(enc_ch[3], decoder_channels[0])

        # ── Decoder ───────────────────────────────────────────────────────
        self.dec3 = DecoderBlock(decoder_channels[0], enc_ch[2], decoder_channels[0])
        self.dec2 = DecoderBlock(decoder_channels[0], enc_ch[1], decoder_channels[1])
        self.dec1 = DecoderBlock(decoder_channels[1], enc_ch[0], decoder_channels[2])

        # ── Segmentation head ─────────────────────────────────────────────
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

        Use discriminative learning rates when calling the optimiser:
        a much lower LR for the backbone than the decoder to avoid
        catastrophic forgetting of pretrained attention patterns.
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
            >>> groups = model.get_param_groups(lr_backbone=5e-6, lr_decoder=8e-5)
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

        Swin stage outputs are token sequences ``(B, N, C)``; this method
        reshapes them to spatial maps ``(B, C, H, W)`` before passing them
        to the convolutional decoder.

        Args:
            x: Batch of RGB images, shape ``(B, 3, H, W)``.
                *H* and *W* must equal ``input_size`` (default 512).

        Returns:
            Per-pixel class logits of shape ``(B, num_classes, H, W)``.
        """
        # Encoder: backbone returns list of stage outputs
        # timm Swin with features_only returns spatial tensors (B, C, H, W)
        features = self.backbone(x)
        f1, f2, f3, f4 = features

        # Ensure spatial format (timm swinv2 features_only already returns
        # (B,C,H,W) but guard against token format just in case)
        f1 = self._ensure_spatial(f1, 0)
        f2 = self._ensure_spatial(f2, 1)
        f3 = self._ensure_spatial(f3, 2)
        f4 = self._ensure_spatial(f4, 3)

        # Bottleneck
        z = self.bottleneck(f4)

        # Decoder with skip connections
        d3 = self.dec3(z, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        return self.seg_head(d1, output_size=(x.shape[-2], x.shape[-1]))

    def _ensure_spatial(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Convert any Swin feature format to ``(B, C, H, W)``.

        Timm's Swin backbone returns features in one of three formats
        depending on the installed timm version:

        - ``(B, C, H, W)`` — channels-first spatial (standard PyTorch).
          Return as-is.
        - ``(B, H, W, C)`` — channels-last spatial (newer timm >= 0.9).
          Detected when the last dimension is larger than dim 1; permute
          to channels-first.
        - ``(B, N, C)`` — flattened token sequence (older timm).
          Reshape using the precomputed ``_spatial_sizes``.

        Args:
            x: Feature tensor from a Swin backbone stage.
            stage_idx: Stage index (0-based) for spatial size lookup.

        Returns:
            Feature map of shape ``(B, C, H, W)``.
        """
        h = w = self._spatial_sizes[stage_idx]
        C = _SWIN_CHANNELS[self.size][stage_idx]
        if x.dim() == 4:
            # Use the known channel count for this stage to distinguish
            # (B, C, H, W) from (B, H, W, C) unambiguously.
            # dim-1 == C  →  channels-first, pass through.
            # dim-1 != C  →  channels-last, permute.
            # This is reliable even when H == C (e.g. stage1 at 256px: H=64, C=96)
            # or when H > C (e.g. stage1 at 512px: H=128, C=96).
            if x.shape[1] == C:
                return x  # already (B, C, H, W)
            return x.permute(0, 3, 1, 2).contiguous()
        # (B, N, C) token sequence
        return _token_to_spatial(x, h, w)

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


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Swin-V2 Unet — self-test (pretrained=False, tiny, input_size=512)")
    model = SwinUNet(num_classes=17, size="tiny", pretrained=False, input_size=512)
    model.eval()

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    counts = model.count_parameters()
    assert out.shape == (2, 17, 512, 512), f"Unexpected output shape: {out.shape}"
    print(f"  Output shape : {out.shape}  ✓")
    print(f"  Backbone     : {counts['backbone'] / 1e6:.1f}M params")
    print(f"  Decoder      : {counts['decoder'] / 1e6:.3f}M params")
    print(f"  Seg Head     : {counts['seg_head'] / 1e6:.3f}M params")
    print(f"  Total        : {counts['total'] / 1e6:.1f}M params")
    print("  All assertions passed ✓")
