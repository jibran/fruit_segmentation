"""ConvNeXt-V2 baseline classifier for fruit segmentation comparison.

Uses the same pretrained ConvNeXt-V2 backbone as ``ConvNeXtUNet`` but
replaces the U-Net decoder with a global-average-pool + linear head.
This gives a fair baseline: identical backbone weights, identical number
of output classes, but no pixel-level segmentation capability.

The baseline operates in *image-level classification* mode — it predicts
a single class label per image (the dominant fruit class) rather than a
per-pixel mask.  Its accuracy on held-out images establishes the ceiling
that pixel-level features from the U-Net decoder add on top of global
backbone representations alone.

Usage::

    python models/convnext_baseline.py  # quick self-test

    # In a training / eval script:
    from models.convnext_baseline import ConvNeXtBaseline
    model = ConvNeXtBaseline(num_classes=17, size="tiny", pretrained=True)
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

# ── Backbone registry (mirrors convnext_unet.py) ──────────────────────────────

_CONVNEXT_TIMM_NAMES: dict[str, str] = {
    "tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "small": "convnextv2_small.fcmae_ft_in22k_in1k",
    "base": "convnextv2_base.fcmae_ft_in22k_in1k",
}

# Final-stage channel widths — used to size the linear head
_CONVNEXT_STAGE4_CHANNELS: dict[str, int] = {
    "tiny": 768,
    "small": 768,
    "base": 1024,
}


class ConvNeXtBaseline(nn.Module):
    """ConvNeXt-V2 backbone with a global-average-pool classification head.

    Intended as a baseline to compare against ``ConvNeXtUNet``.  The
    backbone is identical; the difference is that this model produces one
    class prediction per image rather than one per pixel.

    Architecture::

        Input (B, 3, H, W)
            ↓
        ConvNeXt-V2 encoder  (pretrained, 4 stages)
            ↓
        Global average pool  (B, C)
            ↓
        Dropout(p=0.2)
            ↓
        Linear(C → num_classes)
            ↓
        Logits (B, num_classes)

    Args:
        num_classes: Number of output classes.  Defaults to 17 (16 fruit
            classes + 1 background).
        size: Backbone variant — ``"tiny"``, ``"small"``, or ``"base"``.
            Defaults to ``"tiny"``.
        pretrained: Whether to load ImageNet-22k pretrained weights.
            Defaults to ``True``.
        drop_rate: Dropout probability applied before the linear head.
            Defaults to 0.2.

    Example:
        >>> model = ConvNeXtBaseline(num_classes=17, size="tiny")
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 17])
    """

    def __init__(
        self,
        num_classes: int = 17,
        size: str = "tiny",
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()

        if size not in _CONVNEXT_TIMM_NAMES:
            raise ValueError(
                f"Unsupported size '{size}'. Choose from {list(_CONVNEXT_TIMM_NAMES)}"
            )

        self.size = size
        self.num_classes = num_classes

        # Full backbone (not features_only — we want the pooled representation)
        self.backbone = timm.create_model(
            _CONVNEXT_TIMM_NAMES[size],
            pretrained=pretrained,
            num_classes=0,  # remove timm's own classifier head
            global_pool="avg",  # global average pool before the head
        )

        in_features = _CONVNEXT_STAGE4_CHANNELS[size]
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes),
        )

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze helpers (mirror ConvNeXtUNet API)
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (phase 1 training).

        Only the classification head will receive gradient updates.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (phase 2 fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
    ) -> list[dict[str, Any]]:
        """Return parameter groups for discriminative learning rates.

        Args:
            lr_backbone: Learning rate for backbone parameters.
            lr_head: Learning rate for the classification head.

        Returns:
            List of parameter-group dicts consumable by any
            ``torch.optim`` optimiser.

        Example:
            >>> groups = model.get_param_groups(lr_backbone=1e-5, lr_head=1e-3)
            >>> optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        """
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.head.parameters(), "lr": lr_head},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return class logits.

        Args:
            x: Batch of RGB images, shape ``(B, 3, H, W)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.  No softmax
            applied — pass directly to ``nn.CrossEntropyLoss``.
        """
        features = self.backbone(x)  # (B, C) after global avg pool
        return self.head(features)  # (B, num_classes)

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts broken down by component.

        Returns:
            Dict with keys ``"backbone"``, ``"head"``, and ``"total"``.

        Example:
            >>> counts = model.count_parameters()
            >>> print(f"Total: {counts['total'] / 1e6:.1f}M")
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "backbone": backbone_params,
            "head": head_params,
            "total": backbone_params + head_params,
        }


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ConvNeXt-V2 Baseline — self-test (pretrained=False, tiny)")
    model = ConvNeXtBaseline(num_classes=17, size="tiny", pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    counts = model.count_parameters()
    assert out.shape == (2, 17), f"Unexpected output shape: {out.shape}"
    print(f"  Output shape : {out.shape}  ✓")
    print(f"  Backbone     : {counts['backbone'] / 1e6:.1f}M params")
    print(f"  Head         : {counts['head'] / 1e6:.3f}M params")
    print(f"  Total        : {counts['total'] / 1e6:.1f}M params")
    print("  All assertions passed ✓")
