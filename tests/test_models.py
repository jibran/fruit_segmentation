"""Tests for ConvNeXtUNet and SwinUNet model architectures.

Verifies:
- Correct output shapes for all supported sizes.
- Backbone freeze / unfreeze behaviour.
- Parameter group construction for discriminative LRs.
- build_model factory dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from models.convnext_unet import ConvNeXtUNet
from models.swin_unet import SwinUNet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def convnext_tiny() -> ConvNeXtUNet:
    """Instantiate ConvNeXtUNet tiny without pretrained weights (fast)."""
    return ConvNeXtUNet(num_classes=17, size="tiny", pretrained=False)


@pytest.fixture(scope="module")
def swin_tiny() -> SwinUNet:
    """Instantiate SwinUNet tiny without pretrained weights (fast)."""
    return SwinUNet(num_classes=17, size="tiny", pretrained=False, input_size=256)


@pytest.fixture
def dummy_batch_convnext() -> torch.Tensor:
    """Small batch for ConvNeXtUNet (512 replaced with 256 for speed)."""
    return torch.randn(2, 3, 256, 256)


@pytest.fixture
def dummy_batch_swin() -> torch.Tensor:
    """Small batch matching SwinUNet input_size=256."""
    return torch.randn(2, 3, 256, 256)


# ---------------------------------------------------------------------------
# ConvNeXtUNet tests
# ---------------------------------------------------------------------------


class TestConvNeXtUNet:
    """Tests for the ConvNeXtUNet architecture."""

    def test_output_shape(
        self, convnext_tiny: ConvNeXtUNet, dummy_batch_convnext: torch.Tensor
    ) -> None:
        """Output logit tensor must match (B, num_classes, H, W)."""
        convnext_tiny.eval()
        with torch.no_grad():
            out = convnext_tiny(dummy_batch_convnext)
        assert out.shape == (2, 17, 256, 256), f"Unexpected shape: {out.shape}"

    def test_output_dtype(
        self, convnext_tiny: ConvNeXtUNet, dummy_batch_convnext: torch.Tensor
    ) -> None:
        """Output must be float32 (raw logits, no softmax applied)."""
        convnext_tiny.eval()
        with torch.no_grad():
            out = convnext_tiny(dummy_batch_convnext)
        assert out.dtype == torch.float32

    def test_freeze_backbone(self, convnext_tiny: ConvNeXtUNet) -> None:
        """After freeze_backbone, no backbone param should require grad."""
        convnext_tiny.freeze_backbone()
        backbone_requires_grad = [
            p.requires_grad for p in convnext_tiny.backbone.parameters()
        ]
        assert not any(
            backbone_requires_grad
        ), "Some backbone params still require grad after freeze"

    def test_unfreeze_backbone(self, convnext_tiny: ConvNeXtUNet) -> None:
        """After unfreeze_backbone, all backbone params should require grad."""
        convnext_tiny.unfreeze_backbone()
        backbone_requires_grad = [
            p.requires_grad for p in convnext_tiny.backbone.parameters()
        ]
        assert all(
            backbone_requires_grad
        ), "Some backbone params still frozen after unfreeze"

    def test_param_groups(self, convnext_tiny: ConvNeXtUNet) -> None:
        """get_param_groups must return exactly 2 groups with correct lr keys."""
        groups = convnext_tiny.get_param_groups(lr_backbone=1e-5, lr_decoder=1e-4)
        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-5
        assert groups[1]["lr"] == 1e-4

    def test_count_parameters(self, convnext_tiny: ConvNeXtUNet) -> None:
        """Total param count must equal sum of components."""
        counts = convnext_tiny.count_parameters()
        assert (
            counts["total"]
            == counts["backbone"] + counts["decoder"] + counts["seg_head"]
        )
        assert counts["total"] > 1_000_000, "Suspiciously low parameter count"

    def test_invalid_size_raises(self) -> None:
        """Instantiating with an invalid size must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported size"):
            ConvNeXtUNet(size="xlarge", pretrained=False)

    def test_decoder_trains_when_frozen(
        self, convnext_tiny: ConvNeXtUNet, dummy_batch_convnext: torch.Tensor
    ) -> None:
        """Decoder gradients must accumulate even when backbone is frozen."""
        convnext_tiny.freeze_backbone()
        convnext_tiny.train()
        targets = torch.randint(0, 17, (2, 256, 256))
        loss_fn = torch.nn.CrossEntropyLoss()
        out = convnext_tiny(dummy_batch_convnext)
        loss = loss_fn(out, targets)
        loss.backward()
        decoder_grads = [
            p.grad is not None
            for p in convnext_tiny.dec1.parameters()
            if p.requires_grad
        ]
        assert any(decoder_grads), "No gradients flowed to decoder during phase 1"


# ---------------------------------------------------------------------------
# SwinUNet tests
# ---------------------------------------------------------------------------


class TestSwinUNet:
    """Tests for the SwinUNet architecture."""

    def test_output_shape(
        self, swin_tiny: SwinUNet, dummy_batch_swin: torch.Tensor
    ) -> None:
        """Output logit tensor must match (B, num_classes, H, W)."""
        swin_tiny.eval()
        with torch.no_grad():
            out = swin_tiny(dummy_batch_swin)
        assert out.shape == (2, 17, 256, 256), f"Unexpected shape: {out.shape}"

    def test_output_dtype(
        self, swin_tiny: SwinUNet, dummy_batch_swin: torch.Tensor
    ) -> None:
        """Output must be float32."""
        swin_tiny.eval()
        with torch.no_grad():
            out = swin_tiny(dummy_batch_swin)
        assert out.dtype == torch.float32

    def test_freeze_backbone(self, swin_tiny: SwinUNet) -> None:
        """Backbone freeze must disable all backbone gradients."""
        swin_tiny.freeze_backbone()
        assert not any(p.requires_grad for p in swin_tiny.backbone.parameters())

    def test_unfreeze_backbone(self, swin_tiny: SwinUNet) -> None:
        """Backbone unfreeze must re-enable all backbone gradients."""
        swin_tiny.unfreeze_backbone()
        assert all(p.requires_grad for p in swin_tiny.backbone.parameters())

    def test_param_groups(self, swin_tiny: SwinUNet) -> None:
        """get_param_groups must return 2 groups with the requested LRs."""
        groups = swin_tiny.get_param_groups(lr_backbone=5e-6, lr_decoder=8e-5)
        assert len(groups) == 2
        assert groups[0]["lr"] == 5e-6
        assert groups[1]["lr"] == 8e-5

    def test_count_parameters(self, swin_tiny: SwinUNet) -> None:
        """count_parameters total must be self-consistent."""
        counts = swin_tiny.count_parameters()
        assert (
            counts["total"]
            == counts["backbone"] + counts["decoder"] + counts["seg_head"]
        )

    def test_invalid_size_raises(self) -> None:
        """Instantiating with an invalid size must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported size"):
            SwinUNet(size="huge", pretrained=False)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestBuildModel:
    """Tests for the build_model factory function."""

    def _make_cfg(self, model_name: str, size: str = "tiny") -> dict:
        """Build a minimal config dict for testing.

        Args:
            model_name: One of ``"convnext_unet"`` or ``"swin_unet"``.
            size: Backbone size string.

        Returns:
            Minimal config dict accepted by :func:`build_model`.
        """
        return {
            "model": {"name": model_name, "size": size, "pretrained": False},
            "data": {"num_classes": 17, "image_size": 256},
        }

    def test_builds_convnext(self) -> None:
        """build_model must return a ConvNeXtUNet for 'convnext_unet'."""
        model = build_model(self._make_cfg("convnext_unet"))
        assert isinstance(model, ConvNeXtUNet)

    def test_builds_swin(self) -> None:
        """build_model must return a SwinUNet for 'swin_unet'."""
        model = build_model(self._make_cfg("swin_unet"))
        assert isinstance(model, SwinUNet)

    def test_unknown_model_raises(self) -> None:
        """build_model must raise KeyError for unregistered model names."""
        with pytest.raises(KeyError, match="Unknown model"):
            build_model(self._make_cfg("unknown_model"))
