"""Shared U-Net decoder building blocks.

These modules are backbone-agnostic and are reused by both
``ConvNeXtUNet`` and ``SwinUNet``.  Each block receives a concatenated
tensor (upsampled features + skip-connection features) and refines it
through two convolution layers with residual-style normalisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Single Conv → BatchNorm → ReLU unit.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        kernel_size: Convolution kernel size. Defaults to 3.
        padding: Padding added to both sides. Defaults to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv → BN → ReLU.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of shape ``(B, out_channels, H, W)``.
        """
        return self.block(x)


class DecoderBlock(nn.Module):
    """U-Net decoder stage: bilinear upsample → cat(skip) → 2× ConvBnRelu.

    This block corresponds to one level of the decoder pyramid.  It
    upsamples the incoming feature map, concatenates the matching encoder
    skip-connection tensor, and refines the result with two convolutions.

    Args:
        in_channels: Channels of the upsampled (decoder) feature map.
        skip_channels: Channels of the skip-connection tensor from the encoder.
        out_channels: Channels produced by this block.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample *x*, concatenate *skip*, then apply two conv layers.

        Args:
            x: Decoder feature map of shape ``(B, in_channels, H, W)``.
            skip: Encoder skip tensor of shape ``(B, skip_channels, 2H, 2W)``.

        Returns:
            Refined feature map of shape ``(B, out_channels, 2H, 2W)``.
        """
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BottleneckBlock(nn.Module):
    """Two-layer conv block applied at the lowest spatial resolution.

    Sits between the encoder's final stage and the first decoder block.
    No upsampling or skip connection — purely a feature refinement step.

    Args:
        in_channels: Number of input channels from the backbone's last stage.
        out_channels: Number of output channels fed to the first decoder block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            ConvBnRelu(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck convolutions.

        Args:
            x: Feature map from final backbone stage, shape ``(B, C, H, W)``.

        Returns:
            Refined feature map of shape ``(B, out_channels, H, W)``.
        """
        return self.block(x)


class SegmentationHead(nn.Module):
    """Final 1×1 conv that maps features to per-class logits.

    The forward pass accepts an optional ``output_size`` argument so the
    head can upsample to the exact original input resolution regardless of
    how many decoder stages were used or what the input image size was.
    This avoids the subtle off-by-one errors that arise from chaining fixed
    ``upscale_factor`` multiplications.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of segmentation classes (output channels).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Upsample to *output_size* (if given) then classify every pixel.

        Args:
            x: Feature map of shape ``(B, in_channels, H, W)``.
            output_size: Target ``(H, W)`` for the output logit map.
                If ``None``, no upsampling is applied and the spatial
                dimensions of *x* are preserved.

        Returns:
            Logit map of shape ``(B, num_classes, H_out, W_out)`` where
            ``(H_out, W_out)`` equals *output_size* when provided,
            otherwise the spatial size of *x*.
        """
        if output_size is not None and (x.shape[-2], x.shape[-1]) != output_size:
            x = F.interpolate(
                x,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(x)
