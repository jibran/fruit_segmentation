"""Albumentations-based augmentation pipelines for fruit segmentation.

Provides factory functions that return properly configured
``albumentations.Compose`` transforms for training and validation splits.
All transforms operate jointly on ``image`` and ``mask`` tensors so that
spatial augmentations are applied consistently to both.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(cfg: dict[str, Any]) -> A.Compose:
    """Build the training augmentation pipeline from config.

    Applies spatial augmentations (flip, rotate, zoom, shear), colour
    jitter, and final normalisation to ImageNet statistics.

    Args:
        cfg: Loaded configuration dictionary.  Reads from
            ``augmentation`` and ``data.image_size`` keys.

    Returns:
        An ``albumentations.Compose`` transform that accepts
        ``image`` (HxWx3 uint8 ndarray) and ``mask`` (HxW uint8/int
        ndarray) keyword arguments.

    Example:
        >>> transform = build_train_transform(cfg)
        >>> out = transform(image=img_np, mask=mask_np)
        >>> out["image"].shape   # torch.Size([3, 512, 512])
        >>> out["mask"].shape    # torch.Size([512, 512])
    """
    aug = cfg.get("augmentation", {})
    size = cfg["data"]["image_size"]
    mean = aug.get("normalize_mean", [0.485, 0.456, 0.406])
    std = aug.get("normalize_std", [0.229, 0.224, 0.225])

    transforms = [
        A.Resize(size, size),
    ]

    if aug.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.3))

    rotation = aug.get("rotation_limit", 30)
    if rotation:
        transforms.append(A.Rotate(limit=rotation, p=0.5))

    zoom = aug.get("zoom_limit", 0.2)
    shear = aug.get("shear_limit", 10)
    if zoom or shear:
        transforms.append(
            A.Affine(
                scale=(1 - zoom, 1 + zoom),
                shear=(-shear, shear),
                translate_percent=(-0.05, 0.05),
                p=0.4,
            )
        )

    if aug.get("brightness_contrast", True):
        transforms.append(A.RandomBrightnessContrast(p=0.4))
        transforms.append(A.HueSaturationValue(p=0.3))

    transforms += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def build_val_transform(cfg: dict[str, Any]) -> A.Compose:
    """Build the validation / test transform pipeline from config.

    Applies only resize and normalisation — no spatial or colour
    augmentations that would corrupt evaluation metrics.

    Args:
        cfg: Loaded configuration dictionary.

    Returns:
        An ``albumentations.Compose`` transform accepting ``image``
        and ``mask`` keyword arguments.

    Example:
        >>> transform = build_val_transform(cfg)
        >>> out = transform(image=img_np, mask=mask_np)
    """
    aug = cfg.get("augmentation", {})
    size = cfg["data"]["image_size"]
    mean = aug.get("normalize_mean", [0.485, 0.456, 0.406])
    std = aug.get("normalize_std", [0.229, 0.224, 0.225])

    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
