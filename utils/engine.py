"""Core training and validation engine.

Provides :func:`train_one_epoch` and :func:`validate_one_epoch` which
are called by the main training script.  Both functions return a metrics
dict that the caller can log and checkpoint.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import SegmentationMetrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    log_interval: int = 10,
) -> dict[str, Any]:
    """Run one full training epoch.

    Args:
        model: The segmentation model (``ConvNeXtUNet`` or ``SwinUNet``).
        loader: Training DataLoader.
        criterion: Loss function (e.g. ``nn.CrossEntropyLoss``).
        optimizer: Configured optimiser.
        device: Target compute device.
        epoch: Current epoch number (1-based), used for progress display.
        grad_clip: Maximum gradient norm for clipping.  Set to 0 to
            disable.  Defaults to 1.0.
        log_interval: Print batch-level loss every *N* batches.

    Returns:
        Dictionary with keys ``"loss"``, ``"miou"``, ``"pixel_acc"``,
        ``"mean_acc"``, and ``"epoch_time_s"``.

    Example:
        >>> metrics = train_one_epoch(model, train_loader, criterion,
        ...                           optimizer, device, epoch=1)
        >>> print(f"Train mIoU: {metrics['miou']:.4f}")
    """
    model.train()
    num_classes = (
        criterion.weight.shape[0]
        if hasattr(criterion, "weight") and criterion.weight is not None
        else 17
    )
    seg_metrics = SegmentationMetrics(num_classes)
    total_loss = 0.0
    start = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        preds = logits.detach().argmax(dim=1)
        seg_metrics.update(preds, masks)

        if batch_idx % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    results = seg_metrics.compute()
    results["loss"] = total_loss / len(loader)
    results["epoch_time_s"] = round(time.time() - start, 2)
    return results


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    """Run one full validation epoch without gradient computation.

    Args:
        model: The segmentation model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Target compute device.
        epoch: Current epoch number (used for progress display).

    Returns:
        Dictionary with keys ``"loss"``, ``"miou"``, ``"pixel_acc"``,
        ``"mean_acc"``, and ``"epoch_time_s"``.

    Example:
        >>> val_metrics = validate_one_epoch(model, val_loader, criterion,
        ...                                  device, epoch=1)
        >>> print(f"Val mIoU: {val_metrics['miou']:.4f}")
    """
    model.eval()
    num_classes = (
        criterion.weight.shape[0]
        if hasattr(criterion, "weight") and criterion.weight is not None
        else 17
    )
    seg_metrics = SegmentationMetrics(num_classes)
    total_loss = 0.0
    start = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        seg_metrics.update(preds, masks)

    results = seg_metrics.compute()
    results["loss"] = total_loss / len(loader)
    results["epoch_time_s"] = round(time.time() - start, 2)
    return results
