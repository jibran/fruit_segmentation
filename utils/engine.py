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


# ── Baseline (image-level classification) engine ─────────────────────────────


def _mask_to_image_label(
    masks: torch.Tensor,
    ignore_index: int = 16,
) -> torch.Tensor:
    """Derive a single image-level class label from a pixel mask.

    The dominant fruit class (most pixels, excluding background) is used
    as the image label.  If an image contains only background pixels the
    background index is returned so the loss is still well-defined.

    Args:
        masks: Integer mask tensor of shape ``(B, H, W)``.
        ignore_index: Background/ignore class index.  Defaults to 16.

    Returns:
        Long tensor of shape ``(B,)`` with one class index per image.
    """
    B = masks.shape[0]
    labels = torch.zeros(B, dtype=torch.long, device=masks.device)
    for i in range(B):
        flat = masks[i].flatten()
        fruit = flat[flat != ignore_index]
        if fruit.numel() == 0:
            labels[i] = ignore_index
        else:
            labels[i] = torch.bincount(fruit).argmax()
    return labels


def train_one_epoch_baseline(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    ignore_index: int = 16,
    grad_clip: float = 1.0,
    log_interval: int = 10,
) -> dict[str, Any]:
    """Run one training epoch for image-level baseline classifiers.

    Converts each ``(B, H, W)`` pixel mask to a ``(B,)`` image-level
    label (dominant fruit class) before computing the loss.  Metrics are
    reported in the same format as :func:`train_one_epoch` so the logging
    and checkpointing code requires no changes.

    Args:
        model: Baseline classifier (``ConvNeXtBaseline`` or ``SwinBaseline``).
        loader: Training DataLoader (same one used for the U-Net models).
        criterion: ``nn.CrossEntropyLoss`` or compatible loss.
        optimizer: Configured optimiser.
        device: Target compute device.
        epoch: Current epoch number (1-based).
        ignore_index: Background class index excluded from label derivation.
            Defaults to 16.
        grad_clip: Maximum gradient norm.  Set to 0 to disable.
        log_interval: Print batch-level loss every *N* batches.

    Returns:
        Dictionary with keys ``"loss"``, ``"miou"``, ``"pixel_acc"``,
        ``"mean_acc"``, ``"top1_acc"``, and ``"epoch_time_s"``.
        ``"miou"`` is approximated as mean per-class accuracy over the
        image-level predictions so results are comparable in the log.
    """
    model.train()
    num_classes = (
        criterion.weight.shape[0]
        if hasattr(criterion, "weight") and criterion.weight is not None
        else 17
    )

    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    start = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        labels = _mask_to_image_label(masks, ignore_index=ignore_index)

        optimizer.zero_grad()
        logits = model(images)  # (B, num_classes)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.detach().argmax(dim=1).cpu()
        labels_cpu = labels.cpu()

        correct += (preds == labels_cpu).sum().item()
        total += labels_cpu.numel()

        for c in range(num_classes):
            mask_c = labels_cpu == c
            per_class_correct[c] += (preds[mask_c] == c).sum()
            per_class_total[c] += mask_c.sum()

        if batch_idx % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    top1_acc = correct / max(total, 1)
    valid_cls = per_class_total > 0
    per_cls_acc = torch.where(
        valid_cls,
        per_class_correct.float() / per_class_total.float().clamp(min=1),
        torch.tensor(float("nan")),
    )
    mean_acc = float(per_cls_acc[~per_cls_acc.isnan()].mean())

    return {
        "loss": total_loss / len(loader),
        "miou": mean_acc,  # proxy: mean per-class acc
        "pixel_acc": top1_acc,  # top-1 image classification acc
        "mean_acc": mean_acc,
        "top1_acc": top1_acc,
        "class_iou": [float("nan")] * num_classes,  # N/A for classifier
        "class_acc": per_cls_acc.tolist(),
        "epoch_time_s": round(time.time() - start, 2),
    }


@torch.no_grad()
def validate_one_epoch_baseline(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    ignore_index: int = 16,
) -> dict[str, Any]:
    """Run one validation epoch for image-level baseline classifiers.

    Args:
        model: Baseline classifier.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Target compute device.
        epoch: Current epoch number.
        ignore_index: Background class index.  Defaults to 16.

    Returns:
        Same key structure as :func:`train_one_epoch_baseline`.
    """
    model.eval()
    num_classes = (
        criterion.weight.shape[0]
        if hasattr(criterion, "weight") and criterion.weight is not None
        else 17
    )

    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    start = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        labels = _mask_to_image_label(masks, ignore_index=ignore_index)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu()
        labels_cpu = labels.cpu()

        correct += (preds == labels_cpu).sum().item()
        total += labels_cpu.numel()

        for c in range(num_classes):
            mask_c = labels_cpu == c
            per_class_correct[c] += (preds[mask_c] == c).sum()
            per_class_total[c] += mask_c.sum()

    top1_acc = correct / max(total, 1)
    valid_cls = per_class_total > 0
    per_cls_acc = torch.where(
        valid_cls,
        per_class_correct.float() / per_class_total.float().clamp(min=1),
        torch.tensor(float("nan")),
    )
    mean_acc = float(per_cls_acc[~per_cls_acc.isnan()].mean())

    return {
        "loss": total_loss / len(loader),
        "miou": mean_acc,
        "pixel_acc": top1_acc,
        "mean_acc": mean_acc,
        "top1_acc": top1_acc,
        "class_iou": [float("nan")] * num_classes,
        "class_acc": per_cls_acc.tolist(),
        "epoch_time_s": round(time.time() - start, 2),
    }
