"""Segmentation evaluation metrics.

Implements mean Intersection over Union (mIoU) and pixel accuracy,
both as batch-level accumulators suitable for use inside a training loop.
"""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates per-class IoU and pixel accuracy over a dataset split.

    Call :meth:`update` after each batch, then :meth:`compute` at the end
    of the epoch to get scalar summaries.  Call :meth:`reset` before each
    new epoch.

    Args:
        num_classes: Total number of segmentation classes (including background).
        ignore_index: Class index to exclude from metric computation.
            Defaults to ``-1`` (no class ignored).

    Example:
        >>> metrics = SegmentationMetrics(num_classes=17)
        >>> for images, masks in val_loader:
        ...     logits = model(images)
        ...     preds  = logits.argmax(dim=1)
        ...     metrics.update(preds, masks)
        >>> results = metrics.compute()
        >>> print(f"mIoU: {results['miou']:.4f}")
    """

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Reset the accumulated confusion matrix to zeros."""
        self.confusion_matrix[:] = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions into the confusion matrix.

        Args:
            preds: Predicted class indices, shape ``(B, H, W)``, dtype int.
            targets: Ground truth class indices, shape ``(B, H, W)``, dtype int.
        """
        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        # Mask out ignored pixels
        valid = targets_np != self.ignore_index
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

        # Fast confusion matrix accumulation
        k = self.num_classes
        indices = k * targets_np.astype(np.int64) + preds_np.astype(np.int64)
        cm = np.bincount(indices, minlength=k * k).reshape(k, k)
        self.confusion_matrix += cm

    def compute(self) -> dict[str, float]:
        """Compute mIoU, per-class IoU, and pixel accuracy from accumulated state.

        Returns:
            Dictionary with keys:

            - ``"miou"`` — mean IoU across all classes (float).
            - ``"pixel_acc"`` — global pixel accuracy (float).
            - ``"class_iou"`` — per-class IoU as a list of floats.
            - ``"class_acc"`` — per-class accuracy as a list of floats (NaN for absent classes).
            - ``"mean_acc"`` — mean per-class accuracy (float).

        Example:
            >>> results = metrics.compute()
            >>> results["miou"]
            0.8423
        """
        cm = self.confusion_matrix.astype(np.float64)
        diag = np.diag(cm)

        # Per-class IoU: TP / (TP + FP + FN)
        row_sum = cm.sum(axis=1)  # actual positives per class
        col_sum = cm.sum(axis=0)  # predicted positives per class
        union = row_sum + col_sum - diag

        # Avoid divide-by-zero for absent classes
        valid_classes = union > 0
        iou_per_class = np.where(valid_classes, diag / (union + 1e-10), np.nan)
        miou = float(np.nanmean(iou_per_class))

        # Pixel accuracy
        total_pixels = cm.sum()
        pixel_acc = float(diag.sum() / (total_pixels + 1e-10))

        # Mean per-class accuracy
        per_class_acc = np.where(row_sum > 0, diag / (row_sum + 1e-10), np.nan)
        mean_acc = float(np.nanmean(per_class_acc))

        return {
            "miou": miou,
            "pixel_acc": pixel_acc,
            "class_iou": iou_per_class.tolist(),
            "class_acc": per_class_acc.tolist(),
            "mean_acc": mean_acc,
        }
