"""CSV-based training logger.

Creates two CSV files per run inside ``logs/``:

1. **Epoch log** — one row per epoch/phase with aggregate metrics
   (loss, mIoU, pixel_acc, mean_acc, lr).  This is the same format
   as before, so existing analysis scripts are unaffected.

   Header::

       epoch,phase,loss,miou,pixel_acc,mean_acc,lr

2. **Class log** — one row per epoch/phase with per-class IoU and
   per-class accuracy for every fruit class.  Columns are labelled
   with the class names so the CSV is self-describing.

   Header::

       epoch,phase,<cls0>_iou,<cls1>_iou,...,<cls0>_acc,<cls1>_acc,...

Both files share the same timestamp stem, e.g.::

    logs/convnext_unet_tiny-202401011200.csv
    logs/convnext_unet_tiny-202401011200_classes.csv

File naming convention::

    logs/<model_name>-<YYYYMMDDHHMM>.csv
    logs/<model_name>-<YYYYMMDDHHMM>_classes.csv
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append-mode CSV logger for epoch-level training metrics.

    Writes two files per run: an aggregate epoch log and a per-class
    accuracy/IoU log.  Both are keyed by ``(epoch, phase)`` so they
    can be joined on those columns for analysis.

    Args:
        log_dir: Directory in which to create the log files.
        model_name: Used as the prefix in the filenames, e.g.
            ``"convnext_unet_tiny"``.
        class_names: Ordered list of fruit class name strings matching
            ``dataset.fruit_dataset.CLASS_NAMES``.  Used to label the
            per-class CSV columns.  If ``None``, columns are labelled
            ``class_0``, ``class_1``, etc.

    Attributes:
        filepath: ``pathlib.Path`` to the aggregate epoch CSV.
        class_filepath: ``pathlib.Path`` to the per-class CSV.

    Example:
        >>> from dataset.fruit_dataset import CLASS_NAMES
        >>> logger = CSVLogger("logs", "convnext_unet_tiny", CLASS_NAMES)
        >>> logger.log(epoch=1, phase="train", loss=0.42, miou=0.61,
        ...            pixel_acc=0.88, mean_acc=0.72, lr=1e-3,
        ...            class_iou=[...], class_acc=[...])
    """

    _EPOCH_FIELDS = ["epoch", "phase", "loss", "miou", "pixel_acc", "mean_acc", "lr"]

    def __init__(
        self,
        log_dir: str | Path,
        model_name: str,
        class_names: list[str] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        stem = f"{model_name}-{timestamp}"
        self.filepath = self.log_dir / f"{stem}.csv"
        self.class_filepath = self.log_dir / f"{stem}_classes.csv"

        # Build per-class column names
        n = len(class_names) if class_names else 0
        self._class_names = class_names or [f"class_{i}" for i in range(n)]
        self._class_fields = (
            ["epoch", "phase"]
            + [f"{c}_iou" for c in self._class_names]
            + [f"{c}_acc" for c in self._class_names]
        )

        # Write headers
        with open(self.filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._EPOCH_FIELDS).writeheader()

        with open(self.class_filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._class_fields).writeheader()

    def log(
        self,
        epoch: int,
        phase: str,
        loss: float,
        miou: float,
        pixel_acc: float,
        mean_acc: float,
        lr: float,
        class_iou: list[float] | None = None,
        class_acc: list[float] | None = None,
    ) -> None:
        """Append one row to both CSV logs.

        Args:
            epoch: Current epoch number (1-based).
            phase: Split identifier, ``"train"`` or ``"val"``.
            loss: Average loss for this epoch/phase.
            miou: Mean IoU across all fruit classes (0–1).
            pixel_acc: Global pixel accuracy (0–1).
            mean_acc: Mean per-class accuracy (0–1).
            lr: Current learning rate of the primary parameter group.
            class_iou: Per-class IoU values, one float per class.
                ``float("nan")`` for classes absent from this split.
                If ``None``, the per-class row is skipped.
            class_acc: Per-class accuracy values, one float per class.
                ``float("nan")`` for classes absent from this split.
                If ``None``, the per-class row is skipped.
        """
        # Aggregate epoch row
        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "phase": phase,
            "loss": round(loss, 6),
            "miou": round(miou, 6),
            "pixel_acc": round(pixel_acc, 6),
            "mean_acc": round(mean_acc, 6),
            "lr": f"{lr:.2e}",
        }
        with open(self.filepath, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._EPOCH_FIELDS).writerow(epoch_row)

        # Per-class row
        if class_iou is not None and class_acc is not None:
            class_row: dict[str, Any] = {"epoch": epoch, "phase": phase}
            for i, name in enumerate(self._class_names):
                iou_val = class_iou[i] if i < len(class_iou) else float("nan")
                acc_val = class_acc[i] if i < len(class_acc) else float("nan")
                class_row[f"{name}_iou"] = (
                    ""
                    if (isinstance(iou_val, float) and iou_val != iou_val)
                    else round(float(iou_val), 6)
                )
                class_row[f"{name}_acc"] = (
                    ""
                    if (isinstance(acc_val, float) and acc_val != acc_val)
                    else round(float(acc_val), 6)
                )
            with open(self.class_filepath, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._class_fields).writerow(class_row)

    def __repr__(self) -> str:
        return (
            f"CSVLogger(filepath={self.filepath}, "
            f"class_filepath={self.class_filepath})"
        )
