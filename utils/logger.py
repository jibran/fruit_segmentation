"""CSV-based training logger.

Creates a timestamped CSV file in ``logs/`` and appends one row per
epoch containing loss, mIoU, and pixel accuracy for both train and
validation splits.

File naming convention::

    logs/<model_name>-<YYYYMMDDHHM>.csv

Example log file header::

    epoch,phase,loss,miou,pixel_acc,mean_acc,lr
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append-mode CSV logger for epoch-level training metrics.

    Args:
        log_dir: Directory in which to create the log file.
        model_name: Used as the prefix in the filename, e.g.
            ``"convnext_unet_tiny"``.

    Attributes:
        filepath: ``pathlib.Path`` to the created CSV file.

    Example:
        >>> logger = CSVLogger("logs", "convnext_unet_tiny")
        >>> logger.log(epoch=1, phase="train", loss=0.42, miou=0.61,
        ...            pixel_acc=0.88, mean_acc=0.72, lr=1e-3)
    """

    _FIELDNAMES = ["epoch", "phase", "loss", "miou", "pixel_acc", "mean_acc", "lr"]

    def __init__(self, log_dir: str | Path, model_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{model_name}-{timestamp}.csv"
        self.filepath = self.log_dir / filename

        # Write header
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writeheader()

    def log(
        self,
        epoch: int,
        phase: str,
        loss: float,
        miou: float,
        pixel_acc: float,
        mean_acc: float,
        lr: float,
    ) -> None:
        """Append one row to the CSV log.

        Args:
            epoch: Current epoch number (1-based).
            phase: Split identifier — ``"train"`` or ``"val"``.
            loss: Average loss for this epoch / phase.
            miou: Mean IoU (0.0–1.0).
            pixel_acc: Global pixel accuracy (0.0–1.0).
            mean_acc: Mean per-class accuracy (0.0–1.0).
            lr: Current learning rate of the primary parameter group.
        """
        row: dict[str, Any] = {
            "epoch": epoch,
            "phase": phase,
            "loss": round(loss, 6),
            "miou": round(miou, 6),
            "pixel_acc": round(pixel_acc, 6),
            "mean_acc": round(mean_acc, 6),
            "lr": f"{lr:.2e}",
        }
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writerow(row)

    def __repr__(self) -> str:
        return f"CSVLogger(filepath={self.filepath})"
