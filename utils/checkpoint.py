"""Model checkpoint save and load utilities.

Manages two checkpoint directories:

- ``checkpoints/best/``  — saved when validation mIoU improves.
- ``checkpoints/latest/`` — saved every *N* epochs regardless of metric.

Checkpoint files are named::

    <model_name>_best.pth
    <model_name>_epoch<NNN>.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """Save and restore model + optimiser checkpoints.

    Args:
        best_dir: Path to directory for best-metric checkpoints.
        latest_dir: Path to directory for periodic latest checkpoints.
        model_name: Used as the filename prefix.
        save_every_n_epochs: Frequency of latest checkpoint saves.
            Defaults to 5.

    Example:
        >>> ckpt = CheckpointManager("checkpoints/best",
        ...                          "checkpoints/latest",
        ...                          "convnext_unet_tiny")
        >>> ckpt.save_best(model, optimizer, epoch=10, miou=0.82)
        >>> ckpt.save_latest(model, optimizer, epoch=10)
    """

    def __init__(
        self,
        best_dir: str | Path,
        latest_dir: str | Path,
        model_name: str,
        save_every_n_epochs: int = 5,
    ) -> None:
        self.best_dir = Path(best_dir)
        self.latest_dir = Path(latest_dir)
        self.model_name = model_name
        self.save_every_n_epochs = save_every_n_epochs
        self.best_miou: float = 0.0

        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _build_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the checkpoint state dict.

        Args:
            model: Model whose state will be saved.
            optimizer: Optimiser whose state will be saved.
            epoch: Current training epoch (1-based).
            extra: Optional additional metadata to include.

        Returns:
            Dictionary suitable for ``torch.save``.
        """
        state: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": self.model_name,
        }
        if extra:
            state.update(extra)
        return state

    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        miou: float,
    ) -> bool:
        """Save checkpoint if *miou* exceeds the current best.

        Args:
            model: Model to checkpoint.
            optimizer: Optimiser to checkpoint.
            epoch: Current epoch (1-based).
            miou: Validation mIoU achieved this epoch.

        Returns:
            ``True`` if the checkpoint was saved (new best), else ``False``.
        """
        if miou <= self.best_miou:
            return False

        self.best_miou = miou
        path = self.best_dir / f"{self.model_name}_best.pth"
        torch.save(
            self._build_state(model, optimizer, epoch, {"best_miou": miou}),
            path,
        )
        return True

    def save_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        """Periodically save a latest checkpoint.

        Saves only when ``epoch % save_every_n_epochs == 0``.

        Args:
            model: Model to checkpoint.
            optimizer: Optimiser to checkpoint.
            epoch: Current epoch (1-based).
        """
        if epoch % self.save_every_n_epochs != 0:
            return
        path = self.latest_dir / f"{self.model_name}_epoch{epoch:03d}.pth"
        torch.save(self._build_state(model, optimizer, epoch), path)

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def load_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> int:
        """Load the best checkpoint into *model* (and optionally *optimizer*).

        Args:
            model: Model to load weights into.
            optimizer: If provided, also restores optimiser state.
            device: Device to map tensors to.

        Returns:
            The epoch at which the best checkpoint was saved.

        Raises:
            FileNotFoundError: If no best checkpoint file exists.
        """
        path = self.best_dir / f"{self.model_name}_best.pth"
        return self._load(path, model, optimizer, device)

    def load_checkpoint(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
    ) -> int:
        """Load an arbitrary checkpoint file.

        Args:
            path: Path to the ``.pth`` checkpoint file.
            model: Model to load weights into.
            optimizer: If provided, also restores optimiser state.
            device: Device to map tensors to.

        Returns:
            The epoch stored in the checkpoint.
        """
        return self._load(Path(path), model, optimizer, device)

    def _load(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        device: str | torch.device,
    ) -> int:
        """Internal loader shared by :meth:`load_best` and :meth:`load_checkpoint`.

        Args:
            path: Full path to the checkpoint file.
            model: Target model.
            optimizer: Optional optimiser.
            device: Map-location device.

        Returns:
            Epoch number from the checkpoint.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        return int(state.get("epoch", 0))
