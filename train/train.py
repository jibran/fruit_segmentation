"""Main training script for fruit segmentation models.

Implements two-phase progressive unfreezing:

- **Phase 1**: Backbone frozen, only decoder trains at a high LR.
- **Phase 2**: Full end-to-end fine-tuning with discriminative LRs.

Usage::

    # Train ConvNeXt-V2 U-Net
    python train/train.py --config config/convnext_unet.yaml

    # Train Swin-V2 U-Net with a different model size
    python train/train.py --config config/swin_unet.yaml --size small

    # Resume from a checkpoint
    python train/train.py --config config/convnext_unet.yaml \\
        --resume checkpoints/latest/convnext_unet_tiny_epoch010.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from config.config_loader import get_model_name_with_size, load_config
from dataset.fruit_dataset import build_dataloaders
from models import build_model
from utils.checkpoint import CheckpointManager
from utils.engine import train_one_epoch, validate_one_epoch
from utils.logger import CSVLogger
from utils.transforms import build_train_transform, build_val_transform


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-UNet or Swin-UNet for fruit segmentation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. config/convnext_unet.yaml).",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["tiny", "small", "base"],
        default=None,
        help="Override model size from config (tiny | small | base).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cuda:0', 'cpu'). Auto-detected if omitted.",
    )
    return parser.parse_args()


def get_device(override: str | None) -> torch.device:
    """Resolve the compute device.

    Args:
        override: Optional device string from CLI.

    Returns:
        ``torch.device`` — CUDA if available, else CPU.
    """
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_criterion(cfg: dict, device: torch.device) -> nn.CrossEntropyLoss:
    """Construct the loss function, optionally with class weights.

    Args:
        cfg: Loaded configuration dict.
        device: Device to place weight tensor on.

    Returns:
        ``nn.CrossEntropyLoss`` instance.
    """
    weight_strategy = cfg["training"].get("class_weights", None)
    if weight_strategy == "auto":
        # Would compute from dataset — placeholder returns uniform weights
        print("  [!] class_weights='auto' requested — set uniform for now.")
        print("      Run FruitSegmentationDataset.get_class_weights() on your data.")
    return nn.CrossEntropyLoss().to(device)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    num_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Construct the LR scheduler.

    Args:
        optimizer: Optimiser to wrap.
        cfg: Loaded configuration dict.
        num_epochs: Total epochs for this training phase.

    Returns:
        A PyTorch LR scheduler.
    """
    sched_name = cfg["training"].get("scheduler", "cosine")
    warmup = cfg["training"].get("warmup_epochs", 2)

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_epochs - warmup)
        )
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )


def run_phase(
    phase: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.CrossEntropyLoss,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
    logger: CSVLogger,
    ckpt_manager: CheckpointManager,
    start_epoch: int = 1,
) -> None:
    """Execute one training phase (either freeze or unfreeze).

    Args:
        phase: 1 for frozen-backbone phase, 2 for end-to-end fine-tuning.
        model: Segmentation model.
        optimizer: Configured optimiser for this phase.
        scheduler: LR scheduler for this phase.
        criterion: Loss function.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Compute device.
        cfg: Loaded configuration dict.
        logger: CSV logger instance.
        ckpt_manager: Checkpoint manager.
        start_epoch: Epoch to start from (for resume). Defaults to 1.
    """
    train_cfg = cfg["training"]
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_interval = cfg["logging"].get("log_interval", 10)
    num_epochs = (
        train_cfg["epochs_phase1"] if phase == 1 else train_cfg["epochs_phase2"]
    )

    print(f"\n{'='*60}")
    print(
        f"  Phase {phase} — {'Backbone FROZEN' if phase == 1 else 'End-to-end fine-tuning'}"
    )
    print(f"  Epochs: {num_epochs}  |  Device: {device}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs + 1):
        # ── Train ────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            grad_clip,
            log_interval,
        )
        # ── Validate ─────────────────────────────────────────────────
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch,
        )

        # ── LR step ──────────────────────────────────────────────────
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["miou"])
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ── Log ──────────────────────────────────────────────────────
        logger.log(
            epoch,
            "train",
            train_metrics["loss"],
            train_metrics["miou"],
            train_metrics["pixel_acc"],
            train_metrics["mean_acc"],
            current_lr,
        )
        logger.log(
            epoch,
            "val",
            val_metrics["loss"],
            val_metrics["miou"],
            val_metrics["pixel_acc"],
            val_metrics["mean_acc"],
            current_lr,
        )

        # ── Checkpoints ───────────────────────────────────────────────
        saved_best = ckpt_manager.save_best(
            model, optimizer, epoch, val_metrics["miou"]
        )
        ckpt_manager.save_latest(model, optimizer, epoch)

        best_marker = " ★ NEW BEST" if saved_best else ""
        print(
            f"  Epoch {epoch:03d}/{num_epochs:03d} | "
            f"Train loss={train_metrics['loss']:.4f} mIoU={train_metrics['miou']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} mIoU={val_metrics['miou']:.4f} "
            f"({val_metrics['epoch_time_s']:.1f}s){best_marker}"
        )


def main() -> None:
    """Entry point for the training script.

    Loads config, builds model and dataloaders, then runs Phase 1
    (frozen backbone) followed by Phase 2 (end-to-end fine-tuning).
    """
    args = parse_args()
    cfg = load_config(args.config)

    # CLI size override
    if args.size:
        cfg["model"]["size"] = args.size

    device = get_device(args.device)
    model_name = get_model_name_with_size(cfg)
    train_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoints"]
    log_cfg = cfg["logging"]

    print("\nFruit Segmentation Training")
    print(f"  Model : {model_name}")
    print(f"  Device: {device}")
    print(f"  Config: {args.config}")

    # ── Transforms & Dataloaders ─────────────────────────────────────
    train_tf = build_train_transform(cfg)
    val_tf = build_val_transform(cfg)
    weighted_sampling = cfg["data"].get("weighted_sampling", False)
    train_loader, val_loader, _ = build_dataloaders(
        cfg, train_tf, val_tf, weighted_sampling=weighted_sampling
    )

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    param_counts = model.count_parameters()
    print(
        f"  Params: {param_counts['total'] / 1e6:.1f}M total "
        f"({param_counts['backbone'] / 1e6:.1f}M backbone + "
        f"{param_counts['decoder'] / 1e6:.1f}M decoder)"
    )

    # ── Utilities ────────────────────────────────────────────────────
    criterion = build_criterion(cfg, device)
    logger = CSVLogger(log_cfg["log_dir"], model_name)
    ckpt_manager = CheckpointManager(
        ckpt_cfg["best_dir"],
        ckpt_cfg["latest_dir"],
        model_name,
        ckpt_cfg.get("save_every_n_epochs", 5),
    )
    print(f"  Logs  : {logger.filepath}")

    # ── Resume (optional) ────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        start_epoch = ckpt_manager.load_checkpoint(args.resume, model) + 1
        print(f"  Resumed from epoch {start_epoch - 1}")

    # ──────────────────────────────────────────────────────────────────
    # Phase 1: Backbone frozen — only decoder trains
    # ──────────────────────────────────────────────────────────────────
    model.freeze_backbone()
    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr_decoder"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler_p1 = build_scheduler(optimizer_p1, cfg, train_cfg["epochs_phase1"])

    run_phase(
        phase=1,
        model=model,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        logger=logger,
        ckpt_manager=ckpt_manager,
        start_epoch=start_epoch,
    )

    # ──────────────────────────────────────────────────────────────────
    # Phase 2: End-to-end fine-tuning with discriminative LRs
    # ──────────────────────────────────────────────────────────────────
    model.unfreeze_backbone()
    param_groups = model.get_param_groups(
        lr_backbone=train_cfg["lr_backbone_phase2"],
        lr_decoder=train_cfg["lr_decoder_phase2"],
    )
    optimizer_p2 = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler_p2 = build_scheduler(optimizer_p2, cfg, train_cfg["epochs_phase2"])

    run_phase(
        phase=2,
        model=model,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        logger=logger,
        ckpt_manager=ckpt_manager,
    )

    print(
        f"\nTraining complete. Best checkpoint: {ckpt_manager.best_dir}/{model_name}_best.pth"
    )


if __name__ == "__main__":
    main()
