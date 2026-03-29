"""Visualisation utilities for training metrics and model comparison.

Provides functions to:

- Plot train/val loss and mIoU curves for a single model run.
- Overlay multiple models' curves for side-by-side comparison.
- Save figures to ``logs/`` alongside the CSV files.

All functions read from the CSV log files written by :class:`~utils.logger.CSVLogger`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_log(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a CSV log and split into train and val DataFrames.

    Args:
        csv_path: Path to the CSV file written by :class:`~utils.logger.CSVLogger`.

    Returns:
        A tuple ``(train_df, val_df)`` filtered by the ``phase`` column.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    df = pd.read_csv(path)
    return df[df["phase"] == "train"].copy(), df[df["phase"] == "val"].copy()


def _infer_model_label(csv_path: Path) -> str:
    """Extract a human-readable model label from the CSV filename.

    The filename convention is ``<model_name>-<timestamp>.csv``.
    This returns the ``<model_name>`` portion.

    Args:
        csv_path: Path to the CSV log file.

    Returns:
        Model name string (e.g. ``"convnext_unet_tiny"``).
    """
    return csv_path.stem.rsplit("-", 1)[0]


# ---------------------------------------------------------------------------
# Single model plots
# ---------------------------------------------------------------------------


def plot_training_curves(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> Path:
    """Plot loss and mIoU curves for a single training run.

    Creates a 2×1 figure:

    - Top panel: train + val loss vs epoch.
    - Bottom panel: train + val mIoU vs epoch.

    Args:
        csv_path: Path to the training log CSV.
        output_dir: Directory to save the figure.  Defaults to the
            same directory as *csv_path*.
        show: If ``True``, call ``plt.show()`` after saving.

    Returns:
        Path to the saved PNG figure.

    Example:
        >>> fig_path = plot_training_curves("logs/convnext_unet_tiny-202401011200.csv")
        >>> print(fig_path)
    """
    csv_path = Path(csv_path)
    train_df, val_df = _load_log(csv_path)
    label = _infer_model_label(csv_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Training curves — {label}", fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(train_df["epoch"], train_df["loss"], label="Train loss", linewidth=1.8)
    axes[0].plot(
        val_df["epoch"], val_df["loss"], label="Val loss", linewidth=1.8, linestyle="--"
    )
    axes[0].set_ylabel("Loss")
    axes[0].legend(framealpha=0.7)
    axes[0].grid(True, alpha=0.3)

    # mIoU
    axes[1].plot(train_df["epoch"], train_df["miou"], label="Train mIoU", linewidth=1.8)
    axes[1].plot(
        val_df["epoch"], val_df["miou"], label="Val mIoU", linewidth=1.8, linestyle="--"
    )
    axes[1].set_ylabel("mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(framealpha=0.7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(output_dir) if output_dir else csv_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{label}_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Multi-model comparison plots
# ---------------------------------------------------------------------------


def plot_model_comparison(
    csv_paths: list[str | Path],
    metric: str = "miou",
    phase: str = "val",
    output_dir: str | Path | None = None,
    show: bool = False,
    labels: list[str] | None = None,
) -> Path:
    """Overlay metric curves from multiple model logs on one figure.

    Args:
        csv_paths: List of CSV log file paths to compare.
        metric: Column name to plot — ``"miou"``, ``"loss"``,
            ``"pixel_acc"``, or ``"mean_acc"``.  Defaults to ``"miou"``.
        phase: Which phase to plot — ``"train"`` or ``"val"``.
            Defaults to ``"val"``.
        output_dir: Directory for the saved figure.  Defaults to the
            parent directory of the first CSV file.
        show: If ``True``, display the figure interactively.
        labels: Optional list of display names, one per CSV path.
            If omitted, inferred from filenames.

    Returns:
        Path to the saved PNG figure.

    Raises:
        ValueError: If *csv_paths* is empty.

    Example:
        >>> logs = ["logs/convnext_unet_tiny-*.csv",
        ...         "logs/swin_unet_tiny-*.csv"]
        >>> plot_model_comparison(logs, metric="miou")
    """
    if not csv_paths:
        raise ValueError("csv_paths must not be empty.")

    csv_paths = [Path(p) for p in csv_paths]
    if labels is None:
        labels = [_infer_model_label(p) for p in csv_paths]

    metric_title = {
        "miou": "mIoU",
        "loss": "Loss",
        "pixel_acc": "Pixel accuracy",
        "mean_acc": "Mean per-class accuracy",
    }.get(metric, metric)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(
        f"Model comparison — {phase} {metric_title}",
        fontsize=13,
        fontweight="bold",
    )

    for csv_path, label in zip(csv_paths, labels):
        try:
            train_df, val_df = _load_log(csv_path)
            df = val_df if phase == "val" else train_df
            ax.plot(df["epoch"], df[metric], label=label, linewidth=1.8)
        except (FileNotFoundError, KeyError) as e:
            print(f"  Warning: could not load {csv_path}: {e}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_title)
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_dir = Path(output_dir) if output_dir else csv_paths[0].parent
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"comparison_{phase}_{metric}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
    return out_path


def plot_speed_vs_miou(
    entries: list[dict],
    output_dir: str | Path | None = None,
    show: bool = False,
) -> Path:
    """Scatter plot of inference speed vs best validation mIoU.

    Useful for comparing model efficiency trade-offs.

    Args:
        entries: List of dicts with keys:

            - ``"label"`` (str): Model display name.
            - ``"miou"`` (float): Best validation mIoU.
            - ``"fps"`` (float): Inference frames per second.
            - ``"params_m"`` (float, optional): Total params in millions
              (controls marker size if provided).

        output_dir: Directory to save the figure.
        show: Display interactively.

    Returns:
        Path to the saved PNG figure.

    Example:
        >>> entries = [
        ...     {"label": "ConvNeXt-Tiny", "miou": 0.82, "fps": 45, "params_m": 42},
        ...     {"label": "Swin-Tiny",     "miou": 0.84, "fps": 32, "params_m": 42},
        ... ]
        >>> plot_speed_vs_miou(entries, output_dir="logs/")
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Speed vs accuracy trade-off", fontsize=13, fontweight="bold")

    for entry in entries:
        size = entry.get("params_m", 10) * 5  # scale marker by param count
        ax.scatter(entry["fps"], entry["miou"], s=size, zorder=3)
        ax.annotate(
            entry["label"],
            (entry["fps"], entry["miou"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("Inference speed (FPS)")
    ax.set_ylabel("Validation mIoU")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_dir = Path(output_dir) if output_dir else Path("logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "speed_vs_miou.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
    return out_path


def compare_logs_from_dir(
    log_dir: str | Path,
    metric: str = "miou",
    output_dir: str | Path | None = None,
) -> None:
    """Discover all CSV logs in *log_dir* and plot a comparison figure.

    Convenience wrapper around :func:`plot_model_comparison` that
    automatically discovers all ``*.csv`` files in the given directory.

    Args:
        log_dir: Directory containing ``*.csv`` training logs.
        metric: Metric column to compare.
        output_dir: Directory for the output figure.  Defaults to
            *log_dir*.

    Example:
        >>> compare_logs_from_dir("logs/", metric="miou")
    """
    log_dir = Path(log_dir)
    csv_files = sorted(log_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {log_dir}")
        return

    out = plot_model_comparison(
        csv_files,
        metric=metric,
        output_dir=output_dir or log_dir,
    )
    print(f"Comparison figure saved: {out}")
