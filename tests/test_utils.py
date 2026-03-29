"""Tests for utility modules: metrics, CSV logger, and checkpoint manager."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.checkpoint import CheckpointManager
from utils.logger import CSVLogger
from utils.metrics import SegmentationMetrics

# ---------------------------------------------------------------------------
# SegmentationMetrics tests
# ---------------------------------------------------------------------------


class TestSegmentationMetrics:
    """Tests for SegmentationMetrics accumulator."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions must yield mIoU == 1.0 and pixel_acc == 1.0."""
        metrics = SegmentationMetrics(num_classes=3)
        targets = torch.tensor([[0, 1, 2, 0]])
        metrics.update(targets.clone(), targets.clone())
        results = metrics.compute()
        assert results["miou"] == pytest.approx(1.0, abs=1e-5)
        assert results["pixel_acc"] == pytest.approx(1.0, abs=1e-5)

    def test_all_wrong_predictions(self) -> None:
        """Completely wrong predictions on 2-class problem → mIoU ≈ 0."""
        metrics = SegmentationMetrics(num_classes=2)
        targets = torch.zeros(1, 4, dtype=torch.long)  # all class 0
        preds = torch.ones(1, 4, dtype=torch.long)  # all class 1
        metrics.update(preds, targets)
        results = metrics.compute()
        # Class 0: TP=0, FP=0, FN=4 → IoU=0; Class 1: TP=0, FP=4, FN=0 → IoU=0
        assert results["miou"] == pytest.approx(0.0, abs=1e-5)

    def test_reset_clears_state(self) -> None:
        """reset() must zero out the confusion matrix."""
        metrics = SegmentationMetrics(num_classes=4)
        targets = torch.randint(0, 4, (2, 16))
        metrics.update(targets, targets)
        metrics.reset()
        assert metrics.confusion_matrix.sum() == 0

    def test_accumulate_multiple_batches(self) -> None:
        """Accumulated result must equal single-batch result on same data."""
        metrics_single = SegmentationMetrics(num_classes=5)
        metrics_accum = SegmentationMetrics(num_classes=5)

        t1 = torch.randint(0, 5, (2, 10))
        t2 = torch.randint(0, 5, (2, 10))

        # Single batch (concatenated)
        all_targets = torch.cat([t1, t2], dim=0)
        metrics_single.update(all_targets, all_targets)

        # Accumulated
        metrics_accum.update(t1, t1)
        metrics_accum.update(t2, t2)

        r1 = metrics_single.compute()
        r2 = metrics_accum.compute()
        assert r1["miou"] == pytest.approx(r2["miou"], abs=1e-8)

    def test_output_keys(self) -> None:
        """compute() must return all required metric keys."""
        metrics = SegmentationMetrics(num_classes=3)
        targets = torch.randint(0, 3, (1, 4))
        metrics.update(targets, targets)
        result = metrics.compute()
        for key in ("miou", "pixel_acc", "class_iou", "mean_acc"):
            assert key in result, f"Missing key: {key}"

    def test_class_iou_length(self) -> None:
        """class_iou must have one entry per class."""
        n = 6
        metrics = SegmentationMetrics(num_classes=n)
        targets = torch.randint(0, n, (2, 8))
        metrics.update(targets, targets)
        assert len(metrics.compute()["class_iou"]) == n


# ---------------------------------------------------------------------------
# CSVLogger tests
# ---------------------------------------------------------------------------


class TestCSVLogger:
    """Tests for the CSVLogger append-mode logger."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Instantiation must create a CSV file in log_dir."""
        logger = CSVLogger(tmp_path, "test_model")
        assert logger.filepath.exists()

    def test_writes_header(self, tmp_path: Path) -> None:
        """CSV file must start with the expected header row."""
        logger = CSVLogger(tmp_path, "test_model")
        with open(logger.filepath) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames or []) == {
                "epoch",
                "phase",
                "loss",
                "miou",
                "pixel_acc",
                "mean_acc",
                "lr",
            }

    def test_appends_rows(self, tmp_path: Path) -> None:
        """log() must append one row per call."""
        logger = CSVLogger(tmp_path, "test_model")
        logger.log(1, "train", 0.5, 0.4, 0.8, 0.6, 1e-3)
        logger.log(1, "val", 0.6, 0.38, 0.78, 0.58, 1e-3)
        with open(logger.filepath) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_row_values(self, tmp_path: Path) -> None:
        """Logged values must round-trip correctly from CSV."""
        logger = CSVLogger(tmp_path, "test_model")
        logger.log(2, "val", 0.123456, 0.654321, 0.999, 0.888, 1e-4)
        with open(logger.filepath) as f:
            rows = list(csv.DictReader(f))
        row = rows[0]
        assert int(row["epoch"]) == 2
        assert row["phase"] == "val"
        assert float(row["loss"]) == pytest.approx(0.123456, abs=1e-6)

    def test_filename_contains_model_name(self, tmp_path: Path) -> None:
        """Log filename must start with the model name."""
        logger = CSVLogger(tmp_path, "convnext_unet_tiny")
        assert logger.filepath.name.startswith("convnext_unet_tiny")

    def test_filename_has_timestamp(self, tmp_path: Path) -> None:
        """Log filename must contain a 12-digit timestamp."""
        import re

        logger = CSVLogger(tmp_path, "model")
        # Expect pattern: model-YYYYMMDDHHMM.csv
        assert re.search(r"-\d{12}\.csv$", logger.filepath.name)


# ---------------------------------------------------------------------------
# CheckpointManager tests
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    """Tests for the CheckpointManager save/load utility."""

    def _tiny_model(self) -> torch.nn.Module:
        """Return a minimal linear model for checkpoint testing."""
        return torch.nn.Linear(4, 2)

    def _tiny_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=0.01)

    def test_save_best_creates_file(self, tmp_path: Path) -> None:
        """save_best must create a file on first call (any mIoU > 0)."""
        ckpt = CheckpointManager(tmp_path / "best", tmp_path / "latest", "test_model")
        model = self._tiny_model()
        opt = self._tiny_optimizer(model)
        saved = ckpt.save_best(model, opt, epoch=1, miou=0.5)
        assert saved is True
        assert (tmp_path / "best" / "test_model_best.pth").exists()

    def test_save_best_not_saved_if_lower(self, tmp_path: Path) -> None:
        """save_best must return False when mIoU does not improve."""
        ckpt = CheckpointManager(tmp_path / "best", tmp_path / "latest", "test_model")
        model = self._tiny_model()
        opt = self._tiny_optimizer(model)
        ckpt.save_best(model, opt, epoch=1, miou=0.8)
        saved = ckpt.save_best(model, opt, epoch=2, miou=0.7)
        assert saved is False

    def test_save_latest_respects_frequency(self, tmp_path: Path) -> None:
        """save_latest must only save when epoch is divisible by save_every."""
        ckpt = CheckpointManager(
            tmp_path / "best",
            tmp_path / "latest",
            "test_model",
            save_every_n_epochs=5,
        )
        model = self._tiny_model()
        opt = self._tiny_optimizer(model)
        for epoch in range(1, 11):
            ckpt.save_latest(model, opt, epoch)

        saved_files = list((tmp_path / "latest").glob("*.pth"))
        # Epochs 5 and 10 should be saved
        assert len(saved_files) == 2

    def test_load_best_restores_weights(self, tmp_path: Path) -> None:
        """load_best must restore model weights identical to those saved."""
        ckpt = CheckpointManager(tmp_path / "best", tmp_path / "latest", "test_model")
        model = self._tiny_model()
        opt = self._tiny_optimizer(model)

        # Save known weights
        original_weight = model.weight.data.clone()
        ckpt.save_best(model, opt, epoch=3, miou=0.75)

        # Corrupt weights then reload
        model.weight.data.fill_(0.0)
        epoch = ckpt.load_best(model)

        assert epoch == 3
        assert torch.allclose(model.weight.data, original_weight)

    def test_load_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """load_best must raise FileNotFoundError if no checkpoint exists."""
        ckpt = CheckpointManager(tmp_path / "best", tmp_path / "latest", "no_model")
        model = self._tiny_model()
        with pytest.raises(FileNotFoundError):
            ckpt.load_best(model)
