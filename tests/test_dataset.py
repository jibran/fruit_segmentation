"""Tests for FruitSegmentationDataset, DataLoader builder, and transforms.

Uses a temporary synthetic dataset (generated in a pytest fixture) so
no real image data is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.fruit_dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    FruitSegmentationDataset,
    build_dataloaders,
)
from utils.transforms import build_train_transform, build_val_transform

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory) -> Path:
    """Create a minimal synthetic dataset with 6 images per split.

    Directory layout::

        <tmp>/
        ├── images/
        │   ├── train/  img_00.png … img_05.png
        │   ├── val/
        │   └── test/
        └── masks/
            ├── train/  img_00.png … img_05.png  (grayscale 0-16)
            ├── val/
            └── test/

    Returns:
        Root path of the synthetic dataset.
    """
    root = tmp_path_factory.mktemp("dataset")
    img_size = (64, 64)

    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)

        for i in range(6):
            # RGB image — random noise
            img = Image.fromarray(
                np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
            )
            img.save(root / "images" / split / f"img_{i:02d}.png")

            # Grayscale mask — random class indices 0-16
            mask = Image.fromarray(
                np.random.randint(0, NUM_CLASSES, img_size, dtype=np.uint8)
            )
            mask.save(root / "masks" / split / f"img_{i:02d}.png")

    return root


@pytest.fixture(scope="module")
def minimal_cfg(synthetic_dataset) -> dict:
    """Build a minimal config dict pointing to the synthetic dataset."""
    return {
        "data": {
            "root_dir": str(synthetic_dataset),
            "image_size": 64,
            "num_classes": NUM_CLASSES,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation_limit": 15,
            "zoom_limit": 0.1,
            "shear_limit": 5,
            "brightness_contrast": False,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
    }


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestFruitSegmentationDataset:
    """Tests for FruitSegmentationDataset."""

    def test_len(self, synthetic_dataset: Path) -> None:
        """Dataset length must match the number of created images."""
        ds = FruitSegmentationDataset(synthetic_dataset, split="train")
        assert len(ds) == 6

    def test_item_shapes(self, synthetic_dataset: Path) -> None:
        """Each item must be (image: float, mask: long) with correct shapes."""
        ds = FruitSegmentationDataset(synthetic_dataset, split="train")
        image, mask = ds[0]
        assert image.shape == (3, 64, 64), f"Image shape: {image.shape}"
        assert mask.shape == (64, 64), f"Mask shape: {mask.shape}"

    def test_image_dtype(self, synthetic_dataset: Path) -> None:
        """Image tensor must be float32."""
        import torch

        ds = FruitSegmentationDataset(synthetic_dataset, split="train")
        image, _ = ds[0]
        assert image.dtype == torch.float32

    def test_mask_dtype(self, synthetic_dataset: Path) -> None:
        """Mask tensor must be int64."""
        import torch

        ds = FruitSegmentationDataset(synthetic_dataset, split="train")
        _, mask = ds[0]
        assert mask.dtype == torch.int64

    def test_mask_value_range(self, synthetic_dataset: Path) -> None:
        """Mask pixel values must be within [0, NUM_CLASSES - 1]."""
        ds = FruitSegmentationDataset(synthetic_dataset, split="train")
        for i in range(len(ds)):
            _, mask = ds[i]
            assert mask.min() >= 0
            assert mask.max() < NUM_CLASSES

    def test_invalid_split_raises(self, synthetic_dataset: Path) -> None:
        """Passing an unknown split must raise ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            FruitSegmentationDataset(synthetic_dataset, split="predict")

    def test_missing_root_raises(self) -> None:
        """Passing a non-existent root must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FruitSegmentationDataset("/non/existent/path", split="train")

    def test_all_splits_accessible(self, synthetic_dataset: Path) -> None:
        """All three splits must load without error."""
        for split in ("train", "val", "test"):
            ds = FruitSegmentationDataset(synthetic_dataset, split=split)
            assert len(ds) > 0

    def test_class_names_length(self) -> None:
        """CLASS_NAMES must contain exactly NUM_CLASSES entries."""
        assert len(CLASS_NAMES) == 16  # 16 fruit classes
        assert NUM_CLASSES == 17  # 16 fruit + 1 background


# ---------------------------------------------------------------------------
# DataLoader builder tests
# ---------------------------------------------------------------------------


class TestBuildDataLoaders:
    """Tests for the build_dataloaders factory."""

    def test_returns_three_loaders(self, minimal_cfg: dict) -> None:
        """build_dataloaders must return exactly (train, val, test)."""
        loaders = build_dataloaders(minimal_cfg)
        assert len(loaders) == 3

    def test_batch_shape(self, minimal_cfg: dict) -> None:
        """First batch from train_loader must have correct tensor shapes."""
        import torch

        train_loader, _, _ = build_dataloaders(minimal_cfg)
        images, masks = next(iter(train_loader))
        assert images.shape[1:] == torch.Size([3, 64, 64])
        assert masks.shape[1:] == torch.Size([64, 64])


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestTransforms:
    """Tests for albumentations transform factories."""

    def test_train_transform_output_shape(self, minimal_cfg: dict) -> None:
        """Train transform must produce correct spatial dimensions."""
        tf = build_train_transform(minimal_cfg)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.random.randint(0, 17, (128, 128), dtype=np.uint8)
        out = tf(image=img, mask=mask)
        assert out["image"].shape == (3, 64, 64)
        assert out["mask"].shape == (64, 64)

    def test_val_transform_output_shape(self, minimal_cfg: dict) -> None:
        """Val transform must produce correct spatial dimensions."""
        tf = build_val_transform(minimal_cfg)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.random.randint(0, 17, (128, 128), dtype=np.uint8)
        out = tf(image=img, mask=mask)
        assert out["image"].shape == (3, 64, 64)
        assert out["mask"].shape == (64, 64)

    def test_val_no_random_flip(self, minimal_cfg: dict) -> None:
        """Val transform must produce identical results on repeated calls (deterministic)."""
        tf = build_val_transform(minimal_cfg)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        out1 = tf(image=img.copy(), mask=mask.copy())
        out2 = tf(image=img.copy(), mask=mask.copy())
        import torch

        assert torch.allclose(
            out1["image"], out2["image"]
        ), "Val transform is non-deterministic"
