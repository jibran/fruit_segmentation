"""PyTorch Dataset for the Fresh and Rotten Fruits segmentation dataset.

Expected directory layout after preprocessing::

    data/processed/
    ├── images/
    │   ├── train/   *.jpg or *.png
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/   grayscale PNG, pixel value = class index (0-15, 16=bg)
        ├── val/
        └── test/

Class index mapping — derived from ``folder2label_str.txt`` (alphabetical):

    0  fresh_apple         8  rotten_apple
    1  fresh_banana        9  rotten_banana
    2  fresh_grape        10  rotten_grape
    3  fresh_guava        11  rotten_guava
    4  fresh_jujube       12  rotten_jujube
    5  fresh_orange       13  rotten_orange
    6  fresh_pomegranate  14  rotten_pomegranate
    7  fresh_strawberry   15  rotten_strawberry
   16  background (conveyor belt / unlabelled)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# Indices 0-15 match folder2label_str.txt exactly (alphabetical order).
# Index 16 is reserved for background / conveyor belt pixels that carry
# no fruit annotation.  Pass ignore_index=16 to CrossEntropyLoss so the
# background does not contribute to the segmentation loss.
CLASS_NAMES: list[str] = [
    "fresh_apple",  # 0
    "fresh_banana",  # 1
    "fresh_grape",  # 2
    "fresh_guava",  # 3
    "fresh_jujube",  # 4
    "fresh_orange",  # 5
    "fresh_pomegranate",  # 6
    "fresh_strawberry",  # 7
    "rotten_apple",  # 8
    "rotten_banana",  # 9
    "rotten_grape",  # 10
    "rotten_guava",  # 11
    "rotten_jujube",  # 12
    "rotten_orange",  # 13
    "rotten_pomegranate",  # 14
    "rotten_strawberry",  # 15
]

NUM_CLASSES: int = len(CLASS_NAMES) + 1  # 17 (16 fruit + 1 background at index 16)
BACKGROUND_IDX: int = 16  # pixels with no fruit annotation

# Colour palette for mask visualisation — one RGB tuple per class index.
# Indices 0-7  (fresh):  bright, saturated tones.
# Indices 8-15 (rotten): dark, desaturated versions of the same hue family.
# Index 16     (background): near-black.
CLASS_PALETTE: list[tuple[int, int, int]] = [
    (0, 200, 80),  #  0 fresh_apple
    (255, 220, 0),  #  1 fresh_banana
    (120, 60, 200),  #  2 fresh_grape
    (180, 230, 80),  #  3 fresh_guava
    (255, 140, 0),  #  4 fresh_jujube
    (255, 100, 20),  #  5 fresh_orange
    (220, 30, 60),  #  6 fresh_pomegranate
    (255, 60, 120),  #  7 fresh_strawberry
    (60, 90, 30),  #  8 rotten_apple
    (120, 90, 10),  #  9 rotten_banana
    (50, 20, 90),  # 10 rotten_grape
    (80, 100, 20),  # 11 rotten_guava
    (120, 60, 0),  # 12 rotten_jujube
    (140, 50, 0),  # 13 rotten_orange
    (90, 10, 20),  # 14 rotten_pomegranate
    (100, 20, 40),  # 15 rotten_strawberry
    (30, 30, 30),  # 16 background
]


class FruitSegmentationDataset(Dataset):
    """Pairs RGB fruit images with their integer-encoded segmentation masks.

    Pixel values in the mask PNG encode the class index directly (0–15
    for fruit classes, 16 for background/belt).  No anti-aliasing
    artefacts are introduced because masks are stored as grayscale PNGs
    and loaded without resampling.

    Args:
        root_dir: Path to the processed dataset root (contains
            ``images/`` and ``masks/`` sub-directories).
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        transform: Albumentations transform applied jointly to the image
            and mask.  Must be an *albumentations* ``Compose`` object that
            accepts ``image`` and ``mask`` keyword arguments.
        image_size: Target spatial size ``(H, W)`` after transform.
            Used for validation only — the actual resize is expected to
            be part of *transform*.

    Raises:
        FileNotFoundError: If the images or masks directory does not exist.
        ValueError: If *split* is not one of the allowed values.

    Example:
        >>> ds = FruitSegmentationDataset("data/processed", split="train")
        >>> image, mask = ds[0]
        >>> image.shape   # (3, 512, 512)
        >>> mask.shape    # (512, 512)
    """

    _ALLOWED_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        image_size: int = 512,
    ) -> None:
        if split not in self._ALLOWED_SPLITS:
            raise ValueError(
                f"split must be one of {self._ALLOWED_SPLITS}, got '{split}'"
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size

        self.image_dir = self.root_dir / "images" / split
        self.mask_dir = self.root_dir / "masks" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.image_paths = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        """Return the number of samples in this split.

        Returns:
            Integer count of image-mask pairs.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return one image-mask pair.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            A tuple ``(image, mask)`` where:

            - ``image`` is a float32 tensor of shape ``(3, H, W)`` normalised
              to ImageNet mean/std (if transform includes normalisation).
            - ``mask`` is an int64 tensor of shape ``(H, W)`` with values
              in ``[0, 16]`` (0-15 = fruit, 16 = background).
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # grayscale → (H,W)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        return image, mask_tensor

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights from the training masks.

        Weights are normalised so the mean weight equals 1.0.  The
        background class (index 0) is included in the computation.

        Returns:
            Float tensor of shape ``(NUM_CLASSES,)`` with per-class weights.

        Note:
            This method scans all masks on disk and may take a few minutes
            for large datasets.  Cache the result and pass to
            ``torch.nn.CrossEntropyLoss(weight=...)``.
        """
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for img_path in self.image_paths:
            mask_path = self.mask_dir / (img_path.stem + ".png")
            mask = np.array(Image.open(mask_path).convert("L"))
            for c in range(NUM_CLASSES):
                counts[c] += (mask == c).sum()

        # Inverse frequency — add small epsilon to avoid division by zero
        freq = counts / counts.sum()
        weights = 1.0 / (freq + 1e-6)
        weights = weights / weights.mean()
        return torch.from_numpy(weights.astype(np.float32))

    def get_sample_weights(self) -> torch.Tensor:
        """Compute per-sample weights for ``WeightedRandomSampler``.

        Each sample is assigned the weight of its dominant class (the class
        covering the most pixels in that image's mask).  This ensures images
        dominated by underrepresented fruit classes are sampled more often,
        producing more balanced class exposure per batch.

        Returns:
            Float tensor of shape ``(len(self),)`` — one weight per image.

        Note:
            Scans all masks on disk.  Call once and pass the result to
            ``WeightedRandomSampler``; do not call inside the training loop.

        Example:
            >>> sample_weights = train_ds.get_sample_weights()
            >>> sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        """
        # Count pixel frequency per class across the whole split
        class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        dominant_classes: list[int] = []

        for img_path in self.image_paths:
            mask_path = self.mask_dir / (img_path.stem + ".png")
            mask = np.array(Image.open(mask_path).convert("L"))
            # Exclude background (index 16) from dominant-class calculation
            fruit_mask = mask[mask != BACKGROUND_IDX]
            if fruit_mask.size == 0:
                dominant_classes.append(BACKGROUND_IDX)
            else:
                dominant = int(
                    np.bincount(
                        fruit_mask.flatten(), minlength=NUM_CLASSES - 1
                    ).argmax()
                )
                dominant_classes.append(dominant)
            for c in range(NUM_CLASSES):
                class_pixel_counts[c] += (mask == c).sum()

        # Inverse-frequency weight per class
        freq = class_pixel_counts / (class_pixel_counts.sum() + 1e-10)
        class_weights = 1.0 / (freq + 1e-6)

        # Assign each sample the weight of its dominant class
        sample_weights = np.array(
            [class_weights[c] for c in dominant_classes], dtype=np.float32
        )
        return torch.from_numpy(sample_weights)


def build_dataloaders(
    cfg: dict,
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    weighted_sampling: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, val, and test DataLoaders from a config dict.

    Args:
        cfg: Loaded configuration dictionary.  Uses keys
            ``data.root_dir``, ``data.batch_size``, ``data.num_workers``,
            ``data.pin_memory``, and ``data.image_size``.
        train_transform: Albumentations transform for the training split.
        val_transform: Albumentations transform for val and test splits.
        weighted_sampling: If ``True``, replace the random shuffle with a
            ``WeightedRandomSampler`` so underrepresented fruit classes
            appear more frequently in each training batch.  Has no effect
            on val or test loaders.  Defaults to ``False``.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``.

    Example:
        >>> train_loader, val_loader, test_loader = build_dataloaders(
        ...     cfg, weighted_sampling=True
        ... )
        >>> images, masks = next(iter(train_loader))
        >>> images.shape   # (B, 3, 512, 512)
    """
    data_cfg = cfg["data"]
    root = data_cfg["root_dir"]
    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    pin_memory = data_cfg.get("pin_memory", True)

    train_ds = FruitSegmentationDataset(root, "train", train_transform, image_size)
    val_ds = FruitSegmentationDataset(root, "val", val_transform, image_size)
    test_ds = FruitSegmentationDataset(root, "test", val_transform, image_size)

    if weighted_sampling:
        print("  [sampler] Computing per-sample weights for WeightedRandomSampler...")
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,  # mutually exclusive with shuffle=True
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
