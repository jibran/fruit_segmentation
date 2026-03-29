"""Data preparation pipeline: raw AnyLabeling JSON → grayscale PNG masks.

Reads the original dataset structure::

    data/raw/Original Image/
    ├── FreshApple/
    │   ├── FreshApple (1).jpg
    │   ├── FreshApple (1).json   ← AnyLabeling polygon annotation
    │   └── ...
    ├── RottenApple/
    └── ...  (16 folders total)

Produces the processed dataset layout expected by FruitSegmentationDataset::

    data/processed/
    ├── images/
    │   ├── train/   <stem>.jpg
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/   <stem>.png   (grayscale, pixel = class index 0-16)
        ├── val/
        └── test/

Class index mapping (0 = background, consistent with dataset/fruit_dataset.py):

    0  background            9  fresh_jujube
    1  fresh_apple          10  rotten_jujube
    2  rotten_apple         11  fresh_pomegranate
    3  fresh_banana         12  rotten_pomegranate
    4  rotten_banana        13  fresh_strawberry
    5  fresh_orange         14  rotten_strawberry
    6  rotten_orange        15  fresh_grape
    7  fresh_guava          16  rotten_grape
    8  rotten_guava

AnyLabeling JSON format notes:
- ``shapes[].shape_type == "polygon"`` with ``points: [[x, y], ...]``
- Labels use ``_`` separator: ``Fresh_Apple``, ``Rotten_Strawberry`` etc.
- A sentinel shape with ``Infinity`` coordinates marks the background — skip it.
- JSON ``imageWidth`` / ``imageHeight`` may differ from actual JPEG size
  (images may be stored at half resolution). Points are scaled accordingly.

Usage::

    # Process entire dataset with 80/10/10 split
    python utils/prepare_dataset.py

    # Custom split ratios
    python utils/prepare_dataset.py --train 0.75 --val 0.15 --test 0.1

    # Custom paths
    python utils/prepare_dataset.py \\
        --raw_dir data/raw/Original\\ Image \\
        --out_dir data/processed \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Class index registry — derived from folder2label_str.txt
# Indices 0-15 are alphabetical: Fresh_* (0-7) then Rotten_* (8-15).
# Index 16 is reserved for background pixels (no fruit annotation).
# Must stay in sync with dataset/fruit_dataset.py CLASS_NAMES.
# ---------------------------------------------------------------------------

FOLDER_TO_CLASS: dict[str, int] = {
    "FreshApple": 0,
    "FreshBanana": 1,
    "FreshGrape": 2,
    "FreshGuava": 3,
    "FreshJujube": 4,
    "FreshOrange": 5,
    "FreshPomegranate": 6,
    "FreshStrawberry": 7,
    "RottenApple": 8,
    "RottenBanana": 9,
    "RottenGrape": 10,
    "RottenGuava": 11,
    "RottenJujube": 12,
    "RottenOrange": 13,
    "RottenPomegranate": 14,
    "RottenStrawberry": 15,
}

LABEL_TO_CLASS: dict[str, int] = {
    "Fresh_Apple": 0,
    "Fresh_Banana": 1,
    "Fresh_Grape": 2,
    "Fresh_Guava": 3,
    "Fresh_Jujube": 4,
    "Fresh_Orange": 5,
    "Fresh_Pomegranate": 6,
    "Fresh_Strawberry": 7,
    "Rotten_Apple": 8,
    "Rotten_Banana": 9,
    "Rotten_Grape": 10,
    "Rotten_Guava": 11,
    "Rotten_Jujube": 12,
    "Rotten_Orange": 13,
    "Rotten_Pomegranate": 14,
    "Rotten_Strawberry": 15,
}

BACKGROUND_IDX: int = 16  # written into mask pixels with no fruit polygon


# ---------------------------------------------------------------------------
# Core annotation parsing
# ---------------------------------------------------------------------------


def _is_sentinel_shape(shape: dict) -> bool:
    """Return True for the AnyLabeling background sentinel rectangle.

    The sentinel has ``Infinity`` coordinates and ``"total": true``.

    Args:
        shape: A single shape dict from the JSON ``shapes`` list.

    Returns:
        ``True`` if the shape should be skipped.
    """
    if shape.get("total"):
        return True
    for pt in shape.get("points", []):
        if any(math.isinf(v) for v in pt):
            return True
    return False


def _resolve_label(raw_label: str, folder_name: str) -> int:
    """Resolve a JSON label string to a class integer index.

    Tries the exact JSON label first, then falls back to the folder name.

    Args:
        raw_label: The ``label`` field from the JSON shape, e.g.
            ``"Rotten_Strawberry"``.
        folder_name: The parent folder name, e.g. ``"RottenStrawberry"``.

    Returns:
        Integer class index in ``[1, 16]``.

    Raises:
        ValueError: If neither the label nor folder name is recognised.
    """
    if raw_label in LABEL_TO_CLASS:
        return LABEL_TO_CLASS[raw_label]
    if folder_name in FOLDER_TO_CLASS:
        return FOLDER_TO_CLASS[folder_name]
    raise ValueError(
        f"Unrecognised label '{raw_label}' and folder '{folder_name}'. "
        f"Known labels: {list(LABEL_TO_CLASS)}"
    )


def json_to_mask(
    json_path: Path, folder_name: str
) -> tuple[np.ndarray, int, int] | None:
    """Parse one AnyLabeling JSON and rasterise polygons into a mask array.

    Args:
        json_path: Path to the ``.json`` annotation file.
        folder_name: Name of the parent class folder (used as label fallback).

    Returns:
        A tuple ``(mask, height, width)`` where ``mask`` is a uint8 ndarray
        of shape ``(H, W)`` with pixel values = class index (0 = background),
        and ``height`` / ``width`` are the actual image dimensions read from
        the JSON.  Returns ``None`` if the JSON cannot be parsed or contains
        no valid polygons.

    Note:
        JSON ``imageWidth`` / ``imageHeight`` values are used as the canvas
        size.  If the actual JPEG is stored at a different resolution, the
        caller must resize the mask accordingly.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] Cannot parse {json_path.name}: {e}")
        return None

    h = int(data.get("imageHeight", 0))
    w = int(data.get("imageWidth", 0))
    if h == 0 or w == 0:
        print(f"  [warn] Missing image dimensions in {json_path.name}")
        return None

    # Initialise to BACKGROUND_IDX (16) not 0: class 0 is fresh_apple,
    # so np.zeros would make unfilled pixels indistinguishable from it.
    mask = np.full((h, w), BACKGROUND_IDX, dtype=np.uint8)
    found_any = False

    for shape in data.get("shapes", []):
        if _is_sentinel_shape(shape):
            continue
        if shape.get("shape_type") != "polygon":
            continue

        raw_label = shape.get("label", "")
        try:
            class_idx = _resolve_label(raw_label, folder_name)
        except ValueError as e:
            print(f"  [warn] {e} — skipping shape in {json_path.name}")
            continue

        pts = shape.get("points", [])
        if len(pts) < 3:
            continue

        polygon = np.array(
            [[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32
        )
        cv2.fillPoly(mask, [polygon], color=int(class_idx))
        found_any = True

    if not found_any:
        print(f"  [warn] No valid polygons in {json_path.name}")
        return None

    return mask, h, w


# ---------------------------------------------------------------------------
# Dataset split and copy
# ---------------------------------------------------------------------------


def collect_pairs(raw_dir: Path) -> list[tuple[Path, Path, str]]:
    """Walk the raw directory and collect all (image, json, folder) triples.

    Args:
        raw_dir: Root of the raw dataset, containing one sub-folder per class.

    Returns:
        List of ``(image_path, json_path, folder_name)`` tuples where both
        the image and the JSON annotation exist.
    """
    pairs: list[tuple[Path, Path, str]] = []
    for folder in sorted(raw_dir.iterdir()):
        if not folder.is_dir():
            continue
        folder_name = folder.name
        if folder_name not in FOLDER_TO_CLASS:
            print(f"  [skip] Unknown folder: {folder_name}")
            continue

        for img_path in sorted(folder.glob("*.jpg")):
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                pairs.append((img_path, json_path, folder_name))
            else:
                print(f"  [warn] No JSON for {img_path.name}")

    return pairs


def split_pairs(
    pairs: list,
    train: float = 0.8,
    val: float = 0.1,
    seed: int = 42,
) -> dict[str, list]:
    """Stratified split: preserves class balance across train/val/test.

    Args:
        pairs: List of ``(img_path, json_path, folder_name)`` tuples.
        train: Fraction for training set.
        val: Fraction for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to
        a list of ``(img_path, json_path, folder_name)`` tuples.
    """
    rng = random.Random(seed)

    # Group by class
    by_class: dict[str, list] = {}
    for triple in pairs:
        folder = triple[2]
        by_class.setdefault(folder, []).append(triple)

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for folder, items in by_class.items():
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * val)
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    return splits


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------


def process_split(
    items: list[tuple[Path, Path, str]],
    split: str,
    out_dir: Path,
    target_size: int | None = None,
) -> dict[str, int]:
    """Convert one split's annotations and write images + masks to disk.

    Args:
        items: List of ``(img_path, json_path, folder_name)`` tuples.
        split: Split name — ``"train"``, ``"val"``, or ``"test"``.
        out_dir: Root of the processed dataset output directory.
        target_size: If set, resize images and masks to ``target_size × target_size``
            (preserving square crop).  ``None`` keeps original resolution.

    Returns:
        Dict with keys ``"processed"``, ``"skipped"`` counts.
    """
    img_out = out_dir / "images" / split
    msk_out = out_dir / "masks" / split
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    counts = {"processed": 0, "skipped": 0}
    desc = f"  {split:5s}"

    for img_path, json_path, folder_name in tqdm(items, desc=desc, leave=False):
        # ── Parse annotation ─────────────────────────────────────────
        result = json_to_mask(json_path, folder_name)
        if result is None:
            counts["skipped"] += 1
            continue

        mask_arr, json_h, json_w = result

        # ── Load image ───────────────────────────────────────────────
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError as e:
            print(f"  [warn] Cannot open {img_path.name}: {e}")
            counts["skipped"] += 1
            continue

        actual_w, actual_h = img.size  # PIL is (W, H)

        # ── Scale mask if JSON dims differ from actual image dims ─────
        # JSON stores coordinates at the resolution used during annotation.
        # The stored JPEGs may be at a different (often halved) resolution.
        if (json_h != actual_h) or (json_w != actual_w):
            mask_img = Image.fromarray(mask_arr, mode="L")
            mask_img = mask_img.resize((actual_w, actual_h), Image.NEAREST)
            mask_arr = np.array(mask_img)

        # ── Optional target resize ───────────────────────────────────
        if target_size is not None:
            img = img.resize((target_size, target_size), Image.BILINEAR)
            mask_img = Image.fromarray(mask_arr, mode="L")
            mask_img = mask_img.resize((target_size, target_size), Image.NEAREST)
            mask_arr = np.array(mask_img)

        # ── Sanitise mask values ─────────────────────────────────────
        # Canvas was pre-filled with BACKGROUND_IDX; clip for safety.
        mask_arr = np.clip(mask_arr, 0, BACKGROUND_IDX).astype(np.uint8)

        # ── Write outputs ────────────────────────────────────────────
        stem = img_path.stem
        img.save(img_out / f"{stem}.jpg", quality=95)
        Image.fromarray(mask_arr, mode="L").save(msk_out / f"{stem}.png")
        counts["processed"] += 1

    return counts


# ---------------------------------------------------------------------------
# Stats and verification
# ---------------------------------------------------------------------------


def print_stats(out_dir: Path) -> None:
    """Print per-split file counts and mask value statistics.

    Args:
        out_dir: Root of the processed dataset.
    """
    print("\n── Dataset statistics ──────────────────────────────────────")
    total_imgs = 0
    for split in ("train", "val", "test"):
        imgs = list((out_dir / "images" / split).glob("*.jpg"))
        masks = list((out_dir / "masks" / split).glob("*.png"))
        print(f"  {split:5s}: {len(imgs):4d} images  {len(masks):4d} masks")
        total_imgs += len(imgs)

    print(f"  total: {total_imgs:4d} images")

    # Spot-check one mask to confirm value range
    test_masks = list((out_dir / "masks" / "train").glob("*.png"))
    if test_masks:
        sample = np.array(Image.open(test_masks[0]).convert("L"))
        unique = np.unique(sample)
        print(f"\n  Sample mask unique values: {unique}")
        print("  (0-15 = fruit classes, 16 = background)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Convert AnyLabeling JSON annotations to grayscale PNG masks."
    )
    parser.add_argument(
        "--raw_dir",
        default="data/raw/Original Image",
        help="Path to the raw dataset root containing class sub-folders.",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed",
        help="Output root for processed images and masks.",
    )
    parser.add_argument(
        "--train", type=float, default=0.8, help="Train split fraction."
    )
    parser.add_argument("--val", type=float, default=0.1, help="Val split fraction.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Resize images/masks to this square size (e.g. 512). "
        "None = keep original resolution.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear out_dir before processing.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, collect pairs, split, and process."""
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        print(f"ERROR: raw_dir not found: {raw_dir}")
        sys.exit(1)

    if args.overwrite and out_dir.exists():
        print(f"Removing existing output: {out_dir}")
        shutil.rmtree(out_dir)

    print(f"Raw dir   : {raw_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Split     : train={args.train} val={args.val} test={args.test}")
    if args.target_size:
        print(f"Resize to : {args.target_size}×{args.target_size}")

    # ── Collect all annotated pairs ──────────────────────────────────────
    print("\nCollecting image-annotation pairs...")
    pairs = collect_pairs(raw_dir)
    print(f"Found {len(pairs)} annotated pairs across {len(FOLDER_TO_CLASS)} classes")

    # ── Stratified split ─────────────────────────────────────────────────
    splits = split_pairs(pairs, args.train, args.val, args.seed)
    print(
        f"Split sizes → train:{len(splits['train'])} "
        f"val:{len(splits['val'])} test:{len(splits['test'])}"
    )

    # ── Process each split ───────────────────────────────────────────────
    total_processed = 0
    total_skipped = 0
    for split_name, items in splits.items():
        print(f"\nProcessing {split_name}...")
        counts = process_split(items, split_name, out_dir, args.target_size)
        total_processed += counts["processed"]
        total_skipped += counts["skipped"]
        print(f"  ✓ {counts['processed']} processed  " f"✗ {counts['skipped']} skipped")

    print(f"\nDone. Total: {total_processed} processed, {total_skipped} skipped.")
    print_stats(out_dir)


if __name__ == "__main__":
    main()
