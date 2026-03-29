"""Tests for utils/prepare_dataset.py.

Verifies the AnyLabeling JSON → grayscale mask conversion pipeline using
synthetic JSON fixtures that mirror the real annotation format.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.prepare_dataset import (
    FOLDER_TO_CLASS,
    LABEL_TO_CLASS,
    _is_sentinel_shape,
    _resolve_label,
    collect_pairs,
    json_to_mask,
    split_pairs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_json(tmp_path: Path) -> Path:
    """Write a minimal AnyLabeling JSON with one valid polygon + sentinel.

    The polygon covers the full 100×100 canvas for easy counting.

    Returns:
        Path to the written JSON file.
    """
    data = {
        "version": "2.4.2",
        "flags": {},
        "shapes": [
            {
                "label": "Rotten_Strawberry",
                "score": None,
                "points": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            },
            {
                "kie_linking": [],
                "label": "",
                "score": None,
                "points": [
                    [math.inf, math.inf],
                    [-math.inf, math.inf],
                    [-math.inf, -math.inf],
                    [math.inf, -math.inf],
                ],
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "total": True,
            },
        ],
        "imagePath": "test.jpg",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 100,
    }
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    """Create a synthetic raw dataset with two class folders (2 images each).

    Returns:
        Path to the synthetic raw root directory.
    """
    root = tmp_path / "raw"
    for folder, cls in [("FreshApple", 1), ("RottenApple", 2)]:
        d = root / folder
        d.mkdir(parents=True)
        for i in range(1, 3):
            # Create dummy JPEG
            img = Image.fromarray(
                np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            )
            img.save(d / f"{folder} ({i}).jpg")

            # Create matching JSON
            annotation = {
                "version": "2.4.2",
                "flags": {},
                "shapes": [
                    {
                        "label": folder.replace("Fresh", "Fresh_").replace(
                            "Rotten", "Rotten_"
                        ),
                        "points": [[0, 0], [50, 0], [50, 50], [0, 50]],
                        "shape_type": "polygon",
                        "flags": {},
                        "attributes": {},
                        "kie_linking": [],
                    }
                ],
                "imageHeight": 50,
                "imageWidth": 50,
            }
            (d / f"{folder} ({i}).json").write_text(json.dumps(annotation))

    return root


# ---------------------------------------------------------------------------
# _is_sentinel_shape
# ---------------------------------------------------------------------------


class TestIsSentinelShape:
    """Tests for the sentinel shape detector."""

    def test_detects_total_flag(self) -> None:
        """Shape with total=True must be detected as sentinel."""
        shape = {"total": True, "points": [[0, 0]]}
        assert _is_sentinel_shape(shape) is True

    def test_detects_infinity_coords(self) -> None:
        """Shape with Infinity coordinates must be detected as sentinel."""
        shape = {
            "points": [[math.inf, math.inf], [-math.inf, math.inf]],
            "shape_type": "rectangle",
        }
        assert _is_sentinel_shape(shape) is True

    def test_normal_polygon_not_sentinel(self) -> None:
        """A normal polygon with finite coordinates must not be sentinel."""
        shape = {"points": [[10.0, 20.0], [30.0, 40.0], [50.0, 10.0]]}
        assert _is_sentinel_shape(shape) is False

    def test_empty_points_not_sentinel(self) -> None:
        """A shape with no points and no total flag is not sentinel."""
        assert _is_sentinel_shape({"points": []}) is False


# ---------------------------------------------------------------------------
# _resolve_label
# ---------------------------------------------------------------------------


class TestResolveLabel:
    """Tests for the label-to-class-index resolver."""

    def test_known_json_label(self) -> None:
        """A known JSON label string must return the correct class index."""
        assert _resolve_label("Rotten_Strawberry", "RottenStrawberry") == 15

    def test_fallback_to_folder(self) -> None:
        """Unknown JSON label must fall back to the folder name."""
        assert _resolve_label("Unknown_Label", "FreshApple") == 0

    def test_raises_on_unknown_both(self) -> None:
        """Unknown label AND unknown folder must raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognised"):
            _resolve_label("Bad_Label", "BadFolder")

    def test_all_label_to_class_entries(self) -> None:
        """Every entry in LABEL_TO_CLASS must resolve to a value in [0, 15]."""
        for label, cls in LABEL_TO_CLASS.items():
            result = _resolve_label(label, "FreshApple")
            assert result == cls
            assert 0 <= result <= 15

    def test_all_folder_to_class_entries(self) -> None:
        """Every folder name must resolve via fallback to a value in [0, 15]."""
        for folder, cls in FOLDER_TO_CLASS.items():
            result = _resolve_label("UnknownLabel", folder)
            assert result == cls


# ---------------------------------------------------------------------------
# json_to_mask
# ---------------------------------------------------------------------------


class TestJsonToMask:
    """Tests for the AnyLabeling JSON → mask converter."""

    def test_basic_mask_shape(self, simple_json: Path) -> None:
        """Output mask must match JSON imageHeight × imageWidth."""
        result = json_to_mask(simple_json, "RottenStrawberry")
        assert result is not None
        mask, h, w = result
        assert mask.shape == (100, 100)
        assert h == 100 and w == 100

    def test_class_index_in_mask(self, simple_json: Path) -> None:
        """Polygon pixels must be filled with the correct class index."""
        result = json_to_mask(simple_json, "RottenStrawberry")
        assert result is not None
        mask, _, _ = result
        # The full-canvas polygon should fill the entire mask with class 15
        assert np.all(mask == 15), f"Expected all 15, got unique={np.unique(mask)}"

    def test_sentinel_shape_skipped(self, simple_json: Path) -> None:
        """Sentinel shape (Infinity coords) must not appear in mask."""
        result = json_to_mask(simple_json, "RottenStrawberry")
        assert result is not None
        mask, _, _ = result
        assert np.all(mask == 15), "All pixels should be class 15 (Rotten_Strawberry)"

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Passing a non-existent JSON path must return None."""
        result = json_to_mask(tmp_path / "nonexistent.json", "FreshApple")
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        """A malformed JSON file must return None."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json}")
        result = json_to_mask(bad, "FreshApple")
        assert result is None

    def test_mask_dtype(self, simple_json: Path) -> None:
        """Output mask must be uint8."""
        result = json_to_mask(simple_json, "RottenStrawberry")
        assert result is not None
        mask, _, _ = result
        assert mask.dtype == np.uint8

    def test_background_is_background_idx(self, tmp_path: Path) -> None:
        """Pixels outside all polygons must equal BACKGROUND_IDX (16)."""
        data = {
            "version": "2.4.2",
            "flags": {},
            "shapes": [
                {
                    "label": "Fresh_Apple",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "shape_type": "polygon",
                    "flags": {},
                    "attributes": {},
                    "kie_linking": [],
                }
            ],
            "imageHeight": 100,
            "imageWidth": 100,
        }
        p = tmp_path / "partial.json"
        p.write_text(json.dumps(data))
        result = json_to_mask(p, "FreshApple")
        assert result is not None
        mask, _, _ = result
        assert mask[50, 50] == 16, "Center pixel should be BACKGROUND_IDX=16"
        assert mask[5, 5] == 0, "Inside polygon pixel should be class 0 (fresh_apple)"


# ---------------------------------------------------------------------------
# collect_pairs
# ---------------------------------------------------------------------------


class TestCollectPairs:
    """Tests for the file-pair collector."""

    def test_finds_all_pairs(self, raw_dataset: Path) -> None:
        """collect_pairs must find all JSON-paired images."""
        pairs = collect_pairs(raw_dataset)
        assert len(pairs) == 4  # 2 classes × 2 images each

    def test_pair_structure(self, raw_dataset: Path) -> None:
        """Each pair must be (img_path, json_path, folder_name)."""
        pairs = collect_pairs(raw_dataset)
        for img_path, json_path, folder_name in pairs:
            assert img_path.suffix == ".jpg"
            assert json_path.suffix == ".json"
            assert folder_name in FOLDER_TO_CLASS

    def test_unknown_folder_skipped(self, tmp_path: Path) -> None:
        """Folders not in FOLDER_TO_CLASS must be silently skipped."""
        unknown = tmp_path / "UnknownFruit"
        unknown.mkdir()
        (unknown / "img.jpg").write_bytes(b"fake")
        (unknown / "img.json").write_text("{}")
        pairs = collect_pairs(tmp_path)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# split_pairs
# ---------------------------------------------------------------------------


class TestSplitPairs:
    """Tests for the stratified split function."""

    def _make_pairs(self, n_per_class: int = 10) -> list:
        """Generate synthetic pairs for two classes."""
        pairs = []
        for folder in ("FreshApple", "RottenApple"):
            for i in range(n_per_class):
                pairs.append(
                    (Path(f"{folder}_{i}.jpg"), Path(f"{folder}_{i}.json"), folder)
                )
        return pairs

    def test_all_items_assigned(self) -> None:
        """Every item must appear in exactly one split."""
        pairs = self._make_pairs(10)
        splits = split_pairs(pairs, train=0.8, val=0.1)
        total = sum(len(v) for v in splits.values())
        assert total == len(pairs)

    def test_no_overlap_between_splits(self) -> None:
        """The same item must not appear in multiple splits."""
        pairs = self._make_pairs(10)
        splits = split_pairs(pairs)
        all_items = splits["train"] + splits["val"] + splits["test"]
        assert len(all_items) == len(set(map(str, all_items)))

    def test_approximate_split_ratios(self) -> None:
        """Split fractions must be approximately correct (±10%)."""
        pairs = self._make_pairs(100)
        splits = split_pairs(pairs, train=0.8, val=0.1)
        n = len(pairs)
        assert abs(len(splits["train"]) / n - 0.8) < 0.10
        assert abs(len(splits["val"]) / n - 0.1) < 0.10

    def test_reproducible_with_seed(self) -> None:
        """Same seed must produce identical splits."""
        pairs = self._make_pairs(20)
        s1 = split_pairs(pairs, seed=42)
        s2 = split_pairs(pairs, seed=42)
        assert [str(p) for p in s1["train"]] == [str(p) for p in s2["train"]]

    def test_different_seeds_differ(self) -> None:
        """Different seeds must (almost certainly) produce different splits."""
        pairs = self._make_pairs(50)
        s1 = split_pairs(pairs, seed=1)
        s2 = split_pairs(pairs, seed=99)
        assert s1["train"] != s2["train"]
