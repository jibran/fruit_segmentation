"""Tests for the config loader module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config_loader import get_model_name_with_size, load_config


class TestLoadConfig:
    """Tests for load_config."""

    def test_loads_simple_yaml(self, tmp_path: Path) -> None:
        """A flat YAML file must load correctly."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model:\n  name: foo\n  size: tiny\n")
        cfg = load_config(cfg_file)
        assert cfg["model"]["name"] == "foo"

    def test_resolves_defaults_inheritance(self, tmp_path: Path) -> None:
        """Child config must inherit and override parent values."""
        base = tmp_path / "base.yaml"
        base.write_text("a: 1\nb: 2\n")

        child = tmp_path / "child.yaml"
        child.write_text("defaults:\n  - base\nb: 99\nc: 3\n")

        cfg = load_config(child)
        assert cfg["a"] == 1  # inherited from base
        assert cfg["b"] == 99  # overridden by child
        assert cfg["c"] == 3  # new key in child

    def test_missing_file_raises(self) -> None:
        """Loading a non-existent config must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/non/existent/config.yaml")

    def test_loads_convnext_config(self) -> None:
        """The real convnext_unet.yaml must load without error."""
        cfg = load_config("config/convnext_unet.yaml")
        assert cfg["model"]["name"] == "convnext_unet"

    def test_loads_swin_config(self) -> None:
        """The real swin_unet.yaml must load without error."""
        cfg = load_config("config/swin_unet.yaml")
        assert cfg["model"]["name"] == "swin_unet"

    def test_deep_merge_does_not_mutate_base(self, tmp_path: Path) -> None:
        """Merging must not modify the base config dict in place."""
        base = tmp_path / "base.yaml"
        base.write_text("x:\n  y: 1\n  z: 2\n")
        child = tmp_path / "child.yaml"
        child.write_text("defaults:\n  - base\nx:\n  y: 99\n")

        load_config(child)
        # Re-load base to confirm it wasn't mutated on disk or in memory
        cfg_base = load_config(base)
        assert cfg_base["x"]["y"] == 1


class TestGetModelNameWithSize:
    """Tests for get_model_name_with_size."""

    def test_returns_name_size_string(self) -> None:
        """Must concatenate model name and size with underscore."""
        cfg = {"model": {"name": "convnext_unet", "size": "tiny"}}
        assert get_model_name_with_size(cfg) == "convnext_unet_tiny"

    def test_defaults_to_tiny_if_missing(self) -> None:
        """Missing size key must default to 'tiny'."""
        cfg = {"model": {"name": "swin_unet"}}
        result = get_model_name_with_size(cfg)
        assert result == "swin_unet_tiny"
