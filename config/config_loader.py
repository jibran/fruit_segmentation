"""Configuration loader for fruit segmentation experiments.

Loads and merges YAML config files, resolving ``defaults`` inheritance
so model-specific configs can override only the fields they need.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Args:
        base: The base dictionary (e.g. loaded from base_config.yaml).
        override: The overriding dictionary whose values take precedence.

    Returns:
        A new merged dictionary. Neither input is mutated.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "defaults":
            continue  # handled separately
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving any ``defaults`` inheritance chain.

    If the config contains a ``defaults`` key listing parent config names,
    those parents are loaded first from the same directory, then the child
    config's values are merged on top.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A fully resolved configuration dictionary.

    Raises:
        FileNotFoundError: If ``config_path`` or any parent config is missing.

    Example:
        >>> cfg = load_config("config/convnext_unet.yaml")
        >>> cfg["model"]["name"]
        'convnext_unet'
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Resolve defaults inheritance
    merged: dict[str, Any] = {}
    for parent_name in raw.get("defaults", []):
        parent_path = config_path.parent / f"{parent_name}.yaml"
        parent_cfg = load_config(parent_path)
        merged = _deep_merge(merged, parent_cfg)

    return _deep_merge(merged, raw)


def get_model_name_with_size(cfg: dict[str, Any]) -> str:
    """Return a canonical model identifier string for logging and file naming.

    Args:
        cfg: Loaded configuration dictionary.

    Returns:
        A string like ``"convnext_unet_tiny"`` or ``"swin_unet_small"``.

    Example:
        >>> name = get_model_name_with_size(cfg)
        >>> name
        'convnext_unet_tiny'
    """
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "model")
    size = model_cfg.get("size", "tiny")
    return f"{name}_{size}"
