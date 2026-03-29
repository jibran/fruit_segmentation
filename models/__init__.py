"""Neural network architecture registry for fruit segmentation.

Provides a single :func:`build_model` factory that instantiates the
correct model class from a configuration dict, keeping training scripts
free of direct imports.
"""

from __future__ import annotations

from typing import Any

from models.convnext_unet import ConvNeXtUNet
from models.swin_unet import SwinUNet

__all__ = ["ConvNeXtUNet", "SwinUNet", "build_model"]

_MODEL_REGISTRY: dict[str, type] = {
    "convnext_unet": ConvNeXtUNet,
    "swin_unet": SwinUNet,
}


def build_model(cfg: dict[str, Any]) -> ConvNeXtUNet | SwinUNet:
    """Instantiate a segmentation model from a configuration dictionary.

    Args:
        cfg: Loaded configuration dict (from :func:`config.config_loader.load_config`).
            Must contain a ``model`` key with at minimum ``name`` and ``size``
            sub-keys.

    Returns:
        An instantiated model (``ConvNeXtUNet`` or ``SwinUNet``).

    Raises:
        KeyError: If ``cfg["model"]["name"]`` is not a registered model name.

    Example:
        >>> cfg = load_config("config/convnext_unet.yaml")
        >>> model = build_model(cfg)
        >>> type(model).__name__
        'ConvNeXtUNet'
    """
    model_cfg = cfg["model"]
    name = model_cfg["name"]

    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered models: {list(_MODEL_REGISTRY)}"
        )

    cls = _MODEL_REGISTRY[name]
    num_classes = cfg["data"]["num_classes"]
    size = model_cfg.get("size", "tiny")
    pretrained = model_cfg.get("pretrained", True)

    kwargs: dict[str, Any] = dict(
        num_classes=num_classes,
        size=size,
        pretrained=pretrained,
    )

    # SwinUNet also needs input_size for spatial reshape calculations
    if name == "swin_unet":
        kwargs["input_size"] = cfg["data"]["image_size"]

    return cls(**kwargs)
