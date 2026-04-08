"""Neural network architecture registry for fruit segmentation.

Provides a single :func:`build_model` factory that instantiates the
correct model class from a configuration dict, keeping training scripts
free of direct imports.

Registered models
-----------------
``convnext_unet``
    Hybrid U-Net with ConvNeXt-V2 encoder + custom decoder.  Full
    pixel-level segmentation.

``swin_unet``
    Hybrid U-Net with Swin-V2 encoder + custom decoder.  Full
    pixel-level segmentation.

``convnext_baseline``
    ConvNeXt-V2 backbone + global-average-pool + linear head.
    Image-level classification baseline for ConvNeXtUNet comparison.

``swin_baseline``
    Swin-V2 backbone + global-average-pool + linear head.
    Image-level classification baseline for SwinUNet comparison.
"""

from __future__ import annotations

from typing import Any

from models.convnext_baseline import ConvNeXtBaseline
from models.convnext_unet import ConvNeXtUNet
from models.swin_baseline import SwinBaseline
from models.swin_unet import SwinUNet

__all__ = [
    "ConvNeXtUNet",
    "SwinUNet",
    "ConvNeXtBaseline",
    "SwinBaseline",
    "build_model",
]

_MODEL_REGISTRY: dict[str, type] = {
    "convnext_unet": ConvNeXtUNet,
    "swin_unet": SwinUNet,
    "convnext_baseline": ConvNeXtBaseline,
    "swin_baseline": SwinBaseline,
}


def build_model(
    cfg: dict[str, Any],
) -> ConvNeXtUNet | SwinUNet | ConvNeXtBaseline | SwinBaseline:
    """Instantiate a segmentation or baseline model from a config dict.

    Args:
        cfg: Loaded configuration dict (from
            :func:`config.config_loader.load_config`).  Must contain a
            ``model`` key with at minimum ``name`` and ``size`` sub-keys.

    Returns:
        An instantiated model (``ConvNeXtUNet``, ``SwinUNet``,
        ``ConvNeXtBaseline``, or ``SwinBaseline``).

    Raises:
        KeyError: If ``cfg["model"]["name"]`` is not a registered model.

    Example:
        >>> cfg = load_config("config/convnext_unet.yaml")
        >>> model = build_model(cfg)
        >>> type(model).__name__
        'ConvNeXtUNet'

        >>> cfg = load_config("config/convnext_baseline.yaml")
        >>> model = build_model(cfg)
        >>> type(model).__name__
        'ConvNeXtBaseline'
    """
    model_cfg = cfg["model"]
    name = model_cfg["name"]

    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. " f"Registered models: {list(_MODEL_REGISTRY)}"
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

    # Swin models need input_size for position-bias interpolation
    if name in ("swin_unet", "swin_baseline"):
        kwargs["input_size"] = cfg["data"]["image_size"]

    return cls(**kwargs)
