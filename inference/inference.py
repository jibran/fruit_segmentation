"""Inference script — load a trained model and run predictions.

Supports single-image prediction, directory batch processing, and
optional colourised mask overlay saving.

Usage::

    # Single image
    python inference/inference.py \\
        --config config/convnext_unet.yaml \\
        --checkpoint checkpoints/best/convnext_unet_tiny_best.pth \\
        --input path/to/image.jpg \\
        --output predictions/

    # Entire directory
    python inference/inference.py \\
        --config config/convnext_unet.yaml \\
        --checkpoint checkpoints/best/convnext_unet_tiny_best.pth \\
        --input path/to/images/ \\
        --output predictions/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config.config_loader import load_config

# Colour palette for mask visualisation (one RGB colour per class)
# Import canonical palette from dataset module so colours stay in sync.
# _PALETTE is kept here as a local alias for the inference helpers below.
from dataset.fruit_dataset import CLASS_PALETTE as _PALETTE  # noqa: E402
from models import build_model
from utils.transforms import build_val_transform


def mask_to_colour(mask: np.ndarray) -> np.ndarray:
    """Convert an integer class mask to an RGB colour image.

    Args:
        mask: Integer ndarray of shape ``(H, W)`` with values in
            ``[0, NUM_CLASSES - 1]``.

    Returns:
        RGB uint8 ndarray of shape ``(H, W, 3)``.
    """
    h, w = mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in enumerate(_PALETTE):
        colour[mask == cls_idx] = rgb
    return colour


def overlay_mask(
    image: np.ndarray,
    mask_colour: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a colourised mask over the original image.

    Args:
        image: Original RGB uint8 image of shape ``(H, W, 3)``.
        mask_colour: Colour mask of shape ``(H, W, 3)``.
        alpha: Transparency of the mask overlay (0 = transparent,
            1 = opaque).  Defaults to 0.5.

    Returns:
        Blended RGB uint8 image.
    """
    return (image * (1 - alpha) + mask_colour * alpha).astype(np.uint8)


def predict_single(
    model: torch.nn.Module,
    image_path: Path,
    transform,
    device: torch.device,
) -> np.ndarray:
    """Run inference on a single image and return the predicted mask.

    Args:
        model: Trained segmentation model in eval mode.
        image_path: Path to the input image file.
        transform: Validation transform (resize + normalise).
        device: Compute device.

    Returns:
        Integer ndarray of shape ``(H, W)`` with class indices.
    """
    image_np = np.array(Image.open(image_path).convert("RGB"))
    original_h, original_w = image_np.shape[:2]

    # Dummy mask for transform compatibility
    dummy_mask = np.zeros((original_h, original_w), dtype=np.uint8)
    out = transform(image=image_np, mask=dummy_mask)
    image_tensor = out["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(image_tensor)  # (1, C, H, W)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

    # Resize prediction back to original resolution
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    pred_resized = F.interpolate(
        pred_tensor, size=(original_h, original_w), mode="nearest"
    )
    return pred_resized.squeeze().long().numpy().astype(np.int32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with a trained fruit segmentation model."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single image or a directory of images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write prediction outputs.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Also save colourised mask overlaid on original image.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. 'cuda:0', 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the inference script."""
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Model ──────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {state.get('epoch', '?')} | device: {device}")

    # ── Transform ──────────────────────────────────────────────────
    transform = build_val_transform(cfg)

    # ── Gather inputs ─────────────────────────────────────────────
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = sorted(
            p
            for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        image_paths = [input_path]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(image_paths)} image(s)...")

    for img_path in image_paths:
        pred_mask = predict_single(model, img_path, transform, device)

        # Save grayscale integer mask
        mask_img = Image.fromarray(pred_mask.astype(np.uint8))
        mask_img.save(output_dir / f"{img_path.stem}_mask.png")

        # Optionally save colour overlay
        if args.overlay:
            original = np.array(Image.open(img_path).convert("RGB"))
            colour_mask = mask_to_colour(pred_mask)
            blended = overlay_mask(original, colour_mask)
            Image.fromarray(blended).save(output_dir / f"{img_path.stem}_overlay.png")

        print(f"  {img_path.name} → {img_path.stem}_mask.png")

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
