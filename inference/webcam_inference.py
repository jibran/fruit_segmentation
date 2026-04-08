"""Live webcam inference — real-time fruit segmentation overlay.

Opens a webcam feed, runs the segmentation model on every frame, and
displays the colourised mask blended over the live video.  Class name
labels are drawn at the centroid of each detected region.

Press ``q`` or ``Esc`` to quit.
Press ``s`` to save a snapshot of the current frame to disk.
Press ``l`` to toggle class labels on/off.
Press ``+`` / ``-`` to increase/decrease the overlay alpha.

Usage::

    python inference/webcam_inference.py \\
        --config config/convnext_unet.yaml \\
        --checkpoint checkpoints/best/convnext_unet_tiny_best.pth

    # Specify camera index (default 0)
    python inference/webcam_inference.py \\
        --config config/swin_unet.yaml \\
        --checkpoint checkpoints/best/swin_unet_tiny_best.pth \\
        --camera 1

    # Limit inference resolution for speed (frames are resized back for display)
    python inference/webcam_inference.py \\
        --config config/convnext_unet.yaml \\
        --checkpoint checkpoints/best/convnext_unet_tiny_best.pth \\
        --infer-size 256

    # Save snapshots to a custom directory
    python inference/webcam_inference.py \\
        --config config/convnext_unet.yaml \\
        --checkpoint checkpoints/best/convnext_unet_tiny_best.pth \\
        --snapshot-dir snapshots/

Requirements::

    pip install opencv-python torch torchvision albumentations Pillow numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.config_loader import load_config
from dataset.fruit_dataset import (
    BACKGROUND_IDX,
    CLASS_NAMES,
    CLASS_PALETTE,
    NUM_CLASSES,
)
from models import build_model
from utils.transforms import build_val_transform

# ── Colour lookup table (BGR for OpenCV) ─────────────────────────────────────

_PALETTE_BGR: list[tuple[int, int, int]] = [(b, g, r) for (r, g, b) in CLASS_PALETTE]


# ── Helpers ───────────────────────────────────────────────────────────────────


def build_colour_lut() -> np.ndarray:
    """Build a (NUM_CLASSES, 3) uint8 BGR lookup table for fast mask colouring.

    Returns:
        Array of shape ``(NUM_CLASSES, 3)`` with BGR values.
    """
    lut = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    for idx, (r, g, b) in enumerate(CLASS_PALETTE):
        lut[idx] = (b, g, r)
    return lut


_LUT = build_colour_lut()


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Convert an integer class mask to a BGR colour image via the LUT.

    Args:
        mask: Integer ndarray of shape ``(H, W)`` with class indices.

    Returns:
        BGR uint8 ndarray of shape ``(H, W, 3)``.
    """
    return _LUT[mask]


def overlay_mask_bgr(
    frame: np.ndarray,
    mask_bgr: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend a BGR colour mask over an OpenCV frame.

    Args:
        frame: BGR uint8 frame from OpenCV.
        mask_bgr: BGR colour mask of the same spatial size.
        alpha: Mask opacity (0 = invisible, 1 = fully opaque).

    Returns:
        Blended BGR uint8 frame.
    """
    return cv2.addWeighted(frame, 1.0 - alpha, mask_bgr, alpha, 0)


def draw_labels_cv2(
    frame: np.ndarray,
    mask: np.ndarray,
    font_scale: float = 0.45,
    min_area_frac: float = 0.002,
) -> np.ndarray:
    """Draw class name labels on an OpenCV frame using cv2 drawing primitives.

    Labels are placed at the centroid of each detected class region.
    Regions covering less than ``min_area_frac`` of the frame are skipped.
    The pill background uses the class palette colour; text is white or
    dark depending on fill brightness.

    Args:
        frame: BGR uint8 frame to annotate (modified in-place copy).
        mask: Integer ndarray of shape ``(H, W)`` with class indices.
        font_scale: OpenCV font scale factor.  Defaults to 0.45.
        min_area_frac: Minimum fraction of total pixels a region must
            cover for its label to be shown.  Defaults to 0.002 (0.2%).

    Returns:
        Annotated BGR uint8 frame.
    """
    out = frame.copy()
    h, w = mask.shape
    min_pixels = int(h * w * min_area_frac)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_idx == BACKGROUND_IDX:
            continue
        region = mask == cls_idx
        pixel_count = int(region.sum())
        if pixel_count < min_pixels:
            continue

        ys, xs = np.where(region)
        cx = int(xs.mean())
        cy = int(ys.mean())

        label = cls_name.replace("_", " ")
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        pad_x, pad_y = 6, 4
        x0 = max(0, cx - tw // 2 - pad_x)
        y0 = max(0, cy - th // 2 - pad_y - baseline)
        x1 = min(w - 1, cx + tw // 2 + pad_x)
        y1 = min(h - 1, cy + th // 2 + pad_y)

        r, g, b = CLASS_PALETTE[cls_idx]
        fill_bgr = (b, g, r)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_colour = (255, 255, 255) if brightness < 140 else (20, 20, 20)

        cv2.rectangle(out, (x0, y0), (x1, y1), fill_bgr, cv2.FILLED)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(
            out,
            label,
            (cx - tw // 2, cy + th // 2 - baseline),
            font,
            font_scale,
            text_colour,
            thickness,
            cv2.LINE_AA,
        )

    return out


def draw_hud(
    frame: np.ndarray,
    fps: float,
    alpha: float,
    labels_on: bool,
    model_name: str,
    device: str,
) -> np.ndarray:
    """Draw a minimal HUD (heads-up display) in the top-left corner.

    Shows the model name, device, current FPS, overlay alpha, and
    a label-toggle indicator.

    Args:
        frame: BGR uint8 frame to annotate (modified in-place copy).
        fps: Current frames-per-second estimate.
        alpha: Current overlay alpha value.
        labels_on: Whether class labels are currently shown.
        model_name: Name of the loaded model (for display).
        device: Device string (e.g. ``"cuda:0"`` or ``"cpu"``).

    Returns:
        Annotated BGR uint8 frame.
    """
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"{model_name}  [{device}]",
        f"FPS: {fps:.1f}",
        f"alpha: {alpha:.2f}  [+/-]",
        f"labels: {'on' if labels_on else 'off'}  [l]",
        "s=snapshot  q=quit",
    ]
    fs, th_line = 0.38, 15
    x, y = 8, 18
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, fs, 1)
        cv2.rectangle(out, (x - 2, y - 10), (x + tw + 2, y + 4), (0, 0, 0), cv2.FILLED)
        cv2.putText(out, line, (x, y), font, fs, (200, 230, 200), 1, cv2.LINE_AA)
        y += th_line
    return out


# ── Model helpers ─────────────────────────────────────────────────────────────


def load_model(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained segmentation model from a checkpoint.

    Args:
        cfg: Loaded configuration dictionary.
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Target compute device.

    Returns:
        Model in eval mode, ready for inference.
    """
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    epoch = state.get("epoch", "?")
    miou = state.get("best_miou", None)
    miou_str = f"  best val mIoU={miou:.4f}" if miou else ""
    print(f"  Loaded checkpoint epoch {epoch}{miou_str}")
    return model


@torch.inference_mode()
def infer_frame(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    transform,
    infer_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run segmentation inference on a single BGR OpenCV frame.

    The frame is resized to ``infer_size`` for the model, and the
    predicted mask is resized back to the original frame resolution
    using nearest-neighbour interpolation.

    Args:
        model: Trained segmentation model in eval mode.
        frame_bgr: BGR uint8 frame from OpenCV.
        transform: Albumentations validation transform.
        infer_size: Square inference resolution (e.g. 256 or 512).
        device: Compute device.

    Returns:
        Integer ndarray of shape ``(H, W)`` with predicted class indices,
        at the same spatial resolution as ``frame_bgr``.
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    out = transform(image=frame_rgb, mask=dummy_mask)
    tensor = out["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    if pred.shape != (h, w):
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
        pred_t = F.interpolate(pred_t, size=(h, w), mode="nearest")
        pred = pred_t.squeeze().long().numpy().astype(np.int32)

    return pred


# ── Main loop ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Live webcam fruit segmentation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Defaults to 0.",
    )
    parser.add_argument(
        "--infer-size",
        type=int,
        default=512,
        dest="infer_size",
        help="Square resolution fed to the model. Lower = faster. Default 512.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Initial overlay alpha (0–1). Default 0.5.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        dest="no_labels",
        help="Disable class name labels on startup.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.002,
        dest="min_area",
        help="Min region area fraction for label display. Default 0.002.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="snapshots",
        dest="snapshot_dir",
        help="Directory to save snapshots. Default 'snapshots/'.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. 'cuda:0', 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — opens the webcam and runs the live segmentation loop."""
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("\nFruit Segmentation — Live Webcam")
    print(f"  Config    : {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device    : {device}")
    print(f"  Camera    : {args.camera}")
    print(f"  Infer size: {args.infer_size}px")

    model = load_model(cfg, args.checkpoint, device)
    model_name = cfg.get("model", {}).get("backbone", "model")

    # Override image_size in config to match infer_size so the transform
    # resizes to the requested inference resolution
    cfg["data"]["image_size"] = args.infer_size
    transform = build_val_transform(cfg)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}.")
        sys.exit(1)

    # Try to set a reasonable default resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    alpha_step = 0.05
    labels_on = not args.no_labels

    fps_buf: deque[float] = deque(maxlen=30)
    t_prev = time.perf_counter()

    print("\nControls:")
    print("  q / Esc  — quit")
    print("  s        — save snapshot")
    print("  l        — toggle labels")
    print("  + / -    — increase/decrease overlay alpha\n")

    cv2.namedWindow("Fruit Segmentation", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame, retrying...")
            time.sleep(0.05)
            continue

        # ── Inference ──────────────────────────────────────────────
        pred_mask = infer_frame(model, frame, transform, args.infer_size, device)

        # ── Visualise ──────────────────────────────────────────────
        colour_mask = mask_to_bgr(pred_mask)
        display = overlay_mask_bgr(frame, colour_mask, alpha)

        if labels_on:
            display = draw_labels_cv2(display, pred_mask, min_area_frac=args.min_area)

        # ── FPS ────────────────────────────────────────────────────
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        fps = sum(fps_buf) / len(fps_buf)

        # ── HUD ────────────────────────────────────────────────────
        display = draw_hud(display, fps, alpha, labels_on, model_name, str(device))

        cv2.imshow("Fruit Segmentation", display)

        # ── Key handling ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            break

        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snap_path = snapshot_dir / f"snapshot_{ts}.png"
            cv2.imwrite(str(snap_path), display)
            print(f"  Snapshot saved → {snap_path}")

        elif key == ord("l"):
            labels_on = not labels_on
            print(f"  Labels: {'on' if labels_on else 'off'}")

        elif key in (ord("+"), ord("=")):
            alpha = min(1.0, round(alpha + alpha_step, 2))
            print(f"  Alpha: {alpha:.2f}")

        elif key in (ord("-"), ord("_")):
            alpha = max(0.0, round(alpha - alpha_step, 2))
            print(f"  Alpha: {alpha:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()
