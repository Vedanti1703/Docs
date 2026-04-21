"""
image_cropper.py
────────────────
Utility functions for cropping detected regions from the original
colour image (no preprocessing — colours and quality preserved).

Cropping strategy by region type
──────────────────────────────────
text      → 15 px padding, CLAHE, 1.5× upscale
equation  → 20 px padding, CLAHE, 1.5× upscale, lossless PNG
diagram   → 25 px padding, NO upscale (preserve full resolution),
            mild unsharp-mask sharpening, lossless PNG
            (Fix 4: full resolution + mild sharpening for large diagrams)

All public functions are pure; no Streamlit or Groq imports.
"""

from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np

from region_detector import RegionInfo


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the L-channel of LAB for local contrast enhancement.
    Works on BGR or greyscale input; always returns BGR.
    """
    if len(bgr.shape) == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _apply_mild_sharpen(bgr: np.ndarray) -> np.ndarray:
    """
    Apply a mild unsharp-mask to a diagram crop so that drawn lines
    appear crisper without introducing noise artefacts.

    Fix 4: used ONLY on diagram crops at full resolution.
    """
    # Gaussian blur for the unsharp mask
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.5)
    # amount = 0.6 keeps sharpening gentle
    sharpened = cv2.addWeighted(bgr, 1.6, blurred, -0.6, 0)
    return sharpened


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

# Per-type pixel padding around the bounding box
_TYPE_PADDING_PX: dict = {
    "text":     15,
    "equation": 20,
    "diagram":  25,
}


def _pct_to_pixels(
    region: RegionInfo,
    img_h: int,
    img_w: int,
    extra_pad: int = 0,
) -> Tuple[int, int, int, int]:
    """
    Convert percentage bounding-box to pixel (y1, x1, y2, x2).
    Applies type-aware padding and clamps to image bounds.
    """
    base_pad = _TYPE_PADDING_PX.get(region.type, 15) + extra_pad

    y1 = int(region.top    / 100.0 * img_h) - base_pad
    x1 = int(region.left   / 100.0 * img_w) - base_pad
    y2 = int((region.top  + region.height) / 100.0 * img_h) + base_pad
    x2 = int((region.left + region.width)  / 100.0 * img_w) + base_pad

    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(img_h, y2)
    x2 = min(img_w, x2)

    return y1, x1, y2, x2


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def crop_region(image_bgr: np.ndarray, region: RegionInfo) -> np.ndarray:
    """
    Crop *region* from *image_bgr* with type-aware enhancement.

    text / equation → CLAHE + 1.5× upscale for sharpness.
    diagram         → full resolution + mild unsharp sharpening
                      (Fix 4: no upscale so large diagrams don't become
                       enormous; sharpening preserves drawn-line clarity).

    Parameters
    ----------
    image_bgr : Original BGR image (not preprocessed greyscale).
    region    : RegionInfo with percentage bounding-box coordinates.

    Returns
    -------
    np.ndarray  — cropped BGR image; 1×1 black pixel if degenerate.
    """
    h, w = image_bgr.shape[:2]
    y1, x1, y2, x2 = _pct_to_pixels(region, h, w)

    if y2 <= y1 or x2 <= x1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    crop = image_bgr[y1:y2, x1:x2].copy()

    if region.type == "diagram":
        # Fix 4: full resolution PNG + mild sharpening only
        crop = _apply_mild_sharpen(crop)
    else:
        # text and equation: CLAHE + 1.5× upscale as before
        crop = _apply_clahe(crop)
        new_h = int(crop.shape[0] * 1.5)
        new_w = int(crop.shape[1] * 1.5)
        if new_h > 1 and new_w > 1:
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return crop


def region_to_png_bytes(crop_bgr: np.ndarray) -> io.BytesIO:
    """
    Encode a BGR crop as a lossless PNG and return it as a BytesIO buffer.

    python-docx InlineImage accepts a file-like object; this function
    produces exactly that.

    Parameters
    ----------
    crop_bgr : OpenCV BGR image (the cropped region).

    Returns
    -------
    io.BytesIO  — seeked-to-zero PNG buffer.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    success, buf = cv2.imencode(".png", rgb)
    if not success or len(buf) == 0:
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 97])
    bio = io.BytesIO(bytes(buf))
    bio.seek(0)
    return bio


def region_to_jpeg_bytes(crop_bgr: np.ndarray, quality: int = 92) -> io.BytesIO:
    """
    Encode a BGR crop as JPEG for lightweight UI thumbnails.

    Parameters
    ----------
    crop_bgr : OpenCV BGR image.
    quality  : JPEG quality 1-100 (default 92).

    Returns
    -------
    io.BytesIO  — seeked-to-zero JPEG buffer.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    bio = io.BytesIO(bytes(buf))
    bio.seek(0)
    return bio


def thumbnail(crop_bgr: np.ndarray, max_dim: int = 300) -> np.ndarray:
    """
    Return a downscaled copy of *crop_bgr* fitting within *max_dim*×*max_dim*,
    preserving aspect ratio.  Used for UI previews only.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale >= 1.0:
        return crop_bgr.copy()
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
