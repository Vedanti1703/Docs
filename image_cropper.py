"""
image_cropper.py
────────────────
Utility functions for cropping detected regions from the original
colour image (no preprocessing — colours and quality preserved).

All public functions are pure; no Streamlit or Groq imports.
"""

from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np

from region_detector import RegionInfo


# ─────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────

def _pct_to_pixels(
    region: RegionInfo,
    img_h: int,
    img_w: int,
    padding_px: int = 4,
) -> Tuple[int, int, int, int]:
    """
    Convert percentage bounding-box to pixel (y1, x1, y2, x2).

    Adds a small *padding_px* border so crops don't clip edge strokes.
    All values are clamped to the image dimensions.
    """
    y1 = int(region.top    / 100.0 * img_h) - padding_px
    x1 = int(region.left   / 100.0 * img_w) - padding_px
    y2 = int((region.top + region.height)  / 100.0 * img_h) + padding_px
    x2 = int((region.left + region.width)  / 100.0 * img_w) + padding_px

    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(img_h, y2)
    x2 = min(img_w, x2)

    return y1, x1, y2, x2


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def crop_region(image_bgr: np.ndarray, region: RegionInfo) -> np.ndarray:
    """
    Crop *region* from *image_bgr*, preserving original colours.

    Parameters
    ----------
    image_bgr : Original BGR image (not the preprocessed grayscale one).
    region    : RegionInfo with percentage bounding-box coordinates.

    Returns
    -------
    np.ndarray
        Cropped BGR image.  Returns a 1×1 black pixel if the crop area
        is zero-sized (degenerate bounding box).
    """
    h, w = image_bgr.shape[:2]
    y1, x1, y2, x2 = _pct_to_pixels(region, h, w, padding_px=6)

    if y2 <= y1 or x2 <= x1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return image_bgr[y1:y2, x1:x2].copy()


def region_to_png_bytes(crop_bgr: np.ndarray) -> io.BytesIO:
    """
    Encode a BGR crop as a lossless PNG and return it as a BytesIO buffer.

    python-docx ``InlineImage`` accepts a file-like object; this function
    produces exactly that.

    Parameters
    ----------
    crop_bgr : OpenCV BGR image (the cropped region).

    Returns
    -------
    io.BytesIO
        Seeked-to-zero buffer containing the PNG data.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # Encode as PNG (lossless keeps equation detail crisp)
    success, buf = cv2.imencode(".png", rgb)
    if not success or len(buf) == 0:
        # Fallback: encode as JPEG
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    bio = io.BytesIO(bytes(buf))
    bio.seek(0)
    return bio


def region_to_jpeg_bytes(crop_bgr: np.ndarray, quality: int = 90) -> io.BytesIO:
    """
    Encode a BGR crop as JPEG for lightweight UI thumbnails.

    Parameters
    ----------
    crop_bgr : OpenCV BGR image.
    quality  : JPEG quality 1-100 (default 90).

    Returns
    -------
    io.BytesIO
        Seeked-to-zero JPEG buffer.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    bio = io.BytesIO(bytes(buf))
    bio.seek(0)
    return bio


def thumbnail(crop_bgr: np.ndarray, max_dim: int = 300) -> np.ndarray:
    """
    Return a downscaled copy of *crop_bgr* that fits within *max_dim*×*max_dim*,
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
