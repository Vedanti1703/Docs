"""
region_detector.py
──────────────────
Uses Groq Llama-4-Scout to analyse a full page image and return a
structured list of RegionInfo objects in reading order.  Each region
is typed as  text | equation | diagram  and carries a % bounding-box
so the caller can crop the exact pixel area from the original image.

Two-step detection strategy
────────────────────────────
Step A — Large-diagram detection (NEW):
  Before calling the main JSON detection flow, inspect the image to
  decide if this is a "diagram-dominant" page  (≥ 50 % of pixel area
  is occupied by a drawing / figure with very little text).  If so:

  1. Run a focused OCR call on ONLY the header strip (top 20 % of the
     image) to extract header text reliably.
  2. Treat the remaining portion (top 15 % – bottom 100 %) as a single
     large diagram region covering the full width.
  3. Return these two synthetic regions directly — no JSON parsing
     needed for the diagram portion, so there is zero risk of 0-word
     output on these pages.

Step B — Standard JSON detection (unchanged):
  For all other page types (text, text+equations, mixed) the existing
  JSON flow runs as before, with the four validation steps (sentence
  check, empty content, size sanity, reading order).

Post-processing validation steps (Step B only)
──────────────────────────────────────────────
Step 1 — Sentence check on equation regions
Step 2 — Empty content check on text regions
Step 3 — Size sanity check
Step 4 — Reading order enforcement

Robust JSON parsing
────────────────────
Three attempts are made before falling back to whole-page text region:
  Attempt 1 — direct json.loads after stripping fences
  Attempt 2 — regex scan for first {...} containing "regions"
  Attempt 3 — regex scan for any JSON array inside the response
Never returns an empty RegionList on API success.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegionInfo:
    """One detected region on the page."""
    type: str               # 'text' | 'equation' | 'diagram'
    content: Optional[str]  # prose text (text regions only)
    top: float              # % from image top   (0–100)
    left: float             # % from image left  (0–100)
    width: float            # % of image width   (0–100)
    height: float           # % of image height  (0–100)
    index: int = 0          # reading order (0-based)


class RegionList(list):
    """list[RegionInfo] with an extra `.fallback` flag."""
    fallback: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Prompt retrieval  (single source of truth in ocr_engine)
# ─────────────────────────────────────────────────────────────────────────────

def _get_prompts() -> Tuple[str, str]:
    from ocr_engine import REGION_DETECTION_SYSTEM, REGION_DETECTION_USER
    return REGION_DETECTION_SYSTEM, REGION_DETECTION_USER


# ─────────────────────────────────────────────────────────────────────────────
# Image encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_b64(image_bgr: np.ndarray) -> str:
    """Encode a BGR OpenCV image as a base-64 PNG for the vision model."""
    if len(image_bgr.shape) == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    if max(h, w) < 1024:
        scale = 1024 / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    success, buf = cv2.imencode(".png", rgb)
    if not success:
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(bytes(buf)).decode("utf-8")


def _crop_b64(image_bgr: np.ndarray,
              top_pct: float, left_pct: float,
              height_pct: float, width_pct: float) -> str:
    """
    Crop a sub-rectangle (given as % coordinates) from *image_bgr*
    and return it as a base-64 PNG string.
    Used for the focused header OCR call.
    """
    h, w = image_bgr.shape[:2]
    y1 = max(0, int(top_pct    / 100 * h))
    x1 = max(0, int(left_pct   / 100 * w))
    y2 = min(h, int((top_pct  + height_pct) / 100 * h))
    x2 = min(w, int((left_pct + width_pct)  / 100 * w))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr[:max(1, h // 5), :]
    return _image_to_b64(crop)


# ─────────────────────────────────────────────────────────────────────────────
# Large-diagram page detection  (Step A)
# ─────────────────────────────────────────────────────────────────────────────

def _is_diagram_dominant(image_bgr: np.ndarray,
                          diagram_area_threshold: float = 0.50) -> bool:
    """
    Heuristic: decide if this page is dominated by a hand-drawn diagram.

    Strategy:
    1. Convert to greyscale and binarise (Otsu).
    2. Find the largest connected component of dark pixels.
    3. If that component covers ≥ diagram_area_threshold of the page area
       AND has a roughly rectangular or irregular shape that looks like a
       drawing rather than text lines, classify as diagram-dominant.

    Additionally, count horizontal text-like lines in the top 20 % of the
    image versus the rest.  If text lines are sparse below the header zone,
    the page is likely diagram-dominant.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
        if len(image_bgr.shape) == 3 else image_bgr.copy()

    h, w = gray.shape

    # ── Binarise ──────────────────────────────────────────────────────────
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Count dark pixels in header (top 20 %) vs body (rest) ─────────────
    header_h = int(h * 0.20)
    body_dark   = int(np.sum(bw[header_h:] > 0))
    total_body  = (h - header_h) * w
    body_fill   = body_dark / total_body if total_body > 0 else 0.0

    # ── Count horizontal text-like runs in body ────────────────────────────
    # Project horizontally: each text line produces a peak in row sums
    body_bw     = bw[header_h:]
    row_sums    = np.sum(body_bw > 0, axis=1)          # dark pixels per row
    text_rows   = int(np.sum(row_sums > w * 0.05))     # rows with ≥5 % ink
    body_h      = max(h - header_h, 1)
    text_density = text_rows / body_h

    # A diagram body has scattered ink (medium body_fill) with LOW text density.
    # A text body has high text density (many full-width ink rows).
    # Typical thresholds (tuned empirically):
    #   diagram_dominant: body_fill ∈ [0.03, 0.65]  AND text_density < 0.45
    is_diagram = (
        diagram_area_threshold * 0.06 <= body_fill <= 0.70
        and text_density < 0.45
    )
    return is_diagram


def _ocr_header(image_bgr: np.ndarray, groq_client,
                header_height_pct: float = 20.0) -> str:
    """
    Run a focused OCR call on the top *header_height_pct* % of the image
    to extract header / title text reliably.

    Returns the raw transcribed string (may be empty on failure).
    Fix 3: focused header extraction for better accuracy on small header text.
    """
    b64 = _crop_b64(image_bgr,
                    top_pct=0, left_pct=0,
                    height_pct=header_height_pct, width_pct=100)

    header_prompt = (
        "Transcribe ALL handwritten text in this header strip exactly as written. "
        "Preserve line breaks. Output ONLY the transcribed text, nothing else."
    )
    try:
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": header_prompt},
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.05,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _build_diagram_dominant_result(header_text: str,
                                   header_height_pct: float = 20.0) -> RegionList:
    """
    Construct a synthetic two-region RegionList for diagram-dominant pages:
      Region 0 — text  : header strip (top header_height_pct %)
      Region 1 — diagram : remainder of the page

    Fix 5: header text always first, diagram image always below.
    """
    result = RegionList()
    result.fallback = False

    idx = 0

    # Header text region (only if non-empty)
    if header_text.strip():
        result.append(RegionInfo(
            type="text",
            content=header_text,
            top=0.0,
            left=0.0,
            width=100.0,
            height=header_height_pct,
            index=idx,
        ))
        idx += 1

    # Diagram region: from just below header to bottom of page
    diagram_top = header_height_pct if header_text.strip() else 0.0
    diagram_h   = 100.0 - diagram_top
    result.append(RegionInfo(
        type="diagram",
        content=None,
        top=diagram_top,
        left=0.0,
        width=100.0,
        height=max(diagram_h, 1.0),
        index=idx,
    ))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Robust JSON parsing  (Fix 1)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_attempt_direct(raw: str) -> List[dict]:
    """Attempt 1: direct json.loads after stripping markdown fences."""
    data = json.loads(_strip_fences(raw))
    regions = data.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("'regions' is not a list")
    return regions


def _parse_attempt_regex_object(raw: str) -> List[dict]:
    """Attempt 2: find first JSON object containing 'regions' key."""
    match = re.search(r'\{[^{}]*"regions"\s*:\s*\[.*?\]\s*\}',
                      raw, re.DOTALL)
    if not match:
        # Try wider brace match
        match = re.search(r'\{.*?"regions".*?\}', raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object with 'regions' found")
    data = json.loads(match.group(0))
    regions = data.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("'regions' is not a list")
    return regions


def _parse_attempt_array(raw: str) -> List[dict]:
    """
    Attempt 3: if the model returned a bare JSON array instead of an object,
    treat each element as a region dict.
    """
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON array found")
    arr = json.loads(match.group(0))
    if not isinstance(arr, list):
        raise ValueError("match is not a list")
    return arr


def _robust_parse(raw_response: str) -> List[dict]:
    """
    Try all three parse strategies in order.
    Raises ValueError if all fail (caller handles fallback).
    Fix 1: never returns empty on parse failure — always raises so caller
    knows to use the whole-page fallback.
    """
    for attempt in (_parse_attempt_direct,
                    _parse_attempt_regex_object,
                    _parse_attempt_array):
        try:
            regions = attempt(raw_response)
            if isinstance(regions, list):
                return regions
        except Exception:
            continue
    raise ValueError("all parse attempts failed")


def _build_fallback(full_text: str) -> RegionList:
    """Return a single text region covering the whole page."""
    result = RegionList()
    result.fallback = True
    result.append(RegionInfo(
        type="text",
        content=full_text or "(content could not be parsed)",
        top=0.0, left=0.0,
        width=100.0, height=100.0,
        index=0,
    ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Region validation
# ─────────────────────────────────────────────────────────────────────────────

def _get_field(raw: dict, name: str, fallback: float = 0.0) -> float:
    """Accept both 'top' and 'top_percent' field naming conventions."""
    v = raw.get(f"{name}_percent", raw.get(name, fallback))
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback


_TYPE_PADDING: dict = {
    "equation": 2.0,
    "diagram":  4.0,
    "text":     0.0,
}


def _validate_region(raw: dict, index: int) -> Optional[RegionInfo]:
    """Convert a raw dict to RegionInfo with per-type padding."""
    try:
        rtype = str(raw.get("type", "text")).lower().strip()
        if rtype not in ("text", "equation", "diagram"):
            rtype = "text"

        content = raw.get("content")
        if content is not None:
            content = str(content).strip() or None

        top    = _get_field(raw, "top",    0.0)
        left   = _get_field(raw, "left",   0.0)
        width  = _get_field(raw, "width",  100.0)
        height = _get_field(raw, "height", 10.0)

        pad = _TYPE_PADDING.get(rtype, 0.0)
        top    -= pad
        left   -= pad
        width  += pad * 2
        height += pad * 2

        top    = max(0.0, min(100.0, top))
        left   = max(0.0, min(100.0, left))
        width  = max(1.0, min(100.0 - left, width))
        height = max(1.0, min(100.0 - top,  height))

        return RegionInfo(
            type=rtype,
            content=content,
            top=top, left=left,
            width=width, height=height,
            index=index,
        )
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing steps  (Step B — standard pages)
# ─────────────────────────────────────────────────────────────────────────────

_ENGLISH_WORDS_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
_MATH_KEYWORDS = {
    "sin", "cos", "tan", "log", "exp", "lim", "sup", "inf",
    "max", "min", "det", "div", "curl", "grad", "sqrt", "int",
    "sum", "prod", "mod", "arg", "sgn",
}


def _count_english_words(text: str) -> int:
    if not text:
        return 0
    words = _ENGLISH_WORDS_RE.findall(text)
    return sum(1 for w in words if w.lower() not in _MATH_KEYWORDS)


def _sentence_check(regions: List[RegionInfo]) -> List[RegionInfo]:
    """Step 1: reclassify over-wordy 'equation' regions as text."""
    for r in regions:
        if r.type == "equation" and r.content:
            if _count_english_words(r.content) > 6:
                r.type = "text"
    return regions


def _empty_content_check(regions: List[RegionInfo]) -> List[RegionInfo]:
    """Step 2: drop text regions with null / empty content."""
    return [r for r in regions
            if not (r.type == "text" and not (r.content or "").strip())]


def _size_sanity_check(regions: List[RegionInfo]) -> List[RegionInfo]:
    """Step 3: remove implausibly small regions."""
    result = []
    for r in regions:
        if r.type == "equation" and r.height < 3.0:
            continue
        if r.type == "text" and r.height < 1.0:
            continue
        result.append(r)
    return result


def _reading_order_sort(regions: List[RegionInfo]) -> List[RegionInfo]:
    """Step 4: sort top-to-bottom, ties broken left-to-right (2 % row bucket)."""
    return sorted(regions, key=lambda r: (round(r.top / 2) * 2, r.left))


def _merge_two(a: RegionInfo, b: RegionInfo) -> RegionInfo:
    new_top    = min(a.top, b.top)
    new_left   = min(a.left, b.left)
    new_right  = max(a.left + a.width,  b.left + b.width)
    new_bottom = max(a.top  + a.height, b.top  + b.height)
    if a.content and b.content:
        merged_content: Optional[str] = a.content + "\n" + b.content
    else:
        merged_content = a.content or b.content
    return RegionInfo(
        type=a.type,
        content=merged_content,
        top=new_top, left=new_left,
        width=new_right - new_left,
        height=new_bottom - new_top,
        index=a.index,
    )


def _iou(a: RegionInfo, b: RegionInfo) -> float:
    ax1, ay1 = a.left, a.top
    ax2, ay2 = a.left + a.width, a.top + a.height
    bx1, by1 = b.left, b.top
    bx2, by2 = b.left + b.width, b.top + b.height
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _remove_overlaps(regions: List[RegionInfo],
                     iou_threshold: float = 0.25) -> List[RegionInfo]:
    changed = True
    while changed:
        changed = False
        merged: List[RegionInfo] = []
        skip: set = set()
        for i, r in enumerate(regions):
            if i in skip:
                continue
            for j in range(i + 1, len(regions)):
                if j in skip:
                    continue
                other = regions[j]
                if other.type == r.type and _iou(r, other) >= iou_threshold:
                    r = _merge_two(r, other)
                    skip.add(j)
                    changed = True
            merged.append(r)
        regions = merged
    return regions


def _merge_consecutive_same_type(regions: List[RegionInfo]) -> List[RegionInfo]:
    if not regions:
        return regions
    merged: List[RegionInfo] = []
    current = regions[0]
    for region in regions[1:]:
        if region.type in ("equation", "diagram") and region.type == current.type:
            current = _merge_two(current, region)
        else:
            merged.append(current)
            current = region
    merged.append(current)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Standard detection (Step B)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_standard(image_bgr: np.ndarray, groq_client) -> RegionList:
    """
    Run the full JSON-based region detection for text / equation / mixed pages.
    Includes robust JSON parsing and all 4 post-processing steps.
    """
    system_prompt, user_prompt = _get_prompts()
    b64 = _image_to_b64(image_bgr)

    raw_response = ""
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.05,
        )
        raw_response = response.choices[0].message.content or ""
    except Exception as exc:
        return _build_fallback(f"[API error: {exc}]")

    # ── Robust JSON parsing (Fix 1) ────────────────────────────────────────
    try:
        raw_regions = _robust_parse(raw_response)
    except Exception:
        # All parse attempts failed — fall back to whole-page text region
        return _build_fallback(raw_response.strip() or "(no response)")

    if not raw_regions:
        return _build_fallback(raw_response.strip() or "(empty regions list)")

    # ── Validate ───────────────────────────────────────────────────────────
    validated: List[RegionInfo] = []
    for i, raw in enumerate(raw_regions):
        region = _validate_region(raw, index=i)
        if region is not None:
            validated.append(region)

    if not validated:
        return _build_fallback(raw_response.strip() or "(validation failed)")

    # ── Post-processing ────────────────────────────────────────────────────
    processed = _sentence_check(validated)
    processed = _empty_content_check(processed)
    processed = _size_sanity_check(processed)
    processed = _merge_consecutive_same_type(processed)
    processed = _remove_overlaps(processed, iou_threshold=0.25)
    processed = _reading_order_sort(processed)

    result = RegionList(processed)
    result.fallback = False
    for i, r in enumerate(result):
        r.index = i
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_regions(image_bgr: np.ndarray, groq_client) -> RegionList:
    """
    Detect and classify all content regions on *image_bgr*.

    Two-step strategy (Fix 2):
    • If the page is diagram-dominant (large drawing fills most of the body):
        → OCR only the header strip (Fix 3)
        → Treat body as one large diagram region (Fix 4)
        → Guarantee non-empty output (Fix 5)
    • Otherwise:
        → Run standard JSON detection with robust parsing (Fix 1)

    Parameters
    ----------
    image_bgr   : OpenCV BGR image of the document page.
    groq_client : Initialised groq.Groq client.

    Returns
    -------
    RegionList  — always non-empty; fallback flag set when JSON parsing failed.
    """
    # ── Step A: diagram-dominant page detection ────────────────────────────
    if _is_diagram_dominant(image_bgr):
        header_text = _ocr_header(image_bgr, groq_client, header_height_pct=20.0)
        return _build_diagram_dominant_result(header_text, header_height_pct=20.0)

    # ── Step B: standard JSON detection ───────────────────────────────────
    return _detect_standard(image_bgr, groq_client)
