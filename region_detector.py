"""
region_detector.py
──────────────────
Uses Groq Llama-4-Scout to analyse a full page image and return a
structured list of RegionInfo objects in reading order.  Each region
is typed as  text | equation | diagram  and carries a % bounding-box
so the caller can crop the exact pixel area from the original image.

Fallback behaviour
──────────────────
If the model returns invalid / unparseable JSON the module silently
falls back to treating the entire page as a single text region (the
same result the old plain-text flow produced).  No exception is raised
to the caller; a warning is stored on the returned list's `.fallback`
attribute so the UI can show a soft warning if desired.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────

@dataclass
class RegionInfo:
    """One detected region on the page."""
    type: str           # 'text' | 'equation' | 'diagram'
    content: Optional[str]  # transcribed text (text regions only)
    top: float          # % from image top edge   (0-100)
    left: float         # % from image left edge  (0-100)
    width: float        # % of image width        (0-100)
    height: float       # % of image height       (0-100)
    index: int = 0      # reading order (0-based)


class RegionList(list):
    """list[RegionInfo] with an extra `.fallback` flag."""
    fallback: bool = False


# ─────────────────────────────────────────────────────────────
# JSON prompt
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a document-analysis assistant. "
    "You receive an image of a handwritten notes page. "
    "Your task is to identify every region in reading order (top to bottom, "
    "left to right) and return ONLY a valid JSON object — no markdown fences, "
    "no explanation, no trailing text."
)

_USER_PROMPT = """\
Analyse this handwritten notes image.
Return ONLY valid JSON (no markdown, no extra text) with this exact structure:

{
  "regions": [
    {
      "type": "text",
      "content": "exact transcription of the handwritten text here",
      "top": 5,
      "left": 2,
      "width": 96,
      "height": 8
    },
    {
      "type": "equation",
      "content": null,
      "top": 14,
      "left": 10,
      "width": 80,
      "height": 12
    },
    {
      "type": "diagram",
      "content": null,
      "top": 30,
      "left": 5,
      "width": 90,
      "height": 25
    }
  ]
}

Rules:
- type must be EXACTLY one of: text, equation, diagram
- content: for text regions write the transcribed text; for equation/diagram set null
- top, left, width, height: percentage floats (0-100) relative to full image size
- list ALL regions in reading order top to bottom
- do not skip any region including headers, footers, page numbers
- return ONLY the JSON object, nothing else
"""


# ─────────────────────────────────────────────────────────────
# Core detection
# ─────────────────────────────────────────────────────────────

def _image_to_b64(image_bgr: np.ndarray) -> str:
    """Encode a BGR OpenCV image as a base-64 JPEG string."""
    if len(image_bgr.shape) == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8")


def _strip_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_response(raw: str) -> List[dict]:
    """Attempt to parse the model output; returns list of region dicts."""
    cleaned = _strip_fences(raw)
    data = json.loads(cleaned)
    regions = data.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("'regions' is not a list")
    return regions


def _build_fallback(full_text: str) -> RegionList:
    """Return a single text region covering the whole page."""
    result = RegionList()
    result.fallback = True
    result.append(RegionInfo(
        type="text",
        content=full_text or "",
        top=0.0, left=0.0,
        width=100.0, height=100.0,
        index=0,
    ))
    return result


def _validate_region(raw: dict, index: int) -> Optional[RegionInfo]:
    """Convert a raw dict to RegionInfo; return None if invalid."""
    try:
        rtype = str(raw.get("type", "text")).lower().strip()
        if rtype not in ("text", "equation", "diagram"):
            rtype = "text"

        content = raw.get("content")
        if content is not None:
            content = str(content).strip() or None

        top    = float(raw.get("top",    0))
        left   = float(raw.get("left",   0))
        width  = float(raw.get("width",  100))
        height = float(raw.get("height", 10))

        # Clamp to [0, 100]
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


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def detect_regions(image_bgr: np.ndarray, groq_client) -> RegionList:
    """
    Send *image_bgr* to Groq Llama-4-Scout and return a RegionList.

    Parameters
    ----------
    image_bgr   : OpenCV BGR image (the warped / cropped document page).
    groq_client : An initialised ``groq.Groq`` client instance.

    Returns
    -------
    RegionList
        List of RegionInfo in reading order.
        ``result.fallback`` is True when JSON parsing failed and the
        entire page was returned as a single text region.
    """
    b64 = _image_to_b64(image_bgr)

    raw_response = ""
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.1,
        )
        raw_response = response.choices[0].message.content or ""
    except Exception as exc:
        # Network / API error → fallback
        return _build_fallback(f"[API error: {exc}]")

    # ── First parse attempt ──────────────────────────────────
    raw_regions: List[dict] = []
    try:
        raw_regions = _parse_json_response(raw_response)
    except Exception:
        # ── Second attempt: try to find a JSON object inside the text ──
        try:
            match = re.search(r'\{.*"regions"\s*:.*\}', raw_response, re.DOTALL)
            if match:
                raw_regions = _parse_json_response(match.group(0))
            else:
                raise ValueError("no JSON found")
        except Exception:
            # Give up and fall back to the whole-page-as-text approach
            return _build_fallback(raw_response.strip())

    if not raw_regions:
        return _build_fallback(raw_response.strip())

    result = RegionList()
    result.fallback = False
    for i, raw in enumerate(raw_regions):
        region = _validate_region(raw, index=i)
        if region is not None:
            result.append(region)

    if not result:
        return _build_fallback(raw_response.strip())

    return result
