"""
ocr_engine.py
─────────────
Groq-based OCR engine.  Kept minimal — the new region-aware flow
lives in region_detector.py.  This module retains the original
plain-text extraction path so existing code paths (webcam, legacy
review) continue to work unchanged.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Plain-text OCR prompt  (used by extract_text — whole-page transcription)
# ─────────────────────────────────────────────────────────────────────────────

_OCR_SYSTEM = (
    "You are an expert handwriting transcription specialist with deep knowledge of "
    "mathematics, science, and engineering notation. "
    "Your sole job is to read handwritten text with 100% fidelity. "
    "You NEVER paraphrase, summarise, or interpret — you transcribe character-by-character."
)

_OCR_USER = (
    "Carefully transcribe ALL handwritten text visible in this image.\n\n"
    "STRICT RULES:\n"
    "1. Reproduce every word exactly as written — spelling mistakes included.\n"
    "2. Preserve the original line-break and paragraph structure.\n"
    "3. Headings / titles → place on their own line(s).\n"
    "4. Bullet points / numbered lists → preserve the marker (•, -, 1., a. …).\n"
    "5. Mathematical expressions → use standard notation: "
    "superscripts as ^, subscripts as _, fractions as a/b, "
    "Greek letters spelled out (alpha, beta, …) or Unicode (α, β, …).\n"
    "6. Symbols, arrows, underlines, and emphasis marks → transcribe faithfully "
    "(e.g., --> for arrow, == for double underline).\n"
    "7. If a word is ambiguous, choose the most contextually plausible reading "
    "given surrounding text, then continue.\n"
    "8. Do NOT add commentary, headers, or explanatory text.\n"
    "9. Output ONLY the transcribed text — nothing else."
)


# ─────────────────────────────────────────────────────────────────────────────
# Region-detection prompt  (used by region_detector.detect_regions)
# ─────────────────────────────────────────────────────────────────────────────

REGION_DETECTION_SYSTEM = (
    "You are analyzing a handwritten academic notes image. "
    "Your job is to classify every region of content into exactly one of three "
    "types and return a JSON response."
)

REGION_DETECTION_USER = """\
You are analyzing a handwritten academic notes image. Your job is to classify every
region of content into exactly one of three types and return a JSON response.

THE THREE TYPES ARE STRICTLY DEFINED:

TYPE 1 — text:
Any content that is PRIMARILY made of written words and sentences.
This includes:
- Student name, roll number, class header
- Subject name and assignment title
- Question numbers and question text
- Explanatory paragraphs and theory
- Numbered points and bullet points
- Any sentence that a person would read
Even if a sentence MENTIONS an equation like "the line x+y=4 separates the classes"
the whole sentence is still TYPE text because it is readable as a sentence.
TEXT regions must be transcribed as typed readable text in the output.

TYPE 2 — equation:
ONLY content that is a standalone mathematical expression that exists
on its own line or multiple lines and is NOT a readable sentence.
This includes:
- A formula standing alone:
  Distance = |ax+by+c|
              √a²+b²
- A calculation standing alone:
  d = |3+3-4| = 2 = √2
       √2      √2
- A multi line derivation
- Any expression with fractions where numerator and denominator are on
  separate lines (stacked fraction)
- Any expression with √ square root that spans multiple lines
EQUATION regions must be cropped as image and inserted in Word as picture.

TYPE 3 — diagram:
Any hand drawn visual content that is NOT text and NOT a mathematical formula.
This includes:
- Flowcharts and process diagrams
- Geometric shapes and constructions
- Graphs with axes and plotted points
- Circuit diagrams
- Tree structures
- Any drawing or illustration
DIAGRAM regions must be cropped as image and inserted in Word as picture.

CRITICAL CLASSIFICATION RULES:

Rule 1 — Sentence test:
Ask yourself: Can this be read aloud as a normal English sentence?
If YES → it is TYPE text
If NO → check if equation or diagram

Rule 2 — Standalone equation test:
Is this a mathematical expression that stands alone on its own line(s)?
Does it have stacked fractions or multi line structure?
If YES → it is TYPE equation
If NO and it is in a sentence → TYPE text

Rule 3 — When in doubt:
If you cannot clearly decide between text and equation choose TYPE text.
It is better to have equation as text than to have a text paragraph as image,
because the user can read text in the document but cannot edit an image of text.

Rule 4 — Headers are always text:
Student name, roll number, subject, assignment title are always TYPE text.
Never classify a header as equation or diagram.

Rule 5 — Question labels are always text:
Q1, Q2, a), b), c), d), e) labels are always part of TYPE text regions.
Never separate them as equations.

Rule 6 — Inline equations stay in text:
If an equation appears inside a sentence like "the line x+y=4 is the answer"
keep the whole sentence as TYPE text.
Do not extract the inline equation separately.

BOUNDING BOX RULES:
- top_percent: where region starts vertically as percentage of total image height
- left_percent: where region starts horizontally as percentage of total image width
- width_percent: width of region as percentage of total image width
- height_percent: height of region as percentage of total image height
- All values between 0 and 100
- top_percent + height_percent must be <= 100
- left_percent + width_percent must be <= 100
- For equations add 2% padding on all sides
- For diagrams add 4% padding on all sides
- Regions should not overlap each other

Return ONLY this JSON with no other text:
{
  "regions": [
    {
      "type": "text",
      "content": "exact transcribed text here, preserve line breaks with \\n, transcribe every word accurately",
      "top_percent": 0,
      "left_percent": 0,
      "width_percent": 100,
      "height_percent": 10
    },
    {
      "type": "equation",
      "content": null,
      "top_percent": 12,
      "left_percent": 5,
      "width_percent": 90,
      "height_percent": 15
    },
    {
      "type": "diagram",
      "content": null,
      "top_percent": 30,
      "left_percent": 5,
      "width_percent": 90,
      "height_percent": 25
    }
  ]
}

EXAMPLE FOR THIS TYPE OF PAGE:
If the page has:
- Student name and header → text region
- Q1a question and answer paragraph → text region
- Q1b text explanation → text region
- Standalone distance formula with stacked fraction → equation region
- More calculation lines → equation region
- Q1e conclusion paragraph → text region

The equation region is ONLY the part where the formula is written with
stacked numerator and denominator on separate lines by itself.
Everything else is text region.

OUTPUT ONLY THE JSON — NOTHING ELSE.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_b64_png(image: np.ndarray) -> str:
    """
    Encode an OpenCV image (BGR or greyscale) as a lossless PNG
    base-64 string for transmission to the Groq vision model.
    """
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Upscale small images so the model can read fine detail
    h, w = rgb.shape[:2]
    if max(h, w) < 1024:
        scale = 1024 / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    success, buf = cv2.imencode(".png", rgb)
    if not success:
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 97])
    return base64.b64encode(bytes(buf)).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# OCREngine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OCREngine:
    client: object = None
    backend: str = "none"

    # ── plain-text extraction (original behaviour) ──────────────────────────
    def extract_text(self, image: np.ndarray) -> str:
        """
        Transcribe all handwritten text in *image* and return it as a
        plain string.  Raises RuntimeError if no backend is available.
        """
        if self.client is not None:
            return self._extract_with_groq(image)
        raise RuntimeError("No OCR backend available. Please set GROQ_API_KEY.")

    def _extract_with_groq(self, image: np.ndarray) -> str:
        try:
            b64 = _image_to_b64_png(image)

            response = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": _OCR_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": _OCR_USER,
                            },
                        ],
                    },
                ],
                max_tokens=4096,
                temperature=0.05,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            raise RuntimeError(f"Groq extraction failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading OCR engine…")
def build_ocr_engine() -> OCREngine:
    """Build and cache the OCR engine from the environment GROQ_API_KEY."""
    api_key = os.environ.get("GROQ_API_KEY")

    if api_key:
        try:
            client = Groq(api_key=api_key)
            return OCREngine(client=client, backend="groq")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise Groq: {exc}") from exc

    raise RuntimeError(
        "GROQ_API_KEY not set. Please add it in your .env file or Render environment variables."
    )