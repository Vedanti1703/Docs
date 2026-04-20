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


@dataclass
class OCREngine:
    client: object = None
    backend: str = "none"

    # ── plain-text extraction (original behaviour) ──────────────
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
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            _, buffer = cv2.imencode(
                ".jpg", image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            b64_image = base64.b64encode(buffer).decode("utf-8")

            response = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are a handwriting transcription expert. "
                                    "Carefully read all handwritten text in this image and transcribe it exactly as written. "
                                    "Preserve the original line breaks and paragraph structure. "
                                    "If you see headings or titles, start them on their own line. "
                                    "If you see bullet points or numbered lists, preserve that structure. "
                                    "Output ONLY the transcribed text, no explanations."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            raise RuntimeError(f"Groq extraction failed: {exc}") from exc


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