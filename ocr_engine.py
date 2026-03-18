from dataclasses import dataclass
import base64
import os

import cv2
import numpy as np
import streamlit as st
from groq import Groq


@dataclass
class OCREngine:
    client: object = None
    fallback_reader: object = None
    backend: str = "none"

    def extract_text(self, image: np.ndarray) -> str:
        if self.client is not None:
            return self._extract_with_groq(image)
        elif self.fallback_reader is not None:
            return self._extract_with_easyocr(image)
        else:
            raise RuntimeError("No OCR backend available.")

    def _extract_with_groq(self, image: np.ndarray) -> str:
        try:
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            _, buffer = cv2.imencode(".jpg", image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
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

    def _extract_with_easyocr(self, image: np.ndarray) -> str:
        try:
            results = self.fallback_reader.readtext(image, detail=0, paragraph=True)
            cleaned_lines = [r.strip() for r in results if isinstance(r, str) and r.strip()]
            return "\n".join(cleaned_lines)
        except Exception as exc:
            raise RuntimeError(f"EasyOCR extraction failed: {exc}") from exc


@st.cache_resource(show_spinner="Loading OCR engine...")
def build_ocr_engine() -> OCREngine:
    api_key = os.environ.get("GROQ_API_KEY")

    if api_key:
        try:
            client = Groq(api_key=api_key)
            return OCREngine(client=client, backend="groq")
        except Exception as exc:
            pass

    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        return OCREngine(fallback_reader=reader, backend="easyocr")
    except ImportError:
        raise RuntimeError("No OCR backend available. Set GROQ_API_KEY or install easyocr.")