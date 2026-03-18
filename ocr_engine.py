from dataclasses import dataclass
from typing import List, Optional

import easyocr
import numpy as np
import streamlit as st


@dataclass
class OCREngine:
    """
    Wrapper around EasyOCR Reader for handwritten text recognition.
    """

    reader: easyocr.Reader

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from the given image using EasyOCR.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or RGB/BGR). If BGR, EasyOCR will handle it.

        Returns
        -------
        str
            Concatenated recognized text.
        """
        # EasyOCR expects RGB or grayscale; it can handle most OpenCV images directly.
        try:
            # detail=0 -> only text strings; paragraph=True -> group nearby lines
            results = self.reader.readtext(image, detail=0, paragraph=True)
            # Join non-empty segments with newlines
            cleaned_lines = [r.strip() for r in results if isinstance(r, str) and r.strip()]
            return "\n".join(cleaned_lines)
        except Exception as exc:
            raise RuntimeError(f"EasyOCR extraction failed: {exc}") from exc


@st.cache_resource(show_spinner=True)
def build_ocr_engine(
    languages: Optional[List[str]] = None,
    gpu: bool = False,
) -> OCREngine:
    """
    Build and cache an OCREngine instance using EasyOCR.

    Caching via Streamlit ensures the heavy EasyOCR model is loaded only once,
    significantly improving performance across multiple scans.

    Parameters
    ----------
    languages : list of str, optional
        List of language codes for EasyOCR. Defaults to ["en"].
    gpu : bool, optional
        Whether to use GPU acceleration for OCR. Defaults to False.

    Returns
    -------
    OCREngine
        Initialized OCR engine.
    """
    if languages is None:
        languages = ["en"]

    try:
        reader = easyocr.Reader(languages, gpu=gpu)
    except Exception as exc:
        # If GPU initialization fails, try CPU as a fallback
        if gpu:
            reader = easyocr.Reader(languages, gpu=False)
        else:
            raise RuntimeError(f"Failed to initialize EasyOCR: {exc}") from exc

    return OCREngine(reader=reader)