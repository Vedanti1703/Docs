"""
document_export.py
──────────────────
Assembles a python-docx Word document from a list of processed page results.

New API (Improvement 1 — mixed text + image regions):
    create_docx_document(page_data: list[dict]) -> io.BytesIO

    Each dict must have:
        "regions"      : list[RegionInfo]   — ordered regions from region_detector
        "original_bgr" : np.ndarray         — original colour image for cropping

Backward-compatible legacy API (plain-text pages):
    create_docx_document(page_texts: list[str]) -> io.BytesIO

The function detects which form was passed and dispatches accordingly.
"""

from __future__ import annotations

import io
from typing import List, Union

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from region_detector import RegionInfo
from structure_detector import TextBlock, detect_structure
from text_postprocessor import postprocess_text


# ────────────────────────────────────────
# Style helpers
# ────────────────────────────────────────

def _apply_document_styles(document: Document) -> None:
    """Apply global font and colour settings."""
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    h1 = document.styles["Heading 1"]
    h1.font.name = "Calibri"
    h1.font.size = Pt(16)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    h2 = document.styles["Heading 2"]
    h2.font.name = "Calibri"
    h2.font.size = Pt(13)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)


def _render_blocks(document: Document, blocks: List[TextBlock]) -> None:
    """Write a list of TextBlocks to the document (unchanged from original)."""
    for block in blocks:
        if block.block_type == "heading1":
            document.add_heading(block.text, level=1)
        elif block.block_type == "heading2":
            document.add_heading(block.text, level=2)
        elif block.block_type == "bullet":
            para = document.add_paragraph(style="List Bullet")
            para.add_run(block.text)
        elif block.block_type == "numbered":
            para = document.add_paragraph(style="List Number")
            para.add_run(block.text)
        else:
            para = document.add_paragraph()
            para.add_run(block.text)
            para.paragraph_format.space_after = Pt(6)


def _add_image_region(
    document: Document,
    region: RegionInfo,
    original_bgr: np.ndarray,
    max_width_inches: float = 5.5,
) -> None:
    """
    Crop *region* from *original_bgr*, save as PNG to memory, and insert
    it into the document centered with a caption.
    """
    from image_cropper import crop_region, region_to_png_bytes

    try:
        crop = crop_region(original_bgr, region)
        if crop is None or crop.size == 0:
            return

        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return

        png_bio = region_to_png_bytes(crop)

        # Calculate display width preserving aspect ratio
        aspect = w / h if h else 1.0
        display_w = min(max_width_inches, max_width_inches)  # always full width
        display_h = display_w / aspect

        # Clamp height to reasonable max
        if display_h > 4.0:
            display_h = 4.0
            display_w = display_h * aspect
            display_w = min(display_w, max_width_inches)

        # Spacing above image
        spacer_before = document.add_paragraph()
        spacer_before.paragraph_format.space_before = Pt(8)
        spacer_before.paragraph_format.space_after  = Pt(2)

        # Centred image paragraph
        img_para = document.add_paragraph()
        img_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_para.paragraph_format.space_before = Pt(4)
        img_para.paragraph_format.space_after  = Pt(4)
        run = img_para.add_run()
        run.add_picture(png_bio, width=Inches(display_w))

        # Caption
        caption_text = "Equation" if region.type == "equation" else "Figure"
        cap_para = document.add_paragraph()
        cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_para.paragraph_format.space_after = Pt(10)
        cap_run = cap_para.add_run(caption_text)
        cap_run.italic = True
        cap_run.font.size = Pt(9)
        cap_run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    except Exception:
        # If image insertion fails, add a placeholder text
        label = "[ Equation ]" if region.type == "equation" else "[ Figure / Diagram ]"
        para = document.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(label)
        run.italic = True
        run.font.color.rgb = RGBColor(0x9C, 0xA3, 0xAF)


# ────────────────────────────────────────
# Page assemblers
# ────────────────────────────────────────

def _assemble_region_page(
    document: Document,
    regions: List[RegionInfo],
    original_bgr: np.ndarray,
    spell_check: bool = False,
) -> None:
    """Assemble one page worth of mixed regions into *document*."""
    for region in regions:
        if region.type == "text":
            raw = (region.content or "").strip()
            if not raw:
                continue
            processed = postprocess_text(raw, spell_check=spell_check)
            blocks = detect_structure(processed)
            if blocks:
                _render_blocks(document, blocks)
            else:
                para = document.add_paragraph()
                para.add_run(processed)
                para.paragraph_format.space_after = Pt(6)

        elif region.type in ("equation", "diagram"):
            _add_image_region(document, region, original_bgr)


def _assemble_legacy_page(
    document: Document,
    text: str,
    spell_check: bool = False,
) -> None:
    """Assemble a plain-text page (backward-compat path)."""
    page_text = text.strip()
    if not page_text:
        document.add_paragraph("(Empty page)")
        return
    processed = postprocess_text(page_text, spell_check=spell_check)
    blocks = detect_structure(processed)
    if blocks:
        _render_blocks(document, blocks)
    else:
        document.add_paragraph(processed)


# ────────────────────────────────────────
# Public API
# ────────────────────────────────────────

def create_docx_document(
    page_data: Union[List[dict], List[str]],
    spell_check: bool = False,
) -> io.BytesIO:
    """
    Build and return a .docx file as a BytesIO buffer.

    Parameters
    ----------
    page_data : list[dict] or list[str]
        New form  → list of dicts, each with:
            "regions"      : list[RegionInfo]
            "original_bgr" : np.ndarray
        Legacy form → list of plain-text strings (original API).
    spell_check : bool
        Whether to run spell correction on text regions.

    Returns
    -------
    io.BytesIO  (seeked to position 0, ready for download)
    """
    document = Document()
    _apply_document_styles(document)

    if not page_data:
        document.add_paragraph("No content extracted.")
    else:
        # Determine which API form was used
        is_new_api = isinstance(page_data[0], dict)

        for page_index, page in enumerate(page_data):
            if is_new_api:
                regions: List[RegionInfo] = page.get("regions", [])
                original_bgr: np.ndarray  = page.get("original_bgr", np.zeros((1, 1, 3), dtype=np.uint8))
                _assemble_region_page(document, regions, original_bgr, spell_check=spell_check)
            else:
                _assemble_legacy_page(document, str(page), spell_check=spell_check)

            # Page break between pages (not after the last one)
            if page_index < len(page_data) - 1:
                document.add_page_break()

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer