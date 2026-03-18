import io
from typing import List

from docx import Document
from docx.shared import Pt, RGBColor

from structure_detector import TextBlock, detect_structure


def create_docx_document(page_texts: List[str]) -> io.BytesIO:
    document = Document()

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

    if not page_texts:
        document.add_paragraph("No content extracted.")
    else:
        for page_index, text in enumerate(page_texts):
            page_text = text.strip()
            if not page_text:
                document.add_paragraph(f"(Page {page_index + 1} is empty.)")
            else:
                blocks: List[TextBlock] = detect_structure(page_text)
                if not blocks:
                    document.add_paragraph(page_text)
                else:
                    _render_blocks(document, blocks)

            if page_index < len(page_texts) - 1:
                document.add_page_break()

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer


def _render_blocks(document: Document, blocks: List[TextBlock]) -> None:
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