import io
from typing import List

from docx import Document


def create_docx_document(page_texts: List[str]) -> io.BytesIO:
    """
    Create a DOCX document from a list of page texts.

    Each item in `page_texts` becomes a separate page in the document,
    separated by Word page breaks.

    Parameters
    ----------
    page_texts : list of str
        Text content for each page, in order.

    Returns
    -------
    io.BytesIO
        In-memory DOCX file ready for download.
    """
    document = Document()

    if not page_texts:
        document.add_paragraph("No content extracted.")

    for index, text in enumerate(page_texts):
        paragraph_text = text.strip() or f"(Page {index + 1} is empty.)"
        document.add_paragraph(paragraph_text)
        # Add page break after every page except the last
        if index < len(page_texts) - 1:
            document.add_page_break()

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer