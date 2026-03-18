import io
from typing import List

import cv2
import numpy as np
import streamlit as st

from document_export import create_docx_document
from ocr_engine import build_ocr_engine
from preprocessing import detect_and_warp_document, preprocess_for_ocr
from scanner import uploaded_file_to_cv2_image, camera_input_to_cv2_image


def init_session_state() -> None:
    """
    Initialize Streamlit session state keys used to store page texts.
    """
    if "page_texts" not in st.session_state:
        st.session_state["page_texts"] = []  # List[str]


def process_image(image_bgr: np.ndarray, ocr_engine) -> str:
    """
    Run full pipeline on a single image:
    - Detect and warp document.
    - Preprocess for OCR.
    - Extract text using OCR engine.

    Returns extracted text (possibly empty string on failure).
    """
    try:
        # Detect and warp the document region
        warped = detect_and_warp_document(image_bgr)

        # Preprocess for OCR (grayscale, threshold, denoise)
        preprocessed = preprocess_for_ocr(warped)

        # OCR expects RGB or grayscale. Our preprocessed image is single-channel.
        text = ocr_engine.extract_text(preprocessed)
        return text.strip()
    except Exception as exc:
        st.error(f"Failed to process image: {exc}")
        return ""


def add_pages_to_session(texts: List[str]) -> None:
    """
    Append non-empty page texts to the session state list.
    """
    for text in texts:
        cleaned = text.strip()
        if cleaned:
            st.session_state["page_texts"].append(cleaned)


def main() -> None:
    """
    Main Streamlit application entry point.
    Provides UI for:
    - Image upload
    - Webcam scanning
    - Text preview
    - DOCX export
    """
    st.set_page_config(page_title="Handwritten Notes Scanner", layout="wide")
    st.title("📝 Handwritten Notes Scanner")
    st.write(
        "Upload handwritten note images or scan via webcam, "
        "convert them to editable text, and export as a Word document."
    )

    init_session_state()

    # Build and cache OCR engine (reused for all pages to improve performance)
    ocr_engine = build_ocr_engine()

    tab_upload, tab_webcam, tab_review = st.tabs(
        ["📁 Upload Images", "📷 Webcam Scan", "📄 Review & Export"]
    )

    with tab_upload:
        st.subheader("Upload Handwritten Note Images")

        uploaded_files = st.file_uploader(
            "Select one or more images",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.info(
                "After selecting images, click **Process Uploaded Images** "
                "to extract text from all of them."
            )

        if st.button("Process Uploaded Images") and uploaded_files:
            new_page_texts: List[str] = []
            progress_bar = st.progress(0)
            total = len(uploaded_files)

            for idx, file in enumerate(uploaded_files, start=1):
                st.write(f"Processing image {idx} of {total} ...")
                image_bgr = uploaded_file_to_cv2_image(file)
                text = process_image(image_bgr, ocr_engine)
                if text:
                    new_page_texts.append(text)
                progress_bar.progress(idx / total)

            add_pages_to_session(new_page_texts)

            if new_page_texts:
                st.success(f"Extracted text from {len(new_page_texts)} page(s).")
            else:
                st.warning("No text extracted from uploaded images.")

    with tab_webcam:
        st.subheader("Scan Page Using Webcam")

        st.write(
            "Use your webcam to capture an image of your handwritten page. "
            "Ensure good lighting and the page fills most of the frame."
        )

        camera_image = st.camera_input("Capture handwritten page")

        if camera_image is not None:
            st.image(camera_image, caption="Captured Image", use_column_width=True)

        if st.button("Extract Text from Webcam Capture") and camera_image is not None:
            image_bgr = camera_input_to_cv2_image(camera_image)
            text = process_image(image_bgr, ocr_engine)
            if text:
                add_pages_to_session([text])
                st.success("Extracted text from webcam image and added as a new page.")
                st.text_area("Extracted Text (Latest Page)", value=text, height=200)
            else:
                st.warning("Failed to extract text from the webcam capture.")

    with tab_review:
        st.subheader("Preview Extracted Text")

        page_texts: List[str] = st.session_state.get("page_texts", [])

        if not page_texts:
            st.info(
                "No pages have been processed yet. "
                "Upload images or scan via webcam first."
            )
        else:
            st.write(
                "Review the extracted text below. "
                "You can edit any page before exporting to Word."
            )

            # Allow users to edit individual pages
            updated_page_texts: List[str] = []
            for i, page_text in enumerate(page_texts, start=1):
                edited_text = st.text_area(
                    f"Page {i} Text",
                    value=page_text,
                    height=200,
                    key=f"page_{i}_text",
                )
                updated_page_texts.append(edited_text)

            # Replace with edited content
            st.session_state["page_texts"] = updated_page_texts

            # Option to clear all pages
            if st.button("Clear All Pages"):
                st.session_state["page_texts"] = []
                st.success("Cleared all stored pages.")

            # Export section
            if st.session_state["page_texts"]:
                st.subheader("Export to Word Document")

                if st.button("Generate Word Document"):
                    try:
                        docx_bytes: io.BytesIO = create_docx_document(
                            st.session_state["page_texts"]
                        )
                        st.success("Word document generated successfully.")

                        st.download_button(
                            label="Download Word Document",
                            data=docx_bytes.getvalue(),
                            file_name="handwritten_notes.docx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "wordprocessingml.document"
                            ),
                        )
                    except Exception as exc:
                        st.error(f"Failed to generate Word document: {exc}")


if __name__ == "__main__":
    main()