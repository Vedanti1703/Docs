# Handwritten Notes Scanner

Convert handwritten notes from images or webcam in real time into editable digital text, and export the result as a Word (`.docx`) document.

## Features

- **Image Upload**: Upload one or more images of handwritten notes.
- **Webcam Scanning**: Capture a page directly using your laptop webcam.
- **Image Preprocessing (OpenCV)**:
  - Grayscale conversion
  - Noise reduction
  - Adaptive thresholding
  - Document edge detection
  - Perspective correction (page warping)
- **OCR (EasyOCR + PyTorch)**: Extract handwritten text from processed images.
- **Multiple Page Support**: Process multiple pages and store text for each.
- **Document Compilation**: Combine all page texts with page breaks.
- **Export to Word**: Download the compiled notes as a `.docx` file.
- **Text Preview & Editing**: Review and edit extracted text in the UI before export.

## Tech Stack

- Python
- [OpenCV](https://opencv.org/) for image processing
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) (PyTorch backend) for OCR
- [Streamlit](https://streamlit.io/) for the web UI
- [python-docx](https://python-docx.readthedocs.io/) for Word document creation
- NumPy for array and image operations

## Project Structure

```text
handwritten_scanner/
├── app.py              # Streamlit UI
├── scanner.py          # Image & webcam input handling
├── preprocessing.py    # OpenCV preprocessing and document detection
├── ocr_engine.py       # EasyOCR initialization and text extraction
├── document_export.py  # DOCX generation
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation