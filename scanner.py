from typing import Any

import cv2
import numpy as np


def uploaded_file_to_cv2_image(uploaded_file: Any) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile (from file_uploader) into an OpenCV BGR image.

    Parameters
    ----------
    uploaded_file : Any
        The file-like object returned by Streamlit's file_uploader.

    Returns
    -------
    np.ndarray
        Decoded BGR image.

    Raises
    ------
    ValueError
        If the image cannot be decoded.
    """
    file_bytes = uploaded_file.getvalue()
    np_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode uploaded image.")
    return image


def camera_input_to_cv2_image(camera_input_file: Any) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile (from camera_input) into an OpenCV BGR image.

    Parameters
    ----------
    camera_input_file : Any
        The file-like object returned by Streamlit's camera_input.

    Returns
    -------
    np.ndarray
        Decoded BGR image.

    Raises
    ------
    ValueError
        If the image cannot be decoded.
    """
    # camera_input returns an UploadedFile with image bytes
    file_bytes = camera_input_file.getvalue()
    np_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode webcam image.")
    return image