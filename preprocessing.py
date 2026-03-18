from typing import Tuple

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points (x, y) in the following order:
    top-left, top-right, bottom-right, bottom-left.

    This is used for perspective transform.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform to obtain a top-down view of the document.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    pts : np.ndarray
        Array of four corner points of the document.

    Returns
    -------
    np.ndarray
        Warped (top-down) BGR image of the document.
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of the new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    # Compute height of the new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    # Destination points for transform
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # Perspective transform
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))

    return warped


def detect_and_warp_document(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detect the largest quadrilateral in the image (assumed to be the document)
    and warp it to a top-down view.

    If detection fails, returns the original image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        Warped BGR image of the document or original image if detection fails.
    """
    try:
        # Resize for easier processing while preserving aspect ratio
        ratio = image_bgr.shape[0] / 500.0
        small = cv2.resize(
            image_bgr, (int(image_bgr.shape[1] / ratio), 500), interpolation=cv2.INTER_AREA
        )

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screen_contour = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                screen_contour = approx
                break

        if screen_contour is None:
            # No document contour found; return original image
            return image_bgr

        # Transform back to original image size coordinates
        pts = screen_contour.reshape(4, 2) * ratio
        warped = _four_point_transform(image_bgr, pts)
        return warped
    except Exception:
        # On any failure, fall back to original
        return image_bgr


def preprocess_for_ocr(image_bgr_or_gray: np.ndarray) -> np.ndarray:
    """
    Preprocess an image to enhance readability for OCR.

    Steps:
    - Convert to grayscale if needed.
    - Apply Gaussian blur to reduce noise.
    - Apply adaptive thresholding to create a high-contrast binary image.
    - Optionally apply morphological operations to clean small specks.

    Parameters
    ----------
    image_bgr_or_gray : np.ndarray
        Input BGR or grayscale image.

    Returns
    -------
    np.ndarray
        Preprocessed single-channel image suitable for OCR.
    """
    if len(image_bgr_or_gray.shape) == 3 and image_bgr_or_gray.shape[2] == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray.copy()

    # Mild blur to remove small noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding for varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )

    # Invert if background is dark and text is light
    # Decide by checking mean pixel value
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Morphological opening to remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned