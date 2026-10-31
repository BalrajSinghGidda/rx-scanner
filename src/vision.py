"""OpenCV preprocessing and text-region detection."""

from dataclasses import dataclass

import cv2
import numpy as np

MIN_BOX_WIDTH = 40
MIN_BOX_HEIGHT = 12


def _remove_table_lines(binary: np.ndarray) -> np.ndarray:
    """Remove long rules so table borders do not become text boxes."""
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(30, binary.shape[1] // 8), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, binary.shape[0] // 8))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    return cv2.bitwise_and(binary, cv2.bitwise_not(cv2.bitwise_or(horizontal, vertical)))


@dataclass
class VisionResult:
    """Images and detected text boxes from preprocessing."""

    gray: np.ndarray
    binary: np.ndarray
    closed: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    annotated: np.ndarray


def preprocess(img_bgr: np.ndarray) -> VisionResult:
    """Detect likely printed text regions in a BGR image."""
    if not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if img_bgr.ndim not in (2, 3):
        raise ValueError("image must have two or three dimensions")

    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    line_free = _remove_table_lines(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(line_free, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.boundingRect(contour)[2] > MIN_BOX_WIDTH
        and cv2.boundingRect(contour)[3] > MIN_BOX_HEIGHT
    ]
    boxes.sort(key=lambda box: (box[1], box[0]))
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, width, height in boxes:
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return VisionResult(gray, binary, closed, boxes, annotated)
