"""OpenCV preprocessing and text-region detection."""

from dataclasses import dataclass

import cv2
import numpy as np

MIN_BOX_WIDTH = 40
MIN_BOX_HEIGHT = 12


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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
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
