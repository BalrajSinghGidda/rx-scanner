"""Tesseract OCR helpers."""

import cv2
import numpy as np
import pytesseract


def _read(image: np.ndarray, psm: int = 6) -> str:
    """Read one grayscale image after a modest upscale."""
    enlarged = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(enlarged, config=f"--oem 3 --psm {psm}")


def extract_text(gray: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[str]:
    """Extract text from grayscale boxes, or from the whole image if needed."""
    if gray.size == 0:
        return []
    texts: list[str] = []
    height, width = gray.shape[:2]
    for x, y, box_width, box_height in boxes:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + box_width), min(height, y + box_height)
        if x2 > x1 and y2 > y1:
            text = _read(gray[y1:y2, x1:x2], psm=6).strip()
            if text:
                texts.append(text)
    if not texts:
        whole = _read(gray, psm=6).strip()
        if whole:
            texts.append(whole)
    return texts
