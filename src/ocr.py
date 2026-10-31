"""Tesseract OCR helpers."""

import cv2
import numpy as np
import pytesseract


def _remove_table_lines(image: np.ndarray) -> np.ndarray:
    """Remove long table rules that can distract OCR."""
    binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(30, image.shape[1] // 4), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, image.shape[0] // 5))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    lines = cv2.bitwise_or(horizontal, vertical)
    return cv2.bitwise_not(cv2.bitwise_and(binary, cv2.bitwise_not(lines)))


def _read(image: np.ndarray, psm: int = 6) -> str:
    """Read one grayscale image after line removal and upscaling."""
    cleaned = _remove_table_lines(image)
    enlarged = cv2.resize(cleaned, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(enlarged, config=f"--oem 3 --psm {psm}")


def _text_quality(text: str) -> int:
    """Score OCR output by the amount of readable alphanumeric text."""
    return sum(character.isalnum() for character in text)


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
    whole = _read(gray, psm=6).strip()
    crop_text = "\n".join(texts)
    if whole and _text_quality(whole) >= _text_quality(crop_text):
        return [whole]
    if texts:
        return texts
    return [whole] if whole else []
