import cv2
import numpy as np

from src.nlp import clean_text, match_drugs, tokenize
from src.pipeline import analyze
from src.vision import preprocess


def test_nlp_cleans_tokenizes_and_matches() -> None:
    assert clean_text("Amoxicillin!") == "amoxicillin"
    assert tokenize("The amoxicillin 10mg") == ["amoxicillin", "10mg"]
    result = match_drugs(["amoxiciliin"], ["amoxicillin"])
    assert result[0]["matched"] == "amoxicillin"
    assert match_drugs(["tab", "acid"], ["Lortab", "Acetohydroxamic acid"]) == []


def test_vision_returns_sorted_boxes() -> None:
    image = np.full((120, 300, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Pain", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "Drug", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    result = preprocess(image)
    assert result.gray.shape == image.shape[:2]
    assert result.annotated.shape == image.shape
    assert result.boxes == sorted(result.boxes, key=lambda box: (box[1], box[0]))


def test_pipeline_returns_structured_result() -> None:
    image = np.full((120, 300, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Aspirin", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    result = analyze(image)
    assert result.vision.gray.shape == image.shape[:2]
    assert isinstance(result.raw_lines, list)
