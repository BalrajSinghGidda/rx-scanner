import cv2
import numpy as np

from src.nlp import clean_text, match_brands, match_drugs, tokenize
from src.pipeline import analyze, merge_ingredient_matches
from src.online_lookup import OnlineLookupResult
from src.vision import preprocess


def test_nlp_cleans_tokenizes_and_matches() -> None:
    assert clean_text("Amoxicillin!") == "amoxicillin"
    assert tokenize("The amoxicillin 10mg") == ["amoxicillin", "10mg"]
    result = match_drugs(["amoxiciliin"], ["amoxicillin"])
    assert result[0]["matched"] == "amoxicillin"
    assert match_drugs(["tab", "acid"], ["Lortab", "Acetohydroxamic acid"]) == []
    assert match_brands("Tab. Telvas 40 mg")[0].ingredients == ["telmisartan"]


def test_known_ingredients_are_matched_drugs() -> None:
    brands = match_brands("Telvas 40 mg Glycomet GP")
    matches = merge_ingredient_matches(
        [{"raw": "telmisartan", "matched": "telmisartan", "score": 92}],
        brands,
    )
    assert [match["matched"] for match in matches] == [
        "telmisartan",
        "glimepiride",
        "metformin hydrochloride",
    ]
    assert matches[1]["source"] == "known active ingredient"
    assert matches[1]["brand"] == "Glycomet GP"


def test_vision_returns_sorted_boxes() -> None:
    image = np.full((120, 300, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Pain", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "Drug", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    result = preprocess(image)
    assert result.gray.shape == image.shape[:2]
    assert result.annotated.shape == image.shape
    assert result.boxes == sorted(result.boxes, key=lambda box: (box[1], box[0]))


def test_vision_does_not_return_full_table_rules_as_boxes() -> None:
    image = np.full((180, 400, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 30), (390, 30), (0, 0, 0), 2)
    cv2.line(image, (10, 150), (390, 150), (0, 0, 0), 2)
    cv2.putText(image, "Telvas 40 mg", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    result = preprocess(image)
    assert all(height < 100 for _, _, _, height in result.boxes)


def test_pipeline_returns_structured_result() -> None:
    image = np.full((120, 300, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Aspirin", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    result = analyze(image)
    assert result.vision.gray.shape == image.shape[:2]
    assert isinstance(result.raw_lines, list)


def test_online_lookup_result_is_source_attributed() -> None:
    result = OnlineLookupResult("x", "X", ["ingredient"], "Provider", "https://example.test")
    assert result.citation_url.startswith("https://")
