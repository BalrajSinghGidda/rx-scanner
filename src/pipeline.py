"""End-to-end image analysis pipeline and command-line interface."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.nlp import clean_text, load_vocabulary, match_drugs, tokenize
from src.ocr import extract_text
from src.predict import classify
from src.vision import VisionResult, preprocess


@dataclass
class AnalysisResult:
    """Structured output from one image analysis."""

    vision: VisionResult
    raw_lines: list[str]
    cleaned_text: str
    matches: list[dict]
    category: str | None
    confidence: float | None
    warnings: list[str]


def analyze(img_bgr) -> AnalysisResult:
    """Analyze an image without requiring Streamlit."""
    warnings: list[str] = []
    if img_bgr is None:
        raise ValueError("image could not be loaded")
    vision = preprocess(img_bgr)
    if not vision.boxes:
        warnings.append("No text regions detected; used whole-image OCR.")
    raw_lines = extract_text(vision.gray, vision.boxes)
    cleaned_text = clean_text(" ".join(raw_lines))
    tokens = tokenize(cleaned_text)
    matches = match_drugs(tokens, load_vocabulary())
    prediction_text = " ".join(match["matched"] for match in matches) or cleaned_text
    category = confidence = None
    if prediction_text:
        category, confidence = classify(prediction_text)
        if confidence < 0.50:
            warnings.append("Prediction confidence is below 50%.")
    else:
        warnings.append("No readable text found.")
    return AnalysisResult(
        vision,
        raw_lines,
        cleaned_text,
        matches,
        category,
        confidence,
        warnings,
    )


def main() -> None:
    """Run the pipeline from an image path."""
    parser = argparse.ArgumentParser(description="Analyze a medicine image.")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    result = analyze(image)
    print("Raw OCR:")
    print("\n".join(result.raw_lines) or "(none)")
    print("Matched drugs:")
    for match in result.matches:
        print(f"- {match['raw']} -> {match['matched']} ({match['score']})")
    print(f"Category: {result.category or '(none)'}")
    print(
        f"Confidence: {result.confidence:.3f}"
        if result.confidence is not None
        else "Confidence: (none)"
    )
    for warning in result.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
