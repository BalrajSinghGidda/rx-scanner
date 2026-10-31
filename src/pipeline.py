"""End-to-end image analysis pipeline and command-line interface."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.nlp import (
    BrandIngredientMatch,
    clean_text,
    load_vocabulary,
    match_brands,
    match_drugs,
    tokenize,
)
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
    ingredients: list[str] = field(default_factory=list)
    brands: list[BrandIngredientMatch] = field(default_factory=list)


def merge_ingredient_matches(
    matches: list[dict], brands: list[BrandIngredientMatch]
) -> list[dict]:
    """Include known brand ingredients in the matched-drug results."""
    merged = list(matches)
    known_names = {match["matched"].casefold() for match in merged}
    for brand in brands:
        for ingredient in brand.ingredients:
            if ingredient.casefold() in known_names:
                continue
            merged.append(
                {
                    "raw": ingredient,
                    "matched": ingredient,
                    "score": 100,
                    "source": "known active ingredient",
                    "brand": brand.brand,
                }
            )
            known_names.add(ingredient.casefold())
    return merged


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
    matches = [
        match
        for match in match_drugs(tokens, load_vocabulary(), threshold=85)
        if len(match["raw"]) >= 6
    ]
    brands = match_brands(cleaned_text)
    brand_tokens = {
        token
        for brand in brands
        for token in tokenize(brand.brand)
        if len(token) >= 6
    }
    matches = [match for match in matches if match["raw"] not in brand_tokens]
    ingredients = sorted(
        {ingredient for brand in brands for ingredient in brand.ingredients}
    )
    matches = merge_ingredient_matches(matches, brands)
    prediction_text = " ".join(
        [match["matched"] for match in matches] + ingredients
    )
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
        ingredients,
        brands,
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
    print("Known active ingredients:")
    for brand in result.brands:
        print(f"- {brand.brand}: {', '.join(brand.ingredients)}")
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
