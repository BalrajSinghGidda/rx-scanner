"""OCR text cleaning and fuzzy drug-name matching."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from nltk.corpus import stopwords
from rapidfuzz import fuzz, process

STOP_WORDS = set(stopwords.words("english"))
BRAND_PATH = Path("models/brand_ingredients.json")
NON_DRUG_TOKENS = {
    "acid",
    "after",
    "cap",
    "chief",
    "cot",
    "clinical",
    "complaints",
    "diagnosis",
    "days",
    "dosage",
    "duration",
    "fasting",
    "fever",
    "findings",
    "food",
    "headache",
    "malaria",
    "morning",
    "name",
    "night",
    "patient",
    "prescription",
    "regular",
    "restriction",
    "sample",
    "salt",
    "tab",
    "test",
    "tot",
}


def clean_text(text: str) -> str:
    """Lowercase text and remove non-alphanumeric characters."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text)).strip().lower()


def tokenize(text: str) -> list[str]:
    """Tokenize cleaned text while dropping stop-words and short tokens."""
    return [
        token
        for token in clean_text(text).split()
        if len(token) >= 3 and token not in STOP_WORDS
    ]


def match_drugs(
    tokens: list[str], vocab: list[str], threshold: int = 85
) -> list[dict]:
    """Fuzzy-match each token against the known drug vocabulary."""
    matches: list[dict] = []
    choices = [item for item in vocab if item and item.strip()]
    for token in tokens:
        if not any(character.isalpha() for character in token):
            continue
        if token in NON_DRUG_TOKENS:
            continue
        result = process.extractOne(token, choices, scorer=fuzz.WRatio)
        if result is not None:
            matched, score, _ = result
            if score >= threshold:
                matches.append(
                    {"raw": token, "matched": matched.strip(), "score": round(score)}
                )
    return matches


@dataclass(frozen=True)
class BrandIngredientMatch:
    """A recognized brand and its known active ingredients."""

    brand: str
    ingredients: list[str]


def match_brands(text: str, path: Path = BRAND_PATH) -> list[BrandIngredientMatch]:
    """Find known brand names in cleaned OCR text."""
    if not path.exists():
        return []
    brands = json.loads(path.read_text())
    normalized = clean_text(text)
    matches = []
    for key, details in brands.items():
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            matches.append(
                BrandIngredientMatch(
                    brand=details["display_name"],
                    ingredients=details["ingredients"],
                )
            )
    return matches


def load_vocabulary(path: Path = Path("models/vocab.json")) -> list[str]:
    """Load the drug vocabulary generated during training."""
    import json

    vocabulary = json.loads(path.read_text())
    rxnorm_path = path.with_name("rxnorm_vocab.json")
    if rxnorm_path.exists():
        vocabulary.extend(json.loads(rxnorm_path.read_text()))
    return sorted(set(vocabulary))
