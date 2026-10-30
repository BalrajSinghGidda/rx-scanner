"""Offline medicine-category prediction."""

from pathlib import Path

import joblib

MODEL_PATH = Path("models/model.joblib")


def classify(text: str) -> tuple[str, float]:
    """Return the predicted category and probability-based confidence."""
    if not text.strip():
        raise ValueError("cannot classify empty text")
    model = joblib.load(MODEL_PATH)
    category = str(model.predict([text])[0])
    confidence = float(model.predict_proba([text])[0].max())
    return category, confidence
