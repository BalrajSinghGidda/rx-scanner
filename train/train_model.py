"""Train and compare drug-category classifiers."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

DATASET_PATH = Path("data/processed/drug_categories.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
VOCAB_PATH = MODEL_DIR / "vocab.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset() -> pd.DataFrame:
    """Load the prepared one-row-per-drug dataset."""
    return pd.read_csv(DATASET_PATH)


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split unique drug groups into disjoint training and test rows."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(dataset, dataset["category"], groups=dataset["drugName"])
    )
    return dataset.iloc[train_idx].reset_index(drop=True), dataset.iloc[
        test_idx
    ].reset_index(drop=True)


def make_vectorizer() -> TfidfVectorizer:
    """Create the character n-gram vectorizer."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        lowercase=True,
    )


def make_models() -> dict[str, Pipeline]:
    """Create the required comparable model pipelines."""
    vectorizer = make_vectorizer()
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", vectorizer),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    """Calculate hold-out accuracy and macro-F1."""
    predictions = model.predict(test["drugName"])
    return {
        "accuracy": float(accuracy_score(test["category"], predictions)),
        "macro_f1": float(
            f1_score(test["category"], predictions, average="macro", zero_division=0)
        ),
    }


def save_best_model(model: Pipeline, dataset: pd.DataFrame) -> None:
    """Save the complete inference pipeline and its drug vocabulary."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    vocabulary = sorted(dataset["drugName"].drop_duplicates().tolist())
    VOCAB_PATH.write_text(json.dumps(vocabulary, indent=2) + "\n")


def main() -> None:
    """Train both classifiers and save the better hold-out pipeline."""
    dataset = load_dataset()
    train, test = split_dataset(dataset)
    results: dict[str, dict[str, float]] = {}
    trained: dict[str, Pipeline] = {}
    for name, model in make_models().items():
        model.fit(train["drugName"], train["category"])
        results[name] = evaluate_model(model, test)
        trained[name] = model
        print(
            f"{name}: accuracy={results[name]['accuracy']:.4f}, "
            f"macro_f1={results[name]['macro_f1']:.4f}"
        )

    best_name = max(results, key=lambda name: results[name]["macro_f1"])
    save_best_model(trained[best_name], dataset)
    print(f"Selected {best_name}")
    print(f"Saved {MODEL_PATH} and {VOCAB_PATH}")


if __name__ == "__main__":
    main()
