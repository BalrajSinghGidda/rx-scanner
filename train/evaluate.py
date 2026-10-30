"""Evaluate the saved drug-category pipeline."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from train.train_model import MODEL_PATH, load_dataset, split_dataset

REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"


def main() -> None:
    """Evaluate the saved pipeline on the deterministic hold-out split."""
    dataset = load_dataset()
    _, test = split_dataset(dataset)
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(test["drugName"])
    labels = sorted(dataset["category"].unique())
    report = classification_report(
        test["category"],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": type(model.named_steps["classifier"]).__name__,
        "test_rows": len(test),
        "accuracy": float(accuracy_score(test["category"], predictions)),
        "macro_f1": float(
            f1_score(test["category"], predictions, average="macro", zero_division=0)
        ),
        "per_class": report,
        "labels": labels,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    matrix = confusion_matrix(test["category"], predictions, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted category")
    plt.ylabel("True category")
    plt.title("Drug category confusion matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()

    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"Saved {METRICS_PATH} and {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
