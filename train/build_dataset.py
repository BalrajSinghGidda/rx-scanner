"""Explore raw condition labels and prepare the labelled dataset."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
REPORTS_DIR = Path("reports")
CONDITION_COUNTS_PATH = REPORTS_DIR / "condition_counts.csv"


def load_raw_reviews() -> pd.DataFrame:
    """Load and combine the Kaggle train and test review files."""
    frames = [
        pd.read_csv(path)
        for path in (
            RAW_DIR / "drugsComTrain_raw.csv",
            RAW_DIR / "drugsComTest_raw.csv",
        )
    ]
    return pd.concat(frames, ignore_index=True)


def clean_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without usable condition labels."""
    cleaned = frame.dropna(subset=["drugName", "condition"]).copy()
    condition = cleaned["condition"].astype(str).str.strip()
    junk = condition.str.contains("users found this comment helpful", case=False)
    cleaned = cleaned.loc[~junk].copy()
    cleaned["condition"] = condition.loc[cleaned.index]
    return cleaned


def save_condition_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Save the 100 most common conditions and their row counts."""
    counts = (
        frame["condition"]
        .value_counts()
        .head(100)
        .rename_axis("condition")
        .reset_index(name="count")
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(CONDITION_COUNTS_PATH, index=False)
    return counts


def main() -> None:
    """Load raw reviews, clean labels, and save condition frequencies."""
    counts = save_condition_counts(clean_conditions(load_raw_reviews()))
    print(counts.to_string(index=False))
    print(f"\nSaved {CONDITION_COUNTS_PATH}")


if __name__ == "__main__":
    main()
