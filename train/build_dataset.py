"""Explore raw condition labels and prepare the labelled dataset."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
REPORTS_DIR = Path("reports")
CONDITION_COUNTS_PATH = REPORTS_DIR / "condition_counts.csv"
PROCESSED_DIR = Path("data/processed")
DATASET_PATH = PROCESSED_DIR / "drug_categories.csv"
MIN_PURITY = 0.70
MIN_DRUGS_PER_CATEGORY = 20

CONDITION_TO_CATEGORY = {
    "Pain": "Painkiller",
    "Migraine": "Painkiller",
    "Headache": "Painkiller",
    "Fibromyalgia": "Painkiller",
    "Chronic Pain": "Painkiller",
    "Osteoarthritis": "Painkiller",
    "Rheumatoid Arthritis": "Painkiller",
    "Back Pain": "Painkiller",
    "Neuropathic Pain": "Painkiller",
    "Muscle Spasm": "Painkiller",
    "Migraine Prevention": "Painkiller",
    "Urinary Tract Infection": "Antibiotic",
    "Bacterial Infection": "Antibiotic",
    "Sinusitis": "Antibiotic",
    "Bronchitis": "Antibiotic",
    "Chlamydia Infection": "Antibiotic",
    "Pneumonia": "Antibiotic",
    "Skin or Soft Tissue Infection": "Antibiotic",
    "Helicobacter Pylori Infection": "Antibiotic",
    "Bladder Infection": "Antibiotic",
    "Allergic Rhinitis": "Antihistamine / Allergy",
    "Urticaria": "Antihistamine / Allergy",
    "Hay Fever": "Antihistamine / Allergy",
    "Cough and Nasal Congestion": "Antihistamine / Allergy",
    "Nasal Congestion": "Antihistamine / Allergy",
    "Motion Sickness": "Antihistamine / Allergy",
    "High Blood Pressure": "Antihypertensive",
    "Atrial Fibrillation": "Antihypertensive",
    "Diabetes, Type 2": "Antidiabetic",
    "Depression": "Psychiatric",
    "Anxiety": "Psychiatric",
    "Bipolar Disorder": "Psychiatric",
    "Anxiety and Stress": "Psychiatric",
    "Major Depressive Disorder": "Psychiatric",
    "Panic Disorder": "Psychiatric",
    "Generalized Anxiety Disorder": "Psychiatric",
    "Social Anxiety Disorder": "Psychiatric",
    "Schizophrenia": "Psychiatric",
    "Schizoaffective Disorder": "Psychiatric",
    "Obsessive Compulsive Disorder": "Psychiatric",
    "Post Traumatic Stress Disorder": "Psychiatric",
    "Birth Control": "Contraceptive",
    "Emergency Contraception": "Contraceptive",
}


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
    cleaned["condition"] = condition.loc[cleaned.index].replace(
        {
            "ibromyalgia": "Fibromyalgia",
            "Bipolar Disorde": "Bipolar Disorder",
            "Major Depressive Disorde": "Major Depressive Disorder",
            "Panic Disorde": "Panic Disorder",
            "Generalized Anxiety Disorde": "Generalized Anxiety Disorder",
            "Social Anxiety Disorde": "Social Anxiety Disorder",
            "Schizoaffective Disorde": "Schizoaffective Disorder",
            "Obsessive Compulsive Disorde": "Obsessive Compulsive Disorder",
            "Post Traumatic Stress Disorde": "Post Traumatic Stress Disorder",
        }
    )
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


def build_labelled_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign pure drug-level category labels and filter small categories."""
    mapped = frame.assign(category=frame["condition"].map(CONDITION_TO_CATEGORY))
    mapped = mapped.dropna(subset=["category"])
    category_counts = mapped.groupby(["drugName", "category"]).size()
    drug_totals = category_counts.groupby(level="drugName").sum()
    winning = category_counts.groupby(level="drugName").idxmax()
    winner_counts = category_counts.groupby(level="drugName").max()
    purity = winner_counts / drug_totals
    labels = pd.DataFrame(
        {
            "category": winning.map(lambda value: value[1]),
            "purity": purity,
        }
    )
    labels = labels[labels["purity"] >= MIN_PURITY].reset_index(names="drugName")
    category_sizes = labels.groupby("category")["drugName"].size()
    kept_categories = category_sizes[category_sizes >= MIN_DRUGS_PER_CATEGORY].index
    labels = labels[labels["category"].isin(kept_categories)]
    result = labels[["drugName", "category"]].sort_values("drugName").reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(DATASET_PATH, index=False)
    return result


def main() -> None:
    """Load raw reviews, clean labels, and save condition frequencies."""
    cleaned = clean_conditions(load_raw_reviews())
    counts = save_condition_counts(cleaned)
    dataset = build_labelled_dataset(cleaned)
    print(counts.to_string(index=False))
    print(f"\nSaved {CONDITION_COUNTS_PATH}")
    print(f"Saved {DATASET_PATH}: {len(dataset)} drugs")
    print(dataset["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
