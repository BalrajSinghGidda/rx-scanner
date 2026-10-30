# Project Progress

## Status

- Current phase: Phase 2 - dataset, model training, and evaluation
- Overall progress: 35%
- Next action: Train and compare the required classifiers
- Days left to 3 November 2026: 4

## Checklist

- [x] Phase 0: WSL setup, dependencies, scaffold, and environment check
- [x] Phase 1: Inspect condition values and approve category mapping
- [ ] Phase 2: Build labelled dataset, train models, and evaluate
- [ ] Phase 3: Implement vision, OCR, NLP, and analysis pipeline
- [ ] Phase 4: Build and verify the Streamlit UI
- [ ] Phase 5: Polish README, report outline, and demo script

## Decisions

- 2026-10-30: Use the repository's existing WSL virtual environment and
  `opencv-python-headless` as required by the project specification.
- 2026-10-30: Keep the initial scaffold minimal; implementation modules will be
  added phase by phase so each change can be tested and explained.
- 2026-10-30: Phase 1 will first preserve the real top-100 condition counts
  before any condition-to-category mapping is applied.
- 2026-10-30: Approved seven-category mapping: Painkiller, Antibiotic,
  Antihistamine / Allergy, Antihypertensive, Antidiabetic, Psychiatric, and
  Contraceptive. Normalize truncated dataset labels ending in `Disorde` to
  `Disorder` when defining the mapping; skip ambiguous or unsupported labels.
- 2026-10-30: Apply a 70% per-drug label-purity threshold and retain only
  categories with at least 20 unique drugs, as required by the specification.

## Results

- Environment checks passed: Python 3.14.4, Tesseract 5.5.0, required Python
  imports, and NLTK English stop-words.
- Phase 1 condition counts will be recorded in `reports/condition_counts.csv`.
- Phase 2 labelled data will be saved to `data/processed/drug_categories.csv`.

## Blockers / Questions for user

- None.

## Log

- 2026-10-30: Added the initial scaffold and verified the development environment.
- 2026-10-30: Loaded both raw review files and added condition exploration.
- 2026-10-30: Reviewed the top-100 condition list and approved the category
  mapping with the spelling normalization for truncated `Disorde` labels.
- 2026-10-30: Built the first filtered drug-category dataset using the approved
  mapping and purity rules.
