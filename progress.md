# Project Progress

## Status

- Current phase: Phase 4 - Streamlit UI
- Overall progress: 80%
- Next action: Polish documentation and prepare the report/demo materials
- Days left to 3 November 2026: 4

## Checklist

- [x] Phase 0: WSL setup, dependencies, scaffold, and environment check
- [x] Phase 1: Inspect condition values and approve category mapping
- [x] Phase 2: Build labelled dataset, train models, and evaluate
- [x] Phase 3: Implement vision, OCR, NLP, and analysis pipeline
- [x] Phase 4: Build and verify the Streamlit UI
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
- The best trained inference pipeline will be saved to `models/model.joblib`,
  with its n-gram vocabulary in `models/vocab.json`.
- Hold-out model comparison: Logistic Regression accuracy 0.6800, macro-F1
  0.6829; Random Forest accuracy 0.6400, macro-F1 0.6587. Logistic Regression
  was selected by macro-F1.
- Final labelled dataset: 1,371 unique drugs across seven categories:
  Painkiller 438, Psychiatric 197, Contraceptive 191,
  Antihistamine / Allergy 169, Antihypertensive 152, Antibiotic 127,
  Antidiabetic 97.
- Saved model evaluation on 275 hold-out drugs: accuracy 0.6800 and macro-F1
  0.6829.
- Saved evaluation outputs: `reports/metrics.json` and
  `reports/confusion_matrix.png`.
- Phase 3 includes three clearly labelled synthetic samples because the samples
  directory had no user-provided images. OCR and prediction smoke tests:
  `synthetic_amoxicillin.png` -> Antibiotic (0.515),
  `synthetic_aspirin.png` -> Painkiller (0.730), and
  `synthetic_metformin.png` -> Antidiabetic (0.657).
- Phase 4 UI implemented in `app.py` with upload/sample selection, permanent
  educational-use banner, side-by-side image display, result cards, and
  low-confidence warnings.
- Added `pytest.ini` so test imports resolve from the repository root even when
  the pytest launcher is not the virtualenv's executable.

## Blockers / Questions for user

- None.

## Log

- 2026-10-30: Added the initial scaffold and verified the development environment.
- 2026-10-30: Loaded both raw review files and added condition exploration.
- 2026-10-30: Reviewed the top-100 condition list and approved the category
  mapping with the spelling normalization for truncated `Disorde` labels.
- 2026-10-30: Built the first filtered drug-category dataset using the approved
  mapping and purity rules.
- 2026-10-30: Trained both required classifiers with a unique-drug hold-out
  split and saved the Logistic Regression pipeline as the best model.
- 2026-10-30: Evaluated the saved pipeline and generated per-class metrics and
  a confusion matrix.
- 2026-10-30: Implemented OpenCV vision, Tesseract OCR, NLP matching, offline
  prediction, the plain-Python analysis pipeline, CLI, and focused tests.
- 2026-10-31: Implemented the Streamlit dashboard around the Phase 3 pipeline.
- 2026-10-31: Fixed pytest module discovery for direct `pytest -q` execution.
