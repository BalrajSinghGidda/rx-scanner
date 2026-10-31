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
- 2026-10-31: Treat verified brand-resolved active ingredients as matched drugs
  with score 100 and provenance, while retaining the separate brand summary.

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

- 2026-10-31 15:47 UTC: Included known active ingredients in matched-drug
  results and added a regression test; 7 tests passed.
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
- 2026-10-31: Started OCR experiments on a low-resolution table prescription;
  line removal, stronger upscaling, whole-image quality selection, and
  filtering of prescription boilerplate tokens were added on the test branch.
- On the attached prescription smoke test, OCR now recovers medicine-name
  lines such as ABCIXIMAB, VOMILAST, DOXYLAMINE, ZOCLAR, CLARITHROMYCIN,
  GESTAKIND, and ISOXSUPRINE; the names remain imperfect because the source
  image is only 470x335 pixels and several names are absent from the model
  vocabulary.
- RxNorm ingredient names can be downloaded during setup with
  `python -m train.build_vocabulary`; the generated vocabulary is used offline
  alongside the training vocabulary.
- The pipeline applies a conservative decision gate: only alphabetic tokens of
  at least seven characters and fuzzy scores of 85 or higher can influence the
  category prediction, reducing matches on words such as `tab`, `fever`, and
  `malaria`.
- Table-rule removal was added to vision preprocessing; on the high-resolution
  monochrome sheet, OCR now preserves the medicine rows and on the hypertension
  form it reads `Telvas`, `Glycomet`, and `Atorva` in context. Generic advice
  words are excluded from drug matching.
- Added a small offline brand-to-active-ingredient layer for the synthetic
  hypertension form: Telvas -> telmisartan, Glycomet GP -> glimepiride plus
  metformin hydrochloride, and Atorva -> atorvastatin. Category prediction now
  uses only reliable drug/ingredient evidence and does not fall back to all
  prescription prose.
- Added an opt-in online lookup demonstration using RxNorm or openFDA. Results
  are saved to `models/online_reference.json`, reused offline on future scans,
  and displayed with a provider citation in the UI footer.
- Known active ingredients resolved from recognized brands are now also included
  in the matched-drug results with explicit provenance.
- Focused test suite after the ingredient-match change: 7 passed.
