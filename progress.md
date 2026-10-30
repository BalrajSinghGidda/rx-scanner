# Project Progress

## Status

- Current phase: Phase 1 - data exploration
- Overall progress: 20%
- Next action: Approve or edit the proposed condition-to-category mapping
- Days left to 3 November 2026: 4

## Checklist

- [x] Phase 0: WSL setup, dependencies, scaffold, and environment check
- [ ] Phase 1: Inspect condition values and approve category mapping
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

## Results

- Environment checks passed: Python 3.14.4, Tesseract 5.5.0, required Python
  imports, and NLTK English stop-words.
- Phase 1 condition counts will be recorded in `reports/condition_counts.csv`.

## Blockers / Questions for user

- Mapping approved with the `Disorde` to `Disorder` normalization noted above;
  dataset construction is the next phase.

## Log

- 2026-10-30: Added the initial scaffold and verified the development environment.
- 2026-10-30: Loaded both raw review files and added condition exploration.
- 2026-10-30: Reviewed the top-100 condition list and approved the category
  mapping with the spelling normalization for truncated `Disorde` labels.
