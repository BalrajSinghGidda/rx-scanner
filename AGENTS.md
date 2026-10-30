# AGENTS.md

Instructions for AI coding agents working on this repository. Read `PROJECT.md` first for the full spec. This file says **how to work**; `PROJECT.md` says **what to build**.

## Project in one paragraph

A Streamlit app for the TR-102 training project: the user uploads a photo of a printed medicine box/strip or typed prescription; OpenCV locates text regions, Tesseract extracts text, NLP cleans it and fuzzy-matches drug names, and a custom scikit-learn model predicts the medicine category with a confidence score. **Hard deadline: Tuesday 3 November 2026.** Favour a simple, working, explainable solution over a clever one.

## Environment

- **OS:** WSL2 (Ubuntu). Project lives in `~/rx-scanner` (Linux filesystem, not `/mnt/c`).
- **Python:** 3.10+, virtual environment at `.venv`.
- Activate: `source .venv/bin/activate`
- System dependency: `tesseract-ocr` (installed with `apt`, not pip).
- Use **`opencv-python-headless`**, never `opencv-python` (avoids libGL errors in WSL).

## Commands

```bash
# setup
pip install -r requirements.txt

# data + model
python -m train.build_dataset      # raw CSV/TSV -> data/processed/drug_categories.csv
python -m train.train_model        # trains, compares, saves models/model.joblib
python -m train.evaluate           # metrics + confusion matrix -> reports/

# app
streamlit run app.py               # open http://localhost:8501 in the Windows browser

# tests
pytest -q
```

Always run the relevant command after changing code. Do not claim something works without running it.

## Module contracts

Keep these signatures stable. Other modules depend on them.

```python
# src/vision.py
@dataclass
class VisionResult:
    gray: np.ndarray
    binary: np.ndarray
    closed: np.ndarray
    boxes: list[tuple[int, int, int, int]]   # (x, y, w, h), sorted top-to-bottom, left-to-right
    annotated: np.ndarray                     # BGR copy with boxes drawn

def preprocess(img_bgr: np.ndarray) -> VisionResult: ...

# src/ocr.py
def extract_text(gray: np.ndarray, boxes: list[tuple[int,int,int,int]]) -> list[str]: ...
    # falls back to whole-image OCR if boxes is empty

# src/nlp.py
def clean_text(text: str) -> str: ...
def tokenize(text: str) -> list[str]: ...            # lowercased, stop-words and len<3 removed
def match_drugs(tokens: list[str], vocab: list[str], threshold: int = 85) -> list[dict]: ...
    # [{"raw": "amoxici11in", "matched": "amoxicillin", "score": 92}, ...]

# src/predict.py
def classify(text: str) -> tuple[str, float]: ...    # (category, confidence 0-1)

# src/pipeline.py
@dataclass
class AnalysisResult:
    vision: VisionResult
    raw_lines: list[str]
    cleaned_text: str
    matches: list[dict]
    category: str | None
    confidence: float | None
    warnings: list[str]

def analyze(img_bgr: np.ndarray) -> AnalysisResult: ...
```

`pipeline.analyze()` must work from a plain Python call, with no Streamlit dependency. `app.py` only handles UI.

## Code style

- Python with type hints and short docstrings (one line is fine).
- Pure functions where possible; no global mutable state.
- Paths via `pathlib.Path`, relative to the repo root. No hard-coded absolute paths.
- Constants (thresholds, kernel sizes, model path) at the top of the module or in `src/config.py`.
- Keep functions small and readable. A student must be able to explain every line in a viva.
- Comment the **why** (e.g. why a kernel is 25×5), not the obvious.
- No unnecessary dependencies. Ask before adding any package not listed in `PROJECT.md`.

## Rules for the ML part

- **Split by unique drug name**, never by row. The same drug appears many times in the dataset, and a row split leaks.
- Use `class_weight="balanced"` for imbalanced categories.
- Fix `random_state=42` everywhere.
- Report accuracy, macro-F1, per-class metrics and a confusion matrix. Do not report accuracy alone.
- Never invent metrics. If a number is needed for the report, compute it and save it to `reports/metrics.json`.
- Save the full `Pipeline` (vectorizer + classifier) with `joblib` so inference needs only one load.
- Verify the actual `condition` values in the dataset before finalising the category mapping. Do not assume names.

## Rules for the vision/OCR part

- OCR from the **grayscale** crop (dark text on light), not from the inverted binary.
- Upscale small crops about 2× before OCR.
- If no boxes are found, fall back to OCR on the whole image and add a warning.
- Tune thresholds against the images in `data/samples/`, and note what you changed and why.
- Never crash on a bad image. Return an `AnalysisResult` with warnings instead.

## Rules for the Streamlit UI

- Show "Educational use only. Not medical advice." on every page, always visible.
- Layout: original vs processed side by side, then cards for extracted text, matched drugs, category, confidence.
- Show a warning when confidence is below 0.50.
- Cache the model with `@st.cache_resource`.
- No business logic in `app.py`; call `pipeline.analyze()`.

## Boundaries: do NOT

- Use real patient prescriptions or personal medical data. Use only images in `data/samples/` (own photos or mock-ups).
- Add handwriting recognition, dosage prediction, drug-interaction checking, or any clinical claims.
- Fine-tune DistilBERT or add deep-learning dependencies unless the user explicitly asks (stretch goal only).
- Commit raw datasets or large files (`data/raw/` and `.venv/` are git-ignored).
- Make network calls at inference time. The app must work offline after setup.
- Rewrite working modules "to improve them" in the last 24 hours. After Monday 2 Nov, fix bugs only.

## Definition of done (per module)

1. Runs without errors on every image in `data/samples/`.
2. Has at least one test in `tests/` (even a simple smoke test).
3. Respects the contract above.
4. The user can explain how it works.

## Priorities if time is short

Cut in this order: Random Forest comparison → Otsu vs adaptive comparison → fuzzy-match display → confidence warnings.
**Never cut:** OpenCV stage, OCR, trained model with evaluation, working Streamlit demo.

## Communication style

- When finishing a task, state what changed, how to run it, and what was verified.
- If a requirement is ambiguous, pick the simplest reasonable interpretation, say so, and continue. Do not block.
- Flag risks early (e.g. "OCR accuracy on curved packaging is poor; recommend flat photos").

## Git workflow (mandatory)

The repo is already initialised. Every change you make must be committed.

1. **Commit after EVERY logical change**: one new module, one fix, one doc update, one config tweak = one commit. Never batch unrelated changes. Never end a turn with uncommitted work (`git status` must be clean).
2. **Before each commit:** run the relevant check (import the module, run the script, or `pytest -q`). Do not commit code you have not run.
3. **Stage deliberately.** Read `git status` first. Never commit `data/raw/`, `.venv/`, `kaggle.json`, `.env`, or caches. If something that should be ignored shows up, fix `.gitignore` in its own commit.
4. **Message format: Conventional Commits**, imperative mood, subject at most 72 characters:
   `feat(vision): add adaptive threshold and contour box detection`
   Types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `train`, `data`. Add a short body when the *why* is not obvious.
5. **Update `progress.md` in the same commit** as the change it describes (not in a separate follow-up commit).
6. **Never** run `git push`, `git push --force`, `git reset --hard`, `git rebase`, `git commit --amend`, `git checkout .`, or anything that rewrites or discards history. Never change remotes or branches unless explicitly told. If you introduced a bug in an earlier commit, fix it in a **new** commit.
7. If `git config user.name` / `user.email` are not set, stop and tell the user. Do not set them globally.

## progress.md (mandatory)

`progress.md` in the repo root is the single source of truth for project status. Keep it current. Sections:

- **Status**: current phase, overall % done, next action, days left to 3 Nov 2026.
- **Checklist**: phases and tasks as `- [ ]` / `- [x]` (mirrors section 13 of `PROJECT.md`).
- **Decisions**: dated list of decisions and why (e.g. approved category mapping, chosen thresholds).
- **Results**: real metrics and the end-to-end test table. Only numbers that were actually computed.
- **Blockers / Questions for user**: anything waiting on a human.
- **Log**: dated one-line entries, newest first (no commit hashes needed; `git log` has them).

Use `date` in the terminal for real timestamps. Do not invent dates, numbers, or results.
