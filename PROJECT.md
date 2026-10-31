# AI Medical Prescription & Pharmacy Scanner

**Course:** TR-102 (AI/ML training project) · **Student:** Balraj Singh Gidda · GNDEC
**Deadline:** Tuesday, 3 November 2026 · **Dev environment:** WSL (Ubuntu)

> **Educational use only. This is not medical advice.** The app must show this banner on every page.

---

## 1. Overview

A Streamlit web app where a user uploads a photo of a printed medicine strip, drug box, or typed prescription. The app:

1. Preprocesses the image with **OpenCV** (grayscale, blur, adaptive/Otsu threshold, morphological closing, contour bounding boxes) to locate text regions.
2. Extracts raw text from each region with **Tesseract OCR**.
3. Cleans the text with **NLP** (lowercasing, tokenization, stop-word removal) and repairs OCR typos by fuzzy-matching against a known drug vocabulary.
4. Classifies the medicine into a **therapeutic category** (e.g. Antibiotic, Painkiller, Antihistamine) with a **custom-trained scikit-learn model** and reports a confidence score.
5. Shows everything in a dashboard: original vs processed image, extracted text, predicted category, confidence.

## 2. Scope

### In scope
- Printed text only (boxes, strips, typed prescriptions)
- 5-7 medicine categories
- TF-IDF (character n-grams) + Logistic Regression and Random Forest, compared
- Streamlit dashboard with side-by-side image view and result cards
- Evaluation: accuracy, macro-F1, confusion matrix
- Report + demo

### Out of scope (list under "Limitations / Future Work")
- Handwritten prescriptions
- Dosage-warning prediction
- Fine-tuning DistilBERT (stretch goal only if everything else is finished)
- Real patient data (never use it; use own photos or mock images)
- Any claim of clinical accuracy

## 3. Architecture

```
image upload
   │
   ▼
vision.py   grayscale → blur → adaptive threshold → morph close → contours → boxes
   │
   ▼
ocr.py      crop each box (from grayscale) → pytesseract → list of text lines
   │
   ▼
nlp.py      clean → tokenize → drop stop-words → fuzzy-match tokens to drug vocabulary
   │
   ▼
predict.py  TF-IDF (char n-grams) → classifier → category + confidence (predict_proba)
   │
   ▼
app.py      Streamlit: original vs processed, text card, category card, confidence
```

## 4. Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Vision | `opencv-python-headless`, `numpy` |
| OCR | Tesseract 5 via `pytesseract` (fallback: EasyOCR) |
| NLP | `nltk` (stop-words), `rapidfuzz` (fuzzy match), `scikit-learn` (TF-IDF) |
| Model | scikit-learn Logistic Regression, Random Forest (compare); `joblib` to save |
| UI | Streamlit |
| Data | Kaggle / UCI Drug Review (Drugs.com) dataset |

## 5. WSL setup

Work inside the Linux filesystem (`~/rx-scanner`), **not** `/mnt/c/...`, because it is much faster.

```bash
sudo apt update
sudo apt install -y tesseract-ocr python3-venv python3-pip

mkdir -p ~/rx-scanner && cd ~/rx-scanner
python3 -m venv .venv
source .venv/bin/activate

sudo apt install -y tesseract-ocr
# Configure the Kaggle CLI with your account, then run:
./setup.sh
```

Run the app:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in the **Windows** browser. WSL2 forwards localhost automatically. If it does not, run `hostname -I` in WSL and use that IP.

Use `opencv-python-headless` (not `opencv-python`) to avoid `libGL` errors in WSL. If you ever need GUI windows, `sudo apt install -y libgl1`.

## 6. Repository structure

```
rx-scanner/
├── app.py                    # Streamlit UI
├── requirements.txt
├── README.md
├── PROJECT.md
├── AGENTS.md
├── src/
│   ├── __init__.py
│   ├── vision.py             # preprocessing + text-region detection
│   ├── ocr.py                # Tesseract wrapper + text cleanup
│   ├── nlp.py                # cleaning, tokenizing, fuzzy matching
│   ├── predict.py            # load model, classify, confidence
│   └── pipeline.py           # analyze(image) -> AnalysisResult
├── train/
│   ├── build_dataset.py      # raw TSV -> data/processed/drug_categories.csv
│   ├── train_model.py        # train, compare, save best model
│   └── evaluate.py           # metrics + confusion matrix plots
├── data/
│   ├── raw/                  # downloaded Kaggle files (git-ignored)
│   ├── processed/
│   └── samples/              # test images (own photos / mock prescriptions)
├── models/                   # model.joblib, vocab.json
├── reports/                  # figures, metrics.json
└── tests/
    └── test_pipeline.py
```

## 7. Data and labels

**Source:** Drug Review dataset (Drugs.com). The Kaggle copy (`jessicali9530/kuc-hackathon-winter-2018`) ships `drugsComTrain_raw.csv` / `drugsComTest_raw.csv`; the UCI original ships `.tsv` versions. Same columns either way: `drugName`, `condition`, `review`, `rating`, ...

**Problem:** it has no category column and reviews do not look like box text. **Solution:** derive categories from `condition`, then train on **drug names**.

### Steps (`train/build_dataset.py`)
1. Load train + test files (`sep=","` for Kaggle `.csv`, `sep="\t"` for UCI `.tsv`), concatenate.
2. Drop rows with null `condition` and junk values (some contain "users found this comment helpful").
3. Map each `condition` to a category using the table below. Print the real top-100 condition values first and **verify the names**.
4. Group by `drugName`. Assign the drug the category covering **at least 70%** of its mapped rows. Drop drugs below that threshold (label-purity filter).
5. Keep categories with **20 or more unique drugs**.
6. Save `drugName, category` to `data/processed/drug_categories.csv`.

### Starting condition → category map (verify against real values)

| Category | Example conditions |
|---|---|
| Painkiller | Pain, Migraine, Headache, Fibromyalgia, Osteoarthritis |
| Antibiotic | Urinary Tract Infection, Bacterial Infection, Sinusitis, Bronchitis, Strep Throat |
| Antihistamine / Allergy | Allergic Rhinitis, Allergic Urticaria, Hay Fever |
| Antihypertensive | High Blood Pressure |
| Antidiabetic | Diabetes, Type 2 |
| Psychiatric (Antidepressant / Anxiolytic) | Depression, Anxiety, Bipolar Disorder, Panic Disorder |
| Contraceptive | Birth Control |

Adjust to what has enough drugs. Acne, for example, overlaps several classes, so skip ambiguous conditions rather than force a label.

## 8. Model training (`train/train_model.py`)

- **Input text:** drug name (lowercased). Optionally append a short generic/indication string.
- **Vectorizer:** `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)`. Character n-grams capture suffix patterns such as -cillin, -mycin, -pril, -statin, -azole.
- **Models:** `LogisticRegression(max_iter=1000, class_weight="balanced")` and `RandomForestClassifier(n_estimators=300, class_weight="balanced")`. Optional third: `LinearSVC`.
- **Split:** `GroupShuffleSplit` or split on **unique drug names** (80/20). Never split by review row, because the same drug appears many times and the split would leak.
- **Metrics:** accuracy, macro-F1, per-class precision/recall, confusion matrix (save PNG to `reports/`).
- **Save:** best model + vectorizer as one `sklearn.pipeline.Pipeline` in `models/model.joblib`; drug vocabulary in `models/vocab.json`.
- **Confidence:** `predict_proba().max()`. Warn the user below 0.50.

## 9. Vision stage (`src/vision.py`)

```
bgr → gray → GaussianBlur(5,5) → adaptiveThreshold (INV, block 31, C 15)
    → morphologyEx CLOSE with rect kernel (25×5) → findContours (EXTERNAL)
    → boundingRect → filter (w > 40, h > 12) → sort top-to-bottom, left-to-right
```

- Return the binary image, the closed image, the boxes, and an **annotated copy** with green rectangles for the UI.
- Compare Otsu vs adaptive thresholding on 3-4 samples and put the comparison in the report.
- OCR the crops from the **grayscale** image (dark text on light), not from the inverted binary.
- Upscale small crops (about 2×) before OCR. If nothing is detected, fall back to OCR on the whole image.

## 10. OCR and NLP (`src/ocr.py`, `src/nlp.py`)

- `pytesseract.image_to_string(crop, config="--oem 3 --psm 6")` (try `--psm 7` for single lines).
- Cleanup: strip non-alphanumerics, collapse whitespace, lowercase.
- Tokenize, remove stop-words (NLTK English) and tokens shorter than 3 characters.
- **Fuzzy match:** for each token, `rapidfuzz.process.extractOne(token, vocab, scorer=fuzz.WRatio)`. Accept at a score of 85 or higher. Return both the raw and corrected token so the UI can show what was repaired.
- Pass the best matched drug name (or the cleaned token string if none matched) to the classifier.

## 11. Streamlit UI (`app.py`)

- Persistent banner: "Educational use only. Not medical advice."
- Sidebar: file uploader (jpg/png), "use sample image" dropdown, toggle for showing bounding boxes.
- Main area:
  - Two columns: **Original** | **Processed (boxes drawn)**
  - Card 1: **Extracted text** (raw, then cleaned)
  - Card 2: **Matched drug(s)**: raw token → corrected token
  - Card 3: **Predicted category** + **confidence** (progress bar; warning if below 50%)
- Graceful errors: no text found, blurry image, unsupported file.
- Cache the model with `@st.cache_resource`.

## 12. Evaluation and testing

- **Model:** hold-out set by unique drug; report accuracy, macro-F1, confusion matrix, model comparison table.
- **End to end:** 15-20 own test images. Record per image: OCR text, matched drug, predicted category, correct (Y/N). Report the end-to-end success rate honestly, including failures.
- **Failure analysis:** short section on why it fails (glare, curved packaging, small fonts, stylised logos).

## 13. Timeline

| Date | Goal | Done |
|---|---|---|
| **Fri 30 Oct** | WSL setup, install Tesseract + libs, download dataset, collect 15-20 sample images, inspect `condition` values | ☐ |
| **Sat 31 Oct** | `build_dataset.py`, `train_model.py`, `evaluate.py`; saved model + confusion matrix | ☐ |
| **Sun 1 Nov** | `vision.py`, `ocr.py`, `nlp.py`, `pipeline.analyze()` working from CLI on all samples | ☐ |
| **Mon 2 Nov** | `app.py` UI, error handling, end-to-end test table, screenshots | ☐ |
| **Tue 3 Nov** | Report/PPT, README, demo rehearsal. **No new features.** | ☐ |

**Cut order if behind:** (1) Random Forest comparison, (2) Otsu vs adaptive comparison, (3) fuzzy-match repair display, (4) confidence warnings. Never cut: OpenCV stage, OCR, trained model with evaluation, Streamlit demo.

## 14. Deliverables

- [ ] Working Streamlit app (run from WSL, shown in Windows browser)
- [ ] Source code in a clean repo with README
- [ ] Trained model file + training/evaluation scripts
- [ ] Report: problem, data and label construction, methods, results, limitations, future work
- [ ] Slides/demo with 3-4 live or recorded examples
- [ ] Screenshots of the UI and the confusion matrix

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Tesseract struggles with stylised box text | Crop + upscale + threshold; use `--psm 7`; fuzzy-match to repair; use clear, flat photos |
| Label noise from condition → category mapping | 70% purity filter; drop ambiguous conditions; say so in the report |
| Data leakage inflating accuracy | Split by unique drug name |
| Demo fails live | Pre-load sample images in the UI; keep a screen recording as backup |
| Time runs out | Follow the cut order in section 13 |

## 16. Limitations (for the report)

- Printed text only; no handwriting
- Categories are coarse and derived from review conditions, not an official drug classification
- Classifies by drug name; cannot read dosage or detect interactions
- Not a clinical tool
