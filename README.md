# AI Medical Prescription & Pharmacy Scanner

TR-102 educational project demonstrating a complete printed-prescription
analysis pipeline:

**image -> OpenCV text regions -> Tesseract OCR -> fuzzy drug matching ->
scikit-learn category prediction -> Streamlit dashboard**

> **Educational use only. Not medical advice.** This prototype is not a
> clinical tool and must not be used for diagnosis, treatment, or dosage
> decisions.

## What it does

- Accepts a printed medicine box, strip, or typed prescription image.
- Detects likely text regions with OpenCV.
- Runs OCR on grayscale, upscaled regions using Tesseract.
- Cleans OCR output and repairs likely spelling errors with RapidFuzz.
- Predicts one of seven broad categories:
  Painkiller, Antibiotic, Antihistamine / Allergy, Antihypertensive,
  Antidiabetic, Psychiatric, or Contraceptive.
- Displays the original image, processed image, OCR text, matches, category,
  and confidence in Streamlit.

The model is trained on drug names from the Drugs.com review dataset. Labels
are derived from review conditions, filtered by a 70% per-drug purity rule, and
split by unique drug name to avoid row leakage.

## Setup on WSL Ubuntu

Work in the Linux filesystem rather than `/mnt/c` for better performance.

```bash
cd ~/rx-scanner
python3 -m venv .venv
source .venv/bin/activate
sudo apt update
sudo apt install -y tesseract-ocr
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
tesseract --version
```

The raw Kaggle files must be present in `data/raw/`:

```text
data/raw/drugsComTrain_raw.csv
data/raw/drugsComTest_raw.csv
```

Raw data and the virtual environment are intentionally ignored by Git.

## Train and evaluate

```bash
python -m train.build_dataset
python -m train.train_model
python -m train.evaluate
```

Generated artifacts:

- `data/processed/drug_categories.csv`
- `models/model.joblib`
- `models/vocab.json`
- `reports/condition_counts.csv`
- `reports/metrics.json`
- `reports/confusion_matrix.png`

## Run the application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in the Windows browser. WSL2 normally forwards
localhost automatically. The sidebar also provides safe synthetic sample
images in `data/samples/`.

For a command-line image check:

```bash
python -m src.pipeline data/samples/synthetic_amoxicillin.png
```

## Test

```bash
pytest -q
```

## Results

The deterministic hold-out split contains 275 unique drugs. Logistic
Regression was selected over Random Forest:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 0.6800 | 0.6829 |
| Random Forest | 0.6400 | 0.6587 |

The final labelled dataset contains 1,371 unique drugs:

| Category | Drugs |
|---|---:|
| Painkiller | 438 |
| Psychiatric | 197 |
| Contraceptive | 191 |
| Antihistamine / Allergy | 169 |
| Antihypertensive | 152 |
| Antibiotic | 127 |
| Antidiabetic | 97 |

Synthetic smoke tests produced Antibiotic for amoxicillin (0.515),
Painkiller for aspirin (0.730), and Antidiabetic for metformin (0.657).

## Project structure

```text
app.py                     Streamlit interface
src/vision.py              OpenCV preprocessing and text boxes
src/ocr.py                 Tesseract OCR
src/nlp.py                 cleaning and fuzzy matching
src/predict.py             saved-model inference
src/pipeline.py            plain-Python end-to-end pipeline
train/build_dataset.py     condition cleaning and label construction
train/train_model.py       model comparison and saving
train/evaluate.py          metrics and confusion matrix
data/samples/              synthetic demonstration images
models/                    saved model and vocabulary
reports/                   measured evaluation outputs
tests/                     smoke and regression tests
docs/                      report and demonstration materials
```

## Limitations and future work

- Printed text only; handwriting is unsupported.
- OCR is sensitive to blur, glare, curved packaging, small fonts, and table
  layouts.
- Categories are coarse labels derived from review conditions, not official
  pharmacological classifications.
- The classifier predicts a category from drug-name evidence; it does not
  validate prescriptions, dosage, interactions, or patient safety.
- Confidence is a model score, not medical certainty.
- Future improvements include a larger verified regional vocabulary, a
  measured OCR benchmark, better document layout handling, and human review.
