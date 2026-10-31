# TR-102 Report Outline

## 1. Title and abstract

**AI Medical Prescription & Pharmacy Scanner** is an educational Streamlit
prototype for extracting printed medicine text and predicting a broad
therapeutic category. It combines OpenCV, Tesseract OCR, fuzzy matching, and a
scikit-learn classifier. It is not a clinical decision system.

## 2. Problem statement

Printed medicine names can be difficult to read manually when images are
small, noisy, or arranged in tables. The project demonstrates how a simple
computer-vision and machine-learning pipeline can turn a printed image into
interpretable text and a coarse category prediction.

## 3. Objectives and scope

- Detect printed text regions in medicine images and prescriptions.
- Extract text without using handwritten or patient data.
- Match OCR tokens against a known drug vocabulary.
- Predict one of seven broad categories.
- Present intermediate results so the process is explainable.

Handwriting, dosage validation, drug interactions, diagnosis, and clinical
recommendations are outside the scope.

## 4. System architecture

1. `src/vision.py` converts BGR images to grayscale, thresholds them, removes
   table rules, closes nearby text, and detects sorted contour boxes.
2. `src/ocr.py` runs Tesseract on grayscale, upscaled crops and falls back to
   whole-image OCR when no boxes are found.
3. `src/nlp.py` cleans text, removes stop-words, and uses RapidFuzz to repair
   likely OCR spelling errors.
4. `src/predict.py` loads the saved scikit-learn pipeline and returns category
   plus probability-based confidence.
5. `src/pipeline.py` connects the stages for both Python callers and the CLI.
6. `app.py` provides upload, sample selection, image comparison, and result
   cards in Streamlit.

## 5. Data preparation and label construction

The Drugs.com review dataset was loaded from the Kaggle CSV files. Null and
junk condition rows were removed. Conditions were mapped to seven categories:
Painkiller, Antibiotic, Antihistamine / Allergy, Antihypertensive,
Antidiabetic, Psychiatric, and Contraceptive.

Labels were assigned at the unique-drug level. A drug was retained only when
its winning category covered at least 70% of its mapped rows, and categories
with fewer than 20 unique drugs were removed. The final dataset contains
**1,371 unique drugs**.

## 6. Model method

The input is the drug name. Character TF-IDF features use `char_wb` n-grams
from 2 to 5 characters. Logistic Regression and Random Forest were trained
with balanced class weights and `random_state=42`. The hold-out split was
performed by unique drug name to prevent review-row leakage.

## 7. Results

The deterministic hold-out set contains **275 unique drugs**.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 0.6800 | 0.6829 |
| Random Forest | 0.6400 | 0.6587 |

Logistic Regression was selected and saved as the complete inference pipeline
in `models/model.joblib`. Detailed per-class metrics are in
`reports/metrics.json`; the confusion matrix is in
`reports/confusion_matrix.png`.

Synthetic end-to-end smoke tests produced:

| Image | Prediction | Confidence |
|---|---|---:|
| `synthetic_amoxicillin.png` | Antibiotic | 0.515 |
| `synthetic_aspirin.png` | Painkiller | 0.730 |
| `synthetic_metformin.png` | Antidiabetic | 0.657 |

## 8. Discussion and failure analysis

The model performs better when OCR supplies a clean medicine name. Errors are
expected when text is tiny, blurred, stylised, curved, affected by glare, or
mixed with table borders. OCR mistakes can create false fuzzy matches, while
unseen regional brand names may be absent from the training vocabulary.

The confidence value is useful for flagging uncertain predictions, but it is
not a probability of medical correctness.

## 9. Limitations, ethics, and safety

- The system reads printed text only.
- Category labels are derived from review conditions rather than an official
  pharmacological taxonomy.
- It does not verify prescriptions or provide dosage, interaction, or
  diagnosis advice.
- No real patient prescriptions or personal medical data are used.
- Results require human review and must not guide treatment decisions.

## 10. Future work

- Build a measured OCR benchmark with ground-truth text.
- Improve document layout and multi-column handling.
- Expand the verified regional medicine vocabulary.
- Add user confirmation and audit-friendly error reporting.
- Validate any future clinical use independently and formally.

## 11. Conclusion

The project demonstrates a complete, explainable prototype from image input to
category prediction. Its strongest contribution is showing each intermediate
stage rather than hiding the result behind an opaque model. Its main remaining
challenge is reliable OCR on difficult real-world prescription images.

