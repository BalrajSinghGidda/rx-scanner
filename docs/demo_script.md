# Three-Minute Demonstration Script

## Before the demonstration

From WSL:

```bash
cd ~/rx-scanner
source .venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501` in the Windows browser. Keep the terminal
available as a fallback and preselect a synthetic sample if upload fails.

## 0:00-0:25 — Introduce the problem

Say:

> This is an educational printed-prescription scanner. It is designed to show
> an explainable AI pipeline, not to make medical decisions. It reads printed
> text, matches medicine names, and predicts a broad category.

Point out the permanent **Educational use only. Not medical advice.** banner.

## 0:25-0:55 — Show the input and vision stage

Choose `synthetic_amoxicillin.png` from the sample selector. Explain:

1. The original image is shown on the left.
2. OpenCV converts the image to grayscale and creates a thresholded mask.
3. Morphological processing joins nearby text and removes long table rules.
4. Contours become sorted text-region boxes shown on the processed image.

The UI exposes the intermediate image so the result is not a black box.

## 0:55-1:25 — Show OCR and matching

Point to **Extracted text** and explain that Tesseract runs on grayscale,
upscaled crops rather than the inverted binary image. The cleaned text is
lowercased and tokenized. RapidFuzz compares tokens with the saved vocabulary
and displays raw-to-matched text when a likely correction is found.

Mention that blurry, curved, tiny, or stylised text can still produce OCR
errors, so the extracted text must be reviewed.

## 1:25-2:00 — Show classification

Point to **Prediction**:

> The classifier uses character TF-IDF features and Logistic Regression. It
> was selected because its macro-F1 was 0.6829, compared with 0.6587 for the
> Random Forest baseline. The confidence bar is a model score, not medical
> certainty.

Show the category and confidence for the sample. Explain that the application
warns when confidence is below 50%.

## 2:00-2:30 — Show another sample

Select `synthetic_aspirin.png`, then `synthetic_metformin.png` if time allows.
Use these measured smoke-test results:

- Amoxicillin -> Antibiotic, confidence 0.515
- Aspirin -> Painkiller, confidence 0.730
- Metformin -> Antidiabetic, confidence 0.657

Emphasize that these are clearly labelled synthetic demonstrations, not
patient prescriptions.

## 2:30-2:50 — Explain the training evidence

Show `reports/confusion_matrix.png` or the report outline. Say:

> The training data contains 1,371 unique drugs. Labels were derived from
> review conditions, filtered at 70% label purity, and split by unique drug
> name so repeated review rows could not leak between training and testing.

The hold-out accuracy was 0.6800 and macro-F1 was 0.6829.

## 2:50-3:00 — Close with limitations

Say:

> The prototype demonstrates the full workflow and keeps its intermediate
> steps visible. Its main limitation is OCR reliability on difficult images.
> It does not read handwriting, validate dosage, detect interactions, diagnose
> illness, or replace a pharmacist or doctor.

If a live image fails, return to a synthetic sample and explain the failure as
part of the documented evaluation rather than hiding it.

