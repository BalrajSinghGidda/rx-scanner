#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
DATA_DIR="$ROOT_DIR/data/raw"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing .venv. Create and activate it first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    exit 1
fi

command -v tesseract >/dev/null 2>&1 || {
    echo "Missing Tesseract. Install it with: sudo apt install -y tesseract-ocr"
    exit 1
}
command -v kaggle >/dev/null 2>&1 || {
    echo "Missing Kaggle CLI. Install/configure it before running setup.sh."
    exit 1
}

echo "Installing Python dependencies..."
"$PYTHON" -m pip install -r requirements.txt

echo "Downloading NLTK stop-words..."
"$PYTHON" -c "import nltk; nltk.download('stopwords', quiet=True)"

mkdir -p "$DATA_DIR"
if [[ ! -f "$DATA_DIR/drugsComTrain_raw.csv" ||
      ! -f "$DATA_DIR/drugsComTest_raw.csv" ]]; then
    echo "Downloading the Drugs.com review dataset from Kaggle..."
    kaggle datasets download \
        -d jessicali9530/kuc-hackathon-winter-2018 \
        -p "$DATA_DIR" \
        --unzip
else
    echo "Raw dataset already present; skipping download."
fi

echo "Building the processed dataset..."
"$PYTHON" -m train.build_dataset
echo "Training and comparing models..."
"$PYTHON" -m train.train_model
echo "Evaluating the saved model..."
"$PYTHON" -m train.evaluate
echo "Running tests..."
"$PYTHON" -m pytest -q

echo
echo "Setup complete. Start the app with:"
echo "  streamlit run app.py"
