"""Streamlit interface for the prescription scanner."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import AnalysisResult, analyze

SAMPLES_DIR = Path("data/samples")


@st.cache_resource
def load_sample_paths() -> list[Path]:
    """Cache the available safe sample image paths."""
    return sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR OpenCV image."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The selected file is not a supported image.")
    return image


def show_results(result: AnalysisResult) -> None:
    """Render analysis output cards."""
    st.subheader("Extracted text")
    st.code("\n".join(result.raw_lines) or "(no text found)")
    st.caption(f"Cleaned text: {result.cleaned_text or '(empty)'}")

    left, right = st.columns(2)
    with left:
        st.subheader("Matched drug(s)")
        if result.matches:
            for match in result.matches:
                st.write(
                    f"`{match['raw']}` → **{match['matched']}** "
                    f"(score {match['score']})"
                )
        else:
            st.write("No drug names matched the vocabulary.")
    with right:
        st.subheader("Prediction")
        st.metric("Category", result.category or "Unavailable")
        if result.confidence is not None:
            st.progress(result.confidence, text=f"Confidence: {result.confidence:.1%}")
            if result.confidence < 0.50:
                st.warning("Low confidence. Review the extracted text carefully.")
        else:
            st.info("A category could not be predicted.")

    if result.brands:
        st.subheader("Known active ingredient(s)")
        for brand in result.brands:
            st.write(f"**{brand.brand}:** {', '.join(brand.ingredients)}")

    for warning in result.warnings:
        st.warning(warning)


def main() -> None:
    """Render the scanner dashboard."""
    st.set_page_config(page_title="Rx Scanner", page_icon="💊", layout="wide")
    st.title("AI Medical Prescription & Pharmacy Scanner")
    st.warning("Educational use only. Not medical advice.")
    st.write("Upload a printed medicine image or choose a synthetic sample.")

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
        )
        samples = load_sample_paths()
        sample_names = ["None"] + [path.name for path in samples]
        selected = st.selectbox("Use sample image", sample_names)
        show_boxes = st.checkbox("Show bounding boxes", value=True)

    image_bgr: np.ndarray | None = None
    display_image: Image.Image | None = None
    if uploaded is not None:
        try:
            image_bgr = decode_image(uploaded.getvalue())
            display_image = Image.open(uploaded)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    elif selected != "None":
        sample_path = SAMPLES_DIR / selected
        try:
            image_bgr = cv2.imread(str(sample_path))
            if image_bgr is None:
                raise ValueError("The selected sample could not be read.")
            display_image = Image.open(sample_path)
        except (ValueError, OSError) as exc:
            st.error(str(exc))

    if image_bgr is None or display_image is None:
        st.info("Choose an image from the sidebar to begin.")
        return

    try:
        result = analyze(image_bgr)
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"Analysis failed: {exc}")
        return

    original, processed = st.columns(2)
    with original:
        st.subheader("Original")
        st.image(display_image, use_container_width=True)
    with processed:
        st.subheader("Processed (boxes drawn)")
        processed_image = result.vision.annotated if show_boxes else result.vision.gray
        channels = "BGR" if show_boxes else "GRAY"
        st.image(processed_image, channels=channels, use_container_width=True)
    show_results(result)


if __name__ == "__main__":
    main()
