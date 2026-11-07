from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


st.set_page_config(
    page_title="Xray Vision | COVID-19 Classifier",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "models" / "covid_detection_model.keras"
MODEL_PATH = Path(
    os.environ.get("COVID19_MODEL_PATH", str(DEFAULT_MODEL_PATH))
).expanduser()
LABELS = ("COVID", "Normal", "Viral Pneumonia")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
NOTEBOOK_URL = (
    "https://colab.research.google.com/github/omk1112/Covid19_OpenCV/"
    "blob/main/COVID19_Computer_Vision_Project.ipynb"
)


st.markdown(
    """
    <style>
    :root {
      color-scheme: dark;
      --page: #080e19;
      --panel: rgba(17, 28, 46, 0.82);
      --panel-soft: rgba(21, 36, 57, 0.68);
      --line: rgba(159, 184, 216, 0.14);
      --ink: #edf4ff;
      --muted: #96a9c2;
      --mint: #68e0c2;
      --blue: #83aaff;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
      background:
        radial-gradient(ellipse at 15% 0%, rgba(44, 99, 153, 0.23), transparent 38%),
        radial-gradient(ellipse at 90% 20%, rgba(32, 129, 120, 0.13), transparent 32%),
        var(--page);
      color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
      max-width: 1180px;
      padding-top: 1.8rem;
      padding-bottom: 3rem;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }
    p, li, label { color: #c6d3e5; }
    [data-testid="stCaptionContainer"] p, .stMarkdown small { color: var(--muted); }
    .brand-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      margin-bottom: 3.25rem;
    }
    .brand {
      color: var(--ink);
      font-size: 0.9rem;
      font-weight: 750;
      letter-spacing: 0.12em;
    }
    .brand span { color: var(--mint); }
    .top-chip, .eyebrow {
      color: var(--mint);
      font-size: 0.72rem;
      font-weight: 750;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .top-chip {
      border: 1px solid rgba(104, 224, 194, 0.27);
      border-radius: 999px;
      padding: 0.42rem 0.72rem;
      background: rgba(104, 224, 194, 0.07);
    }
    .hero-title {
      color: var(--ink);
      font-size: clamp(2.6rem, 6vw, 4.8rem);
      font-weight: 760;
      line-height: 1.02;
      letter-spacing: -0.065em;
      margin: 0.6rem 0 0.8rem;
    }
    .hero-title span {
      background: linear-gradient(90deg, var(--mint), var(--blue));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .hero-copy {
      max-width: 690px;
      color: var(--muted);
      font-size: 1.08rem;
      line-height: 1.7;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--line) !important;
      border-radius: 22px !important;
      background: linear-gradient(145deg, rgba(21, 36, 57, 0.92), rgba(12, 21, 35, 0.88));
      box-shadow: 0 22px 70px rgba(0, 0, 0, 0.18);
    }
    .card-kicker {
      color: var(--mint);
      font-size: 0.68rem;
      font-weight: 750;
      letter-spacing: 0.13em;
      text-transform: uppercase;
      margin-bottom: 0.45rem;
    }
    .muted-copy { color: var(--muted); line-height: 1.6; }
    .result-label {
      color: var(--muted);
      font-size: 0.77rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .result-name {
      color: var(--ink);
      font-size: clamp(1.8rem, 4vw, 2.5rem);
      font-weight: 760;
      letter-spacing: -0.05em;
      margin: 0.18rem 0 0.2rem;
    }
    .result-score { color: var(--mint); font-weight: 700; }
    .empty-state {
      border: 1px dashed rgba(159, 184, 216, 0.24);
      border-radius: 16px;
      padding: 2.4rem 1.2rem;
      text-align: center;
      color: var(--muted);
      margin-top: 0.8rem;
    }
    .empty-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    [data-testid="stFileUploaderDropzone"] {
      border: 1px dashed rgba(104, 224, 194, 0.35);
      border-radius: 16px;
      background: rgba(104, 224, 194, 0.035);
    }
    [data-testid="stFileUploaderDropzone"]:hover {
      border-color: rgba(104, 224, 194, 0.75);
    }
    div.stButton > button[kind="primary"] {
      min-height: 3rem;
      border: 0;
      border-radius: 12px;
      color: #071511;
      background: linear-gradient(100deg, #68e0c2, #9debd5);
      font-weight: 760;
      transition: transform 120ms ease, filter 120ms ease;
    }
    div.stButton > button[kind="primary"]:hover {
      filter: brightness(1.06);
      transform: translateY(-1px);
    }
    div.stButton > button[kind="primary"]:disabled {
      background: #344254;
      color: #94a2b2;
    }
    [data-testid="stMetric"] {
      padding: 0.85rem 1rem;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(17, 28, 46, 0.58);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stProgress"] > div > div { background-color: var(--mint); }
    [data-testid="stAlert"] { border-radius: 12px; }
    .footer {
      border-top: 1px solid var(--line);
      margin-top: 2.5rem;
      padding-top: 1.1rem;
      color: var(--muted);
      font-size: 0.82rem;
      text-align: center;
    }
    a { color: var(--mint) !important; }
    @media (max-width: 640px) {
      .brand-row { margin-bottom: 2rem; }
      .top-chip { font-size: 0.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, modified_time_ns: int) -> Any:
    """Load and validate the trained Keras checkpoint once per app process."""
    del modified_time_ns  # Included in the cache key to reload a replaced file.
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = tuple(model.input_shape[1:])
    if input_shape != (224, 224, 3):
        raise ValueError(
            f"Expected model input shape (224, 224, 3); got {input_shape}."
        )
    output_shape = model.output_shape
    if isinstance(output_shape, list) or output_shape[-1] != len(LABELS):
        raise ValueError(
            f"Expected a model with {len(LABELS)} outputs; got {output_shape}."
        )
    return model


def preprocess_xray(image_bytes: bytes) -> np.ndarray:
    """Match the notebook's OpenCV and MobileNetV2 input preprocessing."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode this image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def inspect_upload(image_bytes: bytes) -> Image.Image:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("The image is larger than the 10 MB upload limit.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("This file is not a valid PNG, JPG, or BMP image.") from exc


def make_prediction(model: Any, image_bytes: bytes) -> np.ndarray:
    batch = preprocess_xray(image_bytes)
    output = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
    if output.shape != (1, len(LABELS)):
        raise ValueError(
            f"Expected prediction shape (1, {len(LABELS)}); got {output.shape}."
        )
    scores = output[0]
    if not np.isfinite(scores).all():
        raise ValueError("The model returned a non-finite score.")
    if (scores < 0).any() or (scores > 1.001).any():
        raise ValueError("The model output is not a three-class softmax.")
    total = float(scores.sum())
    if total <= 0:
        raise ValueError("The model returned an empty prediction.")
    return scores / total


st.markdown(
    """
    <div class="brand-row">
      <div class="brand">XRAY<span>VISION</span></div>
      <div class="top-chip">Research prototype · Not for diagnosis</div>
    </div>
    <div class="eyebrow">Computer vision · Chest X-ray analysis</div>
    <div class="hero-title">A clearer view of<br><span>the model output.</span></div>
    <p class="hero-copy">
      Upload one chest X-ray to see the model's predicted category and
      per-class scores. Image preprocessing follows the project notebook.
    </p>
    """,
    unsafe_allow_html=True,
)

stat_a, stat_b, stat_c = st.columns(3)
stat_a.metric("Output categories", "03", help="COVID, Normal, Viral Pneumonia")
stat_b.metric("Model input", "224 × 224", help="RGB image scaled to the 0–1 range")
stat_c.metric("Pipeline", "OpenCV + MobileNetV2", help="OpenCV preprocessing with a TensorFlow/Keras classifier")

st.write("")
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="card-kicker">01 / Add an image</div>', unsafe_allow_html=True)
        st.subheader("Upload a chest X-ray")
        st.markdown(
            '<p class="muted-copy">Use a clear PNG, JPG, JPEG, or BMP image. '
            'The app converts it to RGB, resizes it, and scales pixels to 0–1.</p>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp"],
            label_visibility="collapsed",
            help="Maximum file size: 10 MB.",
        )

        raw_image: bytes | None = None
        preview: Image.Image | None = None
        if uploaded is not None:
            raw_image = uploaded.getvalue()
            try:
                preview = inspect_upload(raw_image)
                st.image(
                    preview,
                    caption=f"Ready to analyze · {uploaded.name}",
                    width="stretch",
                )
            except ValueError as exc:
                st.error(str(exc))
        else:
            st.markdown(
                """
                <div class="empty-state">
                  <div class="empty-icon">＋</div>
                  <div><strong>Drop an image here</strong></div>
                  <div style="margin-top:0.35rem">PNG · JPG · JPEG · BMP</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        image_digest = hashlib.sha256(raw_image).hexdigest() if raw_image else None
        existing_result = st.session_state.get("prediction")
        if existing_result and existing_result.get("image_digest") != image_digest:
            st.session_state.pop("prediction", None)
            st.session_state.pop("prediction_error", None)

        analyze = st.button(
            "Analyze X-ray",
            type="primary",
            width="stretch",
            disabled=preview is None,
        )

    if analyze and raw_image and image_digest:
        if not MODEL_PATH.is_file():
            st.session_state["prediction"] = None
            st.session_state["prediction_error"] = (
                f"Trained model file not found at {MODEL_PATH}. "
                "Add the checkpoint described in the deployment guide."
            )
        else:
            try:
                with st.spinner("Loading the model and analyzing the image..."):
                    model = load_model(
                        str(MODEL_PATH),
                        MODEL_PATH.stat().st_mtime_ns,
                    )
                    scores = make_prediction(model, raw_image)
                order = np.argsort(scores)[::-1]
                st.session_state["prediction"] = {
                    "image_digest": image_digest,
                    "filename": uploaded.name,
                    "label": LABELS[int(order[0])],
                    "scores": {
                        LABELS[index]: float(scores[index])
                        for index in range(len(LABELS))
                    },
                }
                st.session_state.pop("prediction_error", None)
            except Exception as exc:
                st.session_state["prediction"] = None
                st.session_state["prediction_error"] = str(exc)

with right:
    with st.container(border=True):
        st.markdown('<div class="card-kicker">02 / Model output</div>', unsafe_allow_html=True)
        st.subheader("Prediction")
        result = st.session_state.get("prediction")
        prediction_error = st.session_state.get("prediction_error")

        if prediction_error:
            st.error(prediction_error)
            if not MODEL_PATH.is_file():
                st.markdown(
                    f"Follow the model export and setup guide, then place the "
                    f"checkpoint at {MODEL_PATH}."
                )
                st.link_button("Open training notebook", NOTEBOOK_URL)
        elif result:
            top_score = result["scores"][result["label"]]
            st.markdown(
                f"""
                <div class="result-label">Predicted category</div>
                <div class="result-name">{result['label'].replace('_', ' ')}</div>
                <div class="result-score">Top model score · {top_score:.1%}</div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown("**Scores by category**")
            for label in LABELS:
                score = result["scores"][label]
                st.progress(score, text=f"{label.replace('_', ' ')} · {score:.1%}")

            report = json.dumps(
                {
                    "image_filename": result["filename"],
                    "predicted_category": result["label"],
                    "model_scores": result["scores"],
                    "note": "Research output only; not a medical diagnosis.",
                },
                indent=2,
            ).encode("utf-8")
            st.download_button(
                "Download result summary",
                data=report,
                file_name="xray-model-result.json",
                mime="application/json",
                width="stretch",
            )
        else:
            st.markdown(
                """
                <div class="empty-state">
                  <div class="empty-icon">◎</div>
                  <div><strong>Your result will appear here</strong></div>
                  <div style="margin-top:0.35rem">Upload an image, then select Analyze X-ray.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.write("")
st.warning(
    "Research and education only. This model can be wrong and is not validated "
    "for clinical use. Do not use its output to make medical decisions."
)

with st.expander("How this demo handles your image"):
    st.write(
        "The uploaded image is decoded in memory, converted from BGR to RGB, "
        "resized to 224 × 224 with OpenCV, and scaled to the 0–1 range used "
        "by the model's training notebook. This app does not write "
        "the uploaded image to disk. The JSON download contains the filename "
        "and model scores, not the image."
    )

st.markdown(
    f"""
    <div class="footer">
      Model: MobileNetV2 · Source notebook:
      <a href="{NOTEBOOK_URL}" target="_blank" rel="noreferrer">open in Google Colab</a>
      · Educational prototype
    </div>
    """,
    unsafe_allow_html=True,
)

