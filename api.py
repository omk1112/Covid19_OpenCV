"""Small web API that serves the trained COVID-19 chest X-ray model.

Run:  uvicorn api:app --port 8000
Needs mobilenetv2_best.keras in the same folder (or set MODEL_PATH).
"""
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = Path(os.getenv("MODEL_PATH", "mobilenetv2_best.keras"))
# Same order as CLASS_NAMES in the notebook. Do not reorder.
CLASS_NAMES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
MAX_BYTES = 10 * 1024 * 1024

model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(title="COVID-19 chest X-ray classifier")
app.add_middleware(
    CORSMiddleware,
    # Set ALLOWED_ORIGINS to your site, e.g. https://omk1112.github.io
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def preprocess(raw: bytes) -> tf.Tensor:
    """Mirror the notebook's training pipeline: grayscale, 224x224, RGB, [-1, 1]."""
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("not a readable image")
    # Training used tf.image.resize (bilinear), so use the same call here.
    image = tf.image.resize(image[..., None], [224, 224], method="bilinear")
    image = tf.image.grayscale_to_rgb(image)
    image = preprocess_input(image)
    return tf.expand_dims(image, 0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    raw = file.file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    try:
        batch = preprocess(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="Could not read that file as an image.")

    probs = model.predict(batch, verbose=0)[0]
    top = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[top],
        "confidence": float(probs[top]),
        "scores": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }
