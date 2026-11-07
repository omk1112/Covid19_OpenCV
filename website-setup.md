# Xray Vision web app

This Streamlit app gives viewers an upload-and-predict interface for the
three-class MobileNetV2 checkpoint. It mirrors the matching training notebook's
OpenCV preprocessing and returns scores for COVID, Normal, and Viral Pneumonia.

## Add it to the GitHub repository

The app source and trained checkpoint are included in the repository. The
checkpoint is stored at <code>models/covid_detection_model.keras</code>.
<code>models/README.md</code> describes how to export it again from Colab.

The expected layout is:

    app.py
    requirements.txt
    .streamlit/config.toml
    models/covid_detection_model.keras

The app and checkpoint expect the ordered classes COVID, Normal, and Viral
Pneumonia. The separate four-class notebook in the repository does not produce
this three-class checkpoint.

## Run locally

Use Python 3.11, then install the dependencies and start Streamlit:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt
    streamlit run app.py

For macOS or Linux, activate the environment with:

    source .venv/bin/activate

If the checkpoint is stored outside the default <code>models</code> folder,
set <code>COVID19_MODEL_PATH</code> to its full path before starting the app.

## Deploy

Push the app files and checkpoint to a GitHub repository, then create an app
from that repository on [Streamlit Community Cloud](https://share.streamlit.io/).
Choose <code>app.py</code> as the application entry point. The first build
installs TensorFlow and may take a few minutes.

## Prediction flow

The app accepts PNG, JPG, JPEG, and BMP images up to 10 MB. It decodes the
image with OpenCV, converts BGR to RGB, resizes it to 224 × 224, and scales
pixels to the 0–1 range used by this model's training notebook. The three
output scores are displayed and can be downloaded as JSON.

The app processes the upload in memory and does not save the X-ray to disk.
The result file contains the upload filename and class scores, not image data.

> This is an educational research demo, not a medical device. Its output must
> not be used for diagnosis, screening, or treatment decisions.

