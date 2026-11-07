# COVID-19 Chest X-ray Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNetV2-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-preprocessing-5C3EE8?logo=opencv&logoColor=white)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omk1112/Covid19_OpenCV/blob/main/COVID19_Computer_Vision_Project.ipynb)

Upload a chest X-ray and the model sorts it into one of four classes: **COVID-19**, **Normal**, **Lung opacity** or **Viral pneumonia**, with a confidence score for each. Images are prepared with OpenCV and classified by a MobileNetV2 network.

> **Research demo only.** This project is for education and research. It cannot confirm or rule out COVID-19 and is not a medical device.

<!-- TODO: add a screenshot of the website, e.g. ![Screenshot](docs/screenshot.png) -->

## Live demo

<!-- TODO: after publishing, check this link works -->
**https://omk1112.github.io/Covid19_OpenCV/**

Drop in a frontal chest X-ray (JPG, PNG or WebP, up to 10 MB) and choose **Analyse image**.

## Results

Measured on 3,167 held-out test images.

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| COVID-19 | 87.6% | 85.6% | 86.6% | 535 |
| Normal | 91.3% | 85.4% | 88.3% | 1,529 |
| Lung opacity | 79.7% | 88.4% | 83.8% | 902 |
| Viral pneumonia | 90.2% | 96.0% | 93.0% | 201 |

Overall test accuracy is **87.0%** and macro AUC is **0.975**.

## How it works

1. **Read and preprocess.** OpenCV reads the X-ray as grayscale. It is resized to 224 × 224, copied to three channels and scaled to the range −1 to 1.
2. **Classify.** MobileNetV2 (pre-trained on ImageNet, frozen) feeds a global average pooling layer, 30% dropout and a four-way softmax layer. Only the final layer was trained, with class weights to balance the data.
3. **Report.** The most likely class is shown with a score for all four classes.

**Data:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (21,165 images, 21,111 after removing exact duplicates), split 70% training, 15% validation, 15% testing. Please check the dataset's license and citation requirements before reusing it.

## What's in this repository

| File | Purpose |
|---|---|
| `COVID19_Computer_Vision_Project.ipynb` | Data exploration, preprocessing, training and evaluation (written for Google Colab) |
| `mobilenetv2_best.keras` | The trained model |
| `api.py` | Web API that runs the model |
| `index.html` | The website |

## Run it on your computer

1. **Get the code.**
   ```bash
   git clone https://github.com/omk1112/Covid19_OpenCV.git
   cd Covid19_OpenCV
   ```
2. **Install the requirements.**
   ```bash
   pip install fastapi uvicorn python-multipart opencv-python-headless numpy "tensorflow>=2.16"
   ```
3. **Start the API.**
   ```bash
   uvicorn api:app --port 8000
   ```
   Check `http://localhost:8000/health` in your browser. It should show `{"status":"ok"}`.
4. **Open the website.** Double-click `index.html`, or run `python -m http.server 5500` in a second terminal and visit `http://localhost:5500`.
5. **Try it.** Drop in a chest X-ray and choose **Analyse image**.

## Put it online

The website is a static page, so GitHub Pages can host it. The model needs Python, so the API is hosted separately. The steps below use Hugging Face Spaces, which has a free tier.

### 1. Host the API

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) and choose the **Docker** SDK.
2. Upload `api.py` and `mobilenetv2_best.keras` to the Space.
3. Add a file named `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN pip install --no-cache-dir fastapi uvicorn python-multipart opencv-python-headless numpy "tensorflow-cpu>=2.16"
   COPY api.py mobilenetv2_best.keras ./
   EXPOSE 7860
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
4. Make sure the Space's `README.md` starts with:
   ```yaml
   ---
   title: COVID-19 X-ray API
   sdk: docker
   app_port: 7860
   ---
   ```
5. Under **Settings → Variables**, add `ALLOWED_ORIGINS` with the value `https://omk1112.github.io`.
6. When the build finishes, your API address is `https://<your-username>-<space-name>.hf.space/predict`.

### 2. Point the website at your API

In `index.html`, find the `CONFIG` block near the bottom and change `API_URL`:

```js
const CONFIG = {
  API_URL: "https://<your-username>-<space-name>.hf.space/predict",
  // ...
};
```

### 3. Publish the website

1. Commit `index.html` to the root of this repository.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**, select `main` and `/ (root)`, then save.
4. After a minute, the site is live at `https://omk1112.github.io/Covid19_OpenCV/`.

## API

| | |
|---|---|
| `POST /predict` | Multipart form with one field named `file` (the image) |
| `GET /health` | Returns `{"status":"ok"}` |

Example response:

```json
{
  "label": "COVID",
  "confidence": 0.92,
  "scores": { "COVID": 0.92, "Normal": 0.02, "Lung_Opacity": 0.05, "Viral Pneumonia": 0.01 }
}
```

## Limitations

- The model missed about 1 in 7 COVID-19 images in testing (85.6% recall).
- The test images come from the same public dataset as the training images. Performance on X-rays from other hospitals, machines or patient groups is likely lower.
- It only understands frontal chest X-rays. Any other image will still get a prediction, but it will be meaningless.
- It has not been clinically validated. Do not use it to make medical decisions. If you have symptoms or health concerns, speak to a qualified clinician.

## License

<!-- TODO: add a license, e.g. MIT -->
