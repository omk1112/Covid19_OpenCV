<div align="center">

# COVID-19 Chest X-ray Classifier

### A three-class OpenCV + MobileNetV2 web app

Upload a chest X-ray and view the model's scores for **COVID**, **Normal**, and **Viral Pneumonia**.

<p>
  <img src="https://img.shields.io/badge/OpenCV-preprocessing-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/TensorFlow-MobileNetV2-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-web%20app-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

</div>

> **Research and education only.** This model is not a medical device. Do not use its output to diagnose, screen, or guide treatment.

## Live demo

A public app URL has not been created yet. The app needs to be deployed on a hosting service before a live link can be listed here. See [Deploy the app](#deploy-the-app).

## Use the website

The app accepts PNG, JPG, JPEG, and BMP images up to 10 MB. It shows the predicted category and scores for all three classes. The app processes the image in memory and does not save the X-ray to disk.

To run it locally from the repository root:

    python -m pip install -r requirements.txt
    streamlit run app.py

The app loads the checkpoint from:

    models/covid_detection_model.keras

## Deploy the app

1. Make sure the checkpoint file is present at the path above.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from this GitHub repository.
3. Select the main branch and <code>app.py</code> as the entry point.
4. After Streamlit gives the app a public URL, replace the live demo note above with that URL.

See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Model details

| Item | Value |
|---|---|
| Input | 224 × 224 RGB image |
| Pixel scaling | 0–1 |
| Architecture | ImageNet-pretrained MobileNetV2, Dense(128, ReLU), Dropout(0.5), Dense(3, softmax) |
| Class index 0 | COVID |
| Class index 1 | Normal |
| Class index 2 | Viral Pneumonia |

The website preprocessing follows the three-class model's training notebook: decode as RGB, resize with OpenCV, then divide pixel values by 255. The repository also contains a separate four-class notebook, <a href="https://github.com/omk1112/Covid19_OpenCV/blob/main/COVID19_Computer_Vision_Project.ipynb">COVID19_Computer_Vision_Project.ipynb</a>. That notebook predicts a different set of labels; its model and metrics do not describe this web app.

The three-class checkpoint was supplied separately. Its matching training notebook is not included in this repository.

## Recorded validation results

The matching three-class training notebook reports these results on its 3,032-image validation split:

| Class | Precision | Recall | F1 score | Images |
|---|---:|---:|---:|---:|
| COVID | 91% | 84% | 87% | 724 |
| Normal | 93% | 97% | 95% | 2,039 |
| Viral Pneumonia | 97% | 87% | 92% | 269 |
| **Overall accuracy** |  |  | **93.01%** | **3,032** |

These are saved validation results from one training run, not a guarantee for new images or clinical use.

## Responsible use

This model can make mistakes, and validation performance does not establish clinical safety. Do not use it to make medical decisions. Avoid uploading identifiable or private medical images to public services.

