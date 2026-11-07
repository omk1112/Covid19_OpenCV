<div align="center">

# COVID-19 Chest X-ray Classifier

### An OpenCV + MobileNetV2 research project

Classifies chest X-ray images into **COVID**, **Normal**, **Lung Opacity**, or **Viral Pneumonia**. OpenCV handles image loading and preprocessing; the classifier is a TensorFlow/Keras MobileNetV2 model.

<p>
  <a href="https://colab.research.google.com/github/omk1112/Covid19_OpenCV/blob/main/COVID19_Computer_Vision_Project.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open the notebook in Google Colab">
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-image%20preprocessing-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/TensorFlow-MobileNetV2-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
</p>

<p><a href="#run-the-website">Run the website</a> · <a href="#quick-start">Train in Colab</a> · <a href="#results">Results</a> · <a href="#how-it-works">How it works</a></p>

</div>

> **Research and education only.** This project is not a medical device and must not be used to diagnose, screen, or guide treatment. A model score is not a clinical diagnosis.

## Project at a glance

| | |
|---|---|
| **Task** | Four-class chest X-ray image classification |
| **Input** | Chest X-ray image, resized to 224 × 224 |
| **Classes** | COVID · Normal · Lung Opacity · Viral Pneumonia |
| **Model** | ImageNet-pretrained MobileNetV2 with a four-class softmax head |
| **Data split** | 14,777 train · 3,167 validation · 3,167 test images |
| **Notebook** | [COVID19_Computer_Vision_Project.ipynb](https://github.com/omk1112/Covid19_OpenCV/blob/main/COVID19_Computer_Vision_Project.ipynb) |

## Quick start

1. Select **Open in Colab** above.
2. In Colab, choose **Runtime → Change runtime type** and select a GPU if one is available. A GPU is optional, but training can take a long time on a CPU.
3. Run the notebook sections in order. The notebook downloads the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) with KaggleHub (the archive was about 778 MB in the recorded run), prepares the data, trains the model, and evaluates it.
4. After training and loading the best checkpoint, use the code below to upload one image and view the predicted class and scores.

The notebook uses Google Colab paths and Colab download helpers, so Colab is the easiest way to run it without adapting the environment.

The training callback setup is placed before the model training cell so the notebook can run top-to-bottom in a fresh Colab runtime.

## Run the website

The repository includes a Streamlit upload-and-predict interface. It accepts a chest X-ray image and displays the predicted class with scores for all four categories.

Before using predictions, export the trained model from Colab. After the notebook has trained and loaded the best checkpoint, run:

    from google.colab import files
    files.download(str(BEST_MODEL_PATH))

Copy the downloaded <code>mobilenetv2_best.keras</code> file into the repository's <code>models</code> folder. The website expects this path:

    models/mobilenetv2_best.keras

Then start the site locally from the repository root:

    python -m pip install -r requirements.txt
    streamlit run app.py

To make the app public, deploy the GitHub repository with [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and select <code>app.py</code> as the entry point. The app source is ready, but it cannot return predictions until the trained checkpoint is added.

The app processes uploaded images in memory and does not save the X-ray to disk. Its result download contains the filename and class scores only.

## Try one X-ray

Run this in a Colab cell **after the model has been trained and the best checkpoint loaded** (the notebook's Phase 5). The image is processed with the same OpenCV and MobileNetV2 preprocessing used by the project.

    from google.colab import files
    import cv2
    import numpy as np

    uploaded_files = files.upload()
    image_path = next(iter(uploaded_files))

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not open the uploaded image.")

    image = cv2.resize(
        image,
        (IMG_WIDTH, IMG_HEIGHT),
        interpolation=cv2.INTER_AREA
    )
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    image = preprocess_input(image)

    scores = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    ranking = np.argsort(scores)[::-1]

    print(f"Predicted class: {CLASS_NAMES[int(ranking[0])]}")
    print(f"Model score: {scores[ranking[0]]:.1%}")
    print("\nScores by class:")
    for class_id in ranking:
        print(f"  {CLASS_NAMES[int(class_id)]:16s} {scores[class_id]:.1%}")

The displayed score is the model's softmax output. It is not a measure of medical certainty.

## Results

The notebook's saved evaluation reports the following on its 3,167-image held-out test split:

| Metric | Recorded result |
|---|---:|
| Test accuracy | **86.96%** |
| Macro one-vs-rest ROC-AUC | **0.9754** |
| Weighted one-vs-rest ROC-AUC | **0.9663** |

| Class | Precision | Recall | F1 score |
|---|---:|---:|---:|
| COVID | 87.57% | 85.61% | 86.58% |
| Normal | 91.33% | 85.42% | 88.27% |
| Lung Opacity | 79.70% | 88.36% | 83.81% |
| Viral Pneumonia | 90.19% | 96.02% | 93.01% |

These are results saved in the notebook, not a guarantee of performance on new images. Results can vary with the runtime, training, data, and image source. See the notebook for its classification report, confusion matrices, and ROC curves.

## How it works

1. **Load and inspect data:** fetch the four-class radiography dataset and check image readability and class counts.
2. **Prepare splits:** remove exact duplicate images, then create stratified training, validation, and test splits.
3. **Preprocess images:** read with OpenCV, resize to 224 × 224, convert grayscale to three channels, and apply MobileNetV2 preprocessing.
4. **Train:** use an ImageNet-pretrained MobileNetV2 feature extractor with a global average pooling layer, dropout, and a four-class softmax output. Class weights help account for uneven class sizes.
5. **Evaluate:** report accuracy, per-class metrics, ROC-AUC, confusion matrices, and example errors.

## Data and responsible use

The notebook downloads the [COVID-19 Radiography Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). Review the dataset's terms and attribution requirements before reusing or redistributing its images.

This is an educational computer-vision experiment. It may make errors, and its performance on the notebook's test split does not establish clinical safety or suitability. Do not upload identifiable or private medical images to a public notebook or service.

