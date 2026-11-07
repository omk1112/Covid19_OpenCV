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

<p><a href="#quick-start">Quick start</a> · <a href="#try-one-x-ray">Try one X-ray</a> · <a href="#results">Results</a> · <a href="#how-it-works">How it works</a></p>

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

### Fresh-runtime training note

The current notebook places the <code>BEST_MODEL_PATH</code> and <code>callbacks</code> setup cell after the <code>model.fit(...)</code> cell, and that setup cell has extra leading spaces. In a fresh runtime, move that setup above the training cell and remove the leading spaces before selecting **Run all**. Otherwise, training can fail because <code>callbacks</code> has not been defined yet.

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

## What this repository currently provides

The repository contains the training and evaluation notebook. It does **not** currently include a hosted website, prediction API, or a pretrained model file. Viewers can run the notebook in Colab and try a single image there after training.

This README is a project landing page; it does not itself run model inference. A public upload-and-predict website would also need a saved model artifact and a deployed inference service connected to a web interface.

## Data and responsible use

The notebook downloads the [COVID-19 Radiography Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). Review the dataset's terms and attribution requirements before reusing or redistributing its images.

This is an educational computer-vision experiment. It may make errors, and its performance on the notebook's test split does not establish clinical safety or suitability. Do not upload identifiable or private medical images to a public notebook or service.
