# Model checkpoint

Place the trained Keras checkpoint here with this exact filename:

    covid_detection_model.keras

The web app loads this file by default. You can also set the
<code>COVID19_MODEL_PATH</code> environment variable to point to a checkpoint
elsewhere.

## Export the checkpoint from Colab

The matching three-class training notebook saves this file after training. To
download it from Colab, run:

    from google.colab import files
    files.download("/content/covid_detection_model.keras")

Copy the downloaded <code>covid_detection_model.keras</code> file into this
<code>models</code> directory. Its three output classes are ordered as COVID,
Normal, and Viral Pneumonia. The class order and preprocessing match the
three-class training notebook.

