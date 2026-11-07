# Model checkpoint

Place the trained Keras checkpoint here with this exact filename:

    mobilenetv2_best.keras

The web app loads this file by default. You can also set the
<code>COVID19_MODEL_PATH</code> environment variable to point to a checkpoint
elsewhere.

## Export the checkpoint from Colab

After training the notebook and loading the best checkpoint, run:

    from google.colab import files
    import tensorflow as tf

    exported_path = "/content/mobilenetv2_best.keras"
    model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
    model.save(exported_path)
    files.download(exported_path)

Copy the downloaded <code>mobilenetv2_best.keras</code> file into this
<code>models</code> directory. The checkpoint is not included in the source
repository, so predictions remain disabled until the file is present.

