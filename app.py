from flask import Flask, render_template, request
import tensorflow as tf
import audiotospect
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model("cnn.h5")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("index.html", error="No file part")

    file = request.files["file"]
    print(f"Got file: `{file}`")

    if file.filename == "":
        return render_template("index.html", error="No selected file")

    if file:
        filepath = f"./temp/{file.filename}"
        os.makedirs("./temp/", exist_ok=True)
        print(filepath)
        file.save(filepath)
        # delete the file
        # os.remove(filepath)

        prediction = predict_from_audio(filepath)

        # convert to percent
        prediction = [prediction * 100 for prediction in prediction]

        # convert from language code to language name
        languages = [LANGUAGES[lang] for lang in langs]

        # convert to a list of tuples, where each tuple is (language, probability)
        prediction = list(zip(languages, prediction))

        # sort them by probability
        print(prediction)
        prediction.sort(key=lambda x: x[1], reverse=True)
        return render_template("results.html", prediction=prediction)

    # TODO: handle error
    print("Something went wrong")
    return render_template("index.html", error="Something went wrong")


def predict_from_audio(path):
    """
    Args:
        path: path to the audio file

    Returns:
        list containing the probabilities for each language
    """
    model_in = audiotospect.audio_to_spect(path)
    return model.predict(np.array([model_in])).tolist()[0]


from languages import LANGUAGES

langs = ["de", "en", "es", "fr", "it"]


def get_langcode(prediction):
    return langs[np.argmax(prediction)]


def get_language(prediction):
    return LANGUAGES[get_langcode(prediction)]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5678))  # 5678 is the default port
    app.run(debug=True, host="0.0.0.0", port=port)
