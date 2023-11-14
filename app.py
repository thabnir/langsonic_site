from flask import Flask, render_template, request
import tensorflow as tf
import mp3tospect
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model("cnn.h5")
if model is not None:
    model.test_on_batch(np.zeros((1, 13, 250, 1)), np.zeros((1, 5)))


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
        filepath = f"./data/{file.filename}"
        print(filepath)
        file.save(filepath)
        prediction = predict_from_mp3(filepath)
        # delete the file
        # os.remove(filepath)
        # convert to a list of tuples, where each tuple is (language, probability)
        prediction = list(zip(langs, prediction))
        # sort them by probability
        print(prediction)
        prediction.sort(key=lambda x: x[1], reverse=True)
        return render_template("results.html", prediction=prediction)

    # TODO: handle error
    print("Something went wrong")
    return render_template("index.html", error="Something went wrong")


def predict_from_mp3(path):
    """
    Args:
        path: path to the mp3 file

    Returns:
        list containing the probabilities for each language
    """
    return model.predict(np.array([mp3tospect.model_input_from_audio(path)])).tolist()[
        0
    ]


from languages import LANGUAGES

langs = ["de", "en", "es", "fr", "it"]


def get_langcode(prediction):
    return langs[np.argmax(prediction)]


def get_language(prediction):
    return LANGUAGES[get_langcode(prediction)]


def do_tests():
    # get mp3 files in data directory
    mp3path = "../data/mp3/"
    paths = [mp3path + lang + "/" for lang in langs]

    # get the first 20 files from each language directory
    all_files = []
    for path in paths:
        # list all files in directory, filter for mp3 files, take the first 20
        files = [
            f
            for f in os.listdir(path)[:20]
            if os.path.isfile(os.path.join(path, f)) and f.endswith(".mp3")
        ]
        # append the full path to the files
        all_files.extend([os.path.join(path, file) for file in files])

    # get the predictions for each file
    predictions = [get_langcode(predict_from_mp3(path)) for path in all_files]

    # match language based on the path name, for example if it's in `../data/mp3/it/` it's Italian, 'it'
    actual = [path.split("/")[-2] for path in all_files]

    # pair predictions with their corresponding actual language
    results = list(zip(predictions, actual))

    # print or return the results
    correct_predictions = {lang: 0 for lang in langs}
    total_predictions = {lang: 0 for lang in langs}

    # Count correct predictions for each language
    for prediction, truth in results:
        if prediction == truth:
            correct_predictions[truth] += 1
        total_predictions[truth] += 1

    # Calculate and print out percent accuracy by language
    accuracy_by_language = {
        lang: (correct_predictions[lang] / total_predictions[lang]) * 100
        if total_predictions[lang] > 0
        else 0
        for lang in langs
    }

    for lang in sorted(accuracy_by_language.keys()):
        print(f"{lang.upper()} Accuracy: {accuracy_by_language[lang]:.2f}%")


# do_tests()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5678)


# gunicorn version of the app below
# doesn't work beacuse of ssl issues.
# idk how to fix it
# there are tons of bugs with gunicorn in general
# so I'm just going to use the flask dev server for now

# https://stackoverflow.com/questions/50236117/ssl-error-in-gunicorn-when-using-flask

# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

# from gunicorn.app.base import BaseApplication


# class StandaloneApplication(BaseApplication):
#     def __init__(self, app, options=None):
#         self.options = options or {}
#         self.application = app
#         super().__init__()

#     def load_config(self):
#         for key, value in self.options.items():
#             if key in self.cfg.settings and value is not None:
#                 self.cfg.set(key.lower(), value)

#     def load(self):
#         return self.application


# options = {
#     "bind": "0.0.0.0:5678",  # Specify the interface and port
#     "workers": 4,  # Specify the number of worker processes
#     "debug": True,
#     "certfile": "cert.pem",
#     "keyfile": "key.pem",
# }

# if __name__ == "__main__":
#     StandaloneApplication(app, options).run()
