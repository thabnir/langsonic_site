import numpy as np
import librosa
import resampy
import os
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
from transformers import WhisperFeatureExtractor


class CustomWhisperFeatureExtractor(WhisperFeatureExtractor):
    def __init__(self, feature_size=13, **kwargs):
        super().__init__(feature_size=feature_size, **kwargs)


feature_extractor = CustomWhisperFeatureExtractor(
    feature_size=13, sampling_rate=16000, padding_value=0.0, return_attention_mask=False
)


IMAGE_DIMENSIONS = (13, 250, 1)


def model_input_from_audio(
    filepath,
    target_size=(13, 1000),
    feature_extractor=feature_extractor,
):
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def img_to_array_in_memory(spec):
        # Scale the spectrogram as we did before
        scaled_spec = scale_minmax(spec, 0, 255).astype(np.uint8)
        img = np.flip(scaled_spec, axis=0)  # Low frequencies at the bottom

        # TODO: make this less dumb
        # i'm literally just saving the image to disk and then reading it back in
        Image.fromarray(img).save("./data/temp.png")
        # end mydat code section

        # begin realcnn input code section

        img_pil = image.img_to_array(
            image.load_img(
                "./data/temp.png",
                target_size=(IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 3),
            ).convert("L")
        )
        # os.remove("./data/temp.png")

        img_pil = img_pil.reshape(IMAGE_DIMENSIONS)
        img_pil = img_pil / 255

        # buffer = BytesIO()
        # img_pil.save(buffer, format="PNG")
        # buffer.seek(0)

        # img_array = image.img_to_array(image.load_img(buffer, color_mode="grayscale"))
        # img_array = img_array.reshape((target_size[0], target_size[1], 1))
        # img_array = img_array / 255.0  # scale to [0, 1]
        img_array = np.array(img_pil)
        return img_array

    # Load and resample the audio file
    signal, sr = librosa.load(filepath)
    signal = resampy.resample(signal, sr_orig=sr, sr_new=16000, res_type="kaiser_fast")

    # Extract features using the feature extractor
    f = feature_extractor(
        signal,
        sampling_rate=16000,
        padding="max_length",  # pads to 30 seconds
        do_normalize=True,
        feature_size=target_size[0],
        return_attention_mask=False,
    )
    input_features = np.array(f["input_features"])[0]
    input_features = input_features[:, : target_size[1]]

    # Convert the spectrogram to an image array and return
    return img_to_array_in_memory(input_features)


def audio_to_img(path, height=13):
    signal, sr = librosa.load(path)
    signal = resampy.resample(signal, sr_orig=sr, sr_new=16000, res_type="kaiser_fast")

    f = feature_extractor(
        signal,
        sampling_rate=16000,
        padding="max_length",  # pads to 30 seconds
        do_normalize=True,
        feature_size=height,
        return_attention_mask=False,
        # hop_length=hl,
    )

    input_features = np.array(f["input_features"])[0]
    # truncate to width of 1000
    input_features = input_features[:, :1000]

    fname = os.path.basename(path).rsplit(".")[0]
    return input_features, fname
