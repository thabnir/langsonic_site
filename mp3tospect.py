import numpy as np
import librosa
import resampy
import soundfile as sf
from PIL import Image


IMG_DIM = (13, 250, 1)


def model_input_from_audio(
    filepath,
    target_size=(13, 1000),
):
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def img_to_array_in_memory(spec):
        # scale, otherwise it's from -80 to 80 (or something) (cause decibels)
        scaled_spec = scale_minmax(spec, 0, 255).astype(np.uint8)

        img = np.flip(scaled_spec, axis=0)  # Low frequencies at the bottom

        img_pil = Image.fromarray(np.uint8(img))

        img_pil = img_pil.resize((IMG_DIM[1], IMG_DIM[0])).convert("L")
        # crunch it to 250x13

        img_pil.save("./temp/temp.png")

        # return the image as a numpy array in the range [0, 1]
        return np.array(img_pil) / 255.0

    # Load and resample the audio file
    # based on defaults for WhisperFeatureExtractor
    # in order to provide the same input as the model was trained on
    sr_new = 16000
    hop_length = 160
    n_fft = 400

    signal, sr = librosa.load(filepath, res_type="kaiser_fast")
    # print(f"Trying to load {filepath}")
    # signal, sr = sf.read(filepath)

    signal = resampy.resample(signal, sr_orig=sr, sr_new=sr_new, res_type="kaiser_fast")

    spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr_new,
        n_fft=n_fft,
        win_length=n_fft,
        n_mels=target_size[0],
        hop_length=hop_length,
    )

    input_features = librosa.power_to_db(spectrogram, ref=np.max)

    current_length = input_features.shape[1]  # aiming for 1000

    # Pad or crop accordingly
    if current_length < target_size[1]:
        # Pad with silence to the right
        pad_width = ((0, 0), (0, target_size[1] - current_length))
        input_features = np.pad(
            input_features,
            pad_width=pad_width,
            mode="constant",
            constant_values=-80.0,  # -80.0dB is the minimum value for the spectrogram. padding with silence
        )
    elif current_length > target_size[1]:
        input_features = input_features[:, : target_size[1]]  # crop

    return img_to_array_in_memory(input_features)
