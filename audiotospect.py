import numpy as np
import librosa
import resampy


def audio_to_spect(
    audiopath: str,
    target_size: tuple[int, int] = (13, 1000),
    img_dimensions: tuple[int, int, int] = (13, 250, 1),
    output_filepath="./temp/temp.png",
    save=True,
    sr_new=16000,
    hop_length=160,
    n_fft=400,
):
    """
    Based on defaults for WhisperFeatureExtractor in order to provide the same input as the model was trained on
    """

    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def img_to_array_in_memory(spec, filepath="./temp/temp.png", save=False):
        # scale, otherwise it's from -80 to 80 (or something) (cause decibels)
        scaled_spec = scale_minmax(spec, 0, 255).astype(np.uint8)

        img = np.flip(scaled_spec, axis=0)  # Low frequencies at the bottom

        # Convert to grayscale and resize
        img_np = np.uint8(img)
        img_np = np.resize(img_np, img_dimensions)

        # return the image as a numpy array in the range [0, 1]
        return img_np / 255.0

    # TODO: figure out the error with this (it still works though)
    # print(f"Trying to load {filepath}")
    signal, sr = librosa.load(audiopath, res_type="kaiser_fast")
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

    return img_to_array_in_memory(input_features, output_filepath, save=save)
