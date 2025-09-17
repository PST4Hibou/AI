import os
import librosa
import numpy as np


def load_hibou_audio_files(directory, sr=44100, n_mfcc=13):
    features = []
    labels = []
    class_names = sorted(os.listdir(directory))

    for class_name in class_names:
        for filename in os.listdir(os.path.join(directory, class_name)):
            if filename.endswith(".wav"):
                filepath = os.path.join(directory, class_name, filename)
                audio, _ = librosa.load(filepath, sr=sr)
                features.append(audio)
                labels.append(class_name)

    return np.array(features), np.array(labels)