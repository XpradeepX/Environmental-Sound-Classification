import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram(audio_file, output_path):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Generate a mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to decibels (log scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot and save the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, fmax=8000)
    plt.savefig(output_path)
    plt.close()

# Path to the UrbanSound8K dataset
dataset_path = r"E:/archive"

# Output path for saving spectrograms
output_path = r"E:/Spect/fold10"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            output_file = os.path.join(output_path, f"{file.replace('.wav', '.png')}")
            create_spectrogram(file_path, output_file)
            print(f"Spectrogram created for: {file}")

