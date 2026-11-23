"""Day 1 | Slides 21 | Represent tables, text, images, audio, and video (complete solution)."""

# %% Setup

import sys

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from sklearn.feature_extraction.text import CountVectorizer

texts = ["red red blue", "blue green"]
image = np.array([[0, 0, 255], [0, 255, 255]], dtype=np.uint8)
sample_rate = 1024

# %% 1. Structured tables

print("\nStep 1: Structured tables")
table = pd.DataFrame({"letter": ["a", "b"], "value": [3, 5]})
print(table)
print(table.dtypes)

# %% 2. Text tokens, counts, and embeddings

print("\nStep 2: Text tokens, counts, and embeddings")
tokens = texts[0].split()
counts = CountVectorizer()
matrix = counts.fit_transform(texts)
print("Tokens:", tokens)
print("Vocabulary:", counts.get_feature_names_out().tolist())
print("Word counts:\n", matrix.toarray())
embedding = {"red": [1.0, 0.0], "blue": [0.8, 0.2], "green": [0.1, 0.9]}
print("Lookup vectors:\n", np.array([embedding[token] for token in tokens]))

# %% 3. Image pixels and flattening

print("\nStep 3: Image pixels and flattening")
print("Image shape:", image.shape)
print("Flattened pixels:", image.ravel().tolist())

# %% 4. Audio samples and spectrogram

print("\nStep 4: Audio samples and spectrogram")
seconds = np.arange(128) / sample_rate
wave = np.sin(2 * np.pi * 128 * seconds)
frequencies, times, power = spectrogram(wave, fs=sample_rate, nperseg=32, noverlap=16)
print("Audio samples:", wave.shape, "Spectrogram shape:", power.shape)
print("Dominant frequency:", frequencies[power.mean(axis=1).argmax()], "Hz")

# %% 5. Video frames and a visual check

print("\nStep 5: Video frames and a visual check")
video = np.stack([image, np.fliplr(image)])
print("Video shape (frames, rows, columns):", video.shape)

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    axes[0].set(title="Image pixels", xlabel="Column", ylabel="Row")
    axes[1].pcolormesh(times, frequencies, power, shading="auto")
    axes[1].set(title="Audio spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.tight_layout()
    plt.show()
