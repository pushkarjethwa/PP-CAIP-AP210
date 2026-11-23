"""Day 1 | Slides 21 | Represent tables, text, images, audio, and video (starter)."""

# %% Setup

import sys

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from sklearn.feature_extraction.text import CountVectorizer

texts = ["red red blue", "blue green"]
image = np.array([[0, 0, 255], [0, 255, 255]], dtype=np.uint8)
sample_rate = 1024

# Fill each TODO in order. Run the file after you add each step.

# An untouched starter only loads the setup and lists the steps.

# %% 1. Structured tables

# TODO Build a DataFrame with two named columns and print it with its dtypes.
print("\nStep 1: Structured tables")

# %% 2. Text tokens, counts, and embeddings

# TODO Print tokens, count vocabulary terms, then stack a two-number vector for
#      each token.
print("\nStep 2: Text tokens, counts, and embeddings")

# %% 3. Image pixels and flattening

# TODO Print image.shape and the result of image.ravel().
print("\nStep 3: Image pixels and flattening")

# %% 4. Audio samples and spectrogram

# TODO Generate a 128 Hz sine wave and compute a spectrogram for the waveform.
print("\nStep 4: Audio samples and spectrogram")

# %% 5. Video frames and a visual check

# TODO Stack two image frames. If --plot is present, display the pixel grid and
#      spectrogram.
print("\nStep 5: Video frames and a visual check")
